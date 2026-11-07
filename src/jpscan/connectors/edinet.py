"""EDINET API v2 连接器（规划文档 3.2、10.5、19.1 节；附录 G1-30、G1-31、G1-32）。

依据：EDINET API 仕様書 Version 2（2026 年 6 月版）。
- 书类一览 `documents.json?date=YYYY-MM-DD&type=2`；书类取得 `documents/{docID}?type=1..5`。
- 出错时 HTTP 多半仍是 200：书类一览看 `metadata.status`；401/429 的 JSON 是顶层 `StatusCode`；
  书类取得要看 Content-Type（成功为 octet-stream / pdf，失败为 json）。
- 规约禁止短时间大量访问：单线程、≥1 秒一次，429 时充分间隔后再试。
- 内容按公共データ利用規約（PDL1.0）使用，引用时注明出典「EDINET（金融庁）」。

密钥安全：API Key 只能放在查询参数里，而 requests 的异常文本、`Response.url`、urllib3 的 DEBUG 日志都带完整 URL。
因此这里不用 base.get_with_retry（它会把异常文本写进日志），带密钥的请求一律走 `_get`：
异常文本先脱敏再抛出，且不保留原异常链；禁止跟随重定向；账本参数与原始文件名只含 date/type/docID。
"""
from __future__ import annotations

import csv
import io
import json
import logging
import re
import time
import unicodedata
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from ..ledger import Ledger
from ..paths import DataPaths
from ..util.common import RateLimiter, to_date, today_jst
from .base import RawSink, get_with_retry, http_session

log = logging.getLogger("jpscan.edinet")

API_BASE = "https://api.edinet-fsa.go.jp/api/v2/"
CODE_LIST_URL = "https://disclosure2dl.edinet-fsa.go.jp/searchdocument/codelist/Edinetcode.zip"
SOURCE = "edinet"
LIST_DATASET = "documents_list"
DOC_DATASET = "documents"
CODE_DATASET = "codelist"

KEY_PARAM = "Subscription-Key"
KEY_HELP = ("未设置 EDINET_API_KEY：EDINET 书类一览与书类取得需要 API Key（代码表不需要）。"
            "在 EDINET 閲覧サイト（https://disclosure2.edinet-fsa.go.jp/）点「ログイン」注册并发行，"
            "步骤见 EDINET API 仕様書 2-2、2-3 节，然后把 EDINET_API_KEY=... 写入项目根目录的 .env。")

# 书类取得 type → 扩展名
DOC_EXT = {1: ".zip", 2: ".pdf", 3: ".zip", 4: ".zip", 5: ".zip"}
_MAGIC = {".zip": b"PK", ".pdf": b"%PDF"}

DOC_TYPE_LABELS = {
    "030": "有価証券届出書", "040": "訂正有価証券届出書",
    "120": "有価証券報告書", "130": "訂正有価証券報告書",
    "140": "四半期報告書", "150": "訂正四半期報告書",
    "160": "半期報告書", "170": "訂正半期報告書",
    "180": "臨時報告書", "190": "訂正臨時報告書",
    "220": "自己株券買付状況報告書", "230": "訂正自己株券買付状況報告書",
    "235": "内部統制報告書", "236": "訂正内部統制報告書",
    "240": "公開買付届出書", "250": "訂正公開買付届出書",
    "270": "公開買付報告書", "280": "訂正公開買付報告書",
    "350": "大量保有報告書・変更報告書", "360": "訂正大量保有報告書・変更報告書",
}
FORM_LABELS_350 = {"010000": "大量保有報告書", "010002": "変更報告書"}

_KEY_RE = re.compile(r"(Subscription-Key=)[^&\s'\"]+", re.IGNORECASE)
_DOC_ID_RE = re.compile(r"^[A-Za-z0-9]{1,32}$")


class EdinetError(Exception):
    """EDINET 请求失败。消息已脱敏，不含 API Key。"""


class EdinetKeyMissing(EdinetError):
    """未配置 API Key。"""


class EdinetAuthError(EdinetError):
    """API Key 无效（状态 401）。继续请求没有意义，作业应立即停止。"""


class EdinetNotFound(EdinetError):
    """书类不存在或已不可取得（状态 404），例如閲覧期間満了、不开示。"""


class _RedactKeyFilter(logging.Filter):
    """urllib3 的 DEBUG 日志含完整 URL，写入前抹掉密钥。"""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if KEY_PARAM.lower() in msg.lower():
            record.msg, record.args = _KEY_RE.sub(r"\1***", msg), ()
        return True


logging.getLogger("urllib3.connectionpool").addFilter(_RedactKeyFilter())


def doc_kind(doc_type_code: str | None, form_code: str | None = None) -> str | None:
    """书类种别码 → 标签；350 靠 formCode 区分大量保有与变更。"""
    if not doc_type_code:
        return None
    if doc_type_code == "350" and form_code in FORM_LABELS_350:
        return FORM_LABELS_350[form_code]
    return DOC_TYPE_LABELS.get(doc_type_code, f"その他({doc_type_code})")


def normalize_sec_code(sec: Any) -> tuple[str | None, str | None]:
    """证券代码 → (5 位 Code, code4)。4 位补尾 0；其他长度视为无效。"""
    s = unicodedata.normalize("NFKC", str(sec)).strip().upper() if sec is not None else ""
    if len(s) == 4:
        s += "0"
    if len(s) != 5 or not s.isalnum():
        return None, None
    return s, s[:4]


class EdinetClient:
    def __init__(self, api_key: str | None, paths: DataPaths, ledger: Ledger, session: requests.Session | None = None,
                 limiter: RateLimiter | None = None, retries: int = 4, timeout: int = 120):
        self.api_key = api_key or None
        self.paths = paths
        self.ledger = ledger
        self.session = session or http_session()
        self.limiter = limiter or RateLimiter(60)   # 匀速，间隔 ≥ 1.08 秒
        self.retries = retries
        self.timeout = timeout
        self.sink = RawSink(paths, ledger, SOURCE)

    # ---------------- 脱敏 ----------------
    def _clean(self, text: Any) -> str:
        s = _KEY_RE.sub(r"\1***", str(text))
        return s.replace(self.api_key, "***") if self.api_key else s

    # ---------------- 带密钥的请求 ----------------
    def _classify(self, r: requests.Response) -> tuple[int, str, Any]:
        """返回 (有效状态码, 消息, JSON 载荷)。JSON 错误体优先于 HTTP 状态码。"""
        ctype = str(r.headers.get("Content-Type", "")).lower()
        if "json" not in ctype and not r.content[:1] == b"{":
            return r.status_code, "", None
        try:
            payload = json.loads(self._clean(r.content.decode("utf-8-sig")))
        except ValueError:
            return 502, "响应不是合法 JSON", None
        if not isinstance(payload, dict):
            return 502, "响应 JSON 不是对象", None
        meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        raw_status = payload.get("StatusCode") or payload.get("statusCode") or meta.get("status") or r.status_code
        try:
            status = int(raw_status)
        except (TypeError, ValueError):
            return 502, f"无法识别的状态 {raw_status!r}", payload
        return status, str(payload.get("message") or meta.get("message") or ""), payload

    def _get(self, endpoint: str, params: dict[str, Any]) -> tuple[requests.Response, Any]:
        """带密钥的 GET：限速、429 退避、5xx 重试；返回 (响应, JSON 载荷或 None)。"""
        if not self.api_key:
            raise EdinetKeyMissing(KEY_HELP)
        desc = f"{endpoint} {params}"   # 只含公开参数，可安全记录
        last = ""
        for attempt in range(self.retries):
            self.limiter.wait()
            try:
                r = self.session.get(API_BASE + endpoint, params={**params, KEY_PARAM: self.api_key},
                                     timeout=self.timeout, allow_redirects=False)
            except requests.RequestException as e:
                last = self._clean(f"{type(e).__name__}: {e}")
                log.warning("EDINET %s failed (%s); retry", desc, last)
                time.sleep(2.0 ** attempt)
                continue
            status, message, payload = self._classify(r)
            message = self._clean(message)[:200]
            if status == 429:
                wait = _retry_after(r, 60.0)
                log.warning("EDINET 429 on %s; pause %.0fs", desc, wait)
                self.limiter.penalize(wait)
                last = f"429 {message}"
                continue
            if status >= 500 or status == 408:
                last = f"{status} {message}"
                log.warning("EDINET %s returned %s; retry", desc, last)
                time.sleep(2.0 ** attempt)
                continue
            if status == 401:
                raise EdinetAuthError(f"EDINET API Key 无效（401），请检查 .env 的 EDINET_API_KEY：{message}")
            if status == 404:
                raise EdinetNotFound(f"{desc}: 404 {message}")
            if status != 200:
                raise EdinetError(f"{desc}: {status} {message}")
            return r, payload
        raise EdinetError(f"{desc}: 重试 {self.retries} 次后仍失败（{last}）")

    # ---------------- 书类一览 ----------------
    def list_documents(self, d: date | str, use_cache: bool = False) -> list[dict[str, Any]]:
        """取某提出日的书类一览（含元数据），原样落盘后返回 results。use_cache=True 时优先读已落盘的原始 JSON。"""
        day = to_date(d)
        if day is None:
            raise ValueError("date is required")
        key = day.isoformat()
        params = {"date": key, "type": 2}
        if use_cache:
            cached = self.sink.load_json(LIST_DATASET, key, params)
            if isinstance(cached, dict):
                return cached.get("results") or []
        _, payload = self._get("documents.json", params)
        if payload is None:   # 维护时的 Sorry 页是 HTML
            raise EdinetError(f"documents.json {key}: 响应不是 JSON（可能在维护中）")
        rows = payload.get("results") or []
        self.sink.save_json(LIST_DATASET, key, params, payload, http_status=200, n_rows=len(rows))
        return rows

    # ---------------- 书类取得 ----------------
    def doc_path(self, doc_id: str, doc_type: int, subdir: str | None = None) -> Path:
        doc_type = int(doc_type)
        if doc_type not in DOC_EXT:
            raise ValueError(f"unknown document type {doc_type}")
        if not _DOC_ID_RE.match(str(doc_id)):   # docID 来自下载内容且要进文件名
            raise ValueError(f"invalid docID {doc_id!r}")
        return self.paths.raw_dir(SOURCE, DOC_DATASET) / (subdir or "misc") / f"{doc_id}_{doc_type}{DOC_EXT[doc_type]}"

    def download(self, doc_id: str, doc_type: int | str = 5, subdir: str | None = None) -> Path:
        """下载书类（type 1 XBRL / 2 PDF / 3 添付 / 4 英文 / 5 CSV）。已存在则不再请求。"""
        doc_type = int(doc_type)
        path = self.doc_path(doc_id, doc_type, subdir)
        if path.exists():
            return path
        params = {"type": doc_type}
        r, _ = self._get(f"documents/{doc_id}", params)
        key, ext = f"{doc_id}_{doc_type}", DOC_EXT[doc_type]
        if not r.content.startswith(_MAGIC[ext]):
            self.sink.quarantine(DOC_DATASET, key, self._clean_bytes(r.content), f"expected {ext}, got other content")
            raise EdinetError(f"documents/{doc_id} type={doc_type}: 响应不是 {ext} 文件，已隔离")
        saved = self.sink.save_bytes(DOC_DATASET, key, r.content, ext, params={"docID": doc_id, **params},
                                     http_status=200, subdir=subdir or "misc")
        self.ledger.mark(f"edinet_doc_type{doc_type}", doc_id, "ok")
        return saved

    def _clean_bytes(self, content: bytes) -> bytes:
        return content.replace(self.api_key.encode(), b"***") if self.api_key else content

    # ---------------- 代码表（无需密钥） ----------------
    def fetch_code_list(self) -> Path:
        """下载 Edinetcode.zip 并落盘，返回本地路径。"""
        self.limiter.wait()
        r = get_with_retry(self.session, CODE_LIST_URL)
        if r.status_code != 200 or not r.content.startswith(b"PK"):
            raise EdinetError(f"Edinetcode.zip 下载失败：HTTP {r.status_code}")
        key = f"{today_jst().isoformat()}_Edinetcode"
        return self.sink.save_bytes(CODE_DATASET, key, r.content, ".zip", params={"url": CODE_LIST_URL},
                                    http_status=r.status_code)


def _retry_after(r: requests.Response, default: float) -> float:
    try:
        return max(float(r.headers.get("Retry-After", default)), 1.0)
    except (TypeError, ValueError):
        return default


# ---------------- 代码表解析（纯函数） ----------------
_CODE_COLUMNS = {
    "EDINETコード": "edinet_code", "提出者種別": "filer_type", "上場区分": "listed", "連結の有無": "consolidated",
    "資本金": "capital", "決算日": "fiscal_year_end", "提出者名": "name", "提出者名(英字)": "name_en",
    "提出者名(ヨミ)": "name_kana", "所在地": "address", "提出者業種": "industry", "証券コード": "sec_code",
    "提出者法人番号": "jcn",
}
_YES_NO = {"有": True, "無": False}


def parse_code_list(zip_path: Path | str) -> pd.DataFrame:
    """Edinetcode.zip → DataFrame（edinet_codes 表）。CSV 为 cp932：首行含下载日，次行为列头。"""
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise EdinetError("Edinetcode.zip 内没有 CSV")
        name = next((n for n in names if "EdinetcodeDlInfo" in n), names[0])
        text = z.read(name).decode("cp932", errors="replace")
    rows = list(csv.reader(io.StringIO(text, newline="")))
    normalized = [[unicodedata.normalize("NFKC", c).strip() for c in row] for row in rows[:5]]
    head_idx = next((i for i, row in enumerate(normalized) if "EDINETコード" in row), None)
    if head_idx is None:
        raise EdinetError("EdinetcodeDlInfo.csv 找不到列头行（格式可能已变更）")
    columns = [_CODE_COLUMNS.get(c) for c in normalized[head_idx]]
    missing = sorted(set(_CODE_COLUMNS.values()) - set(columns))
    if missing:
        raise EdinetError(f"EdinetcodeDlInfo.csv 缺少列：{missing}")
    as_of = _parse_as_of(" ".join(",".join(r) for r in rows[:head_idx]))
    out: list[dict[str, Any]] = []
    for row in rows[head_idx + 1:]:
        rec = {col: (val.strip() or None) for col, val in zip(columns, row) if col}
        if not rec.get("edinet_code"):
            continue
        code, code4 = normalize_sec_code(rec.pop("sec_code", None))
        out.append({
            "edinet_code": rec["edinet_code"], "filer_type": rec.get("filer_type"),
            "listed": rec.get("listed") == "上場", "consolidated": _YES_NO.get(rec.get("consolidated") or ""),
            "capital": _to_float(rec.get("capital")), "fiscal_year_end": rec.get("fiscal_year_end"),
            "name": rec.get("name"), "name_en": rec.get("name_en"), "name_kana": rec.get("name_kana"),
            "address": rec.get("address"), "industry": rec.get("industry"), "Code": code, "code4": code4,
            "jcn": rec.get("jcn"), "as_of": as_of,
        })
    return pd.DataFrame(out)


def _parse_as_of(text: str) -> date | None:
    m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", text)
    if not m:
        return None
    try:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).date()
    except ValueError:
        return None


def _to_float(s: Any) -> float | None:
    try:
        return float(str(s).replace(",", ""))
    except (TypeError, ValueError):
        return None
