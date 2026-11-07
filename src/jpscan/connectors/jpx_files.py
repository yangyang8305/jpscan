"""JPX 免费文件 / 页面连接器（规划文档 10.5 节 `jpx.files`、10.6 节 B7；附录 A.1、G1-06 … G1-17）。

数据集（全部来自 https://www.jpx.co.jp，无需认证；站点对默认 UA 可能返回 403，故使用浏览器型 UA，请求间隔 ≥ 1 秒）：

| dataset                  | 来源                                                         | staging 表                   |
|--------------------------|--------------------------------------------------------------|------------------------------|
| listed_monthly           | 東証上場銘柄一覧 data_j.xlsx（月次，前月末時点）              | jpx_listed_monthly           |
| topix_weights            | TOPIX 構成銘柄・ウエイト topixweight_j.csv（月次，cp932）     | index_topix_weights          |
| per_pbr_monthly          | 規模別・業種別 PER・PBR perpbrYYYYMM.xlsx（月次，自 04.html 发现） | jpx_per_pbr_monthly      |
| capital_cost_disclosure  | 資本コストや株価を意識した経営 開示企業一覧表 list.xlsx（多 sheet，按月） | jpx_capital_cost_disclosure |
| earnings_schedule        | 決算発表予定日 kessan.xlsx / kessanMM_DDDD.xlsx               | jpx_earnings_schedule        |
| price_limit_expansion    | 制限値幅拡大 対象銘柄 CSV（每営業日，含 ETF 文件）            | jpx_price_limit_expansion    |
| delisted                 | 上場廃止銘柄一覧（HTML 表）                                  | jpx_delisted                 |
| supervision              | 監理・整理銘柄：現在の指定状況 + 指定履歴（HTML 表）          | jpx_supervision              |
| new_listings             | 新規上場会社情報（HTML 表）                                  | jpx_new_listings             |

约定：
- 本模块的解析函数均为纯函数（bytes / str → DataFrame），不联网；抓取由 `JPXFiles.fetch` 完成，原始内容经
  `RawSink.save_bytes` 落盘登记后再解析（作业见 jobs/ingest_jpx.py）。
- `Code` 为 5 位字符串：JPX 文件中的 4 位代码（含 130A 等英数代码）补 "0"，与 J-Quants 一致；另加 `code4`。
- 日期列为 `datetime.date`；数值列 float64；文本列 pandas string；33 業種コード补零为 4 位（"0050"）。
- 文件中的 "－"/"-"/"＊"（无数据或负值 PER）→ NaN / None。
"""
from __future__ import annotations

import csv
import io
import logging
import re
import time
import unicodedata
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urljoin

import numpy as np
import openpyxl
import pandas as pd
import requests
from bs4 import BeautifulSoup

from ..ledger import Ledger
from ..paths import DataPaths
from .base import RawSink, get_with_retry, http_session

log = logging.getLogger("jpscan.jpx")

SOURCE = "jpx"
BASE = "https://www.jpx.co.jp"
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/130.0.0.0 Safari/537.36")
MIN_INTERVAL = 1.2   # 秒；相邻请求最小间隔（礼貌抓取）

URLS: dict[str, str] = {
    "listed_monthly": BASE + "/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xlsx",
    "topix_weights": BASE + "/automation/markets/indices/topix/files/topixweight_j.csv",
    "per_pbr_page": BASE + "/markets/statistics-equities/misc/04.html",
    "capital_cost_disclosure": BASE + "/equities/follow-up/jr4eth0000004vj2-att/list.xlsx",
    "earnings_page": BASE + "/listing/event-schedules/financial-announcement/index.html",
    "price_limit_page": BASE + "/equities/trading/domestic/06.html",
    "delisted": BASE + "/listing/stocks/delisted/index.html",
    "supervision_current": BASE + "/listing/market-alerts/supervision/index.html",
    "supervision_history": BASE + "/listing/market-alerts/supervision/01.html",
    "new_listings": BASE + "/listing/stocks/new/index.html",
}


@dataclass(frozen=True)
class TableSpec:
    table: str
    keys: list[str]
    partition: str   # 按月分区的日期列名，或 "all"


TABLES: dict[str, TableSpec] = {
    "listed_monthly": TableSpec("jpx_listed_monthly", ["date", "Code"], "date"),
    "topix_weights": TableSpec("index_topix_weights", ["date", "Code"], "date"),
    "per_pbr_monthly": TableSpec("jpx_per_pbr_monthly", ["month", "basis", "segment", "market", "category"], "all"),
    "capital_cost_disclosure": TableSpec("jpx_capital_cost_disclosure", ["as_of_month", "Code"], "all"),
    "earnings_schedule": TableSpec("jpx_earnings_schedule", ["sched_date", "Code", "quarter"], "sched_date"),
    "price_limit_expansion": TableSpec("jpx_price_limit_expansion", ["date", "Code"], "date"),
    "delisted": TableSpec("jpx_delisted", ["delist_date", "Code"], "all"),
    "supervision": TableSpec("jpx_supervision", ["designated_date", "Code", "kind"], "all"),
    "new_listings": TableSpec("jpx_new_listings", ["listing_date", "Code"], "all"),
}
DATASETS: list[str] = list(TABLES)

STOP_TYPES = {"1": "ストップ高", "2": "ストップ安", "3": "ストップ高気配", "4": "ストップ安気配"}
KIND_CONFIRMING = "監理銘柄（確認中）"
KIND_UNDER_REVIEW = "監理銘柄（審査中）"
KIND_LIQUIDATION = "整理銘柄"


# ----------------------------------------------------------------------------------------------
# 抓取
# ----------------------------------------------------------------------------------------------
class JPXFetchError(RuntimeError):
    """非 200 响应（403 / 404 等）。"""

    def __init__(self, url: str, status: int):
        super().__init__(f"HTTP {status}: {url}")
        self.url, self.status = url, status


Fetcher = Callable[[str], "bytes | tuple[int, bytes | None]"]


class JPXFiles:
    """抓取 + 原始层落盘。`fetcher` 可注入（测试用：url → bytes 或 (status, bytes)），注入后不联网。"""

    def __init__(self, paths: DataPaths, ledger: Ledger, session: requests.Session | None = None,
                 fetcher: Fetcher | None = None, min_interval: float = MIN_INTERVAL, user_agent: str = USER_AGENT):
        self.paths, self.ledger = paths, ledger
        self.sink = RawSink(paths, ledger, SOURCE)
        self.session = session
        self.fetcher = fetcher
        self.min_interval = min_interval
        self.user_agent = user_agent
        self._last = 0.0

    def fetch(self, url: str) -> bytes:
        if self.fetcher is not None:
            res = self.fetcher(url)
            status, content = res if isinstance(res, tuple) else (200, res)
            if status != 200 or content is None:
                raise JPXFetchError(url, int(status))
            return content
        if self.session is None:
            self.session = http_session(self.user_agent)
            self.session.headers.update({"Accept-Language": "ja,en;q=0.8"})
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        try:
            r = get_with_retry(self.session, url, timeout=120)
        finally:
            self._last = time.monotonic()
        if r.status_code != 200:
            raise JPXFetchError(url, r.status_code)
        return r.content

    def save(self, dataset: str, key: str, content: bytes, ext: str, url: str) -> Path:
        return self.sink.save_bytes(dataset, key, content, ext, params={"url": url}, http_status=200)

    def quarantine(self, dataset: str, key: str, content: bytes, reason: str) -> Path:
        return self.sink.quarantine(dataset, key, content, reason)


# ----------------------------------------------------------------------------------------------
# 通用小工具
# ----------------------------------------------------------------------------------------------
_WS = re.compile(r"[\s　]+")
_NULL_TEXT = {"", "-", "－", "―", "—"}


def _norm(s: Any) -> str:
    """NFKC 归一并去掉全部空白（含全角空格、换行）；用于表头匹配。"""
    if s is None:
        return ""
    return _WS.sub("", unicodedata.normalize("NFKC", str(s)))


def _clean(s: Any) -> str | None:
    """文本单元格：strip；None / 空 / "-" / "－" → None。"""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    if isinstance(s, datetime):
        return s.strftime("%Y-%m-%d")
    t = str(s).replace("　", " ").strip()
    return None if t in _NULL_TEXT else t


def norm_code(v: Any) -> str | None:
    """JPX 4 位代码（含 130A 等英数）→ 5 位（补 "0"，与 J-Quants 一致）；已是 5 位则原样；其余 → None。"""
    if v is None:
        return None
    if isinstance(v, float):
        if np.isnan(v):
            return None
        v = int(v)
    s = unicodedata.normalize("NFKC", str(v)).strip().upper()
    if not re.fullmatch(r"[0-9A-Z]{4,5}", s):
        return None
    return s if len(s) == 5 else s + "0"


def _date(v: Any) -> date | None:
    """接受 datetime / date / int(yyyymmdd) / "2026/11/06" / "2026-11-06" / "2026年11月6日" / "20261106"。"""
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, float):
        if np.isnan(v):
            return None
        v = int(v)
    s = unicodedata.normalize("NFKC", str(v)).strip()
    m = re.search(r"(\d{4})[/\-.年](\d{1,2})[/\-.月](\d{1,2})", s) or re.search(r"(\d{4})(\d{2})(\d{2})", s)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _num(v: Any) -> float:
    """数值单元格：数字；"－"/"＊"/"-"/"" → NaN；公式文本 "=18.2" / '="－"' → 去掉 "=" 与引号；支持 "1,234" 与 "0.0047%"（→ 小数）。"""
    if v is None:
        return np.nan
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    s = unicodedata.normalize("NFKC", str(v)).strip()
    if s.startswith("="):
        s = s[1:].strip()
    s = s.strip('"').strip()
    if s in _NULL_TEXT or s in ("*", "―"):
        return np.nan
    pct = s.endswith("%")
    s = s.rstrip("%").replace(",", "").strip()
    try:
        x = float(s)
    except ValueError:
        return np.nan
    return x / 100.0 if pct else x


def _code_str(v: Any, width: int = 0) -> str | None:
    """业种/规模代码：int 50 → "0050"（width=4）；"-" → None。"""
    t = _clean(v)
    if t is None:
        return None
    if isinstance(v, float) and v.is_integer():
        t = str(int(v))
    t = unicodedata.normalize("NFKC", t)
    return t.zfill(width) if (width and t.isdigit()) else t


def _cell(row: tuple, i: int | None) -> Any:
    return None if i is None or i >= len(row) else row[i]


def _decode(content: bytes) -> str:
    for enc in ("utf-8-sig", "cp932"):
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    return content.decode("cp932", errors="replace")


def _soup(html: str | bytes) -> BeautifulSoup:
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="replace")
    return BeautifulSoup(html, "lxml")


def _xlsx_sheets(content: bytes) -> list[tuple[str, list[tuple]]]:
    """全部工作表 → [(sheet 名, [行元组])]。公式单元格返回公式文本（如 "=18.2"），由 _num 处理。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=False)
        try:
            return [(ws.title, [tuple(r) for r in ws.iter_rows(values_only=True)]) for ws in wb.worksheets]
        finally:
            wb.close()


def _index_by(headers: list[str], spec: dict[str, str | tuple[str, ...]]) -> dict[str, int | None]:
    """按「表头包含关键字」定位列：spec = {字段: 关键字或关键字元组}（已 _norm 的表头）。"""
    out: dict[str, int | None] = {}
    for field, kws in spec.items():
        kws = (kws,) if isinstance(kws, str) else kws
        out[field] = next((i for i, h in enumerate(headers) if any(k in h for k in kws)), None)
    return out


def _href(cell: Any) -> str | None:
    a = cell.find("a", href=True) if cell is not None else None
    return urljoin(BASE, a["href"]) if a else None


def _frame(records: list[dict[str, Any]], cols: list[str], *, text: Iterable[str] = (), dates: Iterable[str] = (),
           nums: Iterable[str] = (), bools: Iterable[str] = (), keys: Iterable[str] = ()) -> pd.DataFrame:
    """统一 dtype：日期 → date 对象、数值 → float64、布尔 → bool、文本 → string；Code → code4；按 keys 去重。"""
    df = pd.DataFrame.from_records(records, columns=cols) if records else pd.DataFrame({c: [] for c in cols})
    if df.empty:
        return df
    for c in dates:
        df[c] = [_date(v) for v in df[c]]
        df[c] = df[c].astype(object)
    for c in nums:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in bools:
        df[c] = df[c].fillna(False).astype(bool)
    for c in text:
        df[c] = df[c].astype("string")
    if "Code" in df.columns:
        df["Code"] = df["Code"].astype("string")
        df["code4"] = df["Code"].str[:4]
    keys = [k for k in keys if k in df.columns]
    if keys:
        df = df.drop_duplicates(subset=keys, keep="last")
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------------------------
# 链接发现
# ----------------------------------------------------------------------------------------------
def find_links(html: str | bytes, pattern: str) -> list[str]:
    """页面内 href 匹配正则的绝对链接（去重、保持顺序）。"""
    out: list[str] = []
    for a in _soup(html).find_all("a", href=True):
        if re.search(pattern, a["href"]):
            u = urljoin(BASE, a["href"])
            if u not in out:
                out.append(u)
    return out


def find_perpbr_links(html: str | bytes) -> dict[str, str]:
    """規模別・業種別 PER・PBR 页 → {"2026-10": ".../perpbr202610.xlsx", ...}。"""
    out: dict[str, str] = {}
    for u in find_links(html, r"perpbr\d{6}\.xlsx?$"):
        m = re.search(r"perpbr(\d{4})(\d{2})\.xlsx?$", u)
        if m:
            out[f"{m.group(1)}-{m.group(2)}"] = u
    return out


def find_kessan_links(html: str | bytes) -> list[str]:
    """決算発表予定日 页 → kessan.xlsx（翌営業日分）与 kessanMM_DDDD.xlsx（月次一覧）。"""
    return find_links(html, r"/kessan[^/]*\.xlsx?$")


def find_price_limit_links(html: str | bytes) -> list[str]:
    """制限値幅 页 → 当日 CSV（MMDD.csv、MMDD_ETF.csv）。"""
    return find_links(html, r"/\d{4}(_[A-Za-z]+)?\.csv$")


# ----------------------------------------------------------------------------------------------
# 1. 東証上場銘柄一覧 data_j.xlsx
# ----------------------------------------------------------------------------------------------
_LISTED_COLS = {"日付": "date", "コード": "Code", "銘柄名": "name", "市場・商品区分": "market_product",
                "33業種コード": "s33", "33業種区分": "s33_name", "17業種コード": "s17", "17業種区分": "s17_name",
                "規模コード": "scale_code", "規模区分": "scale_name"}
LISTED_COLUMNS = ["date", "Code", "name", "market_product", "s33", "s33_name", "s17", "s17_name", "scale_code", "scale_name"]


def parse_listed_monthly(content: bytes) -> pd.DataFrame:
    _, rows = _xlsx_sheets(content)[0]
    hdr = [_norm(c) for c in rows[0]]
    ci = {field: (hdr.index(_norm(jp)) if _norm(jp) in hdr else None) for jp, field in _LISTED_COLS.items()}
    if ci["Code"] is None or ci["date"] is None:
        raise ValueError(f"data_j.xlsx: 未找到表头 コード/日付: {hdr}")
    recs = []
    for r in rows[1:]:
        code = norm_code(_cell(r, ci["Code"]))
        d = _date(_cell(r, ci["date"]))
        if code is None or d is None:
            continue
        recs.append({
            "date": d, "Code": code, "name": _clean(_cell(r, ci["name"])),
            "market_product": _clean(_cell(r, ci["market_product"])),
            "s33": _code_str(_cell(r, ci["s33"]), 4), "s33_name": _clean(_cell(r, ci["s33_name"])),
            "s17": _code_str(_cell(r, ci["s17"])), "s17_name": _clean(_cell(r, ci["s17_name"])),
            "scale_code": _code_str(_cell(r, ci["scale_code"])), "scale_name": _clean(_cell(r, ci["scale_name"])),
        })
    return _frame(recs, LISTED_COLUMNS, text=LISTED_COLUMNS[1:], dates=["date"], keys=TABLES["listed_monthly"].keys)


# ----------------------------------------------------------------------------------------------
# 2. TOPIX 構成銘柄・ウエイト topixweight_j.csv（cp932）
# ----------------------------------------------------------------------------------------------
TOPIX_COLUMNS = ["date", "Code", "name", "sector_name", "weight", "new_index_category"]


def parse_topix_weights(content: bytes) -> pd.DataFrame:
    """weight 为小数（文件中 "0.0047%" → 0.000047）；尾部免责声明行（首列非 8 位日期）忽略。"""
    reader = csv.reader(io.StringIO(_decode(content)))
    header = next(reader, None) or []
    hdr = [_norm(h) for h in header]
    ci = _index_by(hdr, {"date": "日付", "name": "銘柄名", "Code": "コード", "sector_name": "業種",
                         "weight": "ウエイト", "new_index_category": "ニューインデックス"})
    recs = []
    for row in reader:
        if not row or not re.fullmatch(r"\d{8}", row[0].strip()):
            continue
        code = norm_code(_cell(tuple(row), ci["Code"]))
        if code is None:
            continue
        recs.append({"date": _date(row[0]), "Code": code, "name": _clean(_cell(tuple(row), ci["name"])),
                     "sector_name": _clean(_cell(tuple(row), ci["sector_name"])),
                     "weight": _num(_cell(tuple(row), ci["weight"])),
                     "new_index_category": _clean(_cell(tuple(row), ci["new_index_category"]))})
    return _frame(recs, TOPIX_COLUMNS, text=["name", "sector_name", "new_index_category"], dates=["date"],
                  nums=["weight"], keys=TABLES["topix_weights"].keys)


# ----------------------------------------------------------------------------------------------
# 3. 規模別・業種別 PER・PBR perpbrYYYYMM.xlsx
# ----------------------------------------------------------------------------------------------
PER_PBR_COLUMNS = ["month", "basis", "segment", "market", "market_en", "category", "category_en", "sector_no",
                   "n_companies", "per_simple", "pbr_simple", "eps_simple", "bps_simple", "per_weighted", "pbr_weighted",
                   "net_income_sum", "net_assets_sum"]
_SIZE_CATS = {"大型株", "中型株", "小型株"}


def _per_pbr_columns(hdr: list[str]) -> dict[str, int]:
    ci: dict[str, int] = {}
    for i, h in enumerate(hdr):
        if h == "市場区分名":
            ci["market"] = i
        elif h == "Section":
            ci["market_en"] = i
        elif h == "種別":
            ci["category"] = i
        elif h == "Industry":
            ci["category_en"] = i
        elif h == "会社数":
            ci["n_companies"] = i
        elif h.startswith("単純_PER"):
            ci["per_simple"] = i
        elif h.startswith("単純_PBR"):
            ci["pbr_simple"] = i
        elif h.startswith("単純_") and "当期純利益" in h:
            ci["eps_simple"] = i
        elif h.startswith("単純_") and "純資産" in h:
            ci["bps_simple"] = i
        elif h.startswith("加重_PER"):
            ci["per_weighted"] = i
        elif h.startswith("加重_PBR"):
            ci["pbr_weighted"] = i
        elif h.startswith("加重_") and "当期純利益" in h:
            ci["net_income_sum"] = i
        elif h.startswith("加重_") and "純資産" in h:
            ci["net_assets_sum"] = i
    return ci


def parse_per_pbr(content: bytes) -> pd.DataFrame:
    """两张表：「規模別・業種別（連結）」与「市場別（単体）」→ 长表。

    segment：市場区分（種別=総合 / 単体表各行）、規模区分（大型株・中型株・小型株）、業種（"N 業種名"，sector_no=N）、
    業種集計（総合（金融業を除く）、製造業、非製造業）。market 为所属市場区分名（規模区分行在 2022 年后格式中为 "－" → None）。
    """
    recs = []
    for sname, rows in _xlsx_sheets(content):
        basis = "単体" if "単体" in sname else "連結"
        h = next((i for i, r in enumerate(rows) if r and _norm(r[0]) == "年月"), None)
        if h is None:
            continue
        ci = _per_pbr_columns([_norm(c) for c in rows[h]])
        for r in rows[h + 1:]:
            ym = re.fullmatch(r"(\d{4})/(\d{1,2})", _norm(_cell(r, 0)))
            if not ym:
                continue
            month = f"{ym.group(1)}-{int(ym.group(2)):02d}"
            market = _clean(_cell(r, ci.get("market")))
            market_en = _clean(_cell(r, ci.get("market_en")))
            raw_cat = _clean(_cell(r, ci.get("category")))
            cat_en = _clean(_cell(r, ci.get("category_en")))
            sector_no = np.nan
            if raw_cat is None:
                segment, category, cat_en = "市場区分", market, market_en
            else:
                cat = raw_cat.strip()
                mm = re.match(r"^(\d+)\s*(.+)$", unicodedata.normalize("NFKC", cat))
                if cat == "総合":
                    segment, category, cat_en = "市場区分", market, market_en
                elif cat in _SIZE_CATS:
                    segment, category = "規模区分", cat
                elif mm:
                    segment, sector_no = "業種", float(mm.group(1))
                    category = re.sub(r"^\d+\s*", "", cat).strip()
                    cat_en = re.sub(r"^\d+\s*", "", cat_en).strip() if cat_en else None
                else:
                    segment, category = "業種集計", cat
            if category is None:
                continue
            rec = {"month": month, "basis": basis, "segment": segment, "market": market, "market_en": market_en,
                   "category": category, "category_en": cat_en, "sector_no": sector_no}
            for f in PER_PBR_COLUMNS[8:]:
                rec[f] = _num(_cell(r, ci.get(f)))
            recs.append(rec)
    return _frame(recs, PER_PBR_COLUMNS, text=["month", "basis", "segment", "market", "market_en", "category", "category_en"],
                  nums=PER_PBR_COLUMNS[7:], keys=TABLES["per_pbr_monthly"].keys)


# ----------------------------------------------------------------------------------------------
# 4. 資本コストや株価を意識した経営 開示企業一覧表 list.xlsx
# ----------------------------------------------------------------------------------------------
CAPITAL_COST_COLUMNS = ["as_of_month", "as_of_date", "Code", "name", "market", "s33_code", "s33_name", "disclosure_status",
                        "change_from_prev_month", "update_date", "disclosure_content", "contact_wanted",
                        "contact_listed_since", "contact_dept", "english_disclosure"]
_CC_SHEET_RE = re.compile(r"(\d{4})年(\d{1,2})月末")


def _cc_field(name: str) -> str | None:
    if "業種コード" in name:
        return "s33_code"
    if name == "業種":
        return "s33_name"
    if "市場区分" in name:
        return "market"
    if "証券コード" in name:
        return "Code"
    if "銘柄名" in name:
        return "name"
    if "前月から" in name:
        return "change_from_prev_month"
    if "アップデート日" in name:
        return "update_date"
    if name.startswith("開示内容"):
        return "disclosure_content"
    if "開示状況" in name:
        return "disclosure_status"
    if "申請状況" in name or name.startswith("機関投資家"):
        return "contact_wanted"
    if "掲載開始日" in name:
        return "contact_listed_since"
    if "コンタクト先" in name:
        return "contact_dept"
    if "英文開示" in name:
        return "english_disclosure"
    return None


def parse_capital_cost(content: bytes) -> pd.DataFrame:
    """每张「開示企業一覧（YYYY年M月末時点）」sheet → 一行/(as_of_month, Code)。
    表头两行（主表头 + 子表头）按关键字匹配；contact_wanted（"○"）与 english_disclosure（"有"）转为 bool。"""
    recs = []
    for sname, rows in _xlsx_sheets(content):
        ns = _norm(sname)
        m = _CC_SHEET_RE.search(ns)
        if "開示企業一覧" not in ns or not m:
            continue
        as_of_month = f"{m.group(1)}-{int(m.group(2)):02d}"
        h = next((i for i, r in enumerate(rows) if any("証券コード" in _norm(c) for c in r)), None)
        if h is None:
            log.warning("list.xlsx sheet %s: 未找到表头", sname)
            continue
        as_of_date = next((v for r in rows[:h] for v in r if isinstance(v, (datetime, date))), None)
        top = [_norm(c) for c in rows[h]]
        sub = [_norm(c) for c in rows[h + 1]] if h + 1 < len(rows) else []
        ci: dict[str, int] = {}
        for i in range(max(len(top), len(sub))):
            name = (sub[i] if i < len(sub) and sub[i] else "") or (top[i] if i < len(top) else "")
            f = _cc_field(name) if name else None
            if f and f not in ci:
                ci[f] = i
        if "Code" not in ci:
            continue
        start = h + 2 if (h + 1 < len(rows) and norm_code(_cell(rows[h + 1], ci["Code"])) is None) else h + 1
        for r in rows[start:]:
            code = norm_code(_cell(r, ci["Code"]))
            if code is None:
                continue
            recs.append({
                "as_of_month": as_of_month, "as_of_date": as_of_date, "Code": code,
                "name": _clean(_cell(r, ci.get("name"))), "market": _clean(_cell(r, ci.get("market"))),
                "s33_code": _code_str(_cell(r, ci.get("s33_code")), 4), "s33_name": _clean(_cell(r, ci.get("s33_name"))),
                "disclosure_status": _clean(_cell(r, ci.get("disclosure_status"))),
                "change_from_prev_month": _clean(_cell(r, ci.get("change_from_prev_month"))),
                "update_date": _date(_cell(r, ci.get("update_date"))),
                "disclosure_content": _clean(_cell(r, ci.get("disclosure_content"))),
                "contact_wanted": _clean(_cell(r, ci.get("contact_wanted"))) is not None,
                "contact_listed_since": _date(_cell(r, ci.get("contact_listed_since"))),
                "contact_dept": _clean(_cell(r, ci.get("contact_dept"))),
                "english_disclosure": _clean(_cell(r, ci.get("english_disclosure"))) is not None,
            })
    return _frame(recs, CAPITAL_COST_COLUMNS,
                  text=["as_of_month", "name", "market", "s33_code", "s33_name", "disclosure_status", "change_from_prev_month",
                        "disclosure_content", "contact_dept"],
                  dates=["as_of_date", "update_date", "contact_listed_since"], bools=["contact_wanted", "english_disclosure"],
                  keys=TABLES["capital_cost_disclosure"].keys)


# ----------------------------------------------------------------------------------------------
# 5. 決算発表予定日 kessan.xlsx / kessanMM_DDDD.xlsx
# ----------------------------------------------------------------------------------------------
EARNINGS_COLUMNS = ["sched_date", "sched_status", "Code", "name", "name_en", "fiscal_year_end", "sector_name", "sector_name_en",
                    "quarter", "quarter_en", "market", "market_en", "period_end_month", "as_of_date", "source_file"]


def parse_earnings_schedule(content: bytes, source_file: str = "kessan.xlsx", fetch_date: date | None = None) -> pd.DataFrame:
    """決算発表予定日 一覧。sched_date 为 "未定_Undecided" 时置 None 且 sched_status="undecided"；
    as_of_date 取表内「YYYY年M月D日 現在」，缺失（kessan.xlsx）则用 fetch_date；period_end_month 取标题「N月に四半期末…」。"""
    _, rows = _xlsx_sheets(content)[0]
    h = next((i for i, r in enumerate(rows) if _norm(_cell(r, 0)).startswith("決算発表予定日")), None)
    if h is None:
        raise ValueError(f"{source_file}: 未找到表头 決算発表予定日")
    as_of, period_month = None, np.nan
    for r in rows[:h]:
        for v in r:
            if isinstance(v, str):
                t = unicodedata.normalize("NFKC", v)
                m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日\s*現在", t)
                if m:
                    as_of = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                m2 = re.match(r"\s*(\d{1,2})月に", t)
                if m2:
                    period_month = float(m2.group(1))
    hdr = [_norm(c) for c in rows[h]]
    ci: dict[str, int | None] = {}
    for i, hh in enumerate(hdr):
        if hh.startswith("決算発表予定日"):
            ci["sched_date"] = i
        elif hh.startswith("コード"):
            ci["Code"] = i
        elif hh == "会社名":
            ci["name"] = i
        elif hh == "IssueName":
            ci["name_en"] = i
        elif hh.startswith("決算期末"):
            ci["fiscal_year_end"] = i
        elif hh == "業種名":
            ci["sector_name"] = i
        elif hh == "Industry":
            ci["sector_name_en"] = i
        elif hh == "種別":
            ci["quarter"] = i
        elif hh.startswith("FiscalYear/Quarter"):
            ci["quarter_en"] = i
        elif hh == "市場区分":
            ci["market"] = i
        elif hh == "MarketSegment":
            ci["market_en"] = i
    if "Code" not in ci or "sched_date" not in ci:
        raise ValueError(f"{source_file}: 表头不完整: {hdr}")
    recs = []
    for r in rows[h + 1:]:
        code = norm_code(_cell(r, ci["Code"]))
        if code is None:
            continue
        raw = _cell(r, ci["sched_date"])
        d = _date(raw)
        recs.append({
            "sched_date": d, "sched_status": "scheduled" if d else "undecided", "Code": code,
            "name": _clean(_cell(r, ci.get("name"))), "name_en": _clean(_cell(r, ci.get("name_en"))),
            "fiscal_year_end": _date(_cell(r, ci.get("fiscal_year_end"))),
            "sector_name": _clean(_cell(r, ci.get("sector_name"))), "sector_name_en": _clean(_cell(r, ci.get("sector_name_en"))),
            "quarter": _clean(_cell(r, ci.get("quarter"))), "quarter_en": _clean(_cell(r, ci.get("quarter_en"))),
            "market": _clean(_cell(r, ci.get("market"))), "market_en": _clean(_cell(r, ci.get("market_en"))),
            "period_end_month": period_month, "as_of_date": as_of or fetch_date, "source_file": source_file,
        })
    return _frame(recs, EARNINGS_COLUMNS,
                  text=["sched_status", "name", "name_en", "sector_name", "sector_name_en", "quarter", "quarter_en", "market",
                        "market_en", "source_file"],
                  dates=["sched_date", "fiscal_year_end", "as_of_date"], nums=["period_end_month"],
                  keys=TABLES["earnings_schedule"].keys)


# ----------------------------------------------------------------------------------------------
# 6. 制限値幅拡大 対象銘柄 CSV（MMDD.csv / MMDD_ETF.csv，cp932）
# ----------------------------------------------------------------------------------------------
PRICE_LIMIT_COLUMNS = ["date", "Code", "stop_type", "stop_type_name", "consecutive_days", "list_kind"]


def parse_price_limit(content: bytes, list_kind: str = "stock") -> pd.DataFrame:
    """ストップ情報種別 1:ストップ高 2:ストップ安 3:ストップ高気配 4:ストップ安気配；ETF 文件无 連続日 列 → NaN。"""
    reader = csv.reader(io.StringIO(_decode(content)))
    header = next(reader, None) or []
    hdr = [_norm(h) for h in header]
    ci = _index_by(hdr, {"date": "日付", "Code": "コード", "stop_type": "種別", "consecutive_days": "連続日"})
    recs = []
    for row in reader:
        t = tuple(row)
        d, code = _date(_cell(t, ci["date"])), norm_code(_cell(t, ci["Code"]))
        if d is None or code is None:
            continue
        st = _clean(_cell(t, ci["stop_type"]))
        st = str(int(float(st))) if st and re.fullmatch(r"\d+(\.0+)?", st) else st
        recs.append({"date": d, "Code": code, "stop_type": st, "stop_type_name": STOP_TYPES.get(st or ""),
                     "consecutive_days": _num(_cell(t, ci["consecutive_days"])), "list_kind": list_kind})
    return _frame(recs, PRICE_LIMIT_COLUMNS, text=["stop_type", "stop_type_name", "list_kind"], dates=["date"],
                  nums=["consecutive_days"], keys=TABLES["price_limit_expansion"].keys)


# ----------------------------------------------------------------------------------------------
# 7. 上場廃止銘柄一覧（HTML）
# ----------------------------------------------------------------------------------------------
DELISTED_COLUMNS = ["delist_date", "Code", "name", "market", "reason", "detail_url"]


def _tables_with(soup: BeautifulSoup, keyword: str) -> list[Any]:
    out = []
    for t in soup.find_all("table"):
        ths = [_norm(th.get_text(" ", strip=True)) for th in t.find_all("th")]
        if any(keyword in h for h in ths):
            out.append(t)
    return out


def _table_rows(table: Any) -> tuple[list[str], list[list[Any]]]:
    hdr = [_norm(th.get_text(" ", strip=True)) for th in table.find_all("th")]
    body = [tr.find_all("td") for tr in table.find_all("tr")]
    return hdr, [cells for cells in body if cells]


def parse_delisted(html: str | bytes) -> pd.DataFrame:
    recs = []
    for t in _tables_with(_soup(html), "上場廃止日"):
        hdr, body = _table_rows(t)
        ci = _index_by(hdr, {"delist_date": "上場廃止日", "name": "銘柄名", "Code": "コード", "market": "市場区分", "reason": "上場廃止理由"})
        for cells in body:
            tx = [c.get_text(" ", strip=True) for c in cells]
            d, code = _date(_cell(tuple(tx), ci["delist_date"])), norm_code(_cell(tuple(tx), ci["Code"]))
            if d is None or code is None:
                continue
            recs.append({"delist_date": d, "Code": code, "name": _clean(_cell(tuple(tx), ci["name"])),
                         "market": _clean(_cell(tuple(tx), ci["market"])), "reason": _clean(_cell(tuple(tx), ci["reason"])),
                         "detail_url": _href(_cell(tuple(cells), ci["name"])) or _href(_cell(tuple(cells), ci["reason"]))})
    return _frame(recs, DELISTED_COLUMNS, text=["name", "market", "reason", "detail_url"], dates=["delist_date"],
                  keys=TABLES["delisted"].keys)


# ----------------------------------------------------------------------------------------------
# 8. 監理・整理銘柄（HTML：現在の指定状況 index.html + 指定履歴 01.html）
# ----------------------------------------------------------------------------------------------
SUPERVISION_COLUMNS = ["designated_date", "Code", "kind", "name", "market", "released_date", "content", "note", "detail_url",
                       "is_current", "source_page"]


def supervision_kind(text: str | None) -> str | None:
    """标题 / 内容文本 → 指定种类（監理銘柄（確認中）/ 監理銘柄（審査中）/ 整理銘柄）。"""
    t = _norm(text)
    if not t:
        return None
    if "整理銘柄" in t:
        return KIND_LIQUIDATION
    if "審査中" in t:
        return KIND_UNDER_REVIEW
    if "確認中" in t:
        return KIND_CONFIRMING
    if "監理銘柄" in t:
        return "監理銘柄"
    return None


def _supervision_table(table: Any, kind_default: str | None, source_page: str) -> list[dict[str, Any]]:
    hdr, body = _table_rows(table)
    ci = _index_by(hdr, {"designated_date": "指定年月日", "name": "銘柄名", "Code": "コード", "market": "市場区分",
                         "released_date": "解除年月日", "content": "内容", "note": "備考", "detail": "詳細"})
    recs = []
    for cells in body:
        tx = tuple(c.get_text(" ", strip=True) for c in cells)
        d, code = _date(_cell(tx, ci["designated_date"])), norm_code(_cell(tx, ci["Code"]))
        if d is None or code is None:
            continue
        content = _clean(_cell(tx, ci["content"]))
        released = _date(_cell(tx, ci["released_date"]))
        kind = supervision_kind(content) or kind_default
        recs.append({"designated_date": d, "Code": code, "kind": kind, "name": _clean(_cell(tx, ci["name"])),
                     "market": _clean(_cell(tx, ci["market"])), "released_date": released, "content": content,
                     "note": _clean(_cell(tx, ci["note"])),
                     "detail_url": _href(_cell(tuple(cells), ci["detail"])) or _href(_cell(tuple(cells), ci["name"])),
                     "is_current": (source_page == "current") or (ci["released_date"] is not None and released is None),
                     "source_page": source_page})
    return recs


def _supervision_frame(recs: list[dict[str, Any]]) -> pd.DataFrame:
    return _frame(recs, SUPERVISION_COLUMNS, text=["kind", "name", "market", "content", "note", "detail_url", "source_page"],
                  dates=["designated_date", "released_date"], bools=["is_current"], keys=TABLES["supervision"].keys)


def parse_supervision_current(html: str | bytes) -> pd.DataFrame:
    """現在の指定状況：h3 标题（監理銘柄（確認中）/（審査中）/ 整理銘柄）决定其后表格的 kind。"""
    recs, kind = [], None
    for el in _soup(html).find_all(["h2", "h3", "h4", "table"]):
        if el.name != "table":
            k = supervision_kind(el.get_text(" ", strip=True))
            if k:
                kind = k
            continue
        if any("指定年月日" in _norm(th.get_text(" ", strip=True)) for th in el.find_all("th")):
            recs.extend(_supervision_table(el, kind, "current"))
    return _supervision_frame(recs)


def parse_supervision_history(html: str | bytes) -> pd.DataFrame:
    """指定履歴：kind 由「内容」列推断；解除年月日 "-" → None 且 is_current=True。"""
    recs = []
    for t in _tables_with(_soup(html), "指定年月日"):
        recs.extend(_supervision_table(t, None, "history"))
    return _supervision_frame(recs)


def merge_supervision(current: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """合并两页：以履歴为主（含 解除年月日 / 内容），现行页补充未出现在履歴中的记录，并标记 is_current。"""
    keys = TABLES["supervision"].keys
    if history.empty:
        return current
    if current.empty:
        return history
    cur_keys = set(map(tuple, current[keys].astype(object).values.tolist()))
    hist = history.copy()
    hist["is_current"] = [tuple(k) in cur_keys or bool(c) for k, c in zip(hist[keys].astype(object).values.tolist(), hist["is_current"])]
    hist_keys = set(map(tuple, hist[keys].astype(object).values.tolist()))
    extra = current[[tuple(k) not in hist_keys for k in current[keys].astype(object).values.tolist()]]
    return pd.concat([hist, extra], ignore_index=True)


# ----------------------------------------------------------------------------------------------
# 9. 新規上場会社情報（HTML；每家公司两行：行 1 含 rowspan=2 的上場日/会社名）
# ----------------------------------------------------------------------------------------------
NEW_LISTING_COLUMNS = ["listing_date", "approval_date", "Code", "name", "market", "company_url", "price_range", "offer_price",
                       "public_kshares", "secondary_kshares", "oa_kshares", "trading_unit", "outline_url", "confirmation_url",
                       "part1_url", "cg_report_url", "kessan_url"]


def _split_offering(s: str | None) -> tuple[float, float]:
    """"12,216.6(OA2,594.6)" → (12216.6, 2594.6)；"-" → (NaN, NaN)。"""
    if not s:
        return np.nan, np.nan
    t = unicodedata.normalize("NFKC", s)
    m = re.match(r"^\s*([\d,.]+)", t)
    oa = re.search(r"OA\s*([\d,.]+)", t)
    return (_num(m.group(1)) if m else np.nan), (_num(oa.group(1)) if oa else np.nan)


def parse_new_listings(html: str | bytes) -> pd.DataFrame:
    recs: list[dict[str, Any]] = []
    for t in _tables_with(_soup(html), "上場日"):
        pending: dict[str, Any] | None = None
        for tr in t.find_all("tr"):
            tds = tr.find_all("td")
            if not tds:
                continue
            if tds[0].get("rowspan") or len(tds) >= 8:
                dates = re.findall(r"\d{4}/\d{1,2}/\d{1,2}", tds[0].get_text(" ", strip=True))
                name_cell = tds[1]
                for a in name_cell.find_all("a"):
                    if "インタビュー" in a.get_text() or "link-blank" in (a.get("class") or []):
                        a.decompose()
                a = name_cell.find("a", href=True)
                sec, oa = np.nan, np.nan
                pending = {
                    "listing_date": _date(dates[0]) if dates else None, "approval_date": _date(dates[1]) if len(dates) > 1 else None,
                    "Code": norm_code(tds[2].get_text(strip=True)), "name": _clean(name_cell.get_text(" ", strip=True)),
                    "market": None, "company_url": a["href"] if a else None,
                    "price_range": _clean(tds[5].get_text(" ", strip=True)) if len(tds) > 5 else None, "offer_price": np.nan,
                    "public_kshares": _num(tds[6].get_text(strip=True)) if len(tds) > 6 else np.nan,
                    "secondary_kshares": sec, "oa_kshares": oa,
                    "trading_unit": _num(tds[7].get_text(strip=True)) if len(tds) > 7 else np.nan,
                    "outline_url": _href(tds[3]) if len(tds) > 3 else None, "confirmation_url": _href(tds[4]) if len(tds) > 4 else None,
                    "part1_url": None, "cg_report_url": None, "kessan_url": None,
                }
                continue
            if pending is None:
                continue
            sec, oa = _split_offering(tds[4].get_text(strip=True) if len(tds) > 4 else None)
            pending.update({
                "market": _clean(tds[0].get_text(strip=True)), "part1_url": _href(tds[1]) if len(tds) > 1 else None,
                "cg_report_url": _href(tds[2]) if len(tds) > 2 else None,
                "offer_price": _num(tds[3].get_text(strip=True)) if len(tds) > 3 else np.nan,
                "secondary_kshares": sec, "oa_kshares": oa, "kessan_url": _href(tds[5]) if len(tds) > 5 else None,
            })
            if pending["Code"] and pending["listing_date"]:
                recs.append(pending)
            pending = None
    return _frame(recs, NEW_LISTING_COLUMNS,
                  text=["name", "market", "company_url", "price_range", "outline_url", "confirmation_url", "part1_url",
                        "cg_report_url", "kessan_url"],
                  dates=["listing_date", "approval_date"],
                  nums=["offer_price", "public_kshares", "secondary_kshares", "oa_kshares", "trading_unit"],
                  keys=TABLES["new_listings"].keys)


def month_partition(values: Iterable[Any]) -> list[str]:
    """日期 → "YYYY-MM" 分区名；None → "undecided"。"""
    return [d.strftime("%Y-%m") if isinstance(d, (date, datetime)) else "undecided" for d in values]
