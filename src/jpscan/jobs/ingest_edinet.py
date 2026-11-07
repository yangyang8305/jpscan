"""EDINET 采集作业：书类一览 → edinet_documents；CSV → financial_details_edinet、edinet_key_items；代码表 → edinet_codes。

规划文档 10.5 节 `edinet.api`、10.6 节 B5。按提出日逐日循环，可断点续传；单线程、≥1 秒一次（由 EdinetClient 限速）。

几点行为（依据 EDINET API 仕様書 3-1 节）：
- edinet_documents 保留所有书类种别（上市相关），doc_types 只决定下载哪些；临时报告、回购、要约收购等事件逻辑也要用元数据。
- 大量保有报告的提交者常是非上市基金，secCode 为空：用 issuerEdinetCode（大量保有）、subjectEdinetCode（要约收购）
  经 edinet_codes 映射到 target_code，这些书类也算「上市相关」。Code 始终是提交者自身的代码。
- 取下げ、閲覧期間満了后，一览里该 docID 的元数据被置 null：合并时保留已有元数据，只更新状态列（tombstone）。
- 当日的一览会持续追加，不标记为完成；过去日的一览有更新需求时用 resume=False 重取。
- 书类的会计期间列命名为 period_start_date / period_end_date：Store 只把名字含 date 的全空列定为日期类型，
  否则某月全是临时报告时该列成了字符串，跨分区读取会把所有分区的日期都变成字符串。
- financial_details_edinet 不存 TextBlock（大段注记文本，单值可达 3 万字）：原始 zip 已保留，需要时可从原始层再解析。
"""
from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from typing import Any, Iterable

import pandas as pd

from ..config import Settings
from ..connectors.edinet import (DOC_EXT, EdinetAuthError, EdinetClient, EdinetError, EdinetKeyMissing, EdinetNotFound, KEY_HELP,
                                 doc_kind, normalize_sec_code, parse_code_list)
from ..connectors.edinet_csv import extract_key_items, parse_financial_csv_zip
from ..ledger import Ledger
from ..paths import DataPaths
from ..store import Store
from ..util.common import iso_utc, to_date, today_jst

log = logging.getLogger("jpscan.ingest.edinet")

DEFAULT_DOC_TYPES = ("120", "160", "180", "350")
TABLE_DOCS = "edinet_documents"
TABLE_DETAILS = "financial_details_edinet"
TABLE_KEY_ITEMS = "edinet_key_items"
TABLE_CODES = "edinet_codes"
KEYS_DETAILS = ["docID", "file", "element_id", "context_id"]
KEYS_KEY_ITEMS = ["docID", "consolidated", "item"]
FLUSH_DOCS = 200   # 有報高峰日一天上千份，分批落盘控制内存

# 下载类型 → (一览中的有无标志字段, edinet_documents 的路径列)
TYPE_FIELDS = {1: ("xbrl_flag", "path_xbrl"), 2: ("pdf_flag", "path_pdf"), 3: ("attach_doc_flag", "path_attach"),
               4: ("english_doc_flag", "path_english"), 5: ("csv_flag", "path_csv")}
STATUS_COLUMNS = ["withdrawal_status", "doc_info_edit_status", "disclosure_status", "xbrl_flag", "pdf_flag", "attach_doc_flag",
                  "english_doc_flag", "csv_flag", "legal_status", "list_date", "fetched_at"]
_TARGET_FIELD = {"350": "issuer_edinet_code", "360": "issuer_edinet_code", "240": "subject_edinet_code", "250": "subject_edinet_code",
                 "270": "subject_edinet_code", "280": "subject_edinet_code"}


def parse_download_types(download_types: str | Iterable[int | str] | None) -> list[int]:
    """"5"、"1,5"、[1, 5] → [1, 5]。"""
    if download_types is None:
        return []
    text = download_types if isinstance(download_types, str) else ",".join(str(t) for t in download_types)
    types = sorted({int(t) for t in re.findall(r"\d", text)})
    bad = [t for t in types if t not in DOC_EXT]
    if bad:
        raise ValueError(f"unknown download types: {bad}")
    return types


def _str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _date(v: Any) -> date | None:
    try:
        return to_date(_str(v))
    except ValueError:
        return None


def normalize_documents(rows: list[dict[str, Any]], list_date: date, code_map: dict[str, str] | None = None) -> pd.DataFrame:
    """书类一览 results → edinet_documents 的行（未过滤）。"""
    code_map = code_map or {}
    fetched_at = iso_utc()
    out: list[dict[str, Any]] = []
    for r in rows:
        doc_id = _str(r.get("docID"))
        if not doc_id:
            continue
        code, code4 = normalize_sec_code(r.get("secCode"))
        dtype = _str(r.get("docTypeCode"))
        dtype = dtype.zfill(3) if dtype else None
        form = _str(r.get("formCode"))
        submitted = _str(r.get("submitDateTime"))
        issuer, subject = _str(r.get("issuerEdinetCode")), _str(r.get("subjectEdinetCode"))
        target_field = _TARGET_FIELD.get(dtype or "")
        target_edinet = {"issuer_edinet_code": issuer, "subject_edinet_code": subject}.get(target_field or "")
        out.append({
            "docID": doc_id, "seq_number": r.get("seqNumber"), "edinet_code": _str(r.get("edinetCode")), "sec_code": _str(r.get("secCode")),
            "Code": code, "code4": code4, "target_code": code_map.get(target_edinet or ""), "jcn": _str(r.get("JCN")),
            "filer_name": _str(r.get("filerName")), "fund_code": _str(r.get("fundCode")), "ordinance_code": _str(r.get("ordinanceCode")),
            "form_code": form, "doc_type_code": dtype, "doc_kind": doc_kind(dtype, form),
            "period_start_date": _date(r.get("periodStart")), "period_end_date": _date(r.get("periodEnd")),
            "submitted_at": submitted, "submit_date": _date(submitted), "doc_description": _str(r.get("docDescription")),
            "issuer_edinet_code": issuer, "subject_edinet_code": subject, "subsidiary_edinet_code": _str(r.get("subsidiaryEdinetCode")),
            "current_report_reason": _str(r.get("currentReportReason")), "parent_doc_id": _str(r.get("parentDocID")),
            "ope_at": _str(r.get("opeDateTime")), "withdrawal_status": _str(r.get("withdrawalStatus")),
            "doc_info_edit_status": _str(r.get("docInfoEditStatus")), "disclosure_status": _str(r.get("disclosureStatus")),
            "xbrl_flag": _str(r.get("xbrlFlag")) == "1", "pdf_flag": _str(r.get("pdfFlag")) == "1",
            "attach_doc_flag": _str(r.get("attachDocFlag")) == "1", "english_doc_flag": _str(r.get("englishDocFlag")) == "1",
            "csv_flag": _str(r.get("csvFlag")) == "1", "legal_status": _str(r.get("legalStatus")),
            "path_xbrl": None, "path_pdf": None, "path_attach": None, "path_english": None, "path_csv": None,
            "list_date": list_date, "fetched_at": fetched_at,
        })
    return pd.DataFrame(out)


def _is_tombstone(df: pd.DataFrame) -> pd.Series:
    """元数据已被 EDINET 置空的行（取下げ、閲覧期間満了）。"""
    return df["doc_type_code"].isna() & df["submitted_at"].isna() & df["filer_name"].isna()


def filter_listed(df: pd.DataFrame, has_codes: bool) -> pd.DataFrame:
    """只留上市相关书类：提交者有证券代码，或大量保有/要约收购的对象是上市公司。

    tombstone 总是留下（由写入时决定是否命中已有行）。没有 edinet_codes 时无法映射对象，凡有对象代码的都先留下。
    """
    keep = df["Code"].notna() | df["target_code"].notna() | _is_tombstone(df)
    if not has_codes:
        keep |= df["issuer_edinet_code"].notna() | df["subject_edinet_code"].notna()
    return df[keep]


def _month(row_date: Any, list_date: date) -> str:
    """分区月份：提出日，缺失（tombstone）时用一览日期。两者对同一 docID 落在同一个月。"""
    return (row_date if isinstance(row_date, date) else list_date).strftime("%Y-%m")


def write_documents(store: Store, df: pd.DataFrame, listed_only: bool) -> int:
    """按月分区写 edinet_documents。tombstone 与已有行合并：保留元数据与路径，只更新状态列。"""
    n = 0
    months = pd.Series([_month(s, l) for s, l in zip(df["submit_date"], df["list_date"])], index=df.index)
    for month, part in df.groupby(months, sort=True):   # 用 Series 分组：传 list 时单行 df 的组名会变成元组
        tomb = _is_tombstone(part)
        dead = part[tomb]
        if not dead.empty:
            existing = store.read("staging", TABLE_DOCS, partitions=[str(month)])
            old = {r["docID"]: r for r in existing.to_dict("records")} if not existing.empty else {}
            merged: list[dict[str, Any]] = []
            for rec in dead.to_dict("records"):
                if rec["docID"] in old:
                    base = {**old[rec["docID"]], **{c: rec[c] for c in STATUS_COLUMNS}}
                    for c in ("period_start_date", "period_end_date", "submit_date"):   # DuckDB 读回为 Timestamp
                        base[c] = None if pd.isna(base[c]) else _date(base[c])
                    merged.append(base)
                elif not listed_only:
                    merged.append(rec)
            part = pd.concat([part[~tomb], pd.DataFrame(merged)], ignore_index=True) if merged else part[~tomb]
        if part.empty:
            continue
        store.write_merge("staging", TABLE_DOCS, part, str(month), ["docID"])
        n += len(part)
    return n


class _DetailBuffer:
    """明细与关键科目的缓冲：按月分区 merge 写入。落盘之后才在账本标记 edinet_csv，中断时未落盘的书类下次会重新解析。"""

    def __init__(self, store: Store, ledger: Ledger):
        self.store = store
        self.ledger = ledger
        self.frames: list[pd.DataFrame] = []
        self.rows = 0
        self.key_items = 0

    def add(self, df: pd.DataFrame) -> None:
        self.frames.append(df)
        if len(self.frames) >= FLUSH_DOCS:
            self.flush()

    def flush(self) -> None:
        if not self.frames:
            return
        frames, self.frames = self.frames, []
        all_rows = pd.concat(frames, ignore_index=True).drop_duplicates(KEYS_DETAILS, keep="last")
        items = extract_key_items(all_rows.rename(columns={"period_end_date": "period_end"}))
        for month, part in all_rows.groupby("month", sort=True):
            self.store.write_merge("staging", TABLE_DETAILS, part.drop(columns="month"), str(month), KEYS_DETAILS)
        self.rows += len(all_rows)
        if not items.empty:
            item_months = pd.Series([d.strftime("%Y-%m") for d in items["disc_date"]], index=items.index)
            for month, part in items.groupby(item_months, sort=True):
                self.store.write_merge("staging", TABLE_KEY_ITEMS, part, str(month), KEYS_KEY_ITEMS)
            self.key_items += len(items)
        for f in frames:
            self.ledger.mark("edinet_csv", f["docID"].iloc[0], "ok", len(f))


def _load_code_map(store: Store) -> dict[str, str]:
    """edinet_code → 上市 Code。"""
    if not store.exists("staging", TABLE_CODES):
        return {}
    df = store.read("staging", TABLE_CODES, columns=["edinet_code", "Code"])
    df = df[df["Code"].notna()]
    return dict(zip(df["edinet_code"], df["Code"]))


def run(settings: Settings, paths: DataPaths, ledger: Ledger, store: Store, start: date | str, end: date | str,
        doc_types: Iterable[str] | None = None, download_types: str | Iterable[int | str] | None = "5", listed_only: bool = True,
        client: EdinetClient | None = None, resume: bool = True) -> dict[str, int]:
    """逐日采集 [start, end]。缺 Key 抛 EdinetKeyMissing；Key 无效抛 EdinetAuthError；其他错误记入账本并继续。"""
    start_d, end_d = to_date(start), to_date(end)
    if start_d is None or end_d is None:
        raise ValueError("start and end are required")
    if client is None:
        if not settings.edinet_api_key:
            raise EdinetKeyMissing(KEY_HELP)
        client = EdinetClient(settings.edinet_api_key, paths, ledger)
    want_types = {str(t).zfill(3) for t in (doc_types if doc_types is not None else DEFAULT_DOC_TYPES)}
    dl_types = parse_download_types(download_types)
    code_map = _load_code_map(store)
    if not code_map:
        log.warning("edinet_codes 不存在：大量保有/要约收购无法映射到对象代码。请先运行 run_code_list。")
    today = today_jst()
    done_lists = ledger.done_keys("edinet_list") if resume else set()
    done_csv = ledger.done_keys("edinet_csv") if resume else set()
    stats = {"days": 0, "listed": 0, "documents": 0, "downloaded": 0, "parsed": 0, "detail_rows": 0, "key_items": 0, "not_found": 0, "errors": 0}
    buffer = _DetailBuffer(store, ledger)
    try:
        d = start_d
        while d <= end_d:
            key = d.isoformat()
            try:
                rows = client.list_documents(d, use_cache=key in done_lists)
            except (EdinetAuthError, EdinetKeyMissing):
                raise
            except EdinetError as e:
                log.warning("edinet list %s failed: %s", key, e)
                ledger.mark("edinet_list", key, "error", error=str(e)[:200])
                stats["errors"] += 1
                d += timedelta(days=1)
                continue
            stats["days"] += 1
            stats["listed"] += len(rows)
            df = normalize_documents(rows, d, code_map)
            if listed_only and not df.empty:
                df = filter_listed(df, has_codes=bool(code_map))
            if not df.empty:
                df = df.copy()
                _download_and_parse(client, paths, ledger, df, want_types, dl_types, done_csv, buffer, stats)
                stats["documents"] += write_documents(store, df, listed_only)
            if d < today:   # 当日一览还会追加，不算完成
                ledger.mark("edinet_list", key, "ok", len(rows))
            d += timedelta(days=1)
    finally:
        buffer.flush()
        stats["detail_rows"] += buffer.rows
        stats["key_items"] += buffer.key_items
    return stats


def _download_and_parse(client: EdinetClient, paths: DataPaths, ledger: Ledger, df: pd.DataFrame, want_types: set[str], dl_types: list[int],
                        done_csv: set[str], buffer: _DetailBuffer, stats: dict[str, int]) -> None:
    """下载选定种别的书类，把本地路径写回 df，并解析 CSV 进缓冲。认证错误向上抛。"""
    for idx, row in df.iterrows():
        month = _month(row["submit_date"], row["list_date"])
        doc_id = row["docID"]
        wanted = row["doc_type_code"] in want_types
        for t in dl_types:
            flag_col, path_col = TYPE_FIELDS[t]
            path = client.doc_path(doc_id, t, month)
            if not path.exists():
                if not (wanted and row[flag_col]):
                    continue
                try:
                    path = client.download(doc_id, t, subdir=month)
                    stats["downloaded"] += 1
                except EdinetNotFound as e:
                    ledger.mark(f"edinet_doc_type{t}", doc_id, "not_found", error=str(e)[:200])
                    stats["not_found"] += 1
                    continue
                except (EdinetAuthError, EdinetKeyMissing):
                    raise
                except EdinetError as e:
                    log.warning("edinet download %s type=%s failed: %s", doc_id, t, e)
                    ledger.mark(f"edinet_doc_type{t}", doc_id, "error", error=str(e)[:200])
                    stats["errors"] += 1
                    continue
            df.at[idx, path_col] = path.relative_to(paths.root).as_posix()
            if t == 5 and wanted and doc_id not in done_csv:
                _parse_csv(path, row, month, ledger, buffer, stats)
                done_csv.add(doc_id)


def _parse_csv(path: Any, row: pd.Series, month: str, ledger: Ledger, buffer: _DetailBuffer, stats: dict[str, int]) -> None:
    try:
        parsed = parse_financial_csv_zip(path, doc_id=row["docID"])
    except Exception as e:  # noqa: BLE001  单个坏文件不应中断回填；原始 zip 已保留
        log.warning("edinet csv parse %s failed: %s", row["docID"], e)
        ledger.mark("edinet_csv", row["docID"], "error", error=str(e)[:200])
        stats["errors"] += 1
        return
    if parsed.attrs.get("skipped_files"):
        log.info("%s: skipped csv files %s", row["docID"], parsed.attrs["skipped_files"])
    parsed = parsed[~parsed["element_id"].astype("string").str.endswith("TextBlock", na=False)]
    if not parsed.empty:
        parsed = parsed.copy()
        parsed["Code"] = row["Code"]
        parsed["code4"] = row["code4"]
        parsed["doc_type_code"] = row["doc_type_code"]
        parsed["period_end_date"] = row["period_end_date"]
        parsed["submit_date"] = row["submit_date"] if isinstance(row["submit_date"], date) else row["list_date"]
        parsed["month"] = month
        buffer.add(parsed)
    else:
        ledger.mark("edinet_csv", row["docID"], "ok", 0)
    stats["parsed"] += 1


def run_code_list(settings: Settings, paths: DataPaths, ledger: Ledger, store: Store, client: EdinetClient | None = None) -> dict[str, Any]:
    """下载并写入 edinet_codes（不需要 API Key）。退市公司等从代码表消失的行保留在表中。"""
    client = client or EdinetClient(settings.edinet_api_key, paths, ledger)
    zip_path = client.fetch_code_list()
    df = parse_code_list(zip_path)
    if df.empty:
        raise EdinetError("EdinetcodeDlInfo.csv 解析结果为空")
    store.write_merge("staging", TABLE_CODES, df, "all", ["edinet_code"])
    as_of = df["as_of"].iloc[0]
    ledger.mark(TABLE_CODES, str(as_of or today_jst()), "ok", len(df))
    return {"rows": len(df), "listed": int(df["listed"].sum()), "with_code": int(df["Code"].notna().sum()), "as_of": as_of}
