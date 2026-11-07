"""TDnet 采集作业（规划文档 10.5 节 tdnet.daily）：日次列表 → 分类 → disclosures 表 + evidence，可选下载 PDF/XBRL。

账本 dataset：tdnet_list（列表完整）、tdnet_files（文件下载完整），key 为日期。
只有当日已结束后抓取的完整列表才标记 ok；当日标记 partial 且每次重抓，否则傍晚的新披露会被跳过。
"""
from __future__ import annotations

import logging
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .. import __version__
from ..config import Settings, pget
from ..connectors.base import RawSink
from ..connectors.tdnet import SOURCE, WINDOW_DAYS, TdnetBlocked, TdnetClient
from ..events.classify import classify_title, dedupe_ja_en, download_policy, is_correction_title, load_taxonomy
from ..ledger import Ledger
from ..paths import DataPaths
from ..store import Store
from ..util.common import iso_utc, sha256_bytes, sha256_file, to_date, today_jst

log = logging.getLogger("jpscan.ingest.tdnet")

TABLE = "disclosures"
KEYS = ["disclosure_id"]
LIST_DS, FILES_DS = "tdnet_list", "tdnet_files"
COLUMNS = ["disclosure_id", "disclosed_date", "disclosed_time", "disclosed_at", "Code", "code4", "company", "title", "lang", "exchanges",
           "is_correction", "category", "underlying", "direction", "strength", "half_life_days", "hard_exclude", "special", "needs_ai",
           "matched_keyword", "pdf_url", "pdf_path", "xbrl_url", "xbrl_path", "page", "fetched_at"]


def _rel(paths: DataPaths, path: Path) -> str:
    return path.relative_to(paths.root).as_posix()


def _raw_file(paths: DataPaths, dataset: str, d: date, key: str, ext: str) -> Path:
    return paths.raw_dir(SOURCE, dataset) / d.strftime("%Y-%m") / f"{key}{ext}"


def _build_rows(listed: list[dict[str, Any]], d: date, taxonomy: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """列表行 → 表记录（已分类、已日英去重）。返回 (记录, 去掉的英文行数)。"""
    records = []
    fetched_at = iso_utc()
    for r in listed:
        c = classify_title(r["title"], taxonomy)
        hhmm = r["time"] or "00:00"
        records.append({
            "disclosure_id": r["disclosure_id"], "disclosed_date": d, "disclosed_time": f"{hhmm}:00",
            "disclosed_at": f"{d.isoformat()}T{hhmm}:00+09:00", "Code": r["Code"], "code4": r["code4"], "company": r["company"],
            "title": r["title"], "lang": c["lang"], "exchanges": r["exchanges"],
            "is_correction": bool(r["correction_marker"]) or is_correction_title(r["title"], taxonomy),
            "category": c["category"], "underlying": c["underlying"], "direction": c["direction"], "strength": c["strength"],
            "half_life_days": c["half_life_days"], "hard_exclude": c["hard_exclude"], "special": c["special"], "needs_ai": c["needs_ai"],
            "matched_keyword": c["matched_keyword"], "pdf_url": r["pdf_url"], "pdf_path": None, "xbrl_url": r["xbrl_url"], "xbrl_path": None,
            "page": r["page"], "fetched_at": fetched_at,
        })
    kept, dropped = dedupe_ja_en(records, taxonomy)
    return kept, len(dropped)


def _attach_existing_files(records: list[dict[str, Any]], d: date, paths: DataPaths) -> None:
    """路径由磁盘决定，这样 download=False 的重抓不会把已下载文件的路径盖成空。"""
    for r in records:
        for column, dataset, ext in (("pdf_path", "pdf", ".pdf"), ("xbrl_path", "xbrl", ".zip")):
            f = _raw_file(paths, dataset, d, r["disclosure_id"], ext)
            r[column] = _rel(paths, f) if f.exists() else None


def _download_files(records: list[dict[str, Any]], d: date, client: TdnetClient, paths: DataPaths, taxonomy: dict[str, Any],
                    stats: dict[str, Any]) -> int:
    """按 download_policy 下载缺失文件，返回失败件数。TdnetBlocked 向上抛，立即结束运行。"""
    failures = 0
    subdir = d.strftime("%Y-%m")
    for r in records:
        policy = download_policy(r, taxonomy)
        for allowed, url, dataset, ext, magic in ((policy["pdf"], r["pdf_url"], "pdf", ".pdf", b"%PDF"),
                                                  (policy["xbrl"], r["xbrl_url"], "xbrl", ".zip", b"PK")):
            if not allowed or not isinstance(url, str) or _raw_file(paths, dataset, d, r["disclosure_id"], ext).exists():
                continue
            try:
                # 文件按 disclosure_id 命名：XBRL 的 0812… 文件名与 PDF 的 1401… 共用同一 id，便于回查
                saved = client.download_file(url, dataset, r["disclosure_id"], ext, magic, subdir)
            except TdnetBlocked:
                raise
            except Exception as e:  # noqa: BLE001
                log.warning("download failed %s (%s): %s", r["disclosure_id"], dataset, e)
                saved = None
            if saved is None:
                failures += 1
            else:
                stats[f"{dataset}_downloaded"] += 1
    return failures


def _write(records: list[dict[str, Any]], d: date, store: Store) -> None:
    df = pd.DataFrame(records, columns=COLUMNS)
    df["half_life_days"] = pd.to_numeric(df["half_life_days"], errors="coerce").astype("float64")   # 分区间类型一致；NaN = 不衰减
    store.write_merge("staging", TABLE, df, d.strftime("%Y-%m"), KEYS)


def _add_evidence(records: list[dict[str, Any]], paths: DataPaths, ledger: Ledger) -> None:
    """每条披露一条证据。content_hash：已下载则为 PDF 哈希，否则为标题哈希（INSERT OR IGNORE，之后下载不会改写）。"""
    for r in records:
        pdf = paths.root / r["pdf_path"] if r.get("pdf_path") else None
        digest = sha256_file(pdf) if pdf is not None and pdf.exists() else sha256_bytes(r["title"].encode("utf-8"))
        ledger.add_evidence(evidence_id=f"EV-tdnet-{r['disclosure_id']}", source_type="tdnet", doc_id=r["disclosure_id"], url=r["pdf_url"],
                            disclosed_at=r["disclosed_at"], locator={}, snippet=r["title"], snippet_lang=r["lang"], content_hash=digest,
                            internal_id=r["Code"])


def _stored_records(d: date, store: Store) -> list[dict[str, Any]]:
    if not store.exists("staging", TABLE):
        return []
    df = store.read("staging", TABLE, where=f"disclosed_date = DATE '{d.isoformat()}'", partitions=[d.strftime("%Y-%m")])
    df = df.astype(object).where(df.notna(), None)
    records = df.to_dict("records")
    for r in records:
        r["disclosed_date"] = to_date(r["disclosed_date"])
    return records


def _window(start: date | str | None, end: date | str | None, today: date) -> list[date]:
    """默认昨天和今天（JST）；截到站点保留的窗口内。"""
    last = min(to_date(end) or today, today)
    first = to_date(start) or (last - timedelta(days=1))
    earliest = today - timedelta(days=WINDOW_DAYS)
    if first < earliest:
        log.warning("start %s is outside the %d-day window; using %s", first, WINDOW_DAYS, earliest)
        first = earliest
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def run(settings: Settings, paths: DataPaths, ledger: Ledger, store: Store, start: date | str | None = None, end: date | str | None = None,
        download: bool = True, client: TdnetClient | None = None) -> dict[str, Any]:
    """采集 [start, end]，返回统计。站点返回 403/429 时立即结束，已完成的日期保留。"""
    if not pget(settings.params, "data_sources.tdnet.enabled", True):
        log.info("tdnet disabled in params")
        return {"status": "disabled"}
    taxonomy = load_taxonomy(settings.root / "config" / "events.yaml")
    client = client or TdnetClient(RawSink(paths, ledger, SOURCE))
    today = today_jst()
    days = _window(start, end, today)
    stats: dict[str, Any] = {"status": "ok", "days_fetched": 0, "days_skipped": 0, "days_failed": 0, "days_incomplete": [], "rows": 0,
                             "en_deduped": 0, "pdf_downloaded": 0, "xbrl_downloaded": 0, "file_failures": 0, "by_category": {}, "errors": []}
    by_category: Counter[str] = Counter()
    run_id = ledger.start_run("ingest_tdnet", __version__, settings.params_version, note=f"{days[0]}..{days[-1]} download={download}" if days else None)
    for d in days:
        key = d.isoformat()
        list_done = d < today and ledger.status_of(LIST_DS, key) == "ok"
        files_done = d < today and ledger.status_of(FILES_DS, key) == "ok"
        if list_done and (files_done or not download):
            stats["days_skipped"] += 1
            continue
        try:
            complete = True
            if list_done:
                records = _stored_records(d, store)
                stats["days_skipped"] += 1
            else:
                day = client.fetch_day(d)
                records, dropped = _build_rows(day["rows"], d, taxonomy)
                complete = day["complete"]
                stats["days_fetched"] += 1
                stats["rows"] += len(records)
                stats["en_deduped"] += dropped
                by_category.update(r["category"] for r in records)
                if not complete:
                    stats["days_incomplete"].append(key)
                    log.warning("%s: fetched %d rows, page declared %s", key, len(day["rows"]), day["total"])
            failures = _download_files(records, d, client, paths, taxonomy, stats) if download else 0
            stats["file_failures"] += failures
            _attach_existing_files(records, d, paths)
            if records:
                _write(records, d, store)
                _add_evidence(records, paths, ledger)
            if not list_done:
                ledger.mark(LIST_DS, key, "ok" if complete and d < today else "partial", n_rows=len(records))
            if download:
                ledger.mark(FILES_DS, key, "ok" if failures == 0 and d < today else "partial", n_rows=len(records))
        except TdnetBlocked as e:
            log.error("stop run: %s", e)
            ledger.mark(LIST_DS, key, "failed", error=str(e)[:500])
            stats["status"] = "blocked"
            stats["days_failed"] += 1
            stats["errors"].append(f"{key}: {e}")
            break
        except Exception as e:  # noqa: BLE001
            log.exception("%s failed", key)
            ledger.mark(LIST_DS, key, "failed", error=str(e)[:500])
            stats["status"] = "partial"
            stats["days_failed"] += 1
            stats["errors"].append(f"{key}: {e}")
    stats["by_category"] = dict(sorted(by_category.items()))
    stats["e40_share"] = round(by_category["E40"] / stats["rows"], 4) if stats["rows"] else None
    ledger.finish_run(run_id, status=stats["status"])
    log.info("tdnet ingest: %s", {k: v for k, v in stats.items() if k != "by_category"})
    return stats
