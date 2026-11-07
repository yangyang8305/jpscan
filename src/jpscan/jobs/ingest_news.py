"""新闻标题采集作业：每个 feed 取一次 → 原样落盘 → 解析、匹配公司 → news_items（按发布月分区，主键 news_id）。"""
from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
import requests

from .. import __version__
from ..config import Settings
from ..connectors.base import RawSink, get_with_retry, http_session
from ..connectors.news_rss import load_feeds, parse_feed, partition_of
from ..events.company_match import CompanyMatcher
from ..ledger import Ledger
from ..paths import DataPaths
from ..store import Store
from ..util.common import now_jst, sha256_bytes

log = logging.getLogger("jpscan.ingest.news")

TABLE = "news_items"
DATASET = "news"
RAW_SOURCE = "news_rss"
MIN_INTERVAL = 2.0    # 秒；相邻请求最小间隔（礼貌抓取）
MASTER_SQL = ("SELECT Code, arg_max(CoName, Date) AS CoName, arg_max(ProdCat, Date) AS ProdCat, max(Date) AS Date "
              "FROM m GROUP BY Code")


def load_matcher(store: Store) -> CompanyMatcher | None:
    if not store.exists("staging", "security_master_snapshots"):
        log.warning("security_master_snapshots 不存在：本次不做公司匹配")
        return None
    master = store.query(MASTER_SQL, {"m": ("staging", "security_master_snapshots")})
    return CompanyMatcher.from_master(master)


def _fetch_one(session: requests.Session, sink: RawSink, feed: dict[str, str], stamp: str,
               matcher: CompanyMatcher | None) -> tuple[pd.DataFrame, int]:
    resp = get_with_retry(session, feed["url"], timeout=60, retries=2)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}")
    # 内容没变（账本里已有同哈希）就不再存一份
    if not sink.ledger.raw_exists_hash(sha256_bytes(resp.content)):
        sink.save_bytes(feed["source"], f"{stamp}_{feed['source']}", resp.content, ".xml",
                        params={"url": feed["url"]}, http_status=resp.status_code)
    df, n_undated = parse_feed(resp.content, feed, matcher)
    if df.empty and not n_undated:
        sink.quarantine(feed["source"], f"{stamp}_{feed['source']}", resp.content, "no entries parsed")
        raise RuntimeError("no entries parsed")
    return df, n_undated


def run(settings: Settings, paths: DataPaths, ledger: Ledger, store: Store, feeds: list[dict[str, str]] | None = None) -> dict[str, Any]:
    feeds = load_feeds(settings.root / "config" / "news_feeds.yaml") if feeds is None else feeds
    run_id = ledger.start_run("ingest_news", __version__, settings.params_version)
    matcher = load_matcher(store)
    session = http_session()
    sink = RawSink(paths, ledger, RAW_SOURCE)
    stamp = now_jst().strftime("%Y%m%dT%H%M%S")

    report: dict[str, dict[str, Any]] = {}
    frames: list[pd.DataFrame] = []
    for i, feed in enumerate(feeds):
        if i:
            time.sleep(MIN_INTERVAL)
        try:
            df, n_undated = _fetch_one(session, sink, feed, stamp, matcher)
        except Exception as ex:  # noqa: BLE001  单个 feed 失败不影响其他 feed
            log.warning("feed %s 失败：%s", feed["source"], ex)
            ledger.mark(DATASET, feed["source"], "error", error=str(ex)[:300])
            report[feed["source"]] = {"name": feed["name"], "status": "error", "error": str(ex)[:300], "items": 0, "matched": 0}
            continue
        ledger.mark(DATASET, feed["source"], "ok", len(df))
        report[feed["source"]] = {"name": feed["name"], "status": "ok", "items": len(df), "undated": n_undated,
                                  "matched": int((df["matched_codes"] != "").sum())}
        if not df.empty:
            frames.append(df)

    n_written, n_matched, parts = 0, 0, []
    if frames:
        # 同一链接在多个 feed 出现时，保留配置里排在前面的
        all_items = pd.concat(frames, ignore_index=True).drop_duplicates("news_id", keep="first")
        n_written, n_matched = len(all_items), int((all_items["matched_codes"] != "").sum())
        for part, g in all_items.groupby(all_items["published_date"].map(partition_of)):
            store.write_merge("staging", TABLE, g, str(part), ["news_id"])
            parts.append(str(part))

    n_ok = sum(1 for r in report.values() if r["status"] == "ok")
    status = "ok" if n_ok == len(feeds) else ("partial" if n_ok else "error")
    ledger.finish_run(run_id, status, note=f"feeds_ok={n_ok}/{len(feeds)} items={n_written} matched={n_matched}")
    return {"run_id": run_id, "status": status, "feeds": report, "items": n_written, "matched": n_matched, "partitions": parts}
