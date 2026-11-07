"""宏观序列采集作业：財務省国債金利、日銀時系列 API、FRED → macro_series（分区 all，主键 series + date）。

频率（19.1 节）：每日一次即可。日銀 API 禁止短时高频访问，每个序列每次运行只请求一次；已有历史时只取最近的数据，
不必每天下载 5MB 全史。jgbcm_all.csv 每月取一次（首次运行或跨月时），平时只取当月的 jgbcm.csv。
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Any, Callable, Iterable

import pandas as pd
import requests

from .. import __version__
from ..config import Settings
from ..connectors.base import RawSink, get_with_retry, http_session
from ..connectors.macro import (BOJ_SERIES, FRED_SERIES, JGB_COLUMNS, MOF_ALL_URL, MOF_CURRENT_URL, boj_url, fetch_fred, parse_boj_csv,
                                parse_fred_json, parse_jgb_csv)
from ..ledger import Ledger
from ..paths import DataPaths
from ..store import Store
from ..util.common import iso_utc, now_jst, sha256_bytes, today_jst

log = logging.getLogger("jpscan.ingest.macro")

TABLE = "macro_series"
PARTITION = "all"
KEYS = ["series", "date"]
DATASET = "macro"
MIN_INTERVAL = 2.0          # 秒；相邻请求最小间隔
REFETCH_DAYS = 40           # 增量回看天数：覆盖来源对近期值的修订
KV_JGB_ALL_MONTH = "macro.jgb_all_month"


def all_series() -> list[str]:
    return [*JGB_COLUMNS.values(), *BOJ_SERIES, *FRED_SERIES]


def _latest_dates(store: Store) -> dict[str, date]:
    if not store.exists("staging", TABLE):
        return {}
    df = store.query("SELECT series, max(date) AS d FROM m GROUP BY series", {"m": ("staging", TABLE)})
    return {r["series"]: pd.Timestamp(r["d"]).date() for _, r in df.iterrows()}


def _save_raw(sink: RawSink, dataset: str, key: str, content: bytes, ext: str, params: dict[str, Any], status: int) -> None:
    """内容没变（账本里已有同哈希）就不再存一份。"""
    if not sink.ledger.raw_exists_hash(sha256_bytes(content)):
        sink.save_bytes(dataset, key, content, ext, params=params, http_status=status)


def _get(session: requests.Session, url: str) -> requests.Response:
    r = get_with_retry(session, url, timeout=120, retries=3)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    return r


class _Run:
    """一次运行的共享状态：请求间隔、账本标记、结果收集。"""

    def __init__(self, ledger: Ledger):
        self.ledger = ledger
        self.session = http_session()
        self.stamp = now_jst().strftime("%Y%m%dT%H%M%S")
        self.frames: list[pd.DataFrame] = []
        self.sources: dict[str, str] = {}
        self._first = True

    def step(self, key: str, fn: Callable[[], pd.DataFrame]) -> bool:
        """跑一个来源：失败只记账，不影响其他来源。"""
        if not self._first:
            time.sleep(MIN_INTERVAL)
        self._first = False
        try:
            df = fn()
        except Exception as ex:  # noqa: BLE001
            log.warning("macro %s 失败：%s", key, ex)
            self.ledger.mark(DATASET, key, "error", error=str(ex)[:300])
            self.sources[key] = f"error: {str(ex)[:200]}"
            return False
        self.ledger.mark(DATASET, key, "ok", len(df))
        self.sources[key] = "ok"
        if not df.empty:
            self.frames.append(df)
        return True


def _ingest_jgb(ctx: _Run, sink: RawSink, want: set[str], latest: dict[str, date]) -> None:
    def fetch(url: str, key: str) -> pd.DataFrame:
        r = _get(ctx.session, url)
        _save_raw(sink, "jgb", f"{ctx.stamp}_{key}", r.content, ".csv", {"url": url}, r.status_code)
        try:
            df = parse_jgb_csv(r.content)
        except Exception as ex:
            sink.quarantine("jgb", f"{ctx.stamp}_{key}", r.content, str(ex))
            raise
        return df[df["series"].isin(want)]

    month = today_jst().strftime("%Y-%m")
    no_history = any(s in want and s not in latest for s in JGB_COLUMNS.values())
    if no_history or ctx.ledger.get_kv(KV_JGB_ALL_MONTH) != month:
        if ctx.step("jgb_all", lambda: fetch(MOF_ALL_URL, "jgbcm_all")):
            ctx.ledger.set_kv(KV_JGB_ALL_MONTH, month)
    # 当月文件排在全史之后：同一日期以当月文件为准
    ctx.step("jgb_current", lambda: fetch(MOF_CURRENT_URL, "jgbcm"))


def _ingest_boj(ctx: _Run, sink: RawSink, want: set[str], latest: dict[str, date]) -> None:
    for series, (db, code) in BOJ_SERIES.items():
        if series not in want:
            continue

        def fetch(series: str = series, db: str = db, code: str = code) -> pd.DataFrame:
            since = latest.get(series)
            start_ym = (since - timedelta(days=REFETCH_DAYS)).strftime("%Y%m") if since else None
            url = boj_url(db, code, start_ym)
            r = _get(ctx.session, url)
            _save_raw(sink, series, f"{ctx.stamp}_{series}", r.content, ".csv", {"url": url}, r.status_code)
            try:
                return parse_boj_csv(r.content, series, code)
            except Exception as ex:
                sink.quarantine(series, f"{ctx.stamp}_{series}", r.content, str(ex))
                raise

        ctx.step(f"boj_{series}", fetch)


def _ingest_fred(ctx: _Run, sink: RawSink, settings: Settings, ledger: Ledger, want: set[str], latest: dict[str, date]) -> None:
    wanted = [s for s in FRED_SERIES if s in want]
    if not wanted:
        return
    if not settings.fred_api_key:
        ledger.mark(DATASET, "fred", "skipped_no_key")
        ctx.sources["fred"] = "skipped_no_key"
        return
    for series in wanted:
        series_id, unit = FRED_SERIES[series]

        def fetch(series: str = series, series_id: str = series_id, unit: str = unit) -> pd.DataFrame:
            since = latest.get(series)
            start = since - timedelta(days=REFETCH_DAYS) if since else None
            r = fetch_fred(ctx.session, series_id, settings.fred_api_key, start)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            # 账本参数与文件名都不含 key
            _save_raw(sink, series, f"{ctx.stamp}_{series}", r.content, ".json",
                      {"series_id": series_id, "start": str(start) if start else None}, r.status_code)
            return parse_fred_json(r.content, series, unit)

        ctx.step(f"fred_{series}", fetch)


def _summary(store: Store) -> dict[str, dict[str, Any]]:
    if not store.exists("staging", TABLE):
        return {}
    df = store.query("SELECT series, count(*) AS n, max(date) AS d, arg_max(value, date) AS v FROM m GROUP BY series ORDER BY series",
                     {"m": ("staging", TABLE)})
    return {r["series"]: {"rows": int(r["n"]), "latest_date": pd.Timestamp(r["d"]).date().isoformat(), "latest_value": float(r["v"])}
            for _, r in df.iterrows()}


def run(settings: Settings, paths: DataPaths, ledger: Ledger, store: Store, series: Iterable[str] | None = None) -> dict[str, Any]:
    want = set(series) if series is not None else set(all_series())
    unknown = want - set(all_series())
    if unknown:
        raise ValueError(f"未知序列：{sorted(unknown)}")
    run_id = ledger.start_run("ingest_macro", __version__, settings.params_version)
    latest = _latest_dates(store)
    ctx = _Run(ledger)

    if want & set(JGB_COLUMNS.values()):
        _ingest_jgb(ctx, RawSink(paths, ledger, "mof"), want, latest)
    _ingest_boj(ctx, RawSink(paths, ledger, "boj"), want, latest)
    _ingest_fred(ctx, RawSink(paths, ledger, "fred"), settings, ledger, want, latest)

    n_written = 0
    if ctx.frames:
        df = pd.concat(ctx.frames, ignore_index=True).drop_duplicates(KEYS, keep="last")
        df["fetched_at"] = iso_utc()
        df["value"] = df["value"].astype(float)
        store.write_merge("staging", TABLE, df, PARTITION, KEYS)
        n_written = len(df)

    failed = [k for k, v in ctx.sources.items() if v.startswith("error")]
    status = "ok" if not failed else "partial"
    ledger.finish_run(run_id, status, note=f"rows={n_written} failed={failed}")
    return {"run_id": run_id, "status": status, "sources": ctx.sources, "rows_written": n_written, "series": _summary(store)}
