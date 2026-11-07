"""J-Quants 采集作业：全量回填（按交易日循环、可断点续传）与每日增量（规划文档 10.5、10.6、15.1 节）。"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Iterable

import pandas as pd

from ..config import Settings
from ..connectors.jquants import DATASETS, JQuantsClient, PlanAccessError, PlanWindowError, trading_days
from ..ledger import Ledger
from ..paths import DataPaths
from ..staging.jquants_norm import TABLE_NAME, KEYS, normalize, partition_of
from ..store import Store
from ..util.common import to_date

log = logging.getLogger("jpscan.ingest.jquants")

DEFAULT_BACKFILL = ["calendar", "master", "bars_daily", "valuation", "fins_summary", "earnings_date"]
RANGE_CHUNK_DAYS = 366


class JQuantsIngest:
    def __init__(self, settings: Settings, paths: DataPaths, ledger: Ledger, store: Store, client: JQuantsClient):
        self.settings, self.paths, self.ledger, self.store, self.client = settings, paths, ledger, store, client

    # ---------------- calendar ----------------
    def ensure_calendar(self, start: date, end: date) -> list[date]:
        rows: list[dict[str, Any]] = []
        s = start
        while s <= end:
            e = min(end, s + timedelta(days=RANGE_CHUNK_DAYS))
            key = f"{s.isoformat()}_{e.isoformat()}"
            try:
                part = self.client.fetch_range("calendar", s, e)
            except PlanWindowError as ex:
                log.warning("calendar window clamp: %s", ex)
                s2, e2 = self.client.clamp(s, e)
                part = self.client.fetch_range("calendar", s2, e2) if s2 <= e2 else []
            rows.extend(part)
            self.ledger.mark("calendar", key, "ok", len(part))
            s = e + timedelta(days=1)
        df = normalize("calendar", rows)
        if not df.empty:
            self.store.write_merge("staging", TABLE_NAME["calendar"], df, "all", KEYS["calendar"])
        return trading_days(rows, start, end)

    def calendar_set(self) -> set[date]:
        df = self.store.read("staging", TABLE_NAME["calendar"])
        if df.empty:
            return set()
        div = next((c for c in df.columns if c.lower().startswith("hol")), None)
        if div is None:
            return {to_date(d) for d in df["Date"]}
        return {to_date(d) for d, v in zip(df["Date"], df[div]) if str(int(v)) in ("1", "2")} if pd.api.types.is_numeric_dtype(df[div]) \
            else {to_date(d) for d, v in zip(df["Date"], df[div]) if str(v) in ("1", "2")}

    # ---------------- generic by-date backfill ----------------
    def backfill_dataset(self, dataset: str, days: Iterable[date], resume: bool = True) -> dict[str, int]:
        spec = DATASETS[dataset]
        if not self.client.plan_allows(dataset):
            log.info("skip %s: requires plan %s (current %s)", dataset, spec.min_plan, self.client.plan)
            return {"skipped": 1}
        stats = {"fetched": 0, "rows": 0, "empty": 0, "errors": 0, "skipped": 0}
        done = self.ledger.done_keys(dataset) if resume else set()
        cal = self.calendar_set()
        days = [d for d in days]
        if dataset == "master":
            days = _monthly_firsts(days)
        # 按月缓冲写入
        by_month: dict[str, list[date]] = {}
        for d in days:
            by_month.setdefault(d.strftime("%Y-%m"), []).append(d)
        for month, mdays in by_month.items():
            buf: list[dict[str, Any]] = []
            fetched_days: list[tuple[date, int]] = []
            for d in mdays:
                key = d.isoformat()
                if key in done:
                    stats["skipped"] += 1
                    continue
                try:
                    rows = self.client.fetch_by_date(dataset, d)
                except PlanWindowError as ex:
                    log.warning("%s %s outside plan window: %s", dataset, key, ex)
                    self.ledger.mark(dataset, key, "out_of_window", 0, str(ex)[:200])
                    continue
                except PlanAccessError as ex:
                    log.warning("%s not in plan: %s", dataset, ex)
                    self.ledger.mark(dataset, key, "not_in_plan", 0, str(ex)[:200])
                    return stats
                except Exception as ex:  # noqa: BLE001
                    stats["errors"] += 1
                    log.exception("fetch %s %s failed", dataset, key)
                    self.ledger.mark(dataset, key, "error", 0, str(ex)[:300])
                    continue
                stats["fetched"] += 1
                stats["rows"] += len(rows)
                if not rows:
                    stats["empty"] += 1
                buf.extend(rows)
                fetched_days.append((d, len(rows)))
            if fetched_days:
                self._write_rows(dataset, buf, cal)
                for d, n in fetched_days:
                    self.ledger.mark(dataset, d.isoformat(), "ok" if n else "empty", n)
                log.info("%s %s: wrote %d rows for %d days", dataset, month, len(buf), len(fetched_days))
        return stats

    def backfill_range_dataset(self, dataset: str, start: date, end: date, codes: list[str] | None = None) -> dict[str, int]:
        if not self.client.plan_allows(dataset):
            log.info("skip %s: requires plan %s", dataset, DATASETS[dataset].min_plan)
            return {"skipped": 1}
        cal = self.calendar_set()
        total = 0
        targets = codes or [None]
        for code in targets:
            s = start
            while s <= end:
                e = min(end, s + timedelta(days=RANGE_CHUNK_DAYS))
                key = f"{s.isoformat()}_{e.isoformat()}" + (f"_{code}" if code else "")
                if self.ledger.status_of(dataset, key) == "ok":
                    s = e + timedelta(days=1)
                    continue
                try:
                    rows = self.client.fetch_range(dataset, s, e, {"code": code} if code else None)
                except PlanWindowError as ex:
                    self.ledger.mark(dataset, key, "out_of_window", 0, str(ex)[:200])
                    s = e + timedelta(days=1)
                    continue
                self._write_rows(dataset, rows, cal)
                self.ledger.mark(dataset, key, "ok" if rows else "empty", len(rows))
                total += len(rows)
                s = e + timedelta(days=1)
        return {"rows": total}

    def _write_rows(self, dataset: str, rows: list[dict[str, Any]], cal: set[date]) -> None:
        if not rows:
            return
        df = normalize(dataset, rows, cal)
        if df.empty:
            return
        parts = partition_of(dataset, df)
        for part, sub in df.groupby(parts):
            self.store.write_merge("staging", TABLE_NAME[dataset], sub, str(part), KEYS[dataset])

    # ---------------- round-robin backfill ----------------
    def backfill_roundrobin(self, start: date, end: date, datasets: list[str], newest_first: bool = True) -> dict[str, Any]:
        """按交易日（默认从新到旧）轮转各数据集并逐日落库：任何时刻中断，已有数据都是「最近 N 个交易日的完整切片」。"""
        if self.client.window is None:
            self.client.probe_window()
        start, end = self.client.clamp(start, end)
        days = self.ensure_calendar(start, end)
        cal = self.calendar_set()
        days = sorted(d for d in days if not cal or d in cal)
        if not days:
            return {"window": [start.isoformat(), end.isoformat()], "n_days": 0}
        master_days = set(_monthly_firsts(days))
        dsets = [ds for ds in datasets if ds != "calendar" and DATASETS[ds].mode == "date" and self.client.plan_allows(ds)]
        disabled: set[str] = set()
        stats = {ds: {"fetched": 0, "rows": 0, "reused": 0, "errors": 0} for ds in dsets}
        done = {ds: self.ledger.done_keys(ds) | self._empty_keys(ds) for ds in dsets}
        order = sorted(days, reverse=newest_first)
        for i, d in enumerate(order):
            key = d.isoformat()
            for ds in dsets:
                if ds in disabled or key in done[ds]:
                    continue
                if ds == "master" and d not in master_days:
                    continue
                try:
                    rows = self.client.fetch_by_date(ds, d, reuse_raw=True)
                except PlanWindowError as ex:
                    self.ledger.mark(ds, key, "out_of_window", 0, str(ex)[:200])
                    continue
                except PlanAccessError as ex:
                    log.warning("%s not in plan: %s", ds, ex)
                    disabled.add(ds)
                    continue
                except Exception as ex:  # noqa: BLE001
                    stats[ds]["errors"] += 1
                    log.exception("fetch %s %s failed", ds, key)
                    self.ledger.mark(ds, key, "error", 0, str(ex)[:300])
                    continue
                self._write_rows(ds, rows, cal)
                self.ledger.mark(ds, key, "ok" if rows else "empty", len(rows))
                stats[ds]["fetched"] += 1
                stats[ds]["rows"] += len(rows)
            if i % 20 == 0:
                log.info("backfill progress: %d/%d days (at %s)", i + 1, len(order), key)
        report = {"window": [start.isoformat(), end.isoformat()], "n_days": len(days), "datasets": stats, "disabled": sorted(disabled)}
        for ds in datasets:
            if ds != "calendar" and DATASETS[ds].mode == "range":
                report[ds] = self.backfill_range_dataset(ds, start, end)
        self.ledger.set_kv("jquants.last_backfill", report)
        return report

    def _empty_keys(self, dataset: str) -> set[str]:
        with self.ledger.conn() as c:
            return {r[0] for r in c.execute("SELECT key FROM completeness WHERE dataset=? AND status='empty'", (dataset,)).fetchall()}

    def rebuild_from_raw(self, datasets: list[str] | None = None) -> dict[str, int]:
        """用原始层重建 staging，不消耗 API 配额（口径或解析器变更后使用）。"""
        import gzip
        import json
        cal = self.calendar_set()
        out: dict[str, int] = {}
        root = self.paths.raw_dir("jquants", "x").parent
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            ds = d.name
            if ds not in DATASETS or (datasets and ds not in datasets):
                continue
            tbl_dir = self.paths.staging_dir(TABLE_NAME[ds])
            n = 0
            by_part: dict[str, list[pd.DataFrame]] = {}
            for f in sorted(d.rglob("*.json.gz")):
                with gzip.open(f, "rb") as fh:
                    body = json.loads(fh.read().decode("utf-8"))
                rows = body.get("data", []) if isinstance(body, dict) else []
                if not rows:
                    continue
                df = normalize(ds, rows, cal)
                n += len(df)
                for part, sub in df.groupby(partition_of(ds, df)):
                    by_part.setdefault(str(part), []).append(sub)
            for part, frames in by_part.items():
                merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=KEYS[ds], keep="last")
                self.store.write("staging", TABLE_NAME[ds], merged, part)
            out[ds] = n
            log.info("rebuilt %s: %d rows into %s", ds, n, tbl_dir)
        return out

    # ---------------- orchestration ----------------
    def backfill(self, start: date, end: date, datasets: list[str] | None = None, resume: bool = True) -> dict[str, Any]:
        datasets = datasets or DEFAULT_BACKFILL
        if self.client.window is None:
            self.client.probe_window()
        start, end = self.client.clamp(start, end)
        log.info("backfill window %s ~ %s (plan %s)", start, end, self.client.plan)
        days = self.ensure_calendar(start, end) if "calendar" in datasets or not self.store.exists("staging", TABLE_NAME["calendar"]) \
            else trading_days([], start, end)
        if not days:
            days = trading_days([], start, end)
        cal = self.calendar_set()
        if cal:
            days = [d for d in days if d in cal]
        report: dict[str, Any] = {"window": [start.isoformat(), end.isoformat()], "n_days": len(days)}
        for ds in datasets:
            if ds == "calendar":
                continue
            spec = DATASETS[ds]
            if spec.mode == "date":
                report[ds] = self.backfill_dataset(ds, days, resume=resume)
            elif spec.mode == "range":
                report[ds] = self.backfill_range_dataset(ds, start, end)
            elif spec.mode == "static":
                rows = self.client.fetch(ds, {}, "latest")
                self._write_rows(ds, rows, cal)
                report[ds] = {"rows": len(rows)}
        # 最新主数据快照（窗口末日）
        if "master" in datasets and days:
            last = days[-1]
            if self.ledger.status_of("master", last.isoformat()) != "ok":
                rows = self.client.fetch_by_date("master", last)
                self._write_rows("master", rows, cal)
                self.ledger.mark("master", last.isoformat(), "ok", len(rows))
        self.ledger.set_kv("jquants.last_backfill", report)
        return report

    def daily(self, datasets: list[str] | None = None, lookback_days: int = 10) -> dict[str, Any]:
        """增量：补齐窗口末尾最近 lookback_days 个交易日中尚未采集的日期。"""
        datasets = datasets or ["master", "bars_daily", "valuation", "fins_summary", "earnings_date"]
        if self.client.window is None:
            self.client.probe_window()
        end = self.client.window[1] if self.client.window else date.today()
        start = end - timedelta(days=lookback_days * 2)
        start, end = self.client.clamp(start, end)
        days = self.ensure_calendar(start, end)
        report: dict[str, Any] = {"window": [start.isoformat(), end.isoformat()]}
        for ds in datasets:
            if DATASETS[ds].mode == "date":
                dd = days if ds != "master" else days[-1:]
                report[ds] = self.backfill_dataset(ds, dd, resume=True) if ds != "master" else self._master_latest(dd[-1])
        return report

    def _master_latest(self, d: date) -> dict[str, int]:
        rows = self.client.fetch_by_date("master", d)
        self._write_rows("master", rows, self.calendar_set())
        self.ledger.mark("master", d.isoformat(), "ok", len(rows))
        return {"rows": len(rows)}


def _monthly_firsts(days: list[date]) -> list[date]:
    seen: set[str] = set()
    out: list[date] = []
    for d in sorted(days):
        m = d.strftime("%Y-%m")
        if m not in seen:
            seen.add(m)
            out.append(d)
    if days and days[-1] not in out:
        out.append(days[-1])
    return out
