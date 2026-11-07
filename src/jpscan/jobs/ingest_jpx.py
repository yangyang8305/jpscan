"""JPX 免费文件采集作业（规划文档 10.5 节 `jpx.files`、10.6 节 B7、附录 A.1）。

流程：抓取 → 原始层落盘（RawSink）→ 解析 → 写 staging（write_merge 幂等）→ 账本 completeness 标记。
单个文件失败（403/404/格式变化）只记 error 并继续，解析失败的内容进隔离区。

回填与增量共用同一入口：
- 有バックナンバー的数据集（per_pbr_monthly、delisted、supervision 履歴、new_listings）首次全取，
  之后按账本跳过已完成的月份/年份，只取新的；首页每次都取。
- 只有当期文件的数据集每次取当期；原始层按内容哈希去重，同内容不重复落盘。
- `month="YYYY-MM"`：per_pbr_monthly 只取（并强制重取）该月；capital_cost_disclosure 只写该月；其余数据集不受影响。

下游读取约定：
- jpx_improvement_period、jpx_earnings_schedule 是逐次快照的累积。改善期間取最新 published_date 的行作为当前名单；
  決算予定取每个 (Code, fiscal_year_end, quarter) 的最新 as_of_date，改期前的旧日期行仍保留。
- jpx_supervision.is_current 每次按「現在の指定状況」页整表重算；该页取失败时保留上次的值。
- jpx_price_limit_expansion 的 CSV 只有当日，必须每个营业日运行才能累积。
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import PurePosixPath
from typing import Any, Callable, Iterable
from urllib.parse import urlsplit

import pandas as pd
import requests

from .. import __version__
from ..config import Settings
from ..connectors import jpx_files as J
from ..connectors.jpx_files import DATASETS, TABLES, URLS, JPXFetchError, JPXFiles
from ..ledger import Ledger
from ..paths import DataPaths
from ..store import Store
from ..util.common import today_jst

log = logging.getLogger("jpscan.ingest.jpx")

# 报告「数据日期」时取最大值的列
DATA_DATE_COL: dict[str, str] = {
    "listed_monthly": "date", "topix_weights": "date", "per_pbr_monthly": "month",
    "capital_cost_disclosure": "as_of_month", "earnings_schedule": "as_of_date", "price_limit_expansion": "date",
    "delisted": "delist_date", "supervision": "designated_date", "new_listings": "listing_date",
    "improvement_period": "published_date",
}


def _basename(url: str) -> tuple[str, str]:
    p = PurePosixPath(urlsplit(url).path)
    return p.stem, p.suffix


def _latest(values: Iterable[Any]) -> str | None:
    vals = [v for v in values if v is not None and not pd.isna(v)]
    return str(max(vals)) if vals else None


class _Job:
    def __init__(self, ledger: Ledger, store: Store, client: JPXFiles, month: str | None, today: date):
        self.ledger, self.store, self.client, self.month, self.today = ledger, store, client, month, today
        self.results: dict[str, dict[str, Any]] = {}

    # ---------------- 账本与结果 ----------------
    def _res(self, ds: str) -> dict[str, Any]:
        spec = TABLES[ds]
        return self.results.setdefault(ds, {"table": spec.table, "keys": spec.keys, "rows_written": 0, "files_ok": 0,
                                            "files_skipped": 0, "errors": []})

    def _fail(self, ds: str, key: str, msg: str) -> None:
        log.warning("jpx %s/%s: %s", ds, key, msg)
        self.ledger.mark(f"jpx.{ds}", key, "error", error=msg)
        self._res(ds)["errors"].append(f"{key}: {msg}")

    def _ok(self, ds: str, key: str, n_rows: int) -> None:
        self.ledger.mark(f"jpx.{ds}", key, "ok", n_rows)
        self._res(ds)["files_ok"] += 1

    def _done(self, ds: str) -> set[str]:
        return self.ledger.done_keys(f"jpx.{ds}")

    # ---------------- 抓取 / 解析 / 写入 ----------------
    def _get(self, ds: str, url: str, key: str | None = None) -> bytes | None:
        """抓取并存原始层；失败记 error 返回 None。"""
        stem, ext = _basename(url)
        try:
            content = self.client.fetch(url)
        except (JPXFetchError, requests.RequestException) as e:
            self._fail(ds, key or stem, str(e))
            return None
        self.client.save(ds, stem, content, ext, url, self.today)
        return content

    def _parse(self, ds: str, key: str, content: bytes, fn: Callable[[bytes], pd.DataFrame], allow_empty: bool = False) -> pd.DataFrame | None:
        """解析失败或（不应为空的文件）解析出 0 行 → 隔离并记 error，返回 None。"""
        try:
            df = fn(content)
            if df.empty and not allow_empty:
                raise ValueError("解析结果为 0 行（可能改版）")
        except Exception as e:  # noqa: BLE001  格式变化不能中断其他文件
            self.client.quarantine(ds, key.replace(":", "_"), content, str(e))
            self._fail(ds, key, f"解析失败: {e}")
            return None
        return df

    def _write(self, ds: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        spec = TABLES[ds]
        if spec.partition == "all":
            self.store.write_merge("staging", spec.table, df, "all", spec.keys)
        else:
            parts = J.month_partition(df[spec.partition])
            for part, sub in df.groupby(pd.Series(parts, index=df.index), sort=True):
                self.store.write_merge("staging", spec.table, sub.reset_index(drop=True), str(part), spec.keys)
        self._res(ds)["rows_written"] += len(df)

    def _single(self, ds: str, url: str, fn: Callable[[bytes], pd.DataFrame], key_col: str) -> pd.DataFrame | None:
        content = self._get(ds, url)
        if content is None:
            return None
        df = self._parse(ds, _basename(url)[0], content, fn)
        if df is None:
            return None
        self._write(ds, df)
        self._ok(ds, _latest(df[key_col]) or self.today.isoformat(), len(df))
        return df

    def _year_pages(self, ds: str, url: str, parser: Callable[[bytes], pd.DataFrame]) -> list[pd.DataFrame]:
        """当年页（每次）+ 尚未完成的历年存档页（一次）。返回解析结果，写入与账本的 ok 标记由 _ok 完成。"""
        html = self._get(ds, url, "current")
        if html is None:
            return []
        frames: list[pd.DataFrame] = []
        df = self._parse(ds, "current", html, parser)
        if df is not None:
            frames.append(df)
            self._ok(ds, "current", len(df))
        done = self._done(ds)
        for year, page_url in sorted(J.find_year_pages(html).items(), reverse=True):
            if page_url == url:
                continue
            if year in done:
                self._res(ds)["files_skipped"] += 1
                continue
            content = self._get(ds, page_url, year)
            df = self._parse(ds, year, content, parser) if content is not None else None
            if df is not None:
                frames.append(df)
                self._ok(ds, year, len(df))
        return frames

    # ---------------- 各数据集 ----------------
    def listed_monthly(self) -> None:
        self._single("listed_monthly", URLS["listed_monthly"], J.parse_listed_monthly, "date")

    def topix_weights(self) -> None:
        self._single("topix_weights", URLS["topix_weights"], J.parse_topix_weights, "date")

    def per_pbr_monthly(self) -> None:
        ds = "per_pbr_monthly"
        html = self._get(ds, URLS["per_pbr_page"], "page")
        if html is None:
            return
        links = J.find_perpbr_links(html)
        if not links:
            self._fail(ds, "page", "页面未发现 perpbrYYYYMM.xlsx 链接（可能改版）")
            return
        if self.month:
            if self.month not in links:
                self._fail(ds, self.month, "页面没有该月文件")
                return
            todo = [self.month]
        else:
            done = self._done(ds)
            todo = [m for m in sorted(links) if m not in done]
            self._res(ds)["files_skipped"] = len(links) - len(todo)
        for m in todo:
            content = self._get(ds, links[m], m)
            df = self._parse(ds, m, content, J.parse_per_pbr) if content is not None else None
            if df is not None:
                self._write(ds, df)
                self._ok(ds, m, len(df))

    def capital_cost_disclosure(self) -> None:
        ds = "capital_cost_disclosure"
        content = self._get(ds, URLS[ds])
        if content is None:
            return
        df = self._parse(ds, "list", content, J.parse_capital_cost)
        if df is None:
            return
        if self.month:
            df = df[df["as_of_month"] == self.month].reset_index(drop=True)
        self._write(ds, df)
        self._ok(ds, _latest(df["as_of_month"]) or self.month or self.today.isoformat(), len(df))

    def earnings_schedule(self) -> None:
        ds = "earnings_schedule"
        html = self._get(ds, URLS["earnings_page"], "page")
        if html is None:
            return
        links = J.find_kessan_links(html)
        if not links:
            self._fail(ds, "page", "页面未发现 kessan*.xlsx 链接（可能改版）")
            return
        frames = []
        for url in links:
            stem, ext = _basename(url)
            content = self._get(ds, url)
            parse = lambda b, name=stem + ext: J.parse_earnings_schedule(b, source_file=name, fetch_date=self.today)
            df = self._parse(ds, stem, content, parse, allow_empty=True) if content is not None else None  # 翌営業日分可能为空
            if df is not None:
                frames.append(df)
                self._ok(ds, f"{stem}:{self.today.isoformat()}", len(df))
        frames = [f for f in frames if not f.empty]
        if frames:
            # 同一 (日期, Code, 種別) 出现在多个文件时保留 as_of_date 最新者
            all_df = pd.concat(frames, ignore_index=True)
            order = sorted(range(len(all_df)), key=lambda i: all_df["as_of_date"].iloc[i] or date.min)
            all_df = all_df.iloc[order].drop_duplicates(subset=TABLES[ds].keys, keep="last").reset_index(drop=True)
            self._write(ds, all_df)

    def price_limit_expansion(self) -> None:
        ds = "price_limit_expansion"
        html = self._get(ds, URLS["price_limit_page"], "page")
        if html is None:
            return
        links = J.find_price_limit_links(html)
        if not links:
            self._fail(ds, "page", "页面未发现 MMDD.csv 链接（可能改版）")
            return
        for url in links:
            stem, _ = _basename(url)
            kind = "etf" if "ETF" in stem.upper() else "stock"
            content = self._get(ds, url)
            df = self._parse(ds, stem, content, lambda b, k=kind: J.parse_price_limit(b, list_kind=k), allow_empty=True) if content is not None else None
            if df is not None:
                self._write(ds, df)
                self._ok(ds, f"{_latest(df['date']) if not df.empty else self.today.isoformat()}:{kind}", len(df))

    def delisted(self) -> None:
        for df in self._year_pages("delisted", URLS["delisted"], J.parse_delisted):
            self._write("delisted", df)

    def new_listings(self) -> None:
        for df in self._year_pages("new_listings", URLS["new_listings"], J.parse_new_listings):
            self._write("new_listings", df)

    def supervision(self) -> None:
        ds, spec = "supervision", TABLES["supervision"]
        content = self._get(ds, URLS["supervision_current"], "current_page")
        current = self._parse(ds, "current_page", content, J.parse_supervision_current) if content is not None else None
        if current is not None:
            self._ok(ds, "current_page", len(current))
        frames = self._year_pages(ds, URLS["supervision_history"], J.parse_supervision_history)
        history = pd.concat(frames, ignore_index=True) if frames else None
        if current is not None and history is not None:
            merged = J.merge_supervision(current, history)
        else:
            merged = history if history is not None else current
        if merged is None:
            return
        self._write(ds, merged)
        if current is not None:  # 现行状态按最新现行页重算（含此前写入的存档行）
            table = pd.read_parquet(self.store.table_dir("staging", spec.table) / "all.parquet")
            cur_keys = set(J.supervision_keys(current))
            table["is_current"] = [k in cur_keys for k in J.supervision_keys(table)]
            self.store.write("staging", spec.table, table, "all")

    def improvement_period(self) -> None:
        ds = "improvement_period"
        html = self._get(ds, URLS["improvement_page"], "page")
        if html is None:
            return
        url = J.find_improvement_link(html)
        if url is None:
            self.client.quarantine(ds, "page", html, "页面未发现 xlsx 链接")
            self._fail(ds, "page", "页面未发现 xlsx 链接（可能改版）")
            return
        stem, ext = _basename(url)
        self._single(ds, url, lambda b: J.parse_improvement_period(b, source_file=stem + ext), "published_date")

    # ---------------- 汇总 ----------------
    def finish_dataset(self, ds: str) -> dict[str, Any]:
        res = self._res(ds)
        res["status"] = "ok" if not res["errors"] else ("partial" if res["files_ok"] else "error")
        spec = TABLES[ds]
        res["rows_table"], res["data_date"] = 0, None
        if self.store.exists("staging", spec.table):
            col = DATA_DATE_COL[ds]
            out = self.store.query(f'SELECT COUNT(*) AS n, MAX("{col}") AS d FROM t', {"t": ("staging", spec.table)})
            res["rows_table"] = int(out["n"].iloc[0])
            d = out["d"].iloc[0]
            res["data_date"] = None if pd.isna(d) else str(d)[:10]
        return res


def run(settings: Settings, paths: DataPaths, ledger: Ledger, store: Store, datasets: Iterable[str] | None = None,
        month: str | None = None, *, client: JPXFiles | None = None, today: date | None = None) -> dict[str, Any]:
    """采集 JPX 免费文件。`datasets` 缺省 = 全部；`client` 可注入（测试传 fetcher，不联网）；单个数据集失败不影响其他。

    返回 {"run_id", "status", "month", "datasets": {dataset: {table, keys, status, rows_written, rows_table, data_date,
    files_ok, files_skipped, errors}}}；rows_written 为本次写入行数（含覆盖），rows_table 为表的当前总行数。
    """
    names = list(datasets) if datasets else list(DATASETS)
    unknown = [d for d in names if d not in TABLES]
    if unknown:
        raise ValueError(f"未知 JPX 数据集: {unknown}；可选: {DATASETS}")
    run_id = ledger.start_run("ingest_jpx", __version__, settings.params_version, note=",".join(names))
    job = _Job(ledger, store, client or JPXFiles(paths, ledger), month, today or today_jst())
    results: dict[str, dict[str, Any]] = {}
    for ds in names:
        try:
            getattr(job, ds)()
        except Exception as e:  # noqa: BLE001  作业级兜底：一个数据集的意外错误不中断其余
            log.exception("jpx dataset %s failed", ds)
            job._fail(ds, "job", repr(e))
        results[ds] = job.finish_dataset(ds)
        log.info("jpx %s: %s rows_written=%d rows_table=%d data_date=%s", ds, results[ds]["status"], results[ds]["rows_written"],
                 results[ds]["rows_table"], results[ds]["data_date"])
    statuses = {r["status"] for r in results.values()}
    status = "ok" if statuses == {"ok"} else ("error" if statuses == {"error"} else "partial")
    failed = [d for d, r in results.items() if r["status"] != "ok"]
    ledger.finish_run(run_id, status, note=f"failed: {','.join(failed)}" if failed else None)
    return {"run_id": run_id, "status": status, "month": month, "datasets": results}
