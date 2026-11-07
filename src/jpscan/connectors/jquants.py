"""J-Quants API V2 连接器（规划文档 10.5 节）。

- 认证：请求头 x-api-key；Base URL https://api.jquants.com/v2
- 响应：{"data": [...], "pagination_key": ...}；按 pagination_key 翻页
- 限速：按套餐 rpm 滑动窗口；429 时惩罚等待后重试
- 套餐窗口：400 响应中 "Your subscription covers the following dates: A ~ B" 自动解析并记录到账本
- 原始层：每页响应原样 gzip 落盘并登记哈希（RawSink）
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Iterable

import requests

from ..ledger import Ledger
from ..paths import DataPaths
from ..util.common import RateLimiter, to_date
from .base import RawSink, http_session

log = logging.getLogger("jpscan.jquants")

BASE = "https://api.jquants.com/v2"
_WINDOW_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\s*~\s*(\d{4}-\d{2}-\d{2})")


@dataclass(frozen=True)
class DatasetSpec:
    name: str            # 本地数据集名
    path: str            # API 路径
    mode: str            # "date" | "range" | "code" | "static"
    min_plan: str        # Free | Light | Standard | Premium
    key_field: str = ""  # 用于分区/去重的日期字段


DATASETS: dict[str, DatasetSpec] = {
    "master":            DatasetSpec("master", "/equities/master", "date", "Free", "Date"),
    "bars_daily":        DatasetSpec("bars_daily", "/equities/bars/daily", "date", "Free", "Date"),
    "valuation":         DatasetSpec("valuation", "/equities/valuation", "date", "Free", "Date"),
    "fins_summary":      DatasetSpec("fins_summary", "/fins/summary", "date", "Free", "DiscDate"),
    "earnings_date":     DatasetSpec("earnings_date", "/fins/earnings-date", "date", "Free", "PubDate"),
    "earnings_calendar": DatasetSpec("earnings_calendar", "/equities/earnings-calendar", "static", "Free", ""),
    "calendar":          DatasetSpec("calendar", "/markets/calendar", "range", "Free", "Date"),
    "investor_types":    DatasetSpec("investor_types", "/equities/investor-types", "range", "Light", "PubDate"),
    "topix":             DatasetSpec("topix", "/indices/bars/daily/topix", "range", "Light", "Date"),
    "indices":           DatasetSpec("indices", "/indices/bars/daily", "range", "Standard", "Date"),
    "margin_interest":   DatasetSpec("margin_interest", "/markets/margin-interest", "date", "Standard", "Date"),
    "margin_alert":      DatasetSpec("margin_alert", "/markets/margin-alert", "date", "Standard", "Date"),
    "short_ratio":       DatasetSpec("short_ratio", "/markets/short-ratio", "date", "Standard", "Date"),
    "short_sale_report": DatasetSpec("short_sale_report", "/markets/short-sale-report", "date", "Standard", "Date"),
    "breakdown":         DatasetSpec("breakdown", "/markets/breakdown", "date", "Premium", "Date"),
    "fins_details":      DatasetSpec("fins_details", "/fins/details", "date", "Premium", "DiscDate"),
    "dividend":          DatasetSpec("dividend", "/fins/dividend", "date", "Premium", "AnnDate"),
}

PLAN_ORDER = ["Free", "Light", "Standard", "Premium"]


class PlanAccessError(RuntimeError):
    """端点不在当前套餐内（HTTP 403）。"""


class PlanWindowError(RuntimeError):
    """请求日期超出套餐覆盖窗口。"""

    def __init__(self, msg: str, start: date | None, end: date | None):
        super().__init__(msg)
        self.start, self.end = start, end


class JQuantsClient:
    def __init__(self, api_key: str, rpm: int, paths: DataPaths, ledger: Ledger, plan: str = "Free"):
        if not api_key:
            raise RuntimeError("JQUANTS_API_KEY 未设置（请写入 .env）")
        self.api_key = api_key
        self.limiter = RateLimiter(rpm)
        # 财务端点专属限速 60/分：只有 rpm > 60 的套餐才需要单独控制
        self.fins_limiter = RateLimiter(60) if rpm > 60 else None
        self.session = http_session()
        self.session.headers["x-api-key"] = api_key
        self.sink = RawSink(paths, ledger, "jquants")
        self.ledger = ledger
        self.plan = plan
        w = ledger.get_kv("jquants.window")
        self.window: tuple[date, date] | None = (to_date(w[0]), to_date(w[1])) if w else None

    # ---------------- low level ----------------
    def _wait(self, path: str) -> None:
        if self.fins_limiter is not None and (path.startswith("/fins/summary") or path.startswith("/fins/details")):
            self.fins_limiter.wait()
        self.limiter.wait()

    def _request(self, path: str, params: dict[str, Any]) -> tuple[int, Any]:
        url = BASE + path
        for attempt in range(8):
            self._wait(path)
            try:
                r = self.session.get(url, params=params, timeout=180)
            except (requests.ConnectionError, requests.Timeout) as e:
                wait = min(300, 5 * 2 ** attempt)
                log.warning("network error %s on %s; retry in %ss", e, path, wait)
                time.sleep(wait)
                continue
            if r.status_code == 429:
                log.warning("429 rate limited on %s %s; pausing 61s, gap now %.1fs", path, params, self.limiter.gap)
                self.limiter.penalize(61)
                if self.fins_limiter is not None:
                    self.fins_limiter.penalize(61)
                continue
            if r.status_code >= 500:
                wait = min(300, 10 * 2 ** attempt)
                log.warning("HTTP %s on %s; retry in %ss", r.status_code, path, wait)
                time.sleep(wait)
                continue
            try:
                body = r.json()
            except ValueError:
                body = {"message": r.text[:500]}
            if r.status_code == 400 and isinstance(body, dict):
                m = _WINDOW_RE.search(str(body.get("message", "")))
                if m:
                    self.window = (to_date(m.group(1)), to_date(m.group(2)))
                    self.ledger.set_kv("jquants.window", [m.group(1), m.group(2)])
                    raise PlanWindowError(body["message"], *self.window)
            if r.status_code == 403:
                raise PlanAccessError(f"{path}: {body.get('message') if isinstance(body, dict) else body}")
            return r.status_code, body
        raise RuntimeError(f"J-Quants request failed after retries: {path} {params}")

    def get_pages(self, path: str, params: dict[str, Any]) -> Iterable[tuple[int, dict[str, Any]]]:
        """逐页返回 (page_no, body)。"""
        params = dict(params)
        page = 0
        while True:
            status, body = self._request(path, params)
            if status != 200:
                raise RuntimeError(f"J-Quants HTTP {status} {path} {params}: {body}")
            yield page, body
            key = body.get("pagination_key") if isinstance(body, dict) else None
            if not key:
                break
            params["pagination_key"] = key
            page += 1

    # ---------------- dataset fetch ----------------
    def fetch(self, dataset: str, params: dict[str, Any], key: str, reuse_raw: bool = False) -> list[dict[str, Any]]:
        """拉取一个数据集的一次查询（自动翻页），原始页落盘，返回合并后的 data 行。

        reuse_raw=True：原始层已有该查询的完整响应时直接复用，不消耗 API 配额（用于断点续传与重建 staging）。
        """
        spec = DATASETS[dataset]
        rows: list[dict[str, Any]] = []
        if reuse_raw:
            cached = self._load_cached(dataset, params, key)
            if cached is not None:
                return cached
        q = dict(params)
        for page, body in self.get_pages(spec.path, params):
            data = body.get("data", []) if isinstance(body, dict) else []
            self.sink.save_json(dataset, key, q, body, 200, len(data), page)
            rows.extend(data)
            pk = body.get("pagination_key") if isinstance(body, dict) else None
            if pk:
                q = {**q, "pagination_key": pk}
        return rows

    def _load_cached(self, dataset: str, params: dict[str, Any], key: str) -> list[dict[str, Any]] | None:
        rows: list[dict[str, Any]] = []
        page = 0
        q = dict(params)
        while True:
            body = self.sink.load_json(dataset, key, q, page)
            if not isinstance(body, dict):
                return None
            rows.extend(body.get("data", []))
            pk = body.get("pagination_key")
            if not pk:
                return rows
            q["pagination_key"] = pk
            page += 1

    def fetch_by_date(self, dataset: str, d: date, reuse_raw: bool = False) -> list[dict[str, Any]]:
        return self.fetch(dataset, {"date": d.isoformat()}, d.isoformat(), reuse_raw=reuse_raw)

    def fetch_range(self, dataset: str, start: date, end: date, extra: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        params = {"from": start.isoformat(), "to": end.isoformat(), **(extra or {})}
        key = f"{start.isoformat()}_{end.isoformat()}" + (f"_{extra.get('code')}" if extra and extra.get("code") else "")
        return self.fetch(dataset, params, key)

    def fetch_by_code(self, dataset: str, code: str, start: date | None = None, end: date | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"code": code}
        if start:
            params["from"] = start.isoformat()
        if end:
            params["to"] = end.isoformat()
        return self.fetch(dataset, params, f"code_{code}")

    # ---------------- helpers ----------------
    def plan_allows(self, dataset: str) -> bool:
        return PLAN_ORDER.index(DATASETS[dataset].min_plan) <= PLAN_ORDER.index(self.plan)

    def clamp(self, start: date, end: date) -> tuple[date, date]:
        if self.window:
            start = max(start, self.window[0])
            end = min(end, self.window[1])
        return start, end

    def probe_window(self) -> tuple[date, date] | None:
        """用一次必然越界的请求触发窗口提示，得到套餐覆盖区间。"""
        try:
            self._request("/markets/calendar", {"from": "2000-01-01", "to": "2000-01-10"})
        except PlanWindowError as e:
            return (e.start, e.end) if e.start and e.end else None
        return self.window


def trading_days(calendar_rows: list[dict[str, Any]], start: date, end: date) -> list[date]:
    """从交易日历行中取营业日。字段名在 V2 中为 Date + HolDiv（0 非営業日/1 営業日/2 半日/3 祝日取引），
    未知字段时退化为工作日。"""
    out: list[date] = []
    if calendar_rows:
        sample = calendar_rows[0]
        div_field = next((k for k in sample if k.lower().startswith("hol")), None)
        for r in calendar_rows:
            d = to_date(r.get("Date"))
            if d is None or d < start or d > end:
                continue
            if div_field is None or str(r.get(div_field)) in ("1", "2"):
                out.append(d)
        if out:
            return sorted(set(out))
    d = start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out
