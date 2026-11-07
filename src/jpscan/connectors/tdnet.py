"""TDnet 適時開示情報閲覧サービス 连接器（规划文档 10.5 节 tdnet.daily）。

日次列表 I_list_{NNN}_{YYYYMMDD}.html（每页 100 件，过界页 404），PDF 与 XBRL zip 与列表同目录。
站点只保留当日及过去 31 天。

合规注意：站点 robots.txt 为 `User-agent: * / Disallow: /`，列表页带 noindex,nofollow（2026-11-07 确认）。
规划文档 19.1 节未记录这一点；是否运行实时采集由项目所有者决定，见该节。
"""
from __future__ import annotations

import logging
import re
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

from ..util.common import RateLimiter
from .base import RawSink, get_with_retry, http_session

log = logging.getLogger("jpscan.connectors.tdnet")

BASE_URL = "https://www.release.tdnet.info/inbs/"
SOURCE = "tdnet"
BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
WINDOW_DAYS = 31
MAX_PAGES = 200
_FILE_NAME = re.compile(r"^[A-Za-z0-9_\-]+\.(pdf|zip)$")   # 只接受同目录文件名：链接来自页面，不可信
_TOTAL = re.compile(r"全\s*([\d,]+)\s*件")
_DATE = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")


class TdnetBlocked(RuntimeError):
    """站点返回 403/429：应立即停止本次运行，而不是继续请求。"""


class TdnetFormatError(ValueError):
    """200 响应但不是预期的列表页结构（改版或错误页）。"""


def list_url(d: date, page: int) -> str:
    return f"{BASE_URL}I_list_{page:03d}_{d:%Y%m%d}.html"


def normalize_code(raw: str | None) -> tuple[str, str]:
    """(5 位 Code, code4)。4 位代码补尾 0；字母代码（130A0）保留大写。"""
    s = unicodedata.normalize("NFKC", raw or "").strip().upper()
    if len(s) == 4:
        s += "0"
    return s, s[:4]


def _cell_text(td: Any) -> str:
    return td.get_text(" ", strip=True).strip() if td is not None else ""


def _linked_file(td: Any) -> str | None:
    a = td.find("a") if td is not None else None
    href = (a.get("href") or "").strip() if a is not None else ""
    if not href:
        return None
    if not _FILE_NAME.match(href):
        log.warning("ignore unexpected link in list page: %r", href[:80])
        return None
    return href


def parse_list_page(html: str | bytes) -> dict[str, Any]:
    """解析一页列表，返回 {date, total, rows}。结构不符（无 main-list-table）抛 TdnetFormatError。"""
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="main-list-table")
    if table is None:
        raise TdnetFormatError("main-list-table not found")
    m = _DATE.search(_cell_text(soup.find(id="kaiji-date-1")))
    page_date = date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None
    m = _TOTAL.search(_cell_text(soup.find(class_="kaijiSum")))
    total = int(m.group(1).replace(",", "")) if m else None
    rows = []
    for tr in table.find_all("tr"):
        code_td, title_td = tr.find("td", class_="kjCode"), tr.find("td", class_="kjTitle")
        if code_td is None or title_td is None:
            continue
        code, code4 = normalize_code(_cell_text(code_td))
        pdf_name = _linked_file(title_td)
        xbrl_name = _linked_file(tr.find("td", class_="kjXbrl"))
        history = _cell_text(tr.find("td", class_="kjHistroy"))   # 站点拼写即 kjHistroy；空白为全角空格
        rows.append({
            "time": _cell_text(tr.find("td", class_="kjTime")),
            "Code": code,
            "code4": code4,
            "company": _cell_text(tr.find("td", class_="kjName")),
            "title": _cell_text(title_td),
            "pdf_name": pdf_name,
            "pdf_url": BASE_URL + pdf_name if pdf_name else None,
            "xbrl_name": xbrl_name,
            "xbrl_url": BASE_URL + xbrl_name if xbrl_name else None,
            "exchanges": _cell_text(tr.find("td", class_="kjPlace")),
            "history": history,
            "correction_marker": bool(history),
        })
    return {"date": page_date, "total": total, "rows": rows}


def assign_disclosure_ids(rows: list[dict[str, Any]], d: date) -> None:
    """disclosure_id = PDF 文件名主干；无 PDF 时为 {YYYYMMDD}_{code}_{seq}。

    列表新披露在前，序号从最旧一行数起，这样同一天重新抓取时已有编号不变。
    """
    seq: dict[str, int] = {}
    for r in reversed(rows):
        if r.get("pdf_name"):
            r["disclosure_id"] = r["pdf_name"].rsplit(".", 1)[0]
        else:
            seq[r["Code"]] = seq.get(r["Code"], 0) + 1
            r["disclosure_id"] = f"{d:%Y%m%d}_{r['Code']}_{seq[r['Code']]}"


class TdnetClient:
    """单线程、每次请求间隔 ≥ 1 秒。所有响应先经 RawSink 落盘再解析。"""

    def __init__(self, sink: RawSink, session: requests.Session | None = None, limiter: RateLimiter | None = None):
        self.sink = sink
        self.session = session or http_session(BROWSER_UA)
        self.limiter = limiter or RateLimiter(60)   # 60 rpm × 1.08 → 约 1.08 秒一次

    def _get(self, url: str) -> requests.Response:
        self.limiter.wait()
        r = get_with_retry(self.session, url)
        if r.status_code in (403, 429):
            raise TdnetBlocked(f"HTTP {r.status_code} for {url}")
        return r

    def fetch_day(self, d: date) -> dict[str, Any]:
        """抓取一天全部列表页：逐页到 404、空页或达到页面声明的总件数。返回 {rows, total, pages, complete}。"""
        rows: list[dict[str, Any]] = []
        total: int | None = None
        pages = 0
        for page in range(1, MAX_PAGES + 1):
            url = list_url(d, page)
            r = self._get(url)
            if r.status_code == 404:
                break
            r.raise_for_status()
            self.sink.save_bytes("list", f"{d:%Y%m%d}_p{page:03d}", r.content, ".html",
                                 params={"date": d.isoformat(), "page": page}, http_status=r.status_code)
            parsed = parse_list_page(r.content)
            if parsed["date"] is not None and parsed["date"] != d:
                raise TdnetFormatError(f"page date {parsed['date']} != requested {d}")
            if not parsed["rows"]:
                break
            pages = page
            total = parsed["total"] if parsed["total"] is not None else total
            for row in parsed["rows"]:
                row["page"] = page
            rows.extend(parsed["rows"])
            if total is not None and len(rows) >= total:
                break
        assign_disclosure_ids(rows, d)
        return {"rows": rows, "total": total, "pages": pages, "complete": total is None or len(rows) == total}

    def download_file(self, url: str, dataset: str, key: str, ext: str, magic: bytes, subdir: str) -> Path | None:
        """下载 PDF/XBRL 到原始层。404 返回 None；内容头不符则隔离并返回 None。"""
        r = self._get(url)
        if r.status_code == 404:
            log.warning("file not found: %s", url)
            return None
        r.raise_for_status()
        if not r.content.startswith(magic):
            self.sink.quarantine(dataset, key, r.content, f"unexpected content header for {url}")
            return None
        return self.sink.save_bytes(dataset, key, r.content, ext, params={"url": url}, http_status=r.status_code, subdir=subdir)
