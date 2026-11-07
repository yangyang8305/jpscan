"""新闻标题 RSS 连接器（规划文档 3.2、4.7、10.5 节 `news.rss`）：Yahoo!ニュース 媒体 feed、PR TIMES。

条款（19.1 节）：RSS 仅限个人利用，禁止再配信与公开应用，所以只保留标题、链接和 feed 自带的摘要，绝不抓正文；
每次运行每个 feed 只取一次。feed 内容是不可信数据，这里只当文本解析。

本模块的解析函数都是纯函数，不联网；抓取、落盘与写表在 jobs/ingest_news.py。
"""
from __future__ import annotations

import calendar
import hashlib
import html
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import feedparser
import pandas as pd
import yaml

from ..events.company_match import CompanyMatcher
from ..util.common import JST

COLUMNS = ["news_id", "source", "tier", "published_at", "published_date", "title", "link", "summary",
           "matched_codes", "match_method", "category"]

_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")
_CODE_LIKE = re.compile(r"[0-9０-９]{3}[0-9A-Za-z０-９Ａ-Ｚ]")


def load_feeds(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    return list(y.get("feeds", []))


def news_id(link: str) -> str:
    return hashlib.sha256(link.encode("utf-8")).hexdigest()[:24]


def clean_text(text: str | None) -> str:
    """去 HTML 标签与实体，折叠空白。"""
    if not text:
        return ""
    return _SPACE.sub(" ", html.unescape(_TAG.sub(" ", text))).strip()


def title_for_matching(title: str) -> str:
    """去掉 Yahoo 标题末尾的发布媒体，如「…(東洋経済オンライン)」「…(テレビ朝日系（ANN）)」。

    媒体本身可能是上市公司（ITmedia、テレビ朝日 等），不能参与匹配。末尾括号里像证券代码时保留。
    """
    s = title.rstrip()
    if not s.endswith((")", "）")):
        return title
    depth = 0
    for i in range(len(s) - 1, -1, -1):
        if s[i] in ")）":
            depth += 1
        elif s[i] in "(（":
            depth -= 1
            if depth == 0:
                return title if _CODE_LIKE.search(s[i:]) else s[:i]
    return title


def _published(entry: Any) -> datetime | None:
    st = entry.get("published_parsed") or entry.get("updated_parsed")
    if not st:
        return None
    return datetime.fromtimestamp(calendar.timegm(st), tz=timezone.utc).astimezone(JST)


def _match(entry: Any, title: str, source: str, matcher: CompanyMatcher | None) -> tuple[list[str], str]:
    if matcher is None:
        return [], ""
    text = title_for_matching(title) if source.startswith("yahoo_") else title
    codes, method = matcher.match_detail(text)
    corp = clean_text(entry.get("dc_corp"))      # PR TIMES 的发布公司名
    corp_code = matcher.match_name(corp) if corp else None
    if corp_code:
        codes = list(dict.fromkeys([corp_code, *codes]))
        method = f"corp+{method}" if method else "corp"
    return codes, method


def parse_feed(content: bytes, feed: dict[str, str], matcher: CompanyMatcher | None = None) -> tuple[pd.DataFrame, int]:
    """bytes → (news_items 行, 因无发布时间而跳过的条数)。

    没有发布时间的条目不入表：用抓取时间冒充发布时间会破坏时点一致性。
    """
    parsed = feedparser.parse(content)
    rows: list[dict[str, Any]] = []
    n_undated = 0
    for e in parsed.entries:
        link = str(e.get("link") or "").strip()
        title = clean_text(e.get("title"))
        if not link or not title:
            continue
        ts = _published(e)
        if ts is None:
            n_undated += 1
            continue
        codes, method = _match(e, title, feed["source"], matcher)
        rows.append({
            "news_id": news_id(link),
            "source": feed["source"],
            "tier": feed["tier"],
            "published_at": ts.isoformat(),
            "published_date": ts.date(),
            "title": title,
            "link": link,
            "summary": clean_text(e.get("summary")),
            "matched_codes": ",".join(codes),
            "match_method": method,
            "category": "",
        })
    return pd.DataFrame(rows, columns=COLUMNS), n_undated


def partition_of(d) -> str:
    return f"{d.year:04d}-{d.month:02d}"
