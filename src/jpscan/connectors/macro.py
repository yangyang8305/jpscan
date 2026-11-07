"""宏观序列连接器（规划文档 3.1、10.5 节 `macro.*`、19.1 节）。解析函数都是纯函数，不联网；抓取与写表在 jobs/ingest_macro.py。

- 財務省 国債金利（jgbcm.csv 当月、data/jgbcm_all.csv 全史）：cp932，和暦日期（R6.8.15），"-" 表示缺失。出典：財務省（PDL1.0）。
"""
from __future__ import annotations

import csv
import io
import re
import unicodedata
from datetime import date
from typing import Any

import pandas as pd

COLUMNS = ["series", "date", "value", "source", "unit"]

MOF_CURRENT_URL = "https://www.mof.go.jp/jgbs/reference/interest_rate/jgbcm.csv"
MOF_ALL_URL = "https://www.mof.go.jp/jgbs/reference/interest_rate/data/jgbcm_all.csv"
JGB_COLUMNS = {"1年": "jgb_1y", "2年": "jgb_2y", "5年": "jgb_5y", "10年": "jgb_10y", "20年": "jgb_20y", "30年": "jgb_30y", "40年": "jgb_40y"}

ERA_BASE = {"R": 2018, "H": 1988, "S": 1925}
_WAREKI = re.compile(r"^([RHS])(\d{1,2}|元)\.(\d{1,2})\.(\d{1,2})$")
_MISSING = {"", "-", "null", "."}


class MacroParseError(ValueError):
    """来源格式与预期不符（表头变了、API 报错等）。"""


def parse_wareki(text: str) -> date | None:
    """和暦 'R6.8.15' → date；不是和暦或日期非法返回 None。R=2018+n，H=1988+n，S=1925+n。"""
    m = _WAREKI.match(unicodedata.normalize("NFKC", str(text)).strip().upper())
    if not m:
        return None
    era, year, month, day = m.groups()
    n = 1 if year == "元" else int(year)
    try:
        return date(ERA_BASE[era] + n, int(month), int(day))
    except ValueError:
        return None


def _to_float(text: str | None) -> float | None:
    s = unicodedata.normalize("NFKC", str(text if text is not None else "")).strip()
    if s in _MISSING:
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def parse_jgb_csv(content: bytes) -> pd.DataFrame:
    """財務省 国債金利 CSV → 长表。缺失值不出行：长表里缺一行就是缺失，没有必要存 NaN。"""
    rows = list(csv.reader(io.StringIO(content.decode("cp932"))))
    header_idx = next((i for i, r in enumerate(rows) if r and unicodedata.normalize("NFKC", r[0]).strip() == "基準日"), None)
    if header_idx is None:
        raise MacroParseError("jgb csv: 找不到表头（基準日）")
    labels = [unicodedata.normalize("NFKC", c).strip() for c in rows[header_idx]]
    cols = {i: JGB_COLUMNS[c] for i, c in enumerate(labels) if c in JGB_COLUMNS}
    missing = set(JGB_COLUMNS.values()) - set(cols.values())
    if missing:
        raise MacroParseError(f"jgb csv: 缺少期限列 {sorted(missing)}")

    out: list[dict[str, Any]] = []
    n_dates = 0
    for row in rows[header_idx + 1:]:
        d = parse_wareki(row[0]) if row else None
        if d is None:       # 文件末尾的空行与脚注
            continue
        n_dates += 1
        for i, series in cols.items():
            v = _to_float(row[i]) if i < len(row) else None
            if v is not None:
                out.append({"series": series, "date": d, "value": v, "source": "mof", "unit": "%"})
    if not n_dates:
        raise MacroParseError("jgb csv: 没有可解析的数据行")
    return pd.DataFrame(out, columns=COLUMNS)
