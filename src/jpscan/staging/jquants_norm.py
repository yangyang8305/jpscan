"""J-Quants 原始行 → 标准化表（staging）。

规则：
- 代码 Code 保留 5 位字符串，另加 code4（前 4 位）；internal_id = Code（后续代码变更通过 code_history 映射）
- 日期列转 date；数值列 to_numeric(errors='coerce')；空字符串 → NaN
- 标志列（true/false）保留字符串
- fins_summary 增加 disc_dt（披露时刻，JST）与 effective_date（时点生效日）
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any, Iterable

import numpy as np
import pandas as pd

STR_COLS: dict[str, set[str]] = {
    "master": {"Date", "Code", "CoName", "CoNameEn", "S17", "S17Nm", "S33", "S33Nm", "ScaleCat", "Mkt", "MktNm",
               "Mrgn", "MrgnNm", "ProdCat"},
    "bars_daily": {"Date", "Code", "UL", "LL", "ExRT"},
    "valuation": {"Date", "Code"},
    "fins_summary": {"DiscDate", "DiscTime", "Code", "DiscNo", "DocType", "CurPerType", "CurPerSt", "CurPerEn", "CurFYSt",
                     "CurFYEn", "NxtFYSt", "NxtFYEn", "MatChgSub", "SigChgInC", "ChgByASRev", "ChgNoASRev", "ChgAcEst",
                     "RetroRst"},
    "earnings_date": {"PubDate", "SchDate", "FQName", "FYE", "Code", "CoName", "CoNameEn"},
    "earnings_calendar": {"Date", "Code", "CoName", "CoNameEn", "FYE", "SectorName", "Section", "FQName"},
    "calendar": {"Date"},
    "investor_types": {"PubDate", "StDate", "EnDate", "Sec", "SecNm", "Section"},
    "topix": {"Date"},
    "indices": {"Date", "Code"},
    "margin_interest": {"Date", "Code", "IssueType", "Mrgn"},
    "margin_alert": {"Date", "Code"},
    "short_ratio": {"Date", "S33", "S33Nm", "Sector33Code", "Sector33Name"},
    "short_sale_report": {"Date", "Code", "DiscDate", "CalcDate", "Name", "Holder", "HolderAddr", "Entrust"},
    "breakdown": {"Date", "Code"},
    "fins_details": {"DiscDate", "DiscTime", "Code", "DiscNo", "DocType"},
    "dividend": {"AnnDate", "AnnTime", "Code", "RefNo", "StatusCode", "BoardDate", "IntDate", "RecDate", "PayDate",
                 "CAType", "DivType", "FYE", "QuarterStatus"},
}
DATE_COLS: dict[str, set[str]] = {
    "master": {"Date"}, "bars_daily": {"Date"}, "valuation": {"Date"},
    "fins_summary": {"DiscDate", "CurPerSt", "CurPerEn", "CurFYSt", "CurFYEn", "NxtFYSt", "NxtFYEn"},
    "earnings_date": {"PubDate", "SchDate"}, "earnings_calendar": {"Date"}, "calendar": {"Date"},
    "investor_types": {"PubDate", "StDate", "EnDate"}, "topix": {"Date"}, "indices": {"Date"},
    "margin_interest": {"Date"}, "margin_alert": {"Date"}, "short_ratio": {"Date"}, "short_sale_report": {"Date", "DiscDate", "CalcDate"},
    "breakdown": {"Date"}, "fins_details": {"DiscDate"}, "dividend": {"AnnDate", "BoardDate", "IntDate", "RecDate", "PayDate"},
}
KEYS: dict[str, list[str]] = {
    "master": ["Date", "Code"], "bars_daily": ["Date", "Code"], "valuation": ["Date", "Code"],
    "fins_summary": ["Code", "DiscNo"], "earnings_date": ["Code", "PubDate", "SchDate", "FQName"],
    "earnings_calendar": ["Date", "Code"], "calendar": ["Date"], "investor_types": ["PubDate", "Sec", "Section"],
    "topix": ["Date"], "indices": ["Date", "Code"], "margin_interest": ["Date", "Code"], "margin_alert": ["Date", "Code"],
    "short_ratio": ["Date", "S33"], "short_sale_report": ["Date", "Code", "Holder", "CalcDate"], "breakdown": ["Date", "Code"],
    "fins_details": ["Code", "DiscNo"], "dividend": ["Code", "RefNo", "AnnDate"],
}
PARTITION_FIELD: dict[str, str] = {
    "master": "Date", "bars_daily": "Date", "valuation": "Date", "fins_summary": "DiscDate", "earnings_date": "PubDate",
    "earnings_calendar": "Date", "calendar": "Date", "investor_types": "PubDate", "topix": "Date", "indices": "Date",
    "margin_interest": "Date", "margin_alert": "Date", "short_ratio": "Date", "short_sale_report": "Date", "breakdown": "Date",
    "fins_details": "DiscDate", "dividend": "AnnDate",
}
TABLE_NAME: dict[str, str] = {
    "master": "security_master_snapshots", "bars_daily": "prices_daily", "valuation": "valuation_daily",
    "fins_summary": "financial_statements", "earnings_date": "earnings_calendar", "earnings_calendar": "earnings_calendar_next",
    "calendar": "trading_calendar", "investor_types": "investor_types", "topix": "index_topix", "indices": "index_daily",
    "margin_interest": "margin_interest", "margin_alert": "margin_alert", "short_ratio": "short_ratio",
    "short_sale_report": "short_sale_report", "breakdown": "trade_breakdown", "fins_details": "financial_details",
    "dividend": "dividends",
}


def normalize(dataset: str, rows: Iterable[dict[str, Any]], calendar: set[date] | None = None) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    if df.empty:
        return df
    str_cols = STR_COLS.get(dataset, set())
    date_cols = DATE_COLS.get(dataset, set())
    df = df.replace({"": None})
    for c in df.columns:
        if c in date_cols:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
        elif c in str_cols:
            df[c] = df[c].astype("string")
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    if "Code" in df.columns:
        df["Code"] = df["Code"].astype("string")
        df["code4"] = df["Code"].str[:4]
    if dataset == "fins_summary":
        df = _enrich_fins(df, calendar)
    keys = [k for k in KEYS.get(dataset, []) if k in df.columns]
    if keys:
        df = df.drop_duplicates(subset=keys, keep="last")
    return df.reset_index(drop=True)


def next_trading_day(d: date, calendar: set[date] | None, inclusive: bool = False) -> date:
    """返回 d（inclusive=True）或 d 之后的第一个交易日；无日历时按工作日。"""
    cur = d if inclusive else d + timedelta(days=1)
    for _ in range(30):
        ok = (cur in calendar) if calendar else (cur.weekday() < 5)
        if ok:
            return cur
        cur += timedelta(days=1)
    return cur


def effective_date_for(disc_date: date | None, disc_time: str | None, calendar: set[date] | None) -> date | None:
    """披露时刻 < 15:30 → 披露当日（若为交易日）；否则次一交易日（规划文档 3.3 节）。"""
    if disc_date is None:
        return None
    t: time | None = None
    if disc_time:
        try:
            t = datetime.strptime(str(disc_time)[:8], "%H:%M:%S").time()
        except ValueError:
            t = None
    if t is not None and t < time(15, 30):
        return next_trading_day(disc_date, calendar, inclusive=True)
    return next_trading_day(disc_date, calendar, inclusive=False)


def _enrich_fins(df: pd.DataFrame, calendar: set[date] | None) -> pd.DataFrame:
    df["effective_date"] = [effective_date_for(d, t, calendar) for d, t in zip(df["DiscDate"], df.get("DiscTime", pd.Series([None] * len(df))))]
    df["disc_dt"] = [f"{d.isoformat()}T{(t or '00:00:00')}+09:00" if d is not None else None
                     for d, t in zip(df["DiscDate"], df.get("DiscTime", pd.Series([None] * len(df))))]
    # 准则与期间类型
    dt = df["DocType"].astype("string").fillna("")
    df["gaap"] = dt.str.extract(r"_(JP|US|IFRS|JMIS|Foreign|REIT)$")[0]
    df["consolidated"] = np.where(dt.str.contains("NonConsolidated"), "parent", np.where(dt.str.contains("Consolidated"), "consolidated", None))
    df["is_revision"] = dt.str.contains("ForecastRevision")
    df["is_fy"] = dt.str.startswith("FY")
    # 会计期间月数（决算期变更识别）
    def months(a, b):
        if a is None or b is None or pd.isna(a) or pd.isna(b):
            return np.nan
        return (b.year - a.year) * 12 + (b.month - a.month) + 1
    df["fy_months"] = [months(a, b) for a, b in zip(df["CurFYSt"], df["CurFYEn"])]
    return df


def partition_of(dataset: str, df: pd.DataFrame) -> pd.Series:
    """返回每行所属分区（月 'YYYY-MM'；master 按快照日）。"""
    f = PARTITION_FIELD[dataset]
    s = pd.to_datetime(df[f], errors="coerce")
    if dataset == "master":
        return s.dt.strftime("%Y-%m-%d")
    if dataset == "calendar":
        return pd.Series(["all"] * len(df), index=df.index)
    return s.dt.strftime("%Y-%m")
