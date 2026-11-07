"""市场环境标签（规划文档 7.4 节）：趋势、广度、利率、波动、风格 → risk_on / neutral / risk_off，并映射到参数。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_regime(feat: pd.DataFrame, params: dict, macro: pd.DataFrame | None = None) -> pd.DataFrame:
    """feat: features_daily（需 Date, Code, mkt_ret, px_vs_ma200, mcap, ret_1d）。返回按日的环境表。"""
    r = params.get("regime", {})
    th = r.get("thresholds", {})
    d = feat[["Date", "Code", "mkt_ret", "px_vs_ma200", "mcap", "ret_1d", "ret_21d"]].copy()
    daily = d.groupby("Date").agg(mkt_ret=("mkt_ret", "first"), breadth=("px_vs_ma200", lambda s: (s > 0).mean()),
                                  n=("Code", "size")).reset_index().sort_values("Date")
    daily["mkt_level"] = (1 + daily["mkt_ret"].fillna(0)).cumprod()
    daily["mkt_ma200"] = daily["mkt_level"].rolling(200, min_periods=100).mean()
    daily["mkt_vs_ma200"] = daily["mkt_level"] / daily["mkt_ma200"] - 1
    daily["mkt_rvol20"] = daily["mkt_ret"].rolling(20, min_periods=10).std() * np.sqrt(250)
    # 风格：小盘（市值后 50%）vs 大盘（前 20%）6 个月相对表现
    def style(sub: pd.DataFrame) -> float:
        if sub["mcap"].notna().sum() < 100:
            return np.nan
        q20 = sub["mcap"].quantile(0.8)
        q50 = sub["mcap"].quantile(0.5)
        big = sub.loc[sub["mcap"] >= q20, "ret_21d"].median()
        small = sub.loc[sub["mcap"] <= q50, "ret_21d"].median()
        return small - big
    st = d.groupby("Date").apply(style, include_groups=False).rename("small_minus_big_21d").reset_index()
    daily = daily.merge(st, on="Date", how="left")
    if macro is not None and not macro.empty:
        j = macro[macro["series"] == "jgb_10y"].sort_values("date")
        if not j.empty:
            j = j.assign(Date=pd.to_datetime(j["date"]).dt.date)
            daily = daily.merge(j[["Date", "value"]].rename(columns={"value": "jgb10y"}), on="Date", how="left")
            daily["jgb10y"] = daily["jgb10y"].ffill()
            daily["jgb10y_chg_6m"] = daily["jgb10y"] - daily["jgb10y"].shift(125)
    risk_off = (daily["mkt_vs_ma200"] < 0) & (daily["breadth"] <= float(th.get("risk_off", {}).get("breadth_max", 0.35)))
    risk_on = (daily["mkt_vs_ma200"] > 0) & (daily["breadth"] >= float(th.get("risk_on", {}).get("breadth_min", 0.60)))
    daily["regime"] = np.select([risk_off, risk_on], ["risk_off", "risk_on"], default="neutral")
    daily.loc[daily["mkt_ma200"].isna(), "regime"] = "neutral"
    mapping = r.get("mapping", {})
    daily["cash_buffer"] = daily["regime"].map({k: float(v.get("cash_buffer", 0.1)) for k, v in mapping.items()}).fillna(0.1)
    daily["atr_stop_mult"] = daily["regime"].map({k: float(v.get("atr_stop_mult", 2.5)) for k, v in mapping.items()}).fillna(2.5)
    return daily


def regime_for_date(daily: pd.DataFrame, d) -> dict:
    sub = daily[pd.to_datetime(daily["Date"]) <= pd.Timestamp(d)]
    if sub.empty:
        return {"regime": "neutral", "cash_buffer": 0.1, "atr_stop_mult": 2.5}
    row = sub.iloc[-1]
    return {k: (row[k] if k in row else None) for k in ("Date", "regime", "cash_buffer", "atr_stop_mult", "mkt_vs_ma200", "breadth", "mkt_rvol20",
                                                        "small_minus_big_21d", "jgb10y")}
