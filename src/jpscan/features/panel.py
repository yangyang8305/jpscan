"""价格面板与技术/流动性/错杀触发特征（附录 B.5、B.6）。

输入：staging.prices_daily（J-Quants bars），输出按 (Code, Date) 的技术特征；全部用调整后价格 AdjC/AdjVo 计算
趋势类指标，用原始 C 计算当日市值与每股指标（配合调整因子换算）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_price_panel(store, start=None, end=None) -> pd.DataFrame:
    where = []
    if start:
        where.append(f"Date >= DATE '{start}'")
    if end:
        where.append(f"Date <= DATE '{end}'")
    df = store.read("staging", "prices_daily", where=" AND ".join(where) if where else None)
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df = df.sort_values(["Code", "Date"]).reset_index(drop=True)
    # 交易所未成交日全列为空 → 用前值补 AdjC 以便均线连续，但成交量保持 0
    df["AdjC_ff"] = df.groupby("Code")["AdjC"].ffill()
    df["Vo"] = df["Vo"].fillna(0.0)
    df["Va"] = df["Va"].fillna(0.0)
    df["AdjVo"] = df["AdjVo"].fillna(0.0)
    df["AdjFactor"] = df["AdjFactor"].fillna(1.0)
    return df


def market_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """用全市场（市值加权，MktCap 单位百万円）日收益构造 TOPIX 代理（Free 套餐无指数端点）。"""
    d = df[["Code", "Date", "AdjC_ff", "MktCap"]].copy()
    d["ret"] = d.groupby("Code")["AdjC_ff"].pct_change()
    d = d.dropna(subset=["ret"])
    d["w"] = d["MktCap"].fillna(0.0).clip(lower=0)
    g = d.groupby("Date")
    mkt = (g.apply(lambda x: np.average(x["ret"], weights=x["w"]) if x["w"].sum() > 0 else x["ret"].mean(), include_groups=False)
           .rename("mkt_ret").reset_index())
    ew = g["ret"].mean().rename("mkt_ret_ew").reset_index()
    out = mkt.merge(ew, on="Date")
    out["mkt_level"] = (1 + out["mkt_ret"]).cumprod()
    return out


def technical_features(df: pd.DataFrame, mkt: pd.DataFrame, params: dict) -> pd.DataFrame:
    tp = params.get("trend", {})
    op = params.get("oversold", {}).get("trigger", {})
    ma_windows = tp.get("ma_windows", [25, 75, 200])
    rs_windows = tp.get("rs_windows_days", [21, 63, 126])
    atr_w = int(tp.get("atr_window", 14))
    vol_w = int(tp.get("realized_vol_window", 60))
    beta_w = int(tp.get("beta_window", 250))
    hi_lookback = int(op.get("high_lookback_days", 60))

    d = df.sort_values(["Code", "Date"]).copy()
    g = d.groupby("Code", sort=False)
    px = d["AdjC_ff"]
    d["ret_1d"] = g["AdjC_ff"].pct_change()
    d["ret_5d"] = g["AdjC_ff"].pct_change(5)
    d["ret_21d"] = g["AdjC_ff"].pct_change(21)
    for w in ma_windows:
        ma = g["AdjC_ff"].transform(lambda s, w=w: s.rolling(w, min_periods=max(5, w // 2)).mean())
        d[f"ma{w}"] = ma
        d[f"px_vs_ma{w}"] = px / ma - 1
    for w in (25, 75):
        if f"ma{w}" in d:
            d[f"ma{w}_slope20"] = g[f"ma{w}"].pct_change(20)
    d["high_52w"] = g["AdjC_ff"].transform(lambda s: s.rolling(int(tp.get("high_52w_days", 250)), min_periods=60).max())
    d["dist_52w_high"] = px / d["high_52w"] - 1
    d[f"high_{hi_lookback}d"] = g["AdjC_ff"].transform(lambda s: s.rolling(hi_lookback, min_periods=10).max())
    d["drawdown_60d"] = px / d[f"high_{hi_lookback}d"] - 1
    d["low_since_20d"] = g["AdjC_ff"].transform(lambda s: s.rolling(20, min_periods=5).min())
    # 成交额与量比
    d["adv20_turnover"] = g["Va"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    d["adv120_turnover"] = g["Va"].transform(lambda s: s.rolling(120, min_periods=40).mean())
    d["vol_ratio_20_120"] = d["adv20_turnover"] / d["adv120_turnover"]
    d["adv20_volume"] = g["AdjVo"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    d["volume_mult"] = d["AdjVo"] / d["adv20_volume"]
    # ATR / 波动率
    prev_c = g["AdjC_ff"].shift(1)
    tr = pd.concat([(d["AdjH"] - d["AdjL"]).abs(), (d["AdjH"] - prev_c).abs(), (d["AdjL"] - prev_c).abs()], axis=1).max(axis=1)
    d["tr"] = tr
    d["atr14"] = d.groupby("Code", sort=False)["tr"].transform(lambda s, w=atr_w: s.rolling(w, min_periods=5).mean())
    d["atr_pct"] = d["atr14"] / px
    d["atr5"] = d.groupby("Code", sort=False)["tr"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    d["rvol60"] = g["ret_1d"].transform(lambda s, w=vol_w: s.rolling(w, min_periods=20).std()) * np.sqrt(250)
    # 相对强弱与 β（市场代理）
    m = mkt.set_index("Date")
    d = d.merge(mkt[["Date", "mkt_ret"]], on="Date", how="left")
    d = d.sort_values(["Code", "Date"]).reset_index(drop=True)
    g = d.groupby("Code", sort=False)
    d["mkt_level"] = d["Date"].map(m["mkt_level"])
    for w in rs_windows:
        d[f"rs_{w}d"] = g["AdjC_ff"].pct_change(w) - g["mkt_level"].pct_change(w)
    cov = g.apply(lambda x: x["ret_1d"].rolling(beta_w, min_periods=120).cov(x["mkt_ret"]), include_groups=False)
    var = g.apply(lambda x: x["mkt_ret"].rolling(beta_w, min_periods=120).var(), include_groups=False)
    d["beta"] = (cov / var).reset_index(level=0, drop=True).sort_index().values if len(cov) == len(d) else np.nan
    # 阶段分类（T10）
    d["stage"] = _stage(d)
    # 涨跌停标记
    d["limit_up"] = d.get("UL", pd.Series(["0"] * len(d))).astype(str).isin(["1", "1.0"])
    d["limit_down"] = d.get("LL", pd.Series(["0"] * len(d))).astype(str).isin(["1", "1.0"])
    # 累积调整因子（用于把报告期每股数据换算到当日口径）
    d["cumadj"] = g["AdjFactor"].cumprod()
    keep = ["Code", "Date", "C", "AdjC", "AdjC_ff", "Vo", "Va", "AdjVo", "MktCap", "AdjFactor", "cumadj", "ret_1d", "ret_5d", "ret_21d",
            "high_52w", "dist_52w_high", "drawdown_60d", "low_since_20d", "adv20_turnover", "adv120_turnover", "vol_ratio_20_120",
            "adv20_volume", "volume_mult", "atr14", "atr_pct", "atr5", "rvol60", "beta", "stage", "limit_up", "limit_down", "mkt_ret"]
    keep += [c for c in d.columns if c.startswith("ma") or c.startswith("px_vs_ma") or c.startswith("rs_")]
    keep = list(dict.fromkeys([k for k in keep if k in d.columns]))
    return d[keep]


def _stage(d: pd.DataFrame) -> pd.Series:
    px = d["AdjC_ff"]
    ma25, ma75, ma200 = d.get("ma25"), d.get("ma75"), d.get("ma200")
    if ma25 is None or ma75 is None or ma200 is None:
        return pd.Series(["unknown"] * len(d), index=d.index)
    s25 = d.get("ma25_slope20", pd.Series(np.nan, index=d.index))
    up = (px > ma25) & (ma25 > ma75) & (ma75 > ma200)
    down = (px < ma75) & (ma75 < ma200)
    top = (px < ma25) & (s25 < 0) & (px > ma200)
    base = ((px / ma200 - 1).abs() < 0.05)
    out = np.select([up, down, top, base], ["uptrend", "downtrend", "top", "base"], default="transition")
    out = np.where(ma200.isna(), "unknown", out)
    return pd.Series(out, index=d.index)
