"""特征构建：价格面板 + 财务状态 + 主数据 → mart.features_daily（附录 B 全部指标 + 同业/自身百分位 + 质量标志）。"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..config import pget
from .fundamentals import asof_join, build_statement_states, load_statements
from .panel import load_price_panel, market_proxy, technical_features

log = logging.getLogger("jpscan.features")

FINANCIAL_S33 = {"7050", "7100", "7150", "7200"}  # 銀行業 / 証券・商品先物取引業 / 保険業 / その他金融業

# 同业百分位指标：名称 -> 方向（True = 越高越好）
PCT_METRICS: dict[str, bool] = {
    "pbr": False, "per_fwd": False, "per_ttm": False, "ev_ebitda": False, "fcf_yield": True, "div_yield_fwd": True,
    "cash_to_mcap": True, "roe_avg": True, "equity_ratio": True, "accrual_avg": False, "rev_growth_1y": True,
    "op_growth_1y": True, "rev_cagr_3y": True, "op_cagr_3y": True, "f_op_growth": True, "progress_dev": True,
    "op_margin_fy0": True, "op_margin_chg": True, "ocf_to_op": True, "rs_63d": True, "rvol60": False,
}


def load_master_snapshots(store) -> pd.DataFrame:
    m = store.read("staging", "security_master_snapshots")
    if m.empty:
        return m
    m["Date"] = pd.to_datetime(m["Date"]).dt.date
    return m.sort_values(["Code", "Date"])


def attach_master(panel: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        for c in ("CoName", "S17", "S33", "Mkt", "ScaleCat", "ProdCat", "Mrgn"):
            panel[c] = None
        return panel
    cols = ["Code", "Date", "CoName", "S17", "S33", "Mkt", "ScaleCat", "ProdCat", "Mrgn"]
    m = master[[c for c in cols if c in master.columns]].copy()
    m["_d"] = pd.to_datetime(m["Date"])
    m = m.drop(columns=["Date"]).sort_values("_d")
    p = panel.copy()
    p["_d"] = pd.to_datetime(p["Date"])
    p = p.sort_values("_d")
    out = pd.merge_asof(p, m, on="_d", by="Code", direction="backward")
    # 面板早于首个快照的行：用最早快照（forward）补
    miss = out["S33"].isna()
    if miss.any():
        first = master.sort_values("Date").drop_duplicates("Code", keep="first").set_index("Code")
        for c in ("CoName", "S17", "S33", "Mkt", "ScaleCat", "ProdCat", "Mrgn"):
            if c in first.columns:
                out.loc[miss, c] = out.loc[miss, "Code"].map(first[c])
    return out.drop(columns=["_d"])


def compute_fundamental_metrics(df: pd.DataFrame, jgb10y: pd.Series | None = None) -> pd.DataFrame:
    d = df
    # 报告期到当日的调整因子（每股数据换算）
    d["adj_ps"] = (d["cumadj"] / d["cumadj_report"]).fillna(1.0).replace([np.inf, -np.inf], 1.0)
    d["shares_net"] = (d["shares_out"] - d["treasury"].fillna(0.0)) / d["adj_ps"]
    d["mcap"] = d["C"] * d["shares_net"]
    d["mcap_jq"] = d["MktCap"] * 1e6
    d.loc[d["mcap"].isna() | (d["mcap"] <= 0), "mcap"] = d["mcap_jq"]
    eq_latest = d["ta_latest"] * d["eqar_latest"]
    d["equity_sh_latest"] = eq_latest
    bps_calc = eq_latest / d["shares_net"]
    bps_rep = d["bps_reported"] * d["adj_ps"]
    d["bps"] = bps_calc.where(bps_calc.notna() & (bps_calc > 0), bps_rep)
    d["pbr"] = d["C"] / d["bps"]
    d["eps_fwd_adj"] = d["f_eps"] * d["adj_ps"]
    d["per_fwd"] = (d["C"] / d["eps_fwd_adj"]).where(d["eps_fwd_adj"] > 0)
    d["per_ttm"] = (d["mcap"] / d["np_ttm"]).where((d["np_ttm"] > 0) & (d["ttm_quarters"] == 4))
    d["div_yield_fwd"] = (d["f_div_ann"] * d["adj_ps"]) / d["C"]
    d["cash_to_mcap"] = d["cash_latest"] / d["mcap"]
    d["net_cash_ratio"] = np.nan          # 需有利子負債（fins/details 或 EDINET）
    d["ev_ebitda"] = np.nan               # 需減価償却費与有利子負債
    fcf0 = d["cfo_fy0"] + d["cfi_fy0"]
    fcf1 = d["cfo_fy1"] + d["cfi_fy1"]
    fcf2 = d["cfo_fy2"] + d["cfi_fy2"]
    d["fcf_avg"] = pd.concat([fcf0, fcf1, fcf2], axis=1).mean(axis=1, skipna=True)
    d["fcf_yield"] = d["fcf_avg"] / d["mcap"]
    d["earnings_yield_fwd"] = 1.0 / d["per_fwd"]
    if jgb10y is not None:
        d["jgb10y"] = pd.to_datetime(d["Date"]).map(jgb10y)
        d["jgb10y"] = d["jgb10y"].ffill()
        d["ey_spread"] = d["earnings_yield_fwd"] - d["jgb10y"] / 100.0
    else:
        d["jgb10y"] = np.nan
        d["ey_spread"] = np.nan
    # 质量
    eq0 = d["ta_fy0"] * d["eqar_fy0"]
    eq1 = d["ta_fy1"] * d["eqar_fy1"]
    d["roe_fy0"] = d["np_fy0"] / pd.concat([eq0, eq1], axis=1).mean(axis=1, skipna=True)
    d["roe_avg"] = d["roe_fy0"]  # 数据窗口内年度数不足时退化为单年
    d["equity_ratio"] = d["eqar_latest"]
    d["accrual_fy0"] = (d["np_fy0"] - d["cfo_fy0"]) / d["ta_fy0"]
    d["accrual_avg"] = d["accrual_fy0"]
    d["ocf_positive_years"] = (d[["cfo_fy0", "cfo_fy1", "cfo_fy2"]] > 0).sum(axis=1)
    d["ocf_years_available"] = d[["cfo_fy0", "cfo_fy1", "cfo_fy2"]].notna().sum(axis=1)
    d["ocf_to_op"] = (d["cfo_fy0"] / d["op_fy0"]).where(d["op_fy0"] > 0)
    d["dividend_cut"] = (d["div_fy0"] < d["div_fy1"]) & d["div_fy0"].notna() & d["div_fy1"].notna()
    d["negative_equity"] = d["eqar_latest"] < 0
    # 成长
    d["rev_growth_1y"] = d["sales_fy0"] / d["sales_fy1"] - 1
    d["op_growth_1y"] = (d["op_fy0"] / d["op_fy1"] - 1).where(d["op_fy1"] > 0)
    d["rev_cagr_3y"] = (d["sales_fy0"] / d["sales_fy3"]) ** (1 / 3) - 1
    d["op_cagr_3y"] = ((d["op_fy0"] / d["op_fy3"]) ** (1 / 3) - 1).where(d["op_fy3"] > 0)
    d["op_margin_fy0"] = d["op_fy0"] / d["sales_fy0"]
    d["op_margin_fy1"] = d["op_fy1"] / d["sales_fy1"]
    d["op_margin_chg"] = d["op_margin_fy0"] - d["op_margin_fy1"]
    d["f_op_growth"] = (d["f_op"] / d["op_fy0"] - 1).where(d["op_fy0"] > 0)
    d["progress_dev"] = d["progress_rate"] - d["progress_rate_prev_year"]
    inc = ((d["sales_fy0"] > d["sales_fy1"]) & (d["op_fy0"] > d["op_fy1"])).astype(int)
    inc2 = ((d["sales_fy1"] > d["sales_fy2"]) & (d["op_fy1"] > d["op_fy2"])).astype(int)
    d["inc_streak"] = inc + inc * inc2
    d["actual_vs_initial"] = np.nan
    # 数据质量标志
    eff = pd.to_datetime(d["effective_date"], errors="coerce")
    d["days_since_report"] = (pd.to_datetime(d["Date"]) - eff).dt.days
    d["stale_financials"] = d["days_since_report"] > 135
    d["has_forecast"] = d["f_eps"].notna()
    d["is_financial"] = d["S33"].astype(str).isin(FINANCIAL_S33)
    d["is_common_stock"] = d["ProdCat"].astype(str).eq("011") & d["Code"].astype(str).str.endswith("0")
    d["in_tse_main"] = d["Mkt"].astype(str).isin(["0111", "0112", "0113"])
    return d


def peer_percentiles(d: pd.DataFrame, min_group: int = 20) -> pd.DataFrame:
    """同业百分位（S33 → S17 → 全市场回退）；只对普通株且在三市场内的行计算。"""
    base_mask = d["is_common_stock"] & d["in_tse_main"]
    for m in PCT_METRICS:
        if m not in d.columns:
            d[m] = np.nan
    sub = d.loc[base_mask, ["Date", "S33", "S17", *PCT_METRICS.keys()]].copy()
    out = pd.DataFrame(index=sub.index)
    for m in PCT_METRICS:
        s = sub[m].replace([np.inf, -np.inf], np.nan)
        p33 = s.groupby([sub["Date"], sub["S33"]]).rank(pct=True)
        n33 = s.notna().groupby([sub["Date"], sub["S33"]]).transform("sum")
        p17 = s.groupby([sub["Date"], sub["S17"]]).rank(pct=True)
        n17 = s.notna().groupby([sub["Date"], sub["S17"]]).transform("sum")
        pmk = s.groupby(sub["Date"]).rank(pct=True)
        pct = p33.where(n33 >= min_group, p17.where(n17 >= min_group, pmk))
        grp = np.where(n33 >= min_group, "s33", np.where(n17 >= min_group, "s17", "mkt"))
        out[f"pct_{m}"] = pct
        out[f"pct_{m}_grp"] = grp
        out[f"pct_mkt_{m}"] = pmk
    for c in out.columns:
        d[c] = out[c]
    return d


def self_history_percentile(d: pd.DataFrame, metric: str, window: int = 1250, min_periods: int = 250) -> pd.Series:
    """自身历史百分位：当日值在过去 window 个交易日中的分位。"""
    res = pd.Series(np.nan, index=d.index)
    for code, g in d.groupby("Code", sort=False):
        v = g[metric].to_numpy(dtype=float)
        n = len(v)
        if n < min_periods:
            continue
        out = np.full(n, np.nan)
        for i in range(min_periods - 1, n):
            lo = max(0, i - window + 1)
            win = v[lo:i + 1]
            win = win[~np.isnan(win)]
            if len(win) >= min_periods and not np.isnan(v[i]):
                out[i] = (win <= v[i]).mean()
        res.loc[g.index] = out
    return res


def build_features(store, params: dict, start: date | None = None, end: date | None = None, write: bool = True) -> pd.DataFrame:
    hist_days = 420  # 技术指标需要的历史
    panel_start = (start - timedelta(days=hist_days)) if start else None
    panel = load_price_panel(store, panel_start, end)
    if panel.empty:
        log.warning("prices_daily is empty")
        return pd.DataFrame()
    mkt = market_proxy(panel)
    tech = technical_features(panel, mkt, params)
    # 报告日的累积调整因子
    cum = tech[["Code", "Date", "cumadj"]].rename(columns={"Date": "effective_date", "cumadj": "cumadj_report"})
    st = load_statements(store)
    states = build_statement_states(st) if not st.empty else pd.DataFrame()
    if not states.empty:
        states = states.merge(cum, on=["Code", "effective_date"], how="left")
        states["cumadj_report"] = states["cumadj_report"].fillna(1.0)
        feat = asof_join(tech, states)
        feat = feat.rename(columns={"effective_date": "effective_date"})
    else:
        feat = tech.copy()
        feat["cumadj_report"] = 1.0
        for c in ("shares_out", "treasury", "bps_reported", "f_eps", "f_div_ann", "ta_latest", "eqar_latest", "cash_latest", "np_ttm",
                  "ttm_quarters", "cfo_fy0", "cfi_fy0", "cfo_fy1", "cfi_fy1", "cfo_fy2", "cfi_fy2", "ta_fy0", "eqar_fy0", "ta_fy1",
                  "eqar_fy1", "np_fy0", "div_fy0", "div_fy1", "sales_fy0", "sales_fy1", "sales_fy2", "sales_fy3", "op_fy0", "op_fy1",
                  "op_fy2", "op_fy3", "f_op", "progress_rate", "progress_rate_prev_year", "effective_date"):
            feat[c] = np.nan
    if "effective_date" not in feat.columns:
        feat["effective_date"] = pd.NaT
    master = load_master_snapshots(store)
    feat = attach_master(feat, master)
    jgb = _jgb10y(store)
    feat = compute_fundamental_metrics(feat, jgb)
    feat = peer_percentiles(feat, int(pget(params, "peer_group.min_group_size", 20)))
    feat["self_pct_pbr"] = self_history_percentile(feat, "pbr", int(pget(params, "peer_group.self_history_window_days", 1250)),
                                                   int(pget(params, "peer_group.self_history_min_days", 500)) // 2)
    feat["self_pct_per_fwd"] = self_history_percentile(feat, "per_fwd", int(pget(params, "peer_group.self_history_window_days", 1250)),
                                                       int(pget(params, "peer_group.self_history_min_days", 500)) // 2)
    if start:
        feat = feat[pd.to_datetime(feat["Date"]).dt.date >= start]
    feat = feat.reset_index(drop=True)
    feat["metric_version"] = "B.v0"
    if write and not feat.empty:
        part = pd.to_datetime(feat["Date"]).dt.strftime("%Y-%m")
        for p, sub in feat.groupby(part):
            store.write_merge("mart", "features_daily", sub, str(p), ["Code", "Date"])
        log.info("features_daily written: %d rows, %d partitions", len(feat), part.nunique())
    return feat


def _jgb10y(store) -> pd.Series | None:
    try:
        m = store.read("staging", "macro_series", where="series = 'jgb_10y'")
    except Exception:  # noqa: BLE001
        return None
    if m.empty:
        return None
    s = m.sort_values("date").set_index(pd.to_datetime(m["date"]))["value"].astype(float)
    return s[~s.index.duplicated(keep="last")]
