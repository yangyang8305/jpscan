"""错杀辅助（registry price_dislocation 家族与 rev_* 门槛，替代规划文档 4.3 的 O1–O5 打分）。

入池由 1 个月行业调整后跌幅 rev_1m 决定，在 screener/evidence.py 里按当日横截面判断。这里只回答三件事，并写入 mart.oversold_daily：
1. 事件标记：固定阈值急跌（rev_drawdown_triggers，只展示，用于标出事件日）。
2. 下跌归因：有 TDnet 时按事件日 −5…+1 匹配 T1–T4；没有时只有「业绩型（代理）」和「未归因」，未归因不算通过也不算不通过。
3. 基本面完好检验（rev_intact_test）：已知项全部通过；未知项列入未检查。

文档 4.3 的价格反应比 R 和「预想未下修」F1 已按 registry 移除：指引不变时 R 只是跌幅的另一种写法，F1 在 1Q 几乎不筛人。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import pget

log = logging.getLogger("jpscan.oversold")

CATEGORY_TYPE = {
    "E02": "T1", "E04": "T1", "E37": "T1", "E16": "T1",
    "E07": "T2", "E08": "T2", "E09": "T2", "E10": "T2",
    "E17": "T3", "E18": "T3", "E19": "T3", "E22": "T3", "E23": "T3", "E24": "T3",
    "E33": "T4", "E34": "T4",
}
HARD_FAIL_CATS = {"E10", "E17", "E20", "E21", "E24"}
FEATURE_COLUMNS = ["Code", "Date", "S33Nm", "mcap", "C", "ret_1d", "ret_5d", "ret_21d", "drawdown_60d", "volume_mult", "low_since_20d", "atr5", "atr14", "limit_down",
                   "rev_1m", "cause_proxy_earnings", "cause_proxy_earnings_60", "cause_proxy_date", "guid_forecast_innovation", "is_financial", "has_forecast", "guid_fail",
                   "evt_div_down", "evt_dividend_change", "guid_remaining_flag", "gate_distress", "negative_equity"]


def intact_test(d: pd.DataFrame) -> pd.DataFrame:
    """rev_intact_test：每项 1=通过、0=不通过、NaN=未知。金融业只看分红项。至少一项已知且没有不通过才算 pass。"""
    fin = d["is_financial"].fillna(False).astype(bool).to_numpy() if "is_financial" in d.columns else np.zeros(len(d), dtype=bool)
    has_fc = d["has_forecast"].fillna(False).astype(bool)
    rev = pd.Series(np.where(d["guid_fail"].fillna(False).astype(bool), 0.0, 1.0), index=d.index).where(has_fc)
    div = pd.Series(np.where(d["evt_div_down"].fillna(False).astype(bool), 0.0, 1.0), index=d.index).where(d["evt_dividend_change"].notna())
    rem = 1.0 - d["guid_remaining_flag"]
    dis = 1.0 - d["gate_distress"]
    items = pd.DataFrame({"intact_revision": rev, "intact_dividend": div, "intact_remaining": rem, "intact_distress": dis}, index=d.index)
    items.loc[fin, ["intact_revision", "intact_remaining", "intact_distress"]] = np.nan
    known = items.notna().sum(axis=1)
    failed = (items == 0).any(axis=1)
    out = items.copy()
    out["intact_known"] = known
    out["intact_pass"] = (items == 1).sum(axis=1)
    out["intact_status"] = np.select([failed, known > 0], ["fail", "pass"], default="unknown")
    return out


def cause_wait(d: pd.DataFrame) -> pd.Series:
    """业绩型下跌的等待：窗口内最大跌幅落在决算附近，21 个交易日内不入 B3；翌期指引差（SUPF<0）时最长 60 日。"""
    w21 = d["cause_proxy_earnings"].fillna(False).astype(bool)
    w60 = d["cause_proxy_earnings_60"].fillna(False).astype(bool) & (d["guid_forecast_innovation"] < 0)
    return w21 | w60


def build_oversold(feat: pd.DataFrame, disclosures: pd.DataFrame | None, params: dict) -> pd.DataFrame:
    op = params.get("oversold", {})
    trig, att, stab = (op.get(k, {}) for k in ("trigger", "attribution", "stabilization"))
    cols = [c for c in FEATURE_COLUMNS if c in feat.columns]
    d = feat[cols].sort_values(["Code", "Date"]).reset_index(drop=True).copy()
    d["Date"] = pd.to_datetime(d["Date"])
    # ---- 事件标记（只展示）----
    large = d["mcap"] >= float(trig.get("large_cap_mcap_jpy", 3e11))
    dd_th = np.where(large, float(trig.get("drawdown_pct_large_cap", 0.15)), float(trig.get("drawdown_pct", 0.20)))
    sd_th = np.where(large, float(trig.get("single_day_pct_large_cap", 0.08)), float(trig.get("single_day_pct", 0.10)))
    fast = (d["drawdown_60d"] <= -dd_th) & (d["ret_21d"] <= -dd_th * 0.75)
    single = (d["ret_1d"] <= -sd_th) & (d["volume_mult"] >= float(trig.get("volume_mult_vs_adv20", 3.0)))
    five = d["ret_5d"] <= -float(trig.get("five_day_pct", 0.15))
    d["trigger"] = (fast | single | five).fillna(False)
    d["trigger_type"] = np.select([single, five, fast], ["single_day", "five_day", "drawdown"], default="")
    g = d.groupby("Code", sort=False)
    d["_tdate"] = d["Date"].where(d["trigger"])
    d["trigger_date"] = g["_tdate"].ffill()
    d["days_since_trigger"] = (d["Date"] - d["trigger_date"]).dt.days
    validity = int(stab.get("validity_days", 60))
    d["active"] = d["trigger_date"].notna() & (d["days_since_trigger"] <= validity * 1.45)   # 自然日近似 60 交易日
    pre_px = g["C"].shift(1)
    d["_pre_px"] = pre_px.where(d["trigger"])
    d["_pre_px"] = d.groupby("Code", sort=False)["_pre_px"].ffill()
    d["event_move"] = d["C"] / d["_pre_px"] - 1
    # ---- 归因 ----
    sector_ret = d.groupby(["Date", "S33Nm"])["ret_21d"].transform("median")
    mkt_ret = d.groupby("Date")["ret_21d"].transform("median")
    sys_ratio = float(att.get("systemic_ratio", 0.667))
    falling = d["ret_21d"] < 0
    systemic = ((sector_ret / d["ret_21d"]).where(falling) >= sys_ratio) | ((mkt_ret / d["ret_21d"]).where(falling) >= sys_ratio)
    proxy = d["cause_proxy_earnings"].fillna(False).astype(bool)
    d["attr_type"] = np.select([proxy, systemic.fillna(False)], ["T1_proxy", "T5"], default="unattributed")
    d["attr_category"] = ""
    d["hard_fail_event"] = False
    if disclosures is not None and not disclosures.empty:
        d = _attribute_with_disclosures(d, disclosures, int(att.get("event_window_before_days", 5)), int(att.get("event_window_after_days", 1)))
    # ---- 完好检验与等待 ----
    d = pd.concat([d, intact_test(d)], axis=1)
    d["cause_wait"] = cause_wait(d)
    d["hard_fail"] = d["hard_fail_event"].astype(bool) | d["negative_equity"].fillna(False).astype(bool)
    d["intact_ok"] = d["intact_status"].eq("pass") & ~d["hard_fail"]
    # ---- 企稳（只展示）----
    above_low = (d["C"] > d["low_since_20d"]).astype(int)
    run_id = (above_low != above_low.groupby(d["Code"]).shift()).cumsum()
    d["_above_low_run"] = above_low.groupby([d["Code"], run_id]).cumsum()
    s1 = d["_above_low_run"] >= int(stab.get("above_low_days", 3))
    s2 = d["atr5"] <= d["atr14"] * (1 - float(stab.get("atr5_decline_from_peak", 0.30)) * 0.5)
    s3 = d["volume_mult"] < float(stab.get("volume_normal_mult", 1.5))
    d["stab_signals"] = s1.astype(int) + s2.fillna(False).astype(int) + s3.fillna(False).astype(int)
    d["stabilized"] = (d["stab_signals"] >= 2) & ~d["cause_wait"]
    out_cols = ["Code", "Date", "trigger", "trigger_type", "trigger_date", "days_since_trigger", "active", "event_move", "attr_type", "attr_category",
                "intact_revision", "intact_dividend", "intact_remaining", "intact_distress", "intact_known", "intact_pass", "intact_status", "intact_ok", "hard_fail",
                "cause_wait", "cause_proxy_date", "stab_signals", "stabilized"]
    out = d[[c for c in out_cols if c in d.columns]].copy()
    out["Date"] = out["Date"].dt.date
    out["trigger_date"] = pd.to_datetime(out["trigger_date"]).dt.date
    if "cause_proxy_date" in out.columns:
        out["cause_proxy_date"] = pd.to_datetime(out["cause_proxy_date"]).dt.date
    return out


def _attribute_with_disclosures(d: pd.DataFrame, disc: pd.DataFrame, before: int, after: int) -> pd.DataFrame:
    """事件日 −before…+after 自然日匹配 TDnet 事件。只用当时已披露的：事件日之后的公告只影响之后的日子。"""
    ev = disc[["Code", "disclosed_date", "category", "hard_exclude"]].copy()
    ev["disclosed_date"] = pd.to_datetime(ev["disclosed_date"])
    ev["etype"] = ev["category"].map(CATEGORY_TYPE)
    ev = ev.dropna(subset=["etype"])
    trig = d.loc[d["trigger"], ["Code", "Date"]].copy()
    if trig.empty or ev.empty:
        return d
    m = trig.merge(ev, on="Code", how="inner")
    m = m[(m["disclosed_date"] >= m["Date"] - pd.Timedelta(days=before)) & (m["disclosed_date"] <= m["Date"] + pd.Timedelta(days=after))]
    if m.empty:
        return d
    m["pri"] = m["etype"].map({"T2": 0, "T3": 1, "T1": 2, "T4": 3})
    m = m.sort_values(["Code", "Date", "pri"]).drop_duplicates(["Code", "Date"])
    m["hf"] = m["category"].isin(HARD_FAIL_CATS) | m["hard_exclude"].fillna(False).astype(bool)
    d = d.merge(m[["Code", "Date", "etype", "category", "hf"]], on=["Code", "Date"], how="left")
    # 触发日的归因向后传播到有效期内
    g = d.groupby("Code", sort=False)
    filled_t = g["etype"].ffill()
    filled_c = g["category"].ffill()
    filled_h = g["hf"].ffill()
    live = d["active"].to_numpy(dtype=bool)
    d["attr_type"] = np.where(live & filled_t.notna().to_numpy(), filled_t.astype(object), d["attr_type"].astype(object))
    d["attr_category"] = np.where(live & filled_c.notna().to_numpy(), filled_c.astype(object), "")
    d["hard_fail_event"] = (live & filled_h.fillna(False).astype(bool).to_numpy())
    return d.drop(columns=["etype", "category", "hf"])
