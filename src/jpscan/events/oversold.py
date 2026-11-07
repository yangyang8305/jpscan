"""错杀维度（规划文档 4.3 节）：触发 → 归因 → 基本面完好检验 → 过度反应 → 企稳 → 分数。

对整个特征面板按时间向量化计算，输出 mart.oversold_daily（Code, Date, oversold_*）。
归因依赖 disclosures（TDnet）表；缺失时归因为 T6（不明）并用行业/市场同期涨跌判定 T5。
"""
from __future__ import annotations

import logging
from datetime import timedelta

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


def build_oversold(feat: pd.DataFrame, disclosures: pd.DataFrame | None, params: dict) -> pd.DataFrame:
    op = params.get("oversold", {})
    trig, att, it, ovr, stab, sc = (op.get(k, {}) for k in ("trigger", "attribution", "intact_tests", "overreaction", "stabilization", "score"))
    large = feat["mcap"] >= float(trig.get("large_cap_mcap_jpy", 3e11))
    dd_th = np.where(large, float(trig.get("drawdown_pct_large_cap", 0.15)), float(trig.get("drawdown_pct", 0.20)))
    sd_th = np.where(large, float(trig.get("single_day_pct_large_cap", 0.08)), float(trig.get("single_day_pct", 0.10)))
    d = feat[["Code", "Date", "S33", "mcap", "C", "ret_1d", "ret_5d", "ret_21d", "drawdown_60d", "volume_mult", "low_since_20d",
              "atr5", "atr14", "equity_ratio", "cash_to_mcap", "progress_dev", "dividend_cut", "f_div_ann", "div_fy0",
              "last_revision_date", "last_revision_chg", "negative_equity", "limit_down"]].copy()
    d = d.sort_values(["Code", "Date"]).reset_index(drop=True)
    # 触发条件
    fast = (d["drawdown_60d"] <= -dd_th) & (d["ret_21d"] <= -dd_th * 0.75)
    single = (d["ret_1d"] <= -sd_th) & (d["volume_mult"] >= float(trig.get("volume_mult_vs_adv20", 3.0)))
    five = d["ret_5d"] <= -float(trig.get("five_day_pct", 0.15))
    d["trigger"] = (fast | single | five).fillna(False)
    d["trigger_type"] = np.select([single, five, fast], ["single_day", "five_day", "drawdown"], default="")
    g = d.groupby("Code", sort=False)
    validity = int(stab.get("validity_days", 60))
    # 触发日及距今天数
    d["_tdate"] = np.where(d["trigger"], pd.to_datetime(d["Date"]), pd.NaT)
    d["_tdate"] = pd.to_datetime(d["_tdate"])
    d["trigger_date"] = g["_tdate"].ffill()
    d["days_since_trigger"] = (pd.to_datetime(d["Date"]) - d["trigger_date"]).dt.days
    d["active"] = d["trigger_date"].notna() & (d["days_since_trigger"] <= validity * 1.45)  # 自然日近似 60 交易日
    # 事件窗口内的累计跌幅（触发后至今相对触发前 1 日）
    pre_px = g["C"].shift(1)
    d["_pre_px"] = np.where(d["trigger"], pre_px, np.nan)
    d["_pre_px"] = g["_pre_px"].ffill()
    d["event_move"] = d["C"] / d["_pre_px"] - 1
    # ---- 归因 ----
    sector_ret = d.groupby(["Date", "S33"])["ret_21d"].transform("median")
    mkt_ret = d.groupby("Date")["ret_21d"].transform("median")
    sys_ratio = float(att.get("systemic_ratio", 0.667))
    systemic = ((sector_ret / d["ret_21d"]).where(d["ret_21d"] < 0) >= sys_ratio) | ((mkt_ret / d["ret_21d"]).where(d["ret_21d"] < 0) >= sys_ratio)
    d["attr_type"] = np.where(systemic, "T5", "T6")
    d["attr_category"] = ""
    d["hard_fail_event"] = False
    if disclosures is not None and not disclosures.empty:
        d = _attribute_with_disclosures(d, disclosures, int(att.get("event_window_before_days", 5)), int(att.get("event_window_after_days", 1)))
    # ---- 基本面完好检验（五项，NaN 视为未知不计）----
    f1 = (d["last_revision_chg"].isna()) | (d["last_revision_chg"] >= -float(it.get("f1_guidance_cut_max", 0.10)))
    rev_in_window = d["last_revision_date"].notna() & (pd.to_datetime(d["last_revision_date"]) >= d["trigger_date"] - pd.Timedelta(days=7))
    f1 = np.where(rev_in_window, d["last_revision_chg"] >= -float(it.get("f1_guidance_cut_max", 0.10)), True)
    f2 = (d["progress_dev"] >= -float(it.get("f2_progress_tolerance", 0.05))).where(d["progress_dev"].notna(), np.nan)
    f3 = ~d["hard_fail_event"]
    f4 = (d["equity_ratio"] >= float(it.get("f4_equity_ratio_min", 0.30))) & ((d["cash_to_mcap"] >= 0.1) | d["cash_to_mcap"].isna())
    f5 = ~(d["dividend_cut"].fillna(False).astype(bool)) & ((d["f_div_ann"] >= d["div_fy0"]) | d["f_div_ann"].isna() | d["div_fy0"].isna())
    tests = pd.DataFrame({"f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5})
    d["intact_pass"] = tests.fillna(False).astype(bool).sum(axis=1)
    d["intact_known"] = tests.notna().sum(axis=1)
    d["hard_fail"] = d["hard_fail_event"] | d["negative_equity"].fillna(False).astype(bool)
    d["intact_ok"] = (d["intact_pass"] >= int(it.get("required_pass", 3))) & ~d["hard_fail"]
    # ---- 过度反应 ----
    eps_chg = d["last_revision_chg"].abs().where(rev_in_window, np.nan).fillna(float(ovr.get("eps_change_floor", 0.05)))
    eps_chg = eps_chg.clip(lower=float(ovr.get("eps_change_floor", 0.05)))
    d["reaction_ratio"] = d["event_move"].abs() / eps_chg
    d["overreaction"] = d["reaction_ratio"] >= float(ovr.get("ratio_threshold", 2.0))
    # ---- 企稳 ----
    above_low = d["C"] > d["low_since_20d"] * 1.0
    d["_above_low_run"] = above_low.astype(int).groupby(d["Code"]).transform(lambda s: s.groupby((s != s.shift()).cumsum()).cumsum())
    s1 = d["_above_low_run"] >= int(stab.get("above_low_days", 3))
    s2 = d["atr5"] <= d["atr14"] * (1 - float(stab.get("atr5_decline_from_peak", 0.30)) * 0.5)
    s3 = d["volume_mult"] < float(stab.get("volume_normal_mult", 1.5))
    d["stab_signals"] = s1.astype(int) + s2.fillna(False).astype(int) + s3.fillna(False).astype(int)
    # T1 业绩型：公告后 21 个交易日内不判企稳（PEAD）
    t1_wait = (d["attr_type"] == "T1") & (d["days_since_trigger"] < 30)
    d["stabilized"] = (d["stab_signals"] >= 2) & ~t1_wait
    # ---- 分数 ----
    over_pts = np.clip(d["reaction_ratio"].fillna(0) / float(ovr.get("ratio_threshold", 2.0)), 0, 1) * float(sc.get("overreaction_max", 40))
    intact_pts = d["intact_pass"] * float(sc.get("intact_per_test", 10))
    stab_pts = d["stab_signals"] * float(sc.get("stabilization_per_signal", 5))
    score = over_pts + intact_pts + stab_pts
    cap = d["attr_type"].map(sc.get("event_type_cap", {"T2": 70, "T3": 70})).fillna(100)
    score = np.minimum(score, cap)
    score = np.where(d["active"], score, 0.0)
    score = np.where(d["hard_fail"], 0.0, score)
    d["score_oversold"] = np.clip(score, 0, 100)
    d["oversold_candidate"] = d["active"] & d["intact_ok"] & ~d["hard_fail"]
    out_cols = ["Code", "Date", "trigger", "trigger_type", "trigger_date", "days_since_trigger", "active", "event_move", "attr_type",
                "attr_category", "intact_pass", "intact_known", "hard_fail", "intact_ok", "reaction_ratio", "overreaction", "stab_signals",
                "stabilized", "score_oversold", "oversold_candidate"]
    out = d[out_cols].copy()
    out["trigger_date"] = pd.to_datetime(out["trigger_date"]).dt.date
    return out


def _attribute_with_disclosures(d: pd.DataFrame, disc: pd.DataFrame, before: int, after: int) -> pd.DataFrame:
    ev = disc[["Code", "disclosed_date", "category", "hard_exclude"]].copy()
    ev["disclosed_date"] = pd.to_datetime(ev["disclosed_date"])
    ev["etype"] = ev["category"].map(CATEGORY_TYPE)
    ev = ev.dropna(subset=["etype"])
    trig = d.loc[d["trigger"], ["Code", "Date"]].copy()
    trig["Date"] = pd.to_datetime(trig["Date"])
    if trig.empty or ev.empty:
        return d
    m = trig.merge(ev, on="Code", how="inner")
    m = m[(m["disclosed_date"] >= m["Date"] - pd.Timedelta(days=before)) & (m["disclosed_date"] <= m["Date"] + pd.Timedelta(days=after))]
    if m.empty:
        return d
    pri = {"T2": 0, "T3": 1, "T1": 2, "T4": 3}
    m["pri"] = m["etype"].map(pri)
    m = m.sort_values(["Code", "Date", "pri"]).drop_duplicates(["Code", "Date"])
    m["hf"] = m["category"].isin(HARD_FAIL_CATS) | m["hard_exclude"].fillna(False).astype(bool)
    key = d[["Code"]].copy()
    key["Date"] = pd.to_datetime(d["Date"])
    mm = key.merge(m[["Code", "Date", "etype", "category", "hf"]], on=["Code", "Date"], how="left")
    # 触发日的归因向后传播到有效期内
    for col, src in (("attr_type", "etype"), ("attr_category", "category"), ("hard_fail_event", "hf")):
        vals = mm[src]
        tmp = pd.Series(np.where(d["trigger"].values, vals.values, None), index=d.index, dtype=object)
        filled = tmp.groupby(d["Code"]).ffill()
        if col == "hard_fail_event":
            d[col] = filled.fillna(False).astype(bool).values
        elif col == "attr_type":
            d[col] = np.where(filled.notna(), filled, d[col])
        else:
            d[col] = filled.fillna("").values
    return d
