"""六维评分与硬/软过滤（规划文档 4.2–4.8 节）。全部阈值来自 params（附录 C）。

输入：某日的特征行（features_daily）+ 可选 oversold_daily、event_scores_daily、外部名单（监理/整理、改善期间、資本コスト开示）。
输出：每股六维分、排除码、软标记、规则命中、候选标志。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import pget


def _pct_score(df: pd.DataFrame, metric: str, higher_better: bool) -> pd.Series:
    p = df.get(f"pct_{metric}")
    if p is None:
        return pd.Series(np.nan, index=df.index)
    return (p if higher_better else 1 - p) * 100.0


def _nanmean(cols: list[pd.Series]) -> tuple[pd.Series, pd.Series]:
    m = pd.concat(cols, axis=1)
    return m.mean(axis=1, skipna=True), m.notna().mean(axis=1)


def hard_filters(df: pd.DataFrame, params: dict, ext: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """返回 exclusion_codes（列表字符串）与 excluded（bool）。"""
    liq = params.get("liquidity", {})
    codes: list[list[str]] = [[] for _ in range(len(df))]
    def add(mask: pd.Series, code: str) -> None:
        for i in np.flatnonzero(mask.fillna(False).to_numpy(dtype=bool)):
            codes[i].append(code)
    add(~(df["is_common_stock"] & df["in_tse_main"]), "X01")
    ext = ext or {}
    sup = ext.get("supervision")
    if sup is not None and len(sup):
        add(df["Code"].isin(set(sup)), "X02")
    for flag, code in (("flag_going_concern", "X03"), ("flag_improper_accounting", "X04"), ("flag_earnings_release_delay", "X05"),
                       ("flag_ms_warrant", "X06"), ("flag_supervision", "X02"), ("flag_delisting", "X02")):
        if flag in df.columns:
            add(df[flag].fillna(False).astype(bool), code)
    add(df["negative_equity"].fillna(False).astype(bool), "X07")
    watch_to = float(pget(liq, "watch.min_adv20_turnover_jpy", 1e7))
    watch_mc = float(pget(liq, "watch.min_mcap_jpy", 2e9))
    add((df["adv20_turnover"] < watch_to) | (df["mcap"] < watch_mc) | df["adv20_turnover"].isna(), "X08")
    add(df["effective_date"].isna() | (df["n_fy"].fillna(0) < 1), "X09")
    imp = ext.get("improvement_period")
    if imp is not None and len(imp):
        add(df["Code"].isin(set(imp)), "X10")
    if "catalyst_tob" in df.columns:
        add(df["catalyst_tob"].fillna(False).astype(bool) & (df.get("last_event_age", pd.Series(999, index=df.index)) <= 60), "X11")
    df["exclusion_codes"] = [",".join(sorted(set(c))) for c in codes]
    df["excluded"] = df["exclusion_codes"].str.len() > 0
    return df


def soft_flags(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    sx = params.get("exclusions", {}).get("soft", {})
    flags: list[list[str]] = [[] for _ in range(len(df))]
    def add(mask: pd.Series, code: str) -> None:
        for i in np.flatnonzero(mask.fillna(False).to_numpy(dtype=bool)):
            flags[i].append(code)
    s01 = sx.get("S01_low_float_or_parent_control", {})
    if "float_ratio" in df.columns:
        add(df["float_ratio"] < float(s01.get("float_ratio_max", 0.15)), "S01")
    if "days_to_next_earnings" in df.columns:
        add(df["days_to_next_earnings"].between(0, int(sx.get("S03_pre_earnings_window_days", 5))), "S03")
    if "recent_dilution" in df.columns:
        add(df["recent_dilution"].fillna(False).astype(bool), "S04")
    if "n_neg_30d" in df.columns:
        add(df["n_neg_30d"] >= int(pget(sx, "S05_negative_events.count_min", 2)), "S05")
    add(df["is_financial"].fillna(False).astype(bool), "S06")
    add(df["C"] < float(sx.get("S07_low_price_jpy", 100)), "S07")
    add(df["stale_financials"].fillna(False).astype(bool), "STALE")
    add(~df["has_forecast"].fillna(False).astype(bool), "NOFCST")
    df["soft_flags"] = [",".join(f) for f in flags]
    return df


def score_value(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    v = params.get("value", {})
    core = v.get("core_metrics", ["pbr", "per_fwd", "ev_ebitda", "fcf_yield"])
    higher = {"pbr": False, "per_fwd": False, "ev_ebitda": False, "fcf_yield": True, "div_yield_fwd": True, "cash_to_mcap": True,
              "ey_spread": True}
    core_scores = [_pct_score(df, m, higher.get(m, False)) for m in core]
    core_mean, core_cov = _nanmean(core_scores)
    support = [_pct_score(df, "div_yield_fwd", True), _pct_score(df, "cash_to_mcap", True), (1 - df["self_pct_pbr"]) * 100]
    sup_mean, sup_cov = _nanmean(support)
    score = core_mean * float(pget(v, "weights.core_block", 0.7)) + sup_mean.fillna(core_mean) * float(pget(v, "weights.support_block", 0.3))
    # 候选：核心指标 ≥ min_core_hits 项处于行业 ≤ core_metric_pct，且自身历史百分位 ≤ self_pct（缺失则忽略）
    th = float(v.get("core_metric_pct", 0.30))
    hits = sum((df.get(f"pct_{m}", pd.Series(np.nan, index=df.index)) <= th).fillna(False).astype(int) for m in core)
    self_ok = (df["self_pct_pbr"] <= float(v.get("self_pct", 0.40))) | df["self_pct_pbr"].isna()
    df["value_core_hits"] = hits
    df["value_candidate"] = (hits >= int(v.get("min_core_hits", 2))) & self_ok
    # 强低估标签
    s = v.get("strong", {})
    df["value_strong"] = ((df["pbr"] < float(s.get("pbr_max", 1.0))) | (df["per_fwd"] < float(s.get("per_fwd_max", 10))) |
                          (df["fcf_yield"] > float(s.get("fcf_yield_min", 0.08))) | (df["cash_to_mcap"] > float(s.get("net_cash_ratio_min", 0.5))))
    # 价值陷阱
    t = v.get("trap", {})
    rev_g = df["rev_cagr_3y"].where(df["rev_cagr_3y"].notna(), df["rev_growth_1y"])
    trap = ((df["roe_avg"] < float(t.get("roe_3y_max", 0.05))) | ((df["pbr"] < 1.0) & (df["per_fwd"] > 15.0)) |
            (rev_g < float(t.get("rev_cagr_3y_max", 0.0))) |
            ((df["ocf_years_available"] - df["ocf_positive_years"]) >= int(t.get("negative_ocf_years_of_3", 2))) |
            (df["dividend_cut"].fillna(False).astype(bool)) |
            ((df["equity_ratio"] < float(t.get("equity_ratio_min_nonfin", 0.2))) & ~df["is_financial"]))
    df["value_trap"] = trap.fillna(False)
    score = score * np.where(df["value_trap"], float(t.get("multiplier", 0.6)), 1.0)
    # 催化剂加分
    c = v.get("catalyst", {})
    bonus = pd.Series(0.0, index=df.index)
    for col, key in (("catalyst_capital_cost_disclosure", "tse_capital_cost_disclosure_with_targets"), ("catalyst_buyback", "buyback_announced_12m"),
                     ("catalyst_activist", "activist_holding_5pct"), ("catalyst_payout_policy", "cross_holding_reduction_or_payout_policy")):
        if col in df.columns:
            bonus += df[col].fillna(False).astype(float) * float(c.get(key, 0))
    if "jpx_capital_cost_disclosed" in df.columns:
        bonus += df["jpx_capital_cost_disclosed"].fillna(False).astype(float) * float(c.get("tse_capital_cost_disclosure_with_targets", 10)) * 0.5
    bonus = bonus.clip(upper=float(c.get("cap", 15)))
    df["value_catalyst_bonus"] = bonus
    df["score_value"] = (score + bonus).clip(0, 100)
    df["value_coverage"] = core_cov
    return df


def score_growth(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    g = params.get("growth", {})
    rev_g = df["rev_cagr_3y"].where(df["rev_cagr_3y"].notna(), df["rev_growth_1y"])
    op_g = df["op_cagr_3y"].where(df["op_cagr_3y"].notna(), df["op_growth_1y"])
    realized = [_pct_score(df, "rev_cagr_3y", True).where(df["rev_cagr_3y"].notna(), _pct_score(df, "rev_growth_1y", True)),
                _pct_score(df, "op_cagr_3y", True).where(df["op_cagr_3y"].notna(), _pct_score(df, "op_growth_1y", True)),
                (df["inc_streak"].fillna(0) / 2 * 100).clip(0, 100), _pct_score(df, "op_margin_fy0", True), _pct_score(df, "op_margin_chg", True)]
    expected = [_pct_score(df, "f_op_growth", True), _pct_score(df, "progress_dev", True),
                (df["n_up_revisions"].fillna(0).clip(0, 2) / 2 * 100)]
    quality = [_pct_score(df, "ocf_to_op", True)]
    r, rc = _nanmean(realized)
    e, ec = _nanmean(expected)
    q, qc = _nanmean(quality)
    w = g.get("weights", {"realized": 0.4, "expected": 0.4, "quality": 0.2})
    score = r.fillna(50) * w["realized"] + e.fillna(50) * w["expected"] + q.fillna(50) * w["quality"]
    # 降级
    dg = g.get("downgrade", {})
    margin_decl = (df["op_margin_chg"] < 0) & (df["op_margin_fy1"] < (df["op_fy2"] / df["sales_fy2"]))
    strong_after_miss = (df["actual_vs_initial"] < 1.0) & (df["f_op_growth"] > 2 * df.groupby("S33")["f_op_growth"].transform("median"))
    df["growth_downgrade"] = (margin_decl | strong_after_miss).fillna(False)
    score = score * np.where(df["growth_downgrade"], 0.8, 1.0)
    cand_real = (rev_g >= float(g.get("rev_cagr_3y_min", 0.08))) & (op_g >= float(g.get("op_cagr_3y_min", 0.10))) & \
                (df["f_op_growth"] >= float(g.get("op_growth_fwd_min", 0.10))) & (df["ocf_to_op"] >= float(g.get("ocf_to_op_3y_min", 0.7)))
    cand_exp = (df["progress_dev"] >= float(g.get("progress_dev_min", 0.05))) & (df["n_up_revisions"] >= int(g.get("upward_revisions_3y_min", 2)))
    df["growth_candidate"] = (cand_real.fillna(False) | cand_exp.fillna(False)) & ~df["is_financial"].fillna(False)
    df["score_growth"] = score.clip(0, 100)
    df["growth_coverage"] = (rc + ec + qc) / 3
    return df


def score_quality(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    q = params.get("quality", {})
    prof = [_pct_score(df, "roe_avg", True), _pct_score(df, "op_margin_fy0", True)]
    bs = [_pct_score(df, "equity_ratio", True)]
    eq = [_pct_score(df, "accrual_avg", False), (df["ocf_positive_years"] / df["ocf_years_available"].replace(0, np.nan) * 100)]
    pay = [(~df["dividend_cut"].fillna(False).astype(bool)).astype(float) * 100]
    for col in ("catalyst_buyback", "catalyst_payout_policy", "catalyst_capital_cost_disclosure"):
        if col in df.columns:
            pay.append(df[col].fillna(False).astype(float) * 100)
    p, _ = _nanmean(prof)
    b, _ = _nanmean(bs)
    e, _ = _nanmean(eq)
    y, _ = _nanmean(pay)
    w = q.get("weights", {"profitability": 0.4, "balance_sheet": 0.3, "earnings_quality": 0.2, "payout_governance": 0.1})
    score = p.fillna(50) * w["profitability"] + b.fillna(50) * w["balance_sheet"] + e.fillna(50) * w["earnings_quality"] + y.fillna(50) * w["payout_governance"]
    red = df["negative_equity"].fillna(False).astype(bool)
    for col in ("flag_improper_accounting", "flag_going_concern", "flag_ms_warrant", "flag_earnings_release_delay", "flag_supervision", "flag_delisting"):
        if col in df.columns:
            red = red | df[col].fillna(False).astype(bool)
    df["red_flag"] = red
    gate = q.get("gate", {})
    eq_ok = (df["equity_ratio"] >= float(gate.get("equity_ratio_min_nonfin", 0.30))) | df["is_financial"].fillna(False)
    ocf_ok = (df["ocf_positive_years"] >= np.minimum(int(gate.get("positive_ocf_years_min", 2)), df["ocf_years_available"])) & (df["ocf_years_available"] > 0)
    df["quality_gate"] = eq_ok.fillna(False) & ocf_ok.fillna(False) & ~red
    df["score_quality"] = np.where(red, 0.0, score.clip(0, 100))
    return df


def score_trend(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    t = params.get("trend", {})
    blocks = t.get("score_blocks", {"stage": 40, "relative_strength": 30, "volume_supply": 30})
    stage_pts = df["stage"].map({"uptrend": 1.0, "base": 0.6, "transition": 0.4, "top": 0.25, "downtrend": 0.0, "unknown": 0.4}).fillna(0.4)
    rs = _pct_score(df, "rs_63d", True).fillna(50) / 100
    vol = _pct_score(df, "vol_ratio_20_120", True).fillna(50) / 100 if "pct_vol_ratio_20_120" in df.columns else pd.Series(0.5, index=df.index)
    supply = pd.Series(0.5, index=df.index)  # 信用倍率/空売り（Standard 以上）缺失时中性
    if "margin_ratio" in df.columns:
        supply = np.where(df["margin_ratio"] > float(t.get("margin_ratio_warn", 10)), 0.2, 0.6)
    df["score_trend"] = (stage_pts * blocks["stage"] + rs * blocks["relative_strength"] + (vol * 0.5 + supply * 0.5) * blocks["volume_supply"]).clip(0, 100)
    return df


def apply_news(df: pd.DataFrame) -> pd.DataFrame:
    if "news_score" not in df.columns:
        df["news_score"] = 50.0
    df["score_news"] = df["news_score"].fillna(50.0).clip(0, 100)
    return df


def composite_and_buckets(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    b = params.get("bucket", {})
    dims = {"value": "score_value", "quality": "score_quality", "growth": "score_growth", "oversold": "score_oversold",
            "trend": "score_trend", "news": "score_news"}
    if "score_oversold" not in df.columns:
        df["score_oversold"] = 0.0
        df["oversold_candidate"] = False
        df["intact_ok"] = False
    per70 = (df.get("pct_per_fwd", pd.Series(np.nan, index=df.index)) <= 0.70) | df["per_fwd"].isna()
    entry = {
        "B1": df["value_candidate"] & df["quality_gate"],
        "B2": df["growth_candidate"] & df["quality_gate"] & per70,
        "B3": df["oversold_candidate"].fillna(False).astype(bool) & df["intact_ok"].fillna(False).astype(bool),
    }
    eligible = ~df["excluded"]
    for bk, wkey in (("B1", "B1_value_reversal"), ("B2", "B2_quality_growth"), ("B3", "B3_oversold_recovery")):
        w = b.get(wkey, {}).get("weights", {})
        comp = sum(df[col].fillna(0.0) * float(w.get(dim, 0.0)) for dim, col in dims.items())
        df[f"composite_{bk}"] = comp
        ok = entry[bk].fillna(False) & eligible
        df[f"in_{bk}"] = ok
        rank = df.loc[ok, f"composite_{bk}"].rank(ascending=False, method="first")
        df[f"rank_{bk}"] = np.nan
        df.loc[ok, f"rank_{bk}"] = rank
    top_n = int(b.get("top_n", 40))
    df["buckets"] = [",".join(bk for bk in ("B1", "B2", "B3") if row[f"in_{bk}"]) for _, row in df[["in_B1", "in_B2", "in_B3"]].iterrows()]
    df["resonance"] = df[["in_B1", "in_B2", "in_B3"]].sum(axis=1) >= 2
    best = []
    for _, row in df.iterrows():
        cands = [(bk, row[f"rank_{bk}"]) for bk in ("B1", "B2", "B3") if row[f"in_{bk}"] and not pd.isna(row[f"rank_{bk}"])]
        if not cands:
            best.append((None, np.nan, np.nan))
        else:
            bk, rk = min(cands, key=lambda x: x[1])
            best.append((bk, rk, row[f"composite_{bk}"]))
    df["bucket"] = [x[0] for x in best]
    df["bucket_rank"] = [x[1] for x in best]
    df["composite_score"] = [x[2] for x in best]
    df["in_top_n"] = df["bucket_rank"] <= top_n
    return df


def run_scoring(feat_day: pd.DataFrame, params: dict, ext: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    df = feat_day.copy().reset_index(drop=True)
    df = hard_filters(df, params, ext)
    df = soft_flags(df, params)
    df = score_value(df, params)
    df = score_growth(df, params)
    df = score_quality(df, params)
    df = score_trend(df, params)
    df = apply_news(df)
    df = composite_and_buckets(df, params)
    return df
