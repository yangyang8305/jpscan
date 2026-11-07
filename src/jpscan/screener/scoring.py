"""硬排除、软标记与证据判定的编排（规划文档 4.8、4.11 节）。

评分方式已改：不再用六维加权合成。入池由 screener/evidence.py 按注册表的证据家族和篮子要求判定，
排序键是「命中家族数 + anchor 家族分位」，理由由 screener/reasons.py 生成。

score_value 等列保留给报告和回测展示：它们只是家族内指标的同业分位平均（0–100），不影响入池和排序。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import pget
from ..events.oversold import cause_wait, intact_test
from ..features.registry import load_registry
from .evidence import evaluate
from .reasons import attach_reasons

# 旧列名 → 家族，只用于展示
DISPLAY_SCORE_FAMILY = {"score_value": "relative_value", "score_quality": "business_quality", "score_growth": "guidance_change", "score_oversold": "price_dislocation"}


def hard_filters(df: pd.DataFrame, params: dict, ext: dict | None = None) -> pd.DataFrame:
    """X 码：命中即移出。文本类（X03–X06）依赖 TDnet，没有数据时不触发，由 reasons 列入未检查，不算通过。"""
    liq = params.get("liquidity", {})
    n = len(df)
    codes: list[list[str]] = [[] for _ in range(n)]

    def add(mask, code: str) -> None:
        arr = pd.Series(mask, index=df.index).fillna(False).to_numpy(dtype=bool)
        for i in np.flatnonzero(arr):
            codes[i].append(code)

    add(~(df["is_common_stock"].fillna(False).astype(bool) & df["in_tse_main"].fillna(False).astype(bool)), "X01")
    ext = ext or {}
    sup = ext.get("supervision")
    if sup:
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
    if imp:
        add(df["Code"].isin(set(imp)), "X10")
    if "catalyst_tob" in df.columns:
        age = df["last_event_age"] if "last_event_age" in df.columns else pd.Series(999, index=df.index)
        add(df["catalyst_tob"].fillna(False).astype(bool) & (age <= 60), "X11")
    df["exclusion_codes"] = [",".join(sorted(set(c))) for c in codes]
    df["excluded"] = df["exclusion_codes"].str.len() > 0
    return df


def soft_flags(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    sx = params.get("exclusions", {}).get("soft", {})
    flags: list[list[str]] = [[] for _ in range(len(df))]

    def add(mask, code: str) -> None:
        arr = pd.Series(mask, index=df.index).fillna(False).to_numpy(dtype=bool)
        for i in np.flatnonzero(arr):
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
    if "retro_restate" in df.columns:
        add(df["retro_restate"].fillna(False).astype(bool), "RETRO")
    df["soft_flags"] = [",".join(f) for f in flags]
    return df


def apply_news(df: pd.DataFrame) -> pd.DataFrame:
    """披露事件分只展示。数据不覆盖当日时是中性分 + unknown，不是「没有坏消息」。"""
    if "news_score" not in df.columns:
        df["news_score"] = np.nan
    if "news_status" not in df.columns:
        df["news_status"] = "unknown"
    df["news_status"] = df["news_status"].fillna("no_events" if df["news_status"].notna().any() else "unknown")
    df["score_news"] = df["news_score"].fillna(50.0).clip(0, 100)
    return df


def display_scores(df: pd.DataFrame) -> pd.DataFrame:
    """展示用 0–100：家族内已知指标的同业分位平均。没有任何分位时为空。"""
    fam_json = df["family_status"].map(json.loads)
    for col, fam in DISPLAY_SCORE_FAMILY.items():
        vals = []
        for rec in fam_json:
            pcts = [m["pct"] for m in rec.get(fam, {}).get("metrics", {}).values() if m.get("pct") is not None]
            vals.append((1 - float(np.mean(pcts))) * 100 if pcts else np.nan)
        df[col] = vals
    df["score_trend"] = np.nan      # 趋势不再计分（registry trend_ma_52w 等只展示）
    return df


def legacy_flags(df: pd.DataFrame, reg: dict) -> pd.DataFrame:
    """旧的候选标志列，报告和 AI 模块还在读；含义改为对应家族的判定。"""
    rv = df["family_relative_value"]
    df["value_candidate"] = rv.eq("hit")
    df["growth_candidate"] = df["family_guidance_change"].eq("hit")
    df["quality_gate"] = df["family_business_quality"].ne("fail") & ~df["excluded"]
    df["oversold_candidate"] = df["family_price_dislocation"].eq("hit") & df.get("intact_status", pd.Series("unknown", index=df.index)).eq("pass")
    df["intact_ok"] = df.get("intact_status", pd.Series("unknown", index=df.index)).eq("pass")
    trap_cols = [f"gate_status_{g}" for g in ("gate_one_off", "gate_div_cover", "gate_distress", "gate_sector_equity") if f"gate_status_{g}" in df.columns]
    df["value_trap"] = (df[trap_cols] == 1).any(axis=1) if trap_cols else False
    df["growth_downgrade"] = df.get("gate_status_gate_miss_then_bullish", pd.Series(np.nan, index=df.index)).eq(1)
    red = df["negative_equity"].fillna(False).astype(bool)
    for col in ("flag_improper_accounting", "flag_going_concern", "flag_ms_warrant", "flag_earnings_release_delay", "flag_supervision", "flag_delisting"):
        if col in df.columns:
            red = red | df[col].fillna(False).astype(bool)
    df["red_flag"] = red
    hits = []
    for rec in df["family_status"].map(json.loads):
        ms = rec.get("relative_value", {}).get("metrics", {})
        hits.append(sum(1 for m in ms.values() if m.get("hit") == 1))
    df["value_core_hits"] = hits
    return df


def add_peer_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """旧的 pct_* 列（越小越便宜/越好），由 family_status 还原。"""
    alias = {"pct_pbr": ("relative_value", "val_pbr"), "pct_per_fwd": ("relative_value", "val_fwd_ep"), "pct_ocf_yield": ("relative_value", "val_ocf_yield"), "pct_div_yield_fwd": ("shareholder_yield", "yield_dividend")}
    recs = df["family_status"].map(json.loads)
    for col, (fam, m) in alias.items():
        df[col] = [r.get(fam, {}).get("metrics", {}).get(m, {}).get("pct") for r in recs]
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def run_scoring(feat_day: pd.DataFrame, params: dict, ext: dict | None = None, reasons: bool = True) -> pd.DataFrame:
    df = feat_day.copy().reset_index(drop=True)
    reg = load_registry(params)
    df = hard_filters(df, params, ext)
    df = soft_flags(df, params)
    df = apply_news(df)
    if "intact_status" not in df.columns or df["intact_status"].isna().all():
        df = df.drop(columns=[c for c in ("intact_status", "intact_known", "intact_pass", "intact_revision", "intact_dividend", "intact_remaining", "intact_distress") if c in df.columns])
        df = pd.concat([df, intact_test(df)], axis=1)
    if "cause_wait" not in df.columns or df["cause_wait"].isna().all():
        df["cause_wait"] = cause_wait(df)
    df = evaluate(df, params, reg)
    df = display_scores(df)
    df = legacy_flags(df, reg)
    df = add_peer_aliases(df)
    if reasons:
        df = attach_reasons(df, params, reg)
    return df
