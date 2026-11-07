"""筛选运行：读取某日特征 + 事件分 + 错杀分 + 外部名单 → 评分 → mart.scores_daily 与候选清单。"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from ..config import pget
from .scoring import run_scoring

log = logging.getLogger("jpscan.screener")

OUTPUT_COLS = [
    "Code", "Date", "CoName", "Mkt", "S33", "S17", "ScaleCat", "C", "mcap", "adv20_turnover",
    "score_value", "score_oversold", "score_growth", "score_quality", "score_trend", "score_news",
    "bucket", "bucket_rank", "composite_score", "buckets", "resonance", "in_top_n",
    "composite_B1", "composite_B2", "composite_B3", "in_B1", "in_B2", "in_B3", "rank_B1", "rank_B2", "rank_B3",
    "pbr", "per_fwd", "per_ttm", "div_yield_fwd", "cash_to_mcap", "fcf_yield", "ey_spread", "roe_avg", "equity_ratio", "accrual_avg",
    "rev_growth_1y", "op_growth_1y", "rev_cagr_3y", "op_cagr_3y", "f_op_growth", "progress_rate", "progress_dev", "n_up_revisions",
    "n_down_revisions", "last_revision_chg", "drawdown_60d", "rs_63d", "stage", "rvol60", "atr_pct", "beta",
    "pct_pbr", "pct_per_fwd", "pct_fcf_yield", "pct_div_yield_fwd", "self_pct_pbr",
    "value_candidate", "value_core_hits", "value_strong", "value_trap", "value_catalyst_bonus", "growth_candidate", "growth_downgrade",
    "quality_gate", "red_flag", "oversold_candidate", "attr_type", "attr_category", "trigger_date", "intact_pass", "stabilized",
    "reaction_ratio", "excluded", "exclusion_codes", "soft_flags", "stale_financials", "has_forecast", "effective_date",
    "days_to_next_earnings", "next_earnings_date", "news_score", "last_event_cat", "last_event_age", "metric_version",
]


def load_day_inputs(store, d: date) -> dict[str, pd.DataFrame]:
    part = d.strftime("%Y-%m")
    feat = store.read("mart", "features_daily", where=f"Date = DATE '{d.isoformat()}'", partitions=[part])
    out = {"features": feat}
    for tbl in ("oversold_daily", "event_scores_daily"):
        try:
            out[tbl] = store.read("mart", tbl, where=f"Date = DATE '{d.isoformat()}'", partitions=[part])
        except Exception:  # noqa: BLE001
            out[tbl] = pd.DataFrame()
    return out


def external_lists(store, d: date) -> dict[str, object]:
    ext: dict[str, object] = {}
    try:
        sup = store.read("staging", "jpx_supervision")
        if not sup.empty:
            ext["supervision"] = set(sup["Code"].astype(str))
    except Exception:  # noqa: BLE001
        pass
    try:
        cc = store.read("staging", "jpx_capital_cost_disclosure")
        if not cc.empty:
            latest = cc.sort_values("as_of_month").drop_duplicates("Code", keep="last")
            ext["capital_cost"] = latest[latest["disclosure_status"].astype(str).str.contains("開示済", na=False)]
    except Exception:  # noqa: BLE001
        pass
    return ext


def attach_calendar(df: pd.DataFrame, store, d: date) -> pd.DataFrame:
    try:
        cal = store.read("staging", "earnings_calendar")
    except Exception:  # noqa: BLE001
        return df
    if cal.empty:
        return df
    cal["SchDate"] = pd.to_datetime(cal["SchDate"]).dt.date
    cal["PubDate"] = pd.to_datetime(cal["PubDate"]).dt.date
    cal = cal[(cal["PubDate"] <= d) & (cal["SchDate"] >= d)].sort_values("SchDate").drop_duplicates("Code")
    nxt = cal.set_index("Code")["SchDate"]
    df["next_earnings_date"] = df["Code"].map(nxt)
    df["days_to_next_earnings"] = (pd.to_datetime(df["next_earnings_date"]) - pd.Timestamp(d)).dt.days
    return df


def screen_day(store, params: dict, d: date, run_id: str, write: bool = True) -> pd.DataFrame:
    inp = load_day_inputs(store, d)
    feat = inp["features"]
    if feat.empty:
        log.warning("no features for %s", d)
        return pd.DataFrame()
    df = feat
    for tbl in ("oversold_daily", "event_scores_daily"):
        x = inp.get(tbl)
        if x is not None and not x.empty:
            df = df.merge(x.drop(columns=["Date"]), on="Code", how="left")
    ext = external_lists(store, d)
    if "capital_cost" in ext:
        df["jpx_capital_cost_disclosed"] = df["Code"].isin(set(ext["capital_cost"]["Code"].astype(str)))
    df = attach_calendar(df, store, d)
    scored = run_scoring(df, params, ext)
    scored["run_id"] = run_id
    scored["param_version"] = str(params.get("version", ""))
    cols = [c for c in OUTPUT_COLS if c in scored.columns] + ["run_id", "param_version"]
    out = scored[cols].copy()
    if write:
        store.write_merge("mart", "scores_daily", out, d.strftime("%Y-%m"), ["Code", "Date"])
    return out


def candidate_table(scores: pd.DataFrame, params: dict) -> pd.DataFrame:
    top_n = int(pget(params, "bucket.top_n", 40))
    rows = []
    for bk in ("B1", "B2", "B3"):
        sub = scores[scores[f"in_{bk}"] & (scores[f"rank_{bk}"] <= top_n)].sort_values(f"rank_{bk}")
        sub = sub.assign(list_bucket=bk, list_rank=sub[f"rank_{bk}"], list_score=sub[f"composite_{bk}"])
        rows.append(sub)
    cand = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return cand
