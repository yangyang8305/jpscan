"""筛选运行：读取某日特征 + 事件分 + 错杀结果 + 外部名单 → 证据判定与理由 → mart.scores_daily 与候选清单。

外部名单一律按当时的状态取，避免前视（registry gate_regulatory_red_flags）。
"""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from ..config import pget
from ..events.news_score import coverage_status
from .scoring import run_scoring

log = logging.getLogger("jpscan.screener")

OUTPUT_COLS = [
    "Code", "Date", "CoName", "Mkt", "MktNm", "S33", "S33Nm", "S17Nm", "ScaleCat", "C", "mcap", "adv20_turnover", "in_core",
    # 证据
    "family_pass_count", "family_status", "family_coverage", "bucket_check", "reasons", "counter_reasons", "unknowns", "selection_summary_zh", "exclusion_reason_zh",
    "family_relative_value", "family_shareholder_yield", "family_business_quality", "family_guidance_change", "family_capital_action", "family_price_dislocation",
    "max_family_share", "dominant_family", "concentration_flag", "downgraded",
    # 篮子
    "bucket", "bucket_rank", "composite_score", "buckets", "resonance", "in_top_n",
    "composite_B1", "composite_B2", "composite_B3", "entry_B1", "entry_B2", "entry_B3", "in_B1", "in_B2", "in_B3", "rank_B1", "rank_B2", "rank_B3",
    "loo_removed_B1", "loo_removed_B2", "loo_removed_B3", "loo_worst_rank_B1", "loo_worst_rank_B2", "loo_worst_rank_B3", "loo_worst_metric_B1", "loo_worst_metric_B2", "loo_worst_metric_B3",
    # 注册表指标
    "val_pbr", "val_fwd_ep", "ep_ttm", "val_ocf_yield", "yield_dividend", "qual_fscore7", "fscore_n", "qual_op_profitability", "guid_forecast_innovation", "guid_forecast_innovation_pct",
    "guid_innovation_basis", "guid_revision_event", "guid_revision_event_pct", "evt_dividend_change", "rev_1m", "rev_ret_21d", "rev_div_add_back",
    "gate_status_gate_one_off", "gate_status_gate_div_cover", "gate_status_gate_distress", "gate_status_gate_sector_equity", "gate_status_gate_miss_then_bullish",
    "gate_status_guid_remaining_growth", "gate_status_gate_inorganic", "gate_status_gate_dilution", "gate_status_gate_disclosure_delay", "gate_status_gate_guidance_absent",
    "gate_status_gate_zombie", "gate_status_gate_debt_coverage", "gate_status_gate_peak_earnings", "gate_status_gate_tangible_book", "gate_one_off_loss",
    "intact_status", "intact_known", "intact_pass", "cause_wait", "attr_type", "attr_category", "cause_proxy_date",
    # 展示
    "score_value", "score_oversold", "score_growth", "score_quality", "score_trend", "score_news", "news_status",
    "pbr", "per_fwd", "per_ttm", "div_yield_fwd", "cash_to_mcap", "fcf_yield", "ev_ebitda", "net_cash_ratio", "net_debt_ebitda", "interest_coverage", "roe_avg", "equity_ratio", "accrual_avg",
    "rev_growth_1y", "op_growth_1y", "rev_cagr_3y", "op_cagr_3y", "growth_horizon", "fy_irregular", "f_op_growth", "progress_rate", "progress_dev", "n_up_revisions",
    "n_down_revisions", "last_revision_chg", "drawdown_60d", "rs_63d", "stage", "rvol60", "atr14", "atr_pct", "beta", "bps", "eps_fwd_adj",
    "pct_pbr", "pct_per_fwd", "pct_ocf_yield", "pct_div_yield_fwd", "self_pct_pbr", "self_pct_ep", "self_hist_days", "self_pct_low_conf",
    "value_candidate", "value_core_hits", "value_trap", "growth_candidate", "growth_downgrade", "quality_gate", "red_flag", "oversold_candidate", "intact_ok",
    "trigger", "trigger_type", "trigger_date", "stabilized", "excluded", "exclusion_codes", "soft_flags", "supervision_unknown", "stale_financials", "has_forecast", "effective_date",
    "days_to_next_earnings", "next_earnings_date", "news_score", "last_event_cat", "last_event_age", "jpx_capital_cost_disclosed", "mkt_source", "metric_version",
]


def load_day_inputs(store, d: date) -> dict[str, pd.DataFrame]:
    part = d.strftime("%Y-%m")
    feat = store.read("mart", "features_daily", where=f"Date = DATE '{d.isoformat()}'", partitions=[part])
    out = {"features": feat}
    for tbl in ("oversold_daily", "event_scores_daily"):
        out[tbl] = store.read("mart", tbl, where=f"Date = DATE '{d.isoformat()}'", partitions=[part]) if store.exists("mart", tbl) else pd.DataFrame()
    return out


def supervision_as_of(sup: pd.DataFrame, delisted: pd.DataFrame | None, d: date, unknown_days: int = 90) -> tuple[set[str], set[str]]:
    """当时是否在监理、整理名单：(确定在册, 无法确定)。

    designated_date > d 的不算。有解除日的按解除日。is_current 表示今天还在册，且指定早于 d，所以 d 时也在册。
    解除日空且今天已不在册的历史行（占绝大多数）：整理銘柄看退市日；其余只有指定日距 d 很近时算「无法确定」，久远的不算在册。
    """
    if sup.empty:
        return set(), set()
    s = sup.copy()
    s["designated_date"] = pd.to_datetime(s["designated_date"], errors="coerce")
    s["released_date"] = pd.to_datetime(s["released_date"], errors="coerce")
    ts = pd.Timestamp(d)
    s = s[s["designated_date"].notna() & (s["designated_date"] <= ts)]
    released_later = s["released_date"] > ts
    current = s["is_current"].astype(object).map(lambda v: str(v).lower() == "true")
    open_hist = s["released_date"].isna() & ~current
    sure = set(s.loc[released_later | (s["released_date"].isna() & current), "Code"].astype(str))
    if delisted is not None and not delisted.empty:
        dl = delisted.assign(delist_date=pd.to_datetime(delisted["delist_date"], errors="coerce"))
        later = set(dl.loc[dl["delist_date"] > ts, "Code"].astype(str))
        kind = s["kind"].astype(str).str.contains("整理", na=False)
        sure |= set(s.loc[open_hist & kind, "Code"].astype(str)) & later
    recent = (ts - s["designated_date"]).dt.days <= unknown_days
    unknown = set(s.loc[open_hist & recent, "Code"].astype(str)) - sure
    return sure, unknown


def list_as_of(table: pd.DataFrame, d: date, date_col: str, max_age_days: int = 40) -> set[str]:
    """只有当前版本的名单（如改善期间）：只在基准日距离发布日不超过 max_age_days 时使用。回看更早的日子就是前视。"""
    if table.empty:
        return set()
    t = table.assign(_d=pd.to_datetime(table[date_col], errors="coerce"))
    ts = pd.Timestamp(d)
    t = t[t["_d"] <= ts]
    if t.empty:
        return set()
    latest = t["_d"].max()
    if (ts - latest).days > max_age_days:
        return set()
    return set(t.loc[t["_d"] == latest, "Code"].astype(str))


def external_lists(store, d: date, params: dict | None = None) -> dict[str, object]:
    ext: dict[str, object] = {"unchecked": []}
    params = params or {}
    if store.exists("staging", "jpx_supervision"):
        dl = store.read("staging", "jpx_delisted") if store.exists("staging", "jpx_delisted") else None
        sure, unknown = supervision_as_of(store.read("staging", "jpx_supervision"), dl, d, int(pget(params, "evidence.supervision_unknown_days", 90)))
        ext["supervision"], ext["supervision_unknown"] = sure, unknown
    else:
        ext["unchecked"].append("jpx_supervision")
    if store.exists("staging", "jpx_improvement_period"):
        imp = list_as_of(store.read("staging", "jpx_improvement_period"), d, "published_date", int(pget(params, "evidence.current_list_max_age_days", 40)))
        ext["improvement_period"] = imp
        if not imp:
            ext["unchecked"].append("jpx_improvement_period")
    if store.exists("staging", "jpx_capital_cost_disclosure"):
        cc = store.read("staging", "jpx_capital_cost_disclosure", columns=["as_of_month", "Code", "disclosure_status"])
        cc = cc[cc["as_of_month"].astype(str) <= d.strftime("%Y-%m")]
        if not cc.empty:
            latest = cc.sort_values("as_of_month").drop_duplicates("Code", keep="last")
            ext["capital_cost"] = set(latest.loc[latest["disclosure_status"].astype(str).str.contains("開示済", na=False), "Code"].astype(str))
    return ext


def attach_calendar(df: pd.DataFrame, store, d: date) -> pd.DataFrame:
    if not store.exists("staging", "earnings_calendar"):
        return df
    cal = store.read("staging", "earnings_calendar")
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
            dup = [c for c in x.columns if c in df.columns and c != "Code"]
            df = df.merge(x.drop(columns=dup), on="Code", how="left")
    disc = store.read("staging", "disclosures", columns=["disclosed_date"]) if store.exists("staging", "disclosures") else None
    status = coverage_status(disc, d)
    if "news_status" not in df.columns:
        df["news_status"] = status
    else:
        df["news_status"] = df["news_status"].fillna("no_events" if status != "unknown" else "unknown")
    if status != "unknown" and "cause_proxy_earnings" in df.columns:
        # 公告数据覆盖当日时，下跌原因才算检查过；否则 rev_cause_class 留空，理由里列为未检查
        df["rev_cause_class"] = df["cause_proxy_earnings"].fillna(False).astype(float)
    ext = external_lists(store, d, params)
    if "capital_cost" in ext:
        df["jpx_capital_cost_disclosed"] = df["Code"].isin(ext["capital_cost"])
    df["supervision_unknown"] = df["Code"].isin(ext.get("supervision_unknown", set()))
    df = attach_calendar(df, store, d)
    scored = run_scoring(df, params, ext)
    scored["Date"] = pd.to_datetime(scored["Date"]).dt.date
    scored["run_id"] = run_id
    scored["param_version"] = str(params.get("version", ""))
    cols = [c for c in OUTPUT_COLS if c in scored.columns] + ["run_id", "param_version"]
    out = scored[cols].copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.date
    if write:
        _write_scores(store, out, d)
    return out


def _write_scores(store, out: pd.DataFrame, d: date) -> None:
    """同一分区内替换这一天的行。旧行的日期列统一成 date，否则新旧混合写不进 parquet。"""
    part = d.strftime("%Y-%m")
    if part in store.partitions("mart", "scores_daily"):
        old = store.read("mart", "scores_daily", partitions=[part])
        old = old[pd.to_datetime(old["Date"]).dt.date != d]
        for c in old.columns:
            if pd.api.types.is_datetime64_any_dtype(old[c]):
                old[c] = old[c].dt.date
        out = pd.concat([old, out], ignore_index=True) if not old.empty else out
    store.write("mart", "scores_daily", out, part)


def candidate_table(scores: pd.DataFrame, params: dict) -> pd.DataFrame:
    top_n = int(pget(params, "bucket.top_n", 40))
    rows = []
    for bk in ("B1", "B2", "B3"):
        sub = scores[scores[f"in_{bk}"] & (scores[f"rank_{bk}"] <= top_n)].sort_values(f"rank_{bk}")
        sub = sub.assign(list_bucket=bk, list_rank=sub[f"rank_{bk}"], list_score=sub[f"composite_{bk}"])
        rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
