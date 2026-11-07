"""证据判定与筛选规则的测试。公司、行业分布和数字全部是合成的。"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from jpscan.config import load_params
from jpscan.events.news_score import build_event_scores, coverage_status
from jpscan.events.oversold import build_oversold, intact_test
from jpscan.features.panel import cum_after_at
from jpscan.features.registry import load_registry
from jpscan.screener.evidence import HIT, _Ctx, evaluate, evaluate_metrics, family_status, resolve_gates
from jpscan.screener.run import list_as_of, supervision_as_of
from jpscan.screener.scoring import hard_filters, run_scoring, soft_flags

PARAMS = load_params(__import__("pathlib").Path(__file__).resolve().parents[1] / "docs" / "params.v0.yaml")
REG = load_registry(PARAMS)
AS_OF = date(2026, 8, 14)
SECTORS = [("化学", "素材・化学"), ("機械", "機械"), ("小売業", "小売"), ("サービス業", "その他")]


def universe(n_per_sector: int = 40, seed: int = 7) -> pd.DataFrame:
    """每个行业 n 只「普通」公司：指标围绕中位分布，没有事件，所有家族都不命中。"""
    rng = np.random.default_rng(seed)
    rows = []
    for s33, s17 in SECTORS:
        for i in range(n_per_sector):
            # 分三组，各家族的前 30% 落在不同的公司上：默认没有任何公司同时命中两个家族
            g = i % 3
            rows.append({
                "Code": "", "Date": AS_OF, "CoName": f"合成{len(rows)}", "S33": s33, "S33Nm": s33, "S17Nm": s17, "Mkt": "0111", "ProdCat": "011",
                "is_common_stock": True, "in_tse_main": True, "in_core": True, "C": 1000.0, "mcap": 5e10 + i * 1e8, "adv20_turnover": 2e8, "effective_date": date(2026, 8, 1), "n_fy": 2,
                "negative_equity": False, "is_financial": False, "stale_financials": False, "has_forecast": True, "equity_ratio": 0.5, "f_op_growth": 0.02, "gate_miss_prior": 0.0,
                "val_pbr": rng.uniform(1.0, 1.2) if g == 0 else rng.uniform(1.5, 2.0), "val_fwd_ep": rng.uniform(0.07, 0.08) if g == 0 else rng.uniform(0.03, 0.05),
                "ep_ttm": rng.uniform(0.03, 0.08), "val_ocf_yield": rng.uniform(0.07, 0.08) if g == 0 else rng.uniform(0.03, 0.05),
                "yield_dividend": rng.uniform(0.03, 0.04) if g == 1 else rng.uniform(0.01, 0.02), "qual_fscore7": 4.0,
                "qual_op_profitability": rng.uniform(0.07, 0.08) if g == 2 else rng.uniform(0.03, 0.05), "evt_dividend_change": 0.0, "rev_1m": rng.uniform(-0.03, 0.03),
                "guid_forecast_innovation": np.nan, "guid_forecast_innovation_pct": np.nan, "guid_revision_event": np.nan, "guid_revision_event_pct": np.nan, "guid_revision_n_up": np.nan,
                "guid_revision_confirm": np.nan, "guid_fail": False, "guid_fail_low_conf": False, "evt_div_down": False, "guid_remaining_flag": 0.0, "gate_distress": 0.0, "gate_one_off": 0.0,
                "gate_div_cover": 0.0, "gate_inorganic": 0.0, "gate_disclosure_delay": 0.0, "gate_guidance_absent": 0.0, "gate_dilution_shares": 0.0, "cause_proxy_earnings": False,
                "cause_proxy_earnings_60": False,
            })
    df = pd.DataFrame(rows)
    df["Code"] = [f"{1000 + i}0" for i in range(len(df))]
    return df


def _params(**kw) -> dict:
    return {**PARAMS, "evidence": {**PARAMS["evidence"], **kw}}


def make_excellent(df: pd.DataFrame, i: int, metric: str) -> pd.DataFrame:
    """把第 i 只股票在 metric 上调到极好，其余不动。"""
    rule = next(r for f in REG["families"].values() for r in f["threshold"]["hit_rules"] if r["metric"] == metric)
    higher = REG["metrics_by_id"][metric].get("direction") != "lower_better"
    kind = rule["rule"]
    if kind in ("peer_best_pct", "core_best_pct"):
        df.loc[i, metric] = 100.0 if higher else (0.001 if metric != "rev_1m" else -0.5)
    elif kind == "group_best_pct":
        df.loc[i, [metric, f"{metric}_pct"]] = [0.5, 0.01]
    elif kind == "event_up_top_pct":
        df.loc[i, [metric, f"{metric}_pct", "guid_revision_n_up", "guid_revision_confirm"]] = [0.2, 0.01, 100.0, 1.0]
    elif kind == "min_value":
        df.loc[i, metric] = 7.0
    elif kind == "any_up_event":
        df.loc[i, metric] = 1.0
    return df


ALL_SCORE_METRICS = sorted({r["metric"] for f in REG["families"].values() for r in f["threshold"]["hit_rules"]})


@pytest.mark.parametrize("metric", ALL_SCORE_METRICS)
@pytest.mark.parametrize("min_families", [2, 3])
def test_excellent_on_one_metric_never_qualifies(metric, min_families):
    """任何一个指标再好，其他都普通，也进不了任何篮子。所有指标一视同仁。"""
    df = universe()
    for m in ("val_pbr", "val_fwd_ep", "ep_ttm", "val_ocf_yield", "yield_dividend", "qual_op_profitability", "rev_1m"):
        df.loc[5, m] = df.loc[df["S33Nm"] == df.loc[5, "S33Nm"], m].median()      # 其余指标都在同业中位
    df = make_excellent(df, 5, metric)
    out = run_scoring(df, _params(min_families=min_families), reasons=False)
    row = out.iloc[5]
    assert not (row["in_B1"] or row["in_B2"] or row["in_B3"]), metric
    assert row["family_pass_count"] <= 1


def test_unremarkable_universe_has_no_candidates():
    out = run_scoring(universe(), PARAMS, reasons=False)
    assert out[["in_B1", "in_B2", "in_B3"]].sum().sum() == 0
    assert (out["family_pass_count"] <= 2).all()    # 三组各一个家族，再加上随机的 1 个月跌幅


def _strong(df, i, extra=("yield_dividend", "evt_dividend_change")):
    for m in ("val_pbr", "val_fwd_ep", "val_ocf_yield", *extra):
        df = make_excellent(df, i, m)
    return df


def test_three_families_with_two_valuation_metrics_enter_b1():
    df = _strong(universe(), 3)
    out = run_scoring(df, PARAMS, reasons=False)
    row = out.iloc[3]
    assert row["in_B1"] and row["family_pass_count"] == 3
    assert row["family_relative_value"] == "hit" and row["bucket"] == "B1" and row["rank_B1"] == 1


def test_value_hits_are_direction_aware():
    """高收益率算好、低 PBR 算好；反过来不算。"""
    df = universe()
    df.loc[1, "val_ocf_yield"] = 100.0
    df.loc[2, "val_ocf_yield"] = -100.0
    df.loc[3, "val_pbr"] = 0.001
    df.loc[4, "val_pbr"] = 1000.0
    ctx = _Ctx(df, PARAMS, REG)
    gates = resolve_gates(df, ctx)
    m = evaluate_metrics(df, ctx, gates)["relative_value"]
    assert m["val_ocf_yield"]["hit"][1] == 1 and m["val_ocf_yield"]["hit"][2] == 0
    assert m["val_pbr"]["hit"][3] == 1 and m["val_pbr"]["hit"][4] == 0
    assert m["val_ocf_yield"]["pct"][1] < 0.05 and m["val_ocf_yield"]["pct"][2] > 0.95


def test_nan_is_unknown_never_pass():
    df = universe()
    df.loc[0, ["val_pbr", "val_fwd_ep", "ep_ttm", "val_ocf_yield"]] = np.nan
    df.loc[1, ["yield_dividend"]] = np.nan
    out = run_scoring(df, PARAMS, reasons=False)
    assert out.loc[0, "family_relative_value"] == "unknown"
    assert out.loc[1, "family_shareholder_yield"] == "unknown"
    assert out.loc[0, "family_relative_value"] != "hit"


def test_family_unknown_without_coverage_metric():
    """OP/TA 很好但 F-Score 未知：经营质量家族是未知，不是命中（registry coverage_metric）。"""
    df = universe()
    df["qual_fscore7"] = np.nan
    df = make_excellent(df, 2, "qual_op_profitability")
    out = run_scoring(df, PARAMS, reasons=False)
    assert out.loc[2, "family_business_quality"] == "unknown"


def test_low_fscore_is_family_fail():
    df = universe()
    df.loc[1, "qual_fscore7"] = 2.0
    out = run_scoring(df, PARAMS, reasons=False)
    assert out.loc[1, "family_business_quality"] == "fail"


def test_gate_one_off_blocks_earnings_yield_hit():
    df = universe()
    df = make_excellent(df, 1, "val_fwd_ep")
    df.loc[1, "gate_one_off"] = 1.0
    ctx = _Ctx(df, PARAMS, REG)
    m = evaluate_metrics(df, ctx, resolve_gates(df, ctx))["relative_value"]["val_fwd_ep"]
    assert m["hit"][1] == 0 and m["blocked"][1]


def test_ttm_fallback_is_low_confidence_and_compared_with_ttm_peers():
    df = universe()
    df.loc[1, "val_fwd_ep"] = np.nan
    df.loc[1, "ep_ttm"] = 1.0
    df.loc[1, "gate_one_off"] = 1.0     # TTM 回退不受一次性利润门影响
    ctx = _Ctx(df, PARAMS, REG)
    m = evaluate_metrics(df, ctx, resolve_gates(df, ctx))["relative_value"]["val_fwd_ep"]
    assert m["hit"][1] == 1 and m["low_conf"][1] and m["basis"][1] == "fallback"


def test_sector_override_drops_metric_and_market_override_inactivates_family():
    df = universe()
    df.loc[df["S33Nm"] == "機械", "S33Nm"] = "銀行業"
    df.loc[df["S33Nm"] == "化学", "Mkt"] = "0113"
    out = run_scoring(df, PARAMS, reasons=False)
    assert (out.loc[out["S33Nm"] == "銀行業", "family_business_quality"] == "inactive").all()
    assert (out.loc[out["Mkt"] == "0113", "family_shareholder_yield"] == "inactive").all()
    ctx = _Ctx(df, PARAMS, REG)
    m = evaluate_metrics(df, ctx, resolve_gates(df, ctx))["relative_value"]["val_ocf_yield"]
    assert np.isnan(m["hit"][(df["S33Nm"] == "銀行業").to_numpy()]).all()


def test_b1_not_enabled_in_growth_market():
    df = _strong(universe(), 3)
    df.loc[3, "Mkt"] = "0113"
    out = run_scoring(df, _params(min_families=2), reasons=False)
    assert not out.loc[3, "in_B1"]


def test_cyclical_cannot_rely_on_dividend_alone():
    df = universe()
    df = _strong(df, 0, extra=("yield_dividend",))    # 机器人：化学（周期行业）只有股息确认
    out = run_scoring(df, _params(min_families=2), reasons=False)
    assert out.loc[0, "S33Nm"] == "化学" and not out.loc[0, "in_B1"]
    df = universe()
    i = df.index[df["S33Nm"] == "小売業"][0]
    df = _strong(df, i, extra=("yield_dividend",))
    out = run_scoring(df, _params(min_families=2), reasons=False)
    assert out.loc[i, "in_B1"]


def test_guidance_fail_blocks_b1_and_b3():
    df = _strong(universe(), 3)
    df.loc[3, "guid_fail"] = True
    out = run_scoring(df, PARAMS, reasons=False)
    assert out.loc[3, "family_guidance_change"] == "fail" and not out.loc[3, "in_B1"]


def test_b3_needs_intact_and_cause_wait():
    df = universe()
    for m in ("rev_1m", "yield_dividend", "evt_dividend_change"):
        df = make_excellent(df, 3, m)
    out = run_scoring(df, PARAMS, reasons=False)
    assert out.loc[3, "in_B3"]
    d2 = df.copy()
    d2.loc[3, "cause_proxy_earnings"] = True
    assert not run_scoring(d2, PARAMS, reasons=False).loc[3, "in_B3"]
    d3 = df.copy()
    d3.loc[3, "evt_div_down"] = True
    assert not run_scoring(d3, PARAMS, reasons=False).loc[3, "in_B3"]
    d4 = df.copy()
    d4.loc[3, ["guid_remaining_flag", "gate_distress"]] = np.nan
    d4.loc[3, "has_forecast"] = False
    d4.loc[3, "evt_dividend_change"] = np.nan
    assert not run_scoring(d4, PARAMS, reasons=False).loc[3, "in_B3"]     # 完好检验一项都不知道，不能算完好


def _loo_universe():
    """很多股票靠 PBR + E/P + 指引 + 配当上调命中 4 个家族；X 多命中一个股息，但估值分位最弱。"""
    df = universe(n_per_sector=120)
    df["rev_1m"] = 0.0
    idx = np.concatenate([df.index[df["S33Nm"] == sec][:30] for sec, _ in SECTORS])
    df.loc[idx, "val_pbr"] = 0.2 + 0.0001 * np.arange(len(idx))
    df.loc[idx, "val_fwd_ep"] = 0.5
    df.loc[idx, ["yield_dividend", "qual_op_profitability"]] = [0.005, 0.01]
    df.loc[idx, "evt_dividend_change"] = 1.0
    df.loc[idx, ["guid_revision_event", "guid_revision_event_pct", "guid_revision_n_up", "guid_revision_confirm"]] = [0.05, 0.1, 100.0, 1.0]
    x = df.index[df["S33Nm"] == "小売業"][-1]
    df.loc[x, ["val_pbr", "val_fwd_ep", "val_ocf_yield", "yield_dividend", "qual_op_profitability", "evt_dividend_change"]] = [0.9, 0.0805, 0.03, 0.2, 0.01, 1.0]
    df.loc[x, ["guid_revision_event", "guid_revision_event_pct", "guid_revision_n_up", "guid_revision_confirm"]] = [0.05, 0.1, 100.0, 1.0]
    return df, x


def test_leave_one_out_removes_single_metric_dependence():
    df, x = _loo_universe()
    params = {**_params(), "bucket": {**PARAMS["bucket"], "top_n": 5}}
    out = run_scoring(df, params, reasons=False)
    assert out.loc[x, "entry_B1"], "X 符合入池规则"
    assert out.loc[x, "family_pass_count"] == 4     # 估值、股息、指引、资本行动
    assert out.loc[x, "loo_removed_B1"]                       # 基准排名靠前，去掉股息后掉到 10 名之外
    assert not out.loc[x, "in_B1"] and pd.isna(out.loc[x, "rank_B1"])
    assert out.loc[x, "loo_worst_metric_B1"] == "yield_dividend"
    assert out.loc[x, "loo_worst_rank_B1"] > 10
    # 其他股票不受影响，记录了每只的最差排名
    assert out.loc[out["in_B1"], "loo_worst_rank_B1"].notna().all()


def test_loo_treats_all_metrics_identically():
    df, x = _loo_universe()
    params = {**_params(), "bucket": {**PARAMS["bucket"], "top_n": 5}}
    out = run_scoring(df, params, reasons=False)
    import json
    ranks = json.loads(out.loc[x, "bucket_check"])["B1"]["ranks"]
    assert set(ranks) == set(ALL_SCORE_METRICS)


def test_concentration_flag_reports_dominant_family():
    df = _strong(universe(), 3)
    df.loc[3, "yield_dividend"] = df["yield_dividend"].quantile(0.72)   # 股息只是刚过线
    df.loc[3, "evt_dividend_change"] = 1.0
    out = run_scoring(df, PARAMS, reasons=False)
    assert out.loc[3, "in_B1"] and out.loc[3, "max_family_share"] > 0
    assert out.loc[3, "dominant_family"] in REG["families"]


def test_anchor_family_below_coverage_suspends_bucket():
    """估值数据只有四成公司能判断：B1 在这个市场暂停，不能跳过。"""
    df = _strong(universe(), 3)
    n = len(df)
    hide = df.index[np.arange(n) % 5 != 0]
    hide = hide[hide != 3]
    df.loc[hide, ["val_pbr", "val_fwd_ep", "ep_ttm", "val_ocf_yield"]] = np.nan
    out = run_scoring(df, PARAMS, reasons=False)
    assert not out.loc[3, "in_B1"]
    assert "anchor_suspended" in out.loc[3, "bucket_check"]


def test_quality_below_coverage_passes_with_low_confidence_note():
    df = universe()
    df["qual_fscore7"] = np.nan
    for m in ("guid_revision_event", "yield_dividend", "evt_dividend_change"):
        df = make_excellent(df, 3, m)
    out = run_scoring(df, PARAMS, reasons=False)
    assert out.loc[3, "in_B2"]
    assert "low_conf_uncovered:business_quality" in out.loc[3, "bucket_check"]


def test_hard_filter_does_not_exclude_everyone_and_x09_needs_real_gap():
    df = universe()
    df.loc[0, "effective_date"] = None
    df.loc[1, "n_fy"] = 0
    df.loc[2, "adv20_turnover"] = 1.0
    df.loc[3, "negative_equity"] = True
    df.loc[4, "ProdCat"] = "012"
    df.loc[4, "is_common_stock"] = False
    out = soft_flags(hard_filters(df, PARAMS), PARAMS)
    assert out.loc[0, "exclusion_codes"] == "X09" and out.loc[1, "exclusion_codes"] == "X09"
    assert "X08" in out.loc[2, "exclusion_codes"] and out.loc[3, "exclusion_codes"] == "X07" and "X01" in out.loc[4, "exclusion_codes"]
    assert out["excluded"].sum() == 5


def test_candidates_outside_core_are_not_listed():
    df = _strong(universe(), 3)
    df.loc[3, "in_core"] = False
    out = run_scoring(df, PARAMS, reasons=False)
    assert not out.loc[3, "in_B1"]


# ---------- 外部名单不前视 ----------
def _sup(rows):
    cols = ["designated_date", "Code", "kind", "released_date", "is_current"]
    return pd.DataFrame(rows, columns=cols)


def test_supervision_uses_designated_and_released_dates():
    sup = _sup([("2026-03-01", "11110", "監理銘柄（確認中）", "2026-06-01", False),
                ("2026-09-01", "22220", "整理銘柄", None, True),          # 指定晚于基准日：不算
                ("2026-02-01", "33330", "整理銘柄", None, True),          # 今天还在册，指定早于基准日：算
                ("2026-07-01", "44440", "監理銘柄（確認中）", None, False),  # 没有解除日、今天不在册、距基准日近：无法确定
                ("2020-01-01", "55550", "監理銘柄（確認中）", None, False),  # 太久远：不算在册
                ("2026-06-01", "66660", "整理銘柄", None, False)])
    dl = pd.DataFrame({"delist_date": ["2026-09-30"], "Code": ["66660"]})
    sure, unknown = supervision_as_of(sup, dl, date(2026, 5, 1))
    assert sure == {"11110", "33330"} and unknown == set()      # 5/1 时 11110 还没解除
    sure, unknown = supervision_as_of(sup, dl, date(2026, 8, 14))
    assert sure == {"33330", "66660"} and unknown == {"44440"}   # 8/14：11110 已解除；66660 退市日在后


def test_current_only_list_used_only_near_fetch_date():
    imp = pd.DataFrame({"published_date": ["2026-11-05"], "Code": ["77770"]})
    assert list_as_of(imp, date(2026, 8, 14), "published_date", 40) == set()     # 基准日早于名单：前视，不用
    assert list_as_of(imp, date(2026, 11, 7), "published_date", 40) == {"77770"}
    assert list_as_of(imp, date(2027, 3, 1), "published_date", 40) == set()      # 名单太旧


# ---------- 事件分：与行情不重叠时中性且 unknown ----------
def _disc(rows):
    return pd.DataFrame(rows, columns=["disclosure_id", "disclosed_date", "Code", "category", "direction", "strength", "half_life_days", "hard_exclude", "lang"])


def test_news_unknown_when_disclosures_do_not_cover_date():
    disc = _disc([("a", "2026-10-20", "11110", "E05", "pos", 3, 20, False, "ja")])
    assert coverage_status(disc, date(2026, 8, 14)) == "unknown"
    assert coverage_status(None, date(2026, 8, 14)) == "unknown"
    assert build_event_scores(disc, None, [date(2026, 8, 14)], PARAMS).empty
    df = universe()
    out = run_scoring(df, PARAMS, reasons=False)
    assert (out["news_status"] == "unknown").all() and (out["score_news"] == 50.0).all()


def test_event_scores_vectorized_over_many_dates():
    disc = _disc([("a", "2026-10-20", "11110", "E05", "pos", 40, 20, False, "ja"), ("b", "2026-10-25", "11110", "E17", "neg", -60, 20, False, "ja"),
                  ("c", "2026-10-26", "11110", "E10", "neg", -10, 20, False, "ja"), ("d", "2026-10-26", "22220", "E05", "pos", 10, 20, False, "en")])
    dates = list(pd.bdate_range("2025-01-01", "2026-11-06").date)
    ev = build_event_scores(disc, None, dates, PARAMS)
    assert set(ev["Code"]) == {"11110"}                   # 英文重复稿不计
    assert pd.to_datetime(ev["Date"]).min() >= pd.Timestamp("2026-10-20")     # 不覆盖的日期没有行
    last = ev.sort_values("Date").iloc[-1]
    assert bool(last["flag_improper_accounting"]) and bool(last["flag_ms_warrant"]) and bool(last["catalyst_buyback"])
    assert last["news_score"] <= 40                       # 30 日内两个负面事件封顶


# ---------- 错杀 ----------
def test_intact_unknown_items_do_not_count_as_pass_or_fail():
    d = pd.DataFrame({"is_financial": [False, False, True], "has_forecast": [True, False, True], "guid_fail": [False, False, True], "evt_div_down": [False, False, False],
                      "evt_dividend_change": [0.0, np.nan, 0.0], "guid_remaining_flag": [np.nan, np.nan, 1.0], "gate_distress": [0.0, np.nan, 1.0]})
    out = intact_test(d)
    assert out.loc[0, "intact_status"] == "pass" and out.loc[0, "intact_known"] == 3
    assert out.loc[1, "intact_status"] == "unknown" and out.loc[1, "intact_known"] == 0
    assert out.loc[2, "intact_status"] == "pass"           # 金融业只看分红项


def test_oversold_output_has_no_reaction_ratio_and_event_flags_run():
    dates = pd.bdate_range("2026-06-01", periods=30)
    n = len(dates)
    px = np.r_[np.full(20, 100.0), np.full(10, 80.0)]
    feat = pd.DataFrame({"Code": "11110", "Date": dates, "S33Nm": "化学", "mcap": 5e10, "C": px, "ret_1d": pd.Series(px).pct_change(), "ret_5d": pd.Series(px).pct_change(5), "ret_21d": pd.Series(px).pct_change(21),
                         "drawdown_60d": px / 100 - 1, "volume_mult": np.r_[np.ones(20), [4.0], np.ones(9)], "low_since_20d": 80.0, "atr5": 1.0, "atr14": 2.0, "limit_down": False, "rev_1m": 0.0,
                         "cause_proxy_earnings": False, "cause_proxy_earnings_60": False, "cause_proxy_date": dates[0], "guid_forecast_innovation": np.nan, "is_financial": False, "has_forecast": True,
                         "guid_fail": False, "evt_div_down": False, "evt_dividend_change": 0.0, "guid_remaining_flag": 0.0, "gate_distress": 0.0, "negative_equity": False})
    out = build_oversold(feat, None, PARAMS)
    assert "reaction_ratio" not in out.columns and "score_oversold" not in out.columns
    assert out.iloc[20]["trigger"] and out.iloc[20]["trigger_type"] == "single_day"
    assert out.iloc[-1]["intact_status"] == "pass" and len(out) == n


def test_oversold_attributes_with_disclosures_only_on_trigger_window():
    dates = pd.bdate_range("2026-06-01", periods=10)
    px = np.r_[np.full(5, 100.0), np.full(5, 85.0)]
    feat = pd.DataFrame({"Code": "11110", "Date": dates, "S33Nm": "化学", "mcap": 5e10, "C": px, "ret_1d": pd.Series(px).pct_change(), "ret_5d": np.nan, "ret_21d": np.nan,
                         "drawdown_60d": px / 100 - 1, "volume_mult": np.r_[np.ones(5), [4.0], np.ones(4)], "low_since_20d": 85.0, "atr5": 1.0, "atr14": 2.0, "limit_down": False, "rev_1m": 0.0,
                         "cause_proxy_earnings": False, "cause_proxy_earnings_60": False, "cause_proxy_date": dates[0], "guid_forecast_innovation": np.nan, "is_financial": False, "has_forecast": True,
                         "guid_fail": False, "evt_div_down": False, "evt_dividend_change": 0.0, "guid_remaining_flag": 0.0, "gate_distress": 0.0, "negative_equity": False})
    disc = _disc([("a", dates[5].date().isoformat(), "11110", "E10", "neg", -3, 60, True, "ja")])
    out = build_oversold(feat, disc, PARAMS)
    assert out.iloc[4]["attr_type"] != "T2"
    assert out.iloc[5]["attr_type"] == "T2" and out.iloc[6]["attr_type"] == "T2"
    assert bool(out.iloc[6]["hard_fail"])


def test_cum_after_at_is_asof_for_dates_not_in_panel():
    ev = pd.DataFrame({"Date": pd.to_datetime(["2026-03-02", "2026-06-01"]), "Code": ["11110", "11110"], "AdjFactor": [0.5, 0.2]})
    got = cum_after_at(ev, ["11110", "11110", "11110", "22220", "11110"], pd.to_datetime(["2026-01-03", "2026-03-02", "2026-07-01", "2026-01-03", None]))
    assert got[0] == pytest.approx(0.1) and got[1] == pytest.approx(0.2) and got[2] == 1.0 and got[3] == 1.0 and np.isnan(got[4])


def test_family_status_drop_treats_metric_as_absent():
    df = _strong(universe(), 3)
    ctx = _Ctx(df, PARAMS, REG)
    m = evaluate_metrics(df, ctx, resolve_gates(df, ctx))
    base = family_status(ctx, m)
    assert base["shareholder_yield"][3] == HIT
    dropped = family_status(ctx, m, drop="yield_dividend")
    assert dropped["shareholder_yield"][3] != HIT
    assert dropped["relative_value"][3] == HIT
    assert family_status(ctx, m, drop="val_pbr")["relative_value"][3] == HIT     # 三项都命中，去掉一项仍满足 2-of-3


def test_evaluate_is_deterministic():
    df = _strong(universe(), 3)
    a = evaluate(hard_filters(df.copy(), PARAMS), PARAMS, REG)
    b = evaluate(hard_filters(df.copy(), PARAMS), PARAMS, REG)
    assert a["family_status"].tolist() == b["family_status"].tolist() and a["rank_B1"].equals(b["rank_B1"])
