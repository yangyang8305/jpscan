"""筛选理由的测试：数字必须等于存储值，白话不能缺，候选的理由不能少于门槛。数据全部合成。"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

from jpscan.screener.reasons import EXCLUSION_ZH, build_reasons, exclusion_reason, fmt_value
from jpscan.screener.scoring import run_scoring

from test_screener import PARAMS, REG, _params, make_excellent, universe


def _candidates():
    df = universe()
    for i, ms in {1: ("val_pbr", "val_fwd_ep", "val_ocf_yield", "yield_dividend", "evt_dividend_change"),
                  2: ("guid_revision_event", "yield_dividend", "evt_dividend_change"),
                  3: ("rev_1m", "yield_dividend", "evt_dividend_change"),
                  4: ("val_pbr", "val_fwd_ep", "guid_forecast_innovation", "yield_dividend")}.items():
        for m in ms:
            df = make_excellent(df, i, m)
    df.loc[2, "guid_revision_date"] = "2026-08-01"
    df.loc[1, ["evt_div_prev", "evt_div_new", "evt_div_date"]] = [20.0, 25.0, "2026-08-01"]
    df["guid_innovation_basis"] = None
    df.loc[4, ["guid_innovation_basis", "guid_innovation_date", "guid_innovation_disc_no"]] = ["odp", "2026-07-01", "9001"]
    df["DiscNo"] = "1000"
    df["fy_disc_no"] = "2000"
    df["val_source"] = "valuation_daily"
    df["rev_1m_start_date"] = "2026-07-14"
    return run_scoring(df, PARAMS)


@pytest.fixture(scope="module")
def scored():
    return _candidates()


def _numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]


def _allowed(r: dict) -> list[float]:
    """这条理由允许出现的数字：全部来自存储字段。"""
    vals: list[float] = []
    if r["value"] is not None:
        unit_str = fmt_value(r["metric"], r["value"])
        vals += _numbers(unit_str)
    if r.get("threshold") is not None:
        t = float(r["threshold"])
        vals += [t, t * 100, round(t * 100, 1)]
    if r.get("peer_pct") is not None:
        vals.append(round(r["peer_pct"] * 100, 1))
    if r.get("peer_n") is not None:
        vals.append(float(r["peer_n"]))
    if r.get("self_pct") is not None:
        vals.append(round(r["self_pct"] * 100, 1))
    for d in (r.get("as_of"), r.get("detail", {}).get("event_date")):
        if d:
            vals += [float(x) for x in str(d).split("-")]
            vals += [-float(x) for x in str(d).split("-")]
    det = r.get("detail", {})
    for k in ("div_prev", "div_new", "window_days"):
        if det.get(k) is not None:
            vals.append(float(det[k]))
    if det.get("min_abs") is not None:
        vals.append(round(float(det["min_abs"]) * 100, 1))
    return vals


def test_every_number_in_text_is_a_stored_value(scored):
    n = 0
    for _, row in scored[scored[["in_B1", "in_B2", "in_B3"]].any(axis=1)].iterrows():
        for r in json.loads(row["reasons"]):
            body = r["text_zh"].replace(r["label_zh"], "", 1).replace(r["plain_zh"], "", 1)
            allowed = _allowed(r)
            for x in _numbers(body):
                assert any(abs(x - a) < 1e-9 for a in allowed), (x, r["text_zh"], allowed)
            n += 1
    assert n >= 10


def test_value_in_text_equals_stored_value(scored):
    row = scored.iloc[1]
    rs = {r["metric"]: r for r in json.loads(row["reasons"])}
    pbr = rs["val_pbr"]
    assert fmt_value("val_pbr", pbr["value"]) in pbr["text_zh"]
    assert pbr["value"] == pytest.approx(round(pbr["value"], 2))
    # 与 family_status 里的原值一致（只差展示取整）
    raw = json.loads(row["family_status"])["relative_value"]["metrics"]["val_pbr"]["value"]
    assert pbr["value"] == pytest.approx(raw, abs=0.005)


def test_reason_shape_and_plain_language(scored):
    required = {"family", "rule_id", "metric", "label_zh", "plain_zh", "value", "threshold", "comparison", "peer_pct", "self_pct", "peer_group", "as_of", "source", "text_zh"}
    for _, row in scored[scored["in_B1"] | scored["in_B2"] | scored["in_B3"]].iterrows():
        for r in json.loads(row["reasons"]):
            assert required <= set(r)
            meta = REG["metrics_by_id"][r["metric"]]
            assert meta["plain_zh"].rstrip("。") in r["plain_zh"]     # 注册表里那句白话
            assert r["plain_zh"] in r["text_zh"]                      # 和数字写在一起
            assert r["source"]
        for c in json.loads(row["counter_reasons"]):
            assert required - {"self_pct"} <= set(c) and c["text_zh"]


def test_every_registry_score_metric_has_plain_language():
    for fam in REG["families"].values():
        for r in fam["threshold"]["hit_rules"]:
            assert REG["metrics_by_id"][r["metric"]].get("plain_zh")


def test_candidate_cannot_have_fewer_families_than_min(scored):
    mf = PARAMS["evidence"]["min_families"]
    cands = scored[scored[["in_B1", "in_B2", "in_B3"]].any(axis=1)]
    assert len(cands) >= 3
    for _, row in cands.iterrows():
        rs = json.loads(row["reasons"])
        assert len({r["family"] for r in rs}) >= mf
        assert row["family_pass_count"] >= mf


def test_selection_summary_names_bucket_and_numbers(scored):
    row = scored.iloc[1]
    s = row["selection_summary_zh"]
    assert "B1" in s and "低估反转" in s and "相对估值" in s
    assert re.search(r"\d+\.\d+%", s)


def test_unknowns_name_missing_data():
    df = universe()
    df["qual_fscore7"] = np.nan
    for m in ("val_pbr", "val_fwd_ep", "yield_dividend", "evt_dividend_change"):
        df = make_excellent(df, 1, m)
    row = run_scoring(df, PARAMS).iloc[1]
    un = json.loads(row["unknowns"])
    q = next(u for u in un if u.get("family") == "business_quality")
    assert "NP" in q["needed_zh"] or "financial_statements" in q["needed_zh"]
    assert any(u["rule_id"] == "news.unknown" for u in un)
    assert any(u["rule_id"].startswith("gate_unchecked.") for u in un)       # 未检查的门槛必须列出，不能显示为通过


def test_counter_reasons_list_soft_flags_gates_and_failed_families():
    df = universe()
    for m in ("val_pbr", "val_fwd_ep", "val_ocf_yield", "yield_dividend", "evt_dividend_change"):
        df = make_excellent(df, 1, m)
    df.loc[1, ["gate_distress", "stale_financials"]] = [1.0, True]
    df.loc[1, "qual_fscore7"] = 1.0
    df.loc[1, "guid_remaining_flag"] = 1.0
    row = run_scoring(df, _params()).iloc[1]
    # 经营质量 fail 不进 B1，但 build_reasons 可以单独调用
    res = build_reasons(row, PARAMS)
    ids = {c["rule_id"] for c in res["counter_reasons"]}
    assert "family_fail.business_quality" in ids and "gate.gate_distress" in ids and "soft.STALE" in ids and "gate.guid_remaining_growth" in ids
    assert not (row["in_B1"])


def test_build_reasons_works_from_stored_json_only(scored):
    """报告模块只拿 scores_daily 的存储列也能调用。"""
    row = scored.iloc[1]
    stored = {k: row[k] for k in ("family_status", "bucket_check", "Date", "bucket", "bucket_rank", "soft_flags", "exclusion_codes", "in_top_n", "family_pass_count")}
    res = build_reasons(stored, PARAMS)
    assert {r["metric"] for r in res["reasons"]} == {r["metric"] for r in json.loads(row["reasons"])}
    assert res["selection_summary_zh"] == row["selection_summary_zh"]


def test_exclusion_reason_maps_every_x_code_and_bucket_requirement():
    df = universe()
    df.loc[0, "negative_equity"] = True
    df.loc[1, "adv20_turnover"] = 1.0
    out = run_scoring(df, PARAMS)
    assert EXCLUSION_ZH["X07"] in out.loc[0, "exclusion_reason_zh"]
    assert EXCLUSION_ZH["X08"] in out.loc[1, "exclusion_reason_zh"]
    plain = out.loc[2, "exclusion_reason_zh"]
    assert "B1" in plain and "B2" in plain and "B3" in plain
    assert "至少要 3 个" in plain and "相对估值" in plain
    for code in EXCLUSION_ZH:
        row = {"exclusion_codes": code, "bucket_check": "{}"}
        assert EXCLUSION_ZH[code] in exclusion_reason(row, PARAMS)


def test_every_fail_code_gets_a_sentence():
    codes = ["not_universe", "not_core", "hard_excluded", "excluded_sector", "excluded_market", "anchor_suspended:relative_value", "anchor_miss:relative_value", "required_miss:business_quality",
             "confirm_miss", "cyclical_yield_only", "family_fail:guidance_change", "valuation_cap:val_fwd_ep", "intact_fail", "intact_unknown", "cause_wait", "cause_hard_fail", "min_families:3"]
    row = {"exclusion_codes": "", "family_pass_count": 1, "in_top_n": False, "bucket_check": json.dumps({"B1": {"fails": codes}})}
    text = exclusion_reason(row, PARAMS)
    for c in codes:
        assert c not in text, c


def test_loo_removed_explains_in_plain_words():
    row = {"exclusion_codes": "", "in_top_n": False, "bucket_check": json.dumps({"B1": {"fails": [], "loo_removed": True, "worst_metric": "yield_dividend", "base_rank": 3, "worst_rank": 95}})}
    t = exclusion_reason(row, PARAMS)
    assert "予想配当利回り" in t and "第 3" in t and "第 95" in t and "前 80" in t


def test_selected_stock_has_empty_exclusion_reason(scored):
    assert scored.loc[1, "exclusion_reason_zh"] == ""
    assert isinstance(scored.loc[1, "family_status"], str)
    json.loads(scored.loc[1, "family_status"])


def test_self_history_is_labelled_low_confidence_when_short():
    df = universe()
    for m in ("val_pbr", "val_fwd_ep", "val_ocf_yield", "yield_dividend", "evt_dividend_change"):
        df = make_excellent(df, 1, m)
    df["self_pct_pbr"] = np.nan
    df["self_pct_low_conf"] = True
    df.loc[1, "self_pct_pbr"] = 0.2
    row = run_scoring(df, PARAMS).iloc[1]
    r = next(x for x in json.loads(row["reasons"]) if x["metric"] == "val_pbr")
    assert r["self_pct"] == 0.2 and r["self_low_confidence"] is True and "只作参考" in r["text_zh"]
    assert r["threshold"] == 0.30       # 自身分位只展示，不改变命中条件


def test_context_states_growth_horizon_honestly():
    row = {"family_status": "{}", "bucket_check": "{}", "Date": "2026-08-14", "growth_horizon": "1y", "rev_growth_1y": 0.12, "soft_flags": ""}
    ctx = build_reasons(row, PARAMS)["context"]
    assert ctx[0]["horizon"] == "1y" and "算不出 3 年增长" in ctx[0]["text_zh"]


def test_reasons_are_deterministic():
    a, b = _candidates(), _candidates()
    pd.testing.assert_series_equal(a["reasons"], b["reasons"])
    pd.testing.assert_series_equal(a["exclusion_reason_zh"], b["exclusion_reason_zh"])
