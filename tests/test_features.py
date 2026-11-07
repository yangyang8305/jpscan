"""features 单元测试：全部用合成数据（公司与数字都是虚构的）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from jpscan.features.panel import back_adjust, cum_after_matrix, market_proxy, technical_features


def _bars(code="99990", n=30, split_at=None, factor=0.5, start="2026-01-05"):
    dates = pd.bdate_range(start, periods=n)
    rows = []
    for i, d in enumerate(dates):
        c = 100.0 if split_at is None or i < split_at else 100.0 * factor
        vo = 1000.0 if split_at is None or i < split_at else 1000.0 / factor
        af = factor if split_at is not None and i == split_at else 1.0
        rows.append({"Date": d, "Code": code, "O": c, "H": c, "L": c, "C": c, "UL": "0", "LL": "0", "Vo": vo, "Va": c * vo,
                     "AdjFactor": af, "MktCap": 1000.0})
    return pd.DataFrame(rows)


def test_back_adjust_uses_factors_after_date_only():
    d = back_adjust(_bars(n=6, split_at=3))
    # 分割前的价格乘 0.5、量除以 0.5，分割后不变；调整后序列平滑
    assert d["adj_c"].tolist() == [50.0] * 6
    assert d["adj_vo"].tolist() == [2000.0] * 6
    assert d["cum_after"].tolist() == [0.5, 0.5, 0.5, 1.0, 1.0, 1.0]


def test_cum_after_ignores_factors_before_panel():
    dates = pd.bdate_range("2026-02-02", periods=3)
    ev = pd.DataFrame({"Date": [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-02-03")], "Code": ["1", "1"], "AdjFactor": [0.1, 0.5]})
    b = cum_after_matrix(ev, dates, pd.Index(["1"]))
    assert b["1"].tolist() == [0.5, 1.0, 1.0]


def test_split_does_not_create_fake_crash():
    tech = technical_features(_bars(n=40, split_at=20), {})
    assert np.nanmin(tech["ret_1d"]) == 0.0
    assert np.nanmin(tech["ret_21d"]) == 0.0
    # 价位类列按当日股数口径：分割后的均线就是分割后的价格
    assert tech.iloc[-1]["ma25"] == 50.0


def test_price_gap_is_not_treated_as_adjacent_days():
    a = _bars(n=25, start="2025-01-06")
    b = _bars(n=25, start="2026-01-05").assign(C=200.0, O=200.0, H=200.0, L=200.0)
    # 用 store=None 时日期只有实际行情日；手工插入空交易日模拟日历上的回填缺口
    gap = pd.DataFrame({"Date": pd.bdate_range("2025-02-10", "2025-12-31"), "Code": "99990"})
    panel = pd.concat([a, gap, b], ignore_index=True).sort_values(["Code", "Date"]).reset_index(drop=True)
    panel[["Vo", "Va"]] = panel[["Vo", "Va"]].fillna(0.0)
    panel["AdjFactor"] = panel["AdjFactor"].fillna(1.0)
    tech = technical_features(panel, {})
    first_after = tech[tech["Date"] == pd.Timestamp("2026-01-05")].iloc[0]
    assert np.isnan(first_after["ret_21d"])   # 不把一年前的 100 当成 21 天前


def test_market_proxy_weights_by_previous_mcap():
    idx = pd.bdate_range("2026-01-05", periods=2)
    ret = pd.DataFrame({"A": [np.nan, 0.10], "B": [np.nan, -0.10]}, index=idx)
    cap = pd.DataFrame({"A": [300.0, 1.0], "B": [100.0, 1.0]}, index=idx)
    m = market_proxy(ret, cap)
    assert abs(m["mkt_ret"].iloc[1] - 0.05) < 1e-12


# ---------- 财务状态（合成的决算短信行）----------
from datetime import date  # noqa: E402

from jpscan.features.fundamentals import asof_join, build_states_and_events, build_statement_states, fscore7, ttm_from_ytd  # noqa: E402


def _doc(no, disc, doc_type, per, fy_st, fy_en, **kw):
    base = {"Code": "99990", "DiscNo": str(no), "DiscDate": disc, "effective_date": disc, "DocType": doc_type, "CurPerType": per, "CurFYSt": fy_st, "CurFYEn": fy_en,
            "CurPerEn": kw.pop("per_end", fy_en), "NxtFYEn": kw.pop("nxt", None), "gaap": "JP", "is_revision": "ForecastRevision" in doc_type}
    base.update(kw)
    return base


FY = "FYFinancialStatements_Consolidated_JP"
Q1 = "1QFinancialStatements_Consolidated_JP"
Q2 = "2QFinancialStatements_Consolidated_JP"
Q3 = "3QFinancialStatements_Consolidated_JP"
EREV = "EarnForecastRevision"
DREV = "DividendForecastRevision"
S25, E25 = date(2024, 4, 1), date(2025, 3, 31)
S26, E26 = date(2025, 4, 1), date(2026, 3, 31)
E27 = date(2027, 3, 31)


def _fy25():
    return _doc(1, date(2025, 5, 10), FY, "FY", S25, E25, nxt=E26, Sales=1000.0, OP=100.0, OdP=110.0, NP=70.0, TA=2000.0, EqAR=0.5, CashEq=300.0, ShOutFY=100.0, TrShFY=10.0,
                CFO=90.0, DivAnn=20.0, BPS=11.0, NxFOP=120.0, NxFOdP=125.0, NxFNp=80.0, NxFEPS=8.0, NxFDivAnn=22.0)


def test_revision_doc_carries_balance_and_forecast_fields():
    docs = [_fy25(),
            _doc(2, date(2025, 6, 1), EREV, "FY", S26, E26, FOP=150.0, FNP=95.0),                 # 只修 OP、NP
            _doc(3, date(2025, 6, 20), DREV, "FY", S26, E26, FDivAnn=25.0)]                       # 分红修订
    st = build_statement_states(pd.DataFrame(docs)).set_index("DiscNo")
    # 修订文档留空的资产负债字段不被抹掉
    assert st.loc["2", "ta_latest"] == 2000.0 and st.loc["2", "eqar_latest"] == 0.5 and st.loc["2", "cash_latest"] == 300.0 and st.loc["2", "shares_out"] == 100.0
    # 被修的字段更新，没修的字段延续（同一目标财年）
    assert st.loc["2", "f_op"] == 150.0 and st.loc["2", "f_eps"] == 8.0 and st.loc["2", "f_odp"] == 125.0 and st.loc["2", "f_div_ann"] == 22.0
    # 分红修订不抹利润预想
    assert st.loc["3", "f_op"] == 150.0 and st.loc["3", "f_np"] == 95.0 and st.loc["3", "f_div_ann"] == 25.0


def test_forecast_does_not_carry_across_fiscal_years():
    docs = [_fy25(), _doc(2, date(2026, 5, 10), FY, "FY", S26, E26, nxt=E27, Sales=1100.0, OP=130.0, NP=80.0, TA=2100.0, EqAR=0.5)]     # 新年报不给预想
    st = build_statement_states(pd.DataFrame(docs)).set_index("DiscNo")
    assert st.loc["2", "f_fy_end"] == pd.Timestamp(E27)
    assert np.isnan(st.loc["2", "f_op"]) and np.isnan(st.loc["2", "f_eps"]) and bool(st.loc["2", "f_no_guidance_fy"])


def test_quarter_doc_with_some_blank_fields_carries_rest():
    docs = [_fy25(), _doc(2, date(2025, 8, 1), Q1, "1Q", S26, E26, per_end=date(2025, 6, 30), Sales=250.0, OP=30.0, NP=20.0, TA=2050.0, EqAR=0.5, FOP=130.0)]
    st = build_statement_states(pd.DataFrame(docs)).set_index("DiscNo")
    assert st.loc["2", "f_op"] == 130.0 and st.loc["2", "f_np"] == 80.0
    assert st.loc["2", "cash_latest"] == 300.0     # 1Q 通常没有 CashEq
    assert st.loc["2", "ta_latest"] == 2050.0


def test_withdrawn_forecast_is_an_event_and_clears_state():
    docs = [_fy25(), _doc(2, date(2025, 8, 1), Q1, "1Q", S26, E26, per_end=date(2025, 6, 30), Sales=250.0, OP=30.0, NP=20.0, TA=2050.0, EqAR=0.5)]
    st, ev = build_states_and_events(pd.DataFrame(docs))
    assert np.isnan(st.set_index("DiscNo").loc["2", "f_op"])
    assert bool(ev["revision"].iloc[-1]["withdrawn"])


def test_late_filing_does_not_overwrite_latest_balance():
    docs = [_fy25(), _doc(2, date(2025, 8, 1), Q1, "1Q", S26, E26, per_end=date(2025, 6, 30), TA=2050.0, EqAR=0.5, FOP=120.0),
            _doc(3, date(2025, 9, 1), Q3, "3Q", S25, E25, per_end=date(2024, 12, 31), TA=1.0, EqAR=0.01)]   # 上年 3Q 的补报
    st = build_statement_states(pd.DataFrame(docs)).set_index("DiscNo")
    assert st.loc["3", "ta_latest"] == 2050.0 and st.loc["3", "eqar_latest"] == 0.5


def test_ttm_requires_consecutive_quarters():
    fy24 = {"CurFYSt": pd.Timestamp("2023-04-01"), "CurFYEn": pd.Timestamp("2024-03-31"), "OP": 100.0}
    q2_25 = {"CurFYSt": pd.Timestamp("2024-04-01"), "CurFYEn": pd.Timestamp("2025-03-31"), "OP": 70.0}
    q2_24 = {"CurFYSt": pd.Timestamp("2023-04-01"), "CurFYEn": pd.Timestamp("2024-03-31"), "OP": 40.0}
    k24, k25 = pd.Timestamp("2024-03-31"), pd.Timestamp("2025-03-31")
    ok = ttm_from_ytd({(k24, 2): q2_24, (k24, 4): fy24, (k25, 2): q2_25}, {k24: fy24}, (k25, 2), fields=("OP",))
    assert ok["n"] == 4 and ok["OP"] == 130.0            # 100 + 70 - 40
    # 上一财年缺失（跳了一年）：不做近似
    k26 = pd.Timestamp("2026-03-31")
    q2_26 = {"CurFYSt": pd.Timestamp("2025-04-01"), "CurFYEn": k26, "OP": 70.0}
    bad = ttm_from_ytd({(k24, 2): q2_24, (k24, 4): fy24, (k26, 2): q2_26}, {k24: fy24}, (k26, 2), fields=("OP",))
    assert bad["n"] == 0 and np.isnan(bad["OP"])
    # 上一财年只有 9 个月（决算期变更）：差分凑不成 12 个月
    short = {"CurFYSt": pd.Timestamp("2023-07-01"), "CurFYEn": k24, "OP": 100.0}
    bad2 = ttm_from_ytd({(k24, 2): q2_24, (k24, 4): short, (k25, 2): q2_25}, {k24: short}, (k25, 2), fields=("OP",))
    assert bad2["n"] == 0


def test_pit_no_use_of_report_before_effective_date():
    docs = [_fy25(), _doc(2, date(2025, 6, 10), EREV, "FY", S26, E26, FOP=150.0)]
    st = build_statement_states(pd.DataFrame(docs))
    panel = pd.DataFrame({"Code": "99990", "Date": pd.to_datetime(["2025-05-09", "2025-05-10", "2025-06-09", "2025-06-10"])})
    m = asof_join(panel, st).sort_values("Date").reset_index(drop=True)
    assert pd.isna(m.loc[0, "effective_date"]) and np.isnan(m.loc[0, "f_op"])
    assert m.loc[1, "f_op"] == 120.0 and m.loc[2, "f_op"] == 120.0     # 修订生效前仍是旧预想
    assert m.loc[3, "f_op"] == 150.0
    assert m.loc[3, "effective_date"] == pd.Timestamp("2025-06-10")     # effective_date 是真实列


def test_dividend_event_only_compares_same_target_year():
    docs = [_fy25(), _doc(2, date(2025, 6, 20), DREV, "FY", S26, E26, FDivAnn=25.0),
            _doc(3, date(2026, 5, 10), FY, "FY", S26, E26, nxt=E27, Sales=1.0, TA=1.0, EqAR=0.5, DivAnn=25.0, NxFDivAnn=40.0)]
    _, ev = build_states_and_events(pd.DataFrame(docs))
    d = ev["dividend"]
    assert len(d) == 1 and d.iloc[0]["div_new"] == 25.0 and d.iloc[0]["div_prev"] == 22.0   # 新财年指引 40 不算事件


def test_fy_innovation_basis_falls_back_by_field():
    docs = [_doc(1, date(2025, 5, 10), FY, "FY", S25, E25, nxt=E26, OdP=100.0, NxFOdP=130.0, OP=90.0, NxFOP=100.0, NP=60.0, NxFNp=70.0),
            _doc(2, date(2025, 5, 11), FY, "FY", S25, E25, nxt=E26, OP=90.0, NxFOP=100.0, NP=60.0, NxFNp=70.0, Code="88880"),
            _doc(3, date(2025, 5, 12), FY, "FY", S25, E25, nxt=E26, NP=60.0, NxFNp=75.0, Code="77770", gaap="IFRS")]
    _, ev = build_states_and_events(pd.DataFrame(docs))
    got = ev["fy"].set_index("Code")
    assert got.loc["99990", "basis"] == "odp" and got.loc["99990", "diff"] == 30.0
    assert got.loc["88880", "basis"] == "op" and got.loc["88880", "diff"] == 10.0
    assert got.loc["77770", "basis"] == "np" and got.loc["77770", "diff"] == 15.0


def test_fscore_counts_seven_items_and_needs_five_known():
    good = {"np_fy0": 10.0, "np_fy1": 5.0, "ta_fy0": 100.0, "ta_fy1": 100.0, "cfo_fy0": 15.0, "eqar_fy0": 0.5, "eqar_fy1": 0.4, "sales_fy0": 120.0, "sales_fy1": 100.0,
            "share_growth_fy": 0.0, "fy_days_fy0": 365.0, "fy_days_fy1": 365.0}
    one_year = {**good, "np_fy1": np.nan, "ta_fy1": np.nan, "eqar_fy1": np.nan, "sales_fy1": np.nan, "share_growth_fy": np.nan, "fy_days_fy1": np.nan}
    out = fscore7(pd.DataFrame([good, one_year]))
    assert out.loc[0, "qual_fscore7"] == 7.0 and out.loc[0, "fscore_n"] == 7
    assert np.isnan(out.loc[1, "qual_fscore7"]) and out.loc[1, "fscore_n"] == 3     # 只有一个财年：不算
