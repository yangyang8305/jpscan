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
