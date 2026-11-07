"""历史回测（规划文档 7.2、17.1 节）：月度再平衡的篮子组合与分维度分层检验；时点数据、成交约束、成本分层。"""
from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from ..config import pget

log = logging.getLogger("jpscan.backtest")


def rebalance_dates(trading_days: list[date], freq: str = "M") -> list[date]:
    s = pd.Series(sorted(trading_days))
    key = pd.to_datetime(s).dt.to_period("M" if freq == "M" else "W")
    return list(s.groupby(key).first().values)


def forward_returns(prices: pd.DataFrame, start: date, end: date) -> pd.Series:
    """每只股票从 start 收盘持有到 end 收盘的收益（用调整后收盘价）；涨跌停日视为不可成交 → 缺失。"""
    p = prices[(prices["Date"] >= start) & (prices["Date"] <= end)]
    first = p[p["Date"] == start].set_index("Code")
    last = p[p["Date"] == end].set_index("Code")
    common = first.index.intersection(last.index)
    r = last.loc[common, "AdjC"] / first.loc[common, "AdjC"] - 1
    blocked = first.loc[common]
    if "limit_up" in blocked.columns:
        r = r[~blocked["limit_up"].fillna(False).astype(bool)]
    return r


def run_backtest(store, params: dict, scores_all: pd.DataFrame, prices: pd.DataFrame, freq: str = "M",
                 top_n: int | None = None) -> dict:
    v = params.get("validation", {})
    costs = v.get("cost_one_way", {"high_liquidity": 0.001, "mid": 0.002, "low": 0.003})
    top_n = top_n or int(pget(params, "bucket.top_n", 40))
    dates = sorted(scores_all["Date"].unique())
    rdates = rebalance_dates(dates, freq)
    if len(rdates) < 2:
        return {"error": "需要至少 2 个再平衡日的评分数据"}
    prices = prices.copy()
    prices["Date"] = pd.to_datetime(prices["Date"]).dt.date
    mkt_ret = _market_series(prices)
    periods = []
    hold: dict[str, set[str]] = {"B1": set(), "B2": set(), "B3": set(), "ALL": set()}
    for i in range(len(rdates) - 1):
        d0, d1 = rdates[i], rdates[i + 1]
        sc = scores_all[scores_all["Date"] == d0]
        fwd = forward_returns(prices, d0, d1)
        rec = {"start": d0, "end": d1, "n_univ": int(len(sc)), "mkt": float(mkt_ret.get((d0, d1), np.nan))}
        rec["mkt"] = _mkt_period(prices, d0, d1)
        elig = sc[~sc["excluded"]]
        # 篮子组合
        for bk in ("B1", "B2", "B3"):
            sel = elig[elig[f"in_{bk}"] & (elig[f"rank_{bk}"] <= top_n)]
            codes = set(sel["Code"])
            r = fwd.reindex(list(codes)).dropna()
            turnover = len(codes ^ hold[bk]) / max(1, len(codes | hold[bk])) if (codes or hold[bk]) else 0.0
            cost = _cost(sel, costs) * turnover
            rec[f"{bk}_n"] = int(len(r))
            rec[f"{bk}_ret"] = float(r.mean() - cost) if len(r) else np.nan
            rec[f"{bk}_hit"] = float((r > 0).mean()) if len(r) else np.nan
            rec[f"{bk}_turnover"] = turnover
            hold[bk] = codes
        # 分层检验：各维度分 5 层
        for dim in ("score_value", "score_growth", "score_quality", "score_trend", "score_oversold"):
            s = elig[["Code", dim]].dropna()
            if len(s) < 50:
                continue
            q = pd.qcut(s[dim].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
            for lv in (1, 2, 3, 4, 5):
                codes = s.loc[q == lv, "Code"]
                r = fwd.reindex(list(codes)).dropna()
                rec[f"{dim}_q{lv}"] = float(r.mean()) if len(r) else np.nan
        periods.append(rec)
    per = pd.DataFrame(periods)
    summary = {"periods": int(len(per)), "start": str(per["start"].min()), "end": str(per["end"].max()), "freq": freq, "top_n": top_n}
    for bk in ("B1", "B2", "B3"):
        summary[bk] = _stats(per[f"{bk}_ret"], per["mkt"], per[f"{bk}_hit"], per[f"{bk}_turnover"], per[f"{bk}_n"])
    summary["market"] = _stats(per["mkt"], per["mkt"], None, None, None)
    layers = {}
    for dim in ("score_value", "score_growth", "score_quality", "score_trend", "score_oversold"):
        cols = [f"{dim}_q{lv}" for lv in (1, 2, 3, 4, 5) if f"{dim}_q{lv}" in per.columns]
        if cols:
            layers[dim] = {c.split("_")[-1]: float(per[c].mean()) for c in cols}
            layers[dim]["monotonic"] = bool(all(np.diff([per[c].mean() for c in cols]) >= 0)) if len(cols) == 5 else None
    summary["layers"] = layers
    return {"summary": summary, "periods": per}


def _cost(sel: pd.DataFrame, costs: dict) -> float:
    if sel.empty:
        return 0.0
    adv = sel["adv20_turnover"]
    tier = np.where(adv >= 5e8, costs.get("high_liquidity", 0.001), np.where(adv >= 5e7, costs.get("mid", 0.002), costs.get("low", 0.003)))
    return float(np.mean(tier)) * 2  # 买卖双边


def _market_series(prices: pd.DataFrame) -> dict:
    return {}


def _mkt_period(prices: pd.DataFrame, d0: date, d1: date) -> float:
    p0 = prices[prices["Date"] == d0].set_index("Code")
    p1 = prices[prices["Date"] == d1].set_index("Code")
    common = p0.index.intersection(p1.index)
    if len(common) == 0:
        return np.nan
    r = p1.loc[common, "AdjC"] / p0.loc[common, "AdjC"] - 1
    w = p0.loc[common, "MktCap"].fillna(0).clip(lower=0)
    return float(np.average(r.fillna(0), weights=w)) if w.sum() > 0 else float(r.mean())


def _stats(ret: pd.Series, mkt: pd.Series, hit: pd.Series | None, turnover: pd.Series | None, n: pd.Series | None) -> dict:
    r = ret.dropna()
    if r.empty:
        return {}
    periods_per_year = 12
    cum = float((1 + r).prod() - 1)
    years = len(r) / periods_per_year
    cagr = (1 + cum) ** (1 / years) - 1 if years > 0 else np.nan
    vol = float(r.std() * np.sqrt(periods_per_year))
    dd = (1 + r).cumprod()
    mdd = float((dd / dd.cummax() - 1).min())
    excess = (ret - mkt).dropna()
    return {"cum_return": cum, "cagr": float(cagr), "vol": vol, "sharpe": float(cagr / vol) if vol else np.nan, "max_drawdown": mdd,
            "avg_excess_vs_mkt": float(excess.mean()) if len(excess) else np.nan, "excess_hit_rate": float((excess > 0).mean()) if len(excess) else np.nan,
            "avg_hit_rate": float(hit.dropna().mean()) if hit is not None and len(hit.dropna()) else np.nan,
            "avg_turnover": float(turnover.dropna().mean()) if turnover is not None and len(turnover.dropna()) else np.nan,
            "avg_n": float(n.dropna().mean()) if n is not None and len(n.dropna()) else np.nan}
