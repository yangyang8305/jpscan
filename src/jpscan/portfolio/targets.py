"""估值区间、买入/卖出区间与触发条件（规划文档 6.3 节）。全部为区间与情景，不是收益承诺。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import pget


def fair_value_methods(row: pd.Series, peers: pd.DataFrame, params: dict) -> list[dict]:
    """三法：PBR 回归、PER 回归、EV/EBITDA（数据不足时跳过）。peers = 同日同业（S33）行。"""
    out: list[dict] = []
    roe_th = float(pget(params, "targets.pbr_roe_adjust_threshold", 0.08))
    # PBR 回归
    if pd.notna(row.get("bps")) and row["bps"] > 0:
        sector_pbr = peers["pbr"].median() if peers is not None and len(peers) >= 10 else np.nan
        own_hist = row.get("pbr_hist_median", np.nan)
        cands = [x for x in (sector_pbr, own_hist) if pd.notna(x) and x > 0]
        if cands:
            target_pbr = min(cands)
            roe = row.get("roe_avg", np.nan)
            adj = min(1.0, roe / roe_th) if pd.notna(roe) and roe > 0 else 0.6
            target_pbr *= adj if (pd.isna(roe) or roe < roe_th) else 1.0
            out.append({"method": "pbr_reversion", "target_multiple": float(target_pbr), "fair_value": float(target_pbr * row["bps"]),
                        "assumptions": f"目标 PBR = min(行业中位 {sector_pbr:.2f}, 自身历史 {own_hist if pd.notna(own_hist) else float('nan'):.2f}) × ROE 调整 {adj:.2f}"})
    # PER 回归
    eps = row.get("eps_fwd_adj", np.nan)
    if pd.notna(eps) and eps > 0:
        sector_per = peers["per_fwd"].median() if peers is not None and len(peers) >= 10 else np.nan
        own_hist = row.get("per_hist_median", np.nan)
        cands = [x for x in (sector_per, own_hist) if pd.notna(x) and x > 0]
        if cands:
            target_per = min(cands)
            g = row.get("f_op_growth", np.nan)
            if pd.notna(g) and g > 0:
                target_per = min(target_per, float(pget(params, "targets.peg_cap", 1.5)) * g * 100) if g * 100 * 1.5 < target_per else target_per
            out.append({"method": "per_reversion", "target_multiple": float(target_per), "fair_value": float(target_per * eps),
                        "assumptions": f"目标予想 PER = min(行业中位 {sector_per:.1f}, 自身历史) ；予想 EPS = {eps:.1f}"})
    return out


def scenarios(methods: list[dict], params: dict) -> dict:
    sc = params.get("targets", {}).get("scenario", {})
    vals = np.array([m["fair_value"] for m in methods if np.isfinite(m["fair_value"]) and m["fair_value"] > 0])
    if len(vals) == 0:
        return {"bear": np.nan, "base": np.nan, "bull": np.nan}
    q = lambda p: float(np.quantile(vals, p))
    return {
        "bear": q(float(sc.get("bear", {}).get("quantile", 0.25))) * (1 + float(sc.get("bear", {}).get("earnings_adj", -0.1))),
        "base": q(float(sc.get("base", {}).get("quantile", 0.5))) * (1 + float(sc.get("base", {}).get("earnings_adj", 0.0))),
        "bull": q(float(sc.get("bull", {}).get("quantile", 0.75))) * (1 + float(sc.get("bull", {}).get("earnings_adj", 0.1))),
    }


def entry_exit(row: pd.Series, scen: dict, bucket: str, params: dict, regime: dict | None = None) -> dict:
    t = params.get("targets", {})
    sm = float(t.get("safety_margin", {}).get(bucket or "B1", 0.25))
    base, bear, bull = scen.get("base"), scen.get("bear"), scen.get("bull")
    price = float(row["C"])
    atr = float(row.get("atr14", np.nan)) if pd.notna(row.get("atr14", np.nan)) else np.nan
    atr_mult = float((regime or {}).get("atr_stop_mult") or pget(t, "exit.atr_stop_mult", 2.5))
    zone_high = base * (1 - sm) if pd.notna(base) else np.nan
    zone_low = bear if pd.notna(bear) else np.nan
    if pd.notna(zone_high) and pd.notna(zone_low) and zone_low > zone_high:
        zone_low = zone_high * 0.9
    conds = []
    if bucket == "B3":
        conds.append("企稳信号 ≥ 2（高于 20 日低点 ≥ 3 日 / ATR 收敛 / 量能回归）")
    if bucket == "B2":
        conds.append("阶段为 uptrend 或 base，且相对市场 63 日强弱 ≥ 行业中位")
    blocked = None
    dte = row.get("days_to_next_earnings", np.nan)
    if pd.notna(dte) and 0 <= dte <= int(t.get("no_entry_pre_earnings_days", 5)):
        blocked = f"決算発表前 {int(dte)} 日，禁止新建仓"
    exit_ = {
        "trim_at": base if pd.notna(base) else np.nan,
        "exit_at": bull if pd.notna(bull) else np.nan,
        "fundamental_stops": [f"通期営業利益予想下修 > {int(float(pget(t, 'exit.guidance_cut_stop', 0.2)) * 100)}%", "出现任一硬排除红旗",
                              f"连续 {int(pget(t, 'exit.progress_dev_stop_quarters', 2))} 个季度進捗率偏差 < {int(float(pget(t, 'exit.progress_dev_stop_pp', -0.1)) * 100)}pp"],
        "technical_stop": (price - atr_mult * atr) if (bucket in ("B2", "B3") and pd.notna(atr)) else np.nan,
        "time_stop_months": int(pget(t, "exit.time_stop_months", 12)),
    }
    return {
        "price": price, "zone_high": zone_high, "zone_low": zone_low, "tranches": list(t.get("tranches", [0.4, 0.3, 0.3])),
        "conditions": conds, "blocked_reason": blocked, "price_position": (price / base - 1) if pd.notna(base) and base > 0 else np.nan,
        "upside_base": (base / price - 1) if pd.notna(base) else np.nan, "upside_bull": (bull / price - 1) if pd.notna(bull) else np.nan,
        "downside_bear": (bear / price - 1) if pd.notna(bear) else np.nan, "exit": exit_,
    }


def build_signals(scores_day: pd.DataFrame, hist_medians: pd.DataFrame | None, params: dict, regime: dict | None = None) -> pd.DataFrame:
    """对候选池逐股计算估值区间与买卖区间。hist_medians: Code -> pbr_hist_median, per_hist_median（自身历史中位）。"""
    df = scores_day.copy()
    if hist_medians is not None and not hist_medians.empty:
        df = df.merge(hist_medians, on="Code", how="left")
    else:
        df["pbr_hist_median"] = np.nan
        df["per_hist_median"] = np.nan
    rows = []
    for s33, grp in df.groupby("S33"):
        peers = grp
        for _, row in grp.iterrows():
            if not row.get("in_top_n", False):
                continue
            methods = fair_value_methods(row, peers, params)
            scen = scenarios(methods, params)
            ee = entry_exit(row, scen, row.get("bucket"), params, regime)
            rows.append({"Code": row["Code"], "Date": row["Date"], "bucket": row.get("bucket"), "methods": methods, "scenarios": scen, **ee})
    return pd.DataFrame(rows)
