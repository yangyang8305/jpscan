"""组合构建（规划文档 6.4、14 章）：篮子配额 → 约束贪心填充 → 分层等权 × 波动率调整 → 变动建议与风险指标。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import pget


def construct(scores: pd.DataFrame, params: dict, regime: dict | None = None, current: pd.DataFrame | None = None) -> dict:
    p = params.get("portfolio", {})
    n_min, n_max = int(pget(p, "n_positions.min", 15)), int(pget(p, "n_positions.max", 30))
    max_single = float(p.get("max_single", 0.07))
    max_sector = float(p.get("max_sector17", 0.25))
    max_growth_mkt = float(p.get("max_growth_market", 0.20))
    alloc = dict(p.get("bucket_allocation", {"B1": 0.5, "B2": 0.3, "B3": 0.2}))
    tilt_max = float(p.get("bucket_tilt_max", 0.15))
    cash = float((regime or {}).get("cash_buffer") or pget(p, "cash_buffer.min", 0.05))
    tilt = (regime or {}).get("bucket_tilt") or {}
    for k, v in tilt.items():
        alloc[k] = alloc.get(k, 0) + max(-tilt_max, min(tilt_max, float(v)))
    tot = sum(max(0.0, v) for v in alloc.values())
    alloc = {k: max(0.0, v) / tot for k, v in alloc.items()}
    invest = 1.0 - cash
    df = scores[(~scores["excluded"]) & scores["bucket"].notna()].copy()
    # 禁止建仓窗口与红旗
    if "days_to_next_earnings" in df.columns:
        df = df[~df["days_to_next_earnings"].between(0, 5)]
    df = df[~df.get("red_flag", pd.Series(False, index=df.index)).fillna(False).astype(bool)]
    liq = params.get("liquidity", {})
    part = float(liq.get("participation_rate", 0.05))
    exit_days = float(liq.get("exit_days", 5))
    chosen: list[dict] = []
    sector_w: dict[str, float] = {}
    growth_w = 0.0
    n_target = n_max
    slots = {bk: max(1, round(n_target * alloc.get(bk, 0))) for bk in ("B1", "B2", "B3")}
    for bk in ("B1", "B2", "B3"):
        sub = df[df["bucket"] == bk].sort_values("bucket_rank")
        used = 0
        for _, row in sub.iterrows():
            if used >= slots[bk]:
                break
            if any(c["Code"] == row["Code"] for c in chosen):
                continue
            base_w = invest / n_target
            sec = str(row.get("S17"))
            if sector_w.get(sec, 0.0) + base_w > max_sector:
                continue
            if str(row.get("Mkt")) == "0113" and growth_w + base_w > max_growth_mkt:
                continue
            chosen.append({"Code": row["Code"], "CoName": row.get("CoName"), "bucket": bk, "rank": row["bucket_rank"], "S17": sec,
                           "Mkt": row.get("Mkt"), "rvol60": row.get("rvol60"), "adv20_turnover": row.get("adv20_turnover"),
                           "composite": row.get("composite_score"), "price": row.get("C")})
            sector_w[sec] = sector_w.get(sec, 0.0) + base_w
            if str(row.get("Mkt")) == "0113":
                growth_w += base_w
            used += 1
    if not chosen:
        return {"positions": pd.DataFrame(), "cash": 1.0, "note": "no eligible candidates"}
    pos = pd.DataFrame(chosen)
    # 分层等权 × 波动率倒数
    top_tier = int(p.get("top_tier_n", 10))
    tier_mult = np.where(pos["rank"] <= top_tier, float(p.get("top_tier_multiplier", 1.2)), 1.0)
    vol = pos["rvol60"].fillna(pos["rvol60"].median()).clip(lower=0.1)
    raw = tier_mult * (vol.median() / vol)
    w = raw / raw.sum() * invest
    w = w.clip(upper=max_single)
    w = w / w.sum() * invest
    pos["weight"] = w.values
    # 流动性检查（相对资金规模未知 → 报告可退出天数）
    pos["liquidity_cap_jpy"] = pos["adv20_turnover"] * part * exit_days
    pos["cash_buffer"] = cash
    changes = []
    if current is not None and not current.empty:
        cur = set(current["Code"])
        new = set(pos["Code"])
        for c in sorted(new - cur):
            changes.append({"Code": c, "action": "add", "reason": "entered top ranks"})
        for c in sorted(cur - new):
            changes.append({"Code": c, "action": "remove", "reason": "left candidate pool or constraints"})
    exposures = {
        "sector17": pos.groupby("S17")["weight"].sum().round(4).to_dict(),
        "market": pos.groupby("Mkt")["weight"].sum().round(4).to_dict(),
        "bucket": pos.groupby("bucket")["weight"].sum().round(4).to_dict(),
        "hhi": float((pos["weight"] ** 2).sum()),
        "n": int(len(pos)),
    }
    return {"positions": pos, "cash": cash, "exposures": exposures, "changes": changes, "allocation": alloc}
