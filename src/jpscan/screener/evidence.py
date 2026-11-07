"""证据家族判定（规划文档 4.11 节，定义全部来自 config/metric_registry.yaml）。

每只股票每天每个家族得到 hit / miss / fail / unknown / inactive：
- unknown：已知指标数少于 min_known。数据缺失绝不算 hit，也绝不算 miss。
- inactive：行业、市场或公司覆盖后这个家族不适用，不计入覆盖率分母。
- 同一家族内的指标合起来只算一份证据。

新增或删除指标：改注册表，并让 features 提供与指标 id 同名的列。列不存在时该指标是未知，不会报错，也不会算通过。
这里只有「规则类型」的实现和少量绑定（回退列、事件列后缀、fail 规则），没有给任何指标开特例。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import pget
from ..features.registry import load_registry

HIT, MISS, FAIL, UNKNOWN, INACTIVE = 2, 1, -1, 0, -2
STATUS_NAME = {HIT: "hit", MISS: "miss", FAIL: "fail", UNKNOWN: "unknown", INACTIVE: "inactive"}

# 没有前瞻窗口的回退列：指标为空时改用回退列，只与同业的回退值比较并标低置信（registry val_fwd_ep）
FALLBACK_COLUMN = {"val_fwd_ep": "ep_ttm"}
# 排序用的 anchor 连续分位怎么聚合（registry family_policy.ranking）：mean_hit=已命中指标分位的平均；best=较好的分位
ANCHOR_AGG = {"relative_value": "mean_hit", "guidance_change": "best"}


def _peer_pct(values: np.ndarray, higher_better: bool, s33: np.ndarray, s17: np.ndarray, mask: np.ndarray, min_group: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """同业分位，越小越好（0 附近是同业最好）。S33 → S17 → 全市场回退；返回 (pct, 同业名, 同业已知只数)。"""
    n = len(values)
    pct = np.full(n, np.nan)
    grp = np.full(n, None, dtype=object)
    cnt = np.full(n, np.nan)
    ok = mask & ~np.isnan(values)
    if not ok.any():
        return pct, grp, cnt
    s = pd.DataFrame({"v": values[ok], "s33": s33[ok], "s17": s17[ok]})
    def ranks(by):
        g = s.groupby(by, dropna=False, sort=False)["v"] if by is not None else s["v"]
        return g.rank(ascending=not higher_better), g.transform("count") if by is not None else pd.Series(len(s), index=s.index)
    r33, n33 = ranks("s33")
    r17, n17 = ranks("s17")
    rall, nall = ranks(None)
    use33 = (n33 >= min_group).to_numpy()
    use17 = (n17 >= min_group).to_numpy()
    p = np.where(use33, r33 / n33, np.where(use17, r17 / n17, rall / nall))
    c = np.where(use33, n33, np.where(use17, n17, nall))
    g_lab = np.where(use33, s["s33"].astype(object), np.where(use17, s["s17"].astype(object), "全市场"))
    pct[ok], grp[ok], cnt[ok] = p, g_lab, c
    return pct, grp, cnt


class _Ctx:
    """当日横截面的共用数组。"""

    def __init__(self, df: pd.DataFrame, params: dict, reg: dict):
        self.df = df
        self.params = params
        self.reg = reg
        self.n = len(df)
        self.s33 = df["S33Nm"].astype(object).to_numpy() if "S33Nm" in df.columns else np.full(self.n, None, dtype=object)
        self.s17 = df["S17Nm"].astype(object).to_numpy() if "S17Nm" in df.columns else self.s33
        self.codes = df["Code"].astype(str).to_numpy(dtype=object)
        market_keys = reg.get("market_keys", {})
        self.mkt = df["Mkt"].astype(str).to_numpy(dtype=object) if "Mkt" in df.columns else np.full(self.n, "", dtype=object)
        self.market_keys = market_keys
        common = df["is_common_stock"].fillna(False).astype(bool).to_numpy() if "is_common_stock" in df.columns else np.ones(self.n, dtype=bool)
        main = df["in_tse_main"].fillna(False).astype(bool).to_numpy() if "in_tse_main" in df.columns else np.ones(self.n, dtype=bool)
        self.universe = common & main
        self.core = (df["in_core"].fillna(False).astype(bool).to_numpy() if "in_core" in df.columns else np.ones(self.n, dtype=bool)) & self.universe
        self.min_group = int(pget(params, "peer_group.min_group_size", 20))
        self.applicable, self.inactive, self.gates_off = self._overrides()

    def col(self, name: str) -> np.ndarray:
        if name not in self.df.columns:
            return np.full(self.n, np.nan)
        return pd.to_numeric(self.df[name], errors="coerce").to_numpy(dtype=float)

    def _overrides(self):
        """sector / market / firm 覆盖叠加：metric → 是否适用；family → 是否 inactive；gate → 是否关闭。"""
        reg = self.reg
        fams = reg["families"]
        applicable = {f: {r["metric"]: np.ones(self.n, dtype=bool) for r in cfg["threshold"]["hit_rules"]} for f, cfg in fams.items()}
        inactive = {f: np.zeros(self.n, dtype=bool) for f in fams}
        gates_off: dict[str, np.ndarray] = {}

        def apply(mask: np.ndarray, spec: dict) -> None:
            if not mask.any():
                return
            for fam, ch in (spec.get("family_changes") or {}).items():
                if fam not in fams:
                    continue
                if ch.get("inactive"):
                    inactive[fam] |= mask
                for m in ch.get("drop") or []:
                    if m in applicable[fam]:
                        applicable[fam][m] &= ~mask
            for g in spec.get("gates_off") or []:
                gates_off.setdefault(g, np.zeros(self.n, dtype=bool))
                gates_off[g] |= mask

        for sector, spec in (reg.get("sector_overrides") or {}).items():
            apply(self.s33 == sector, spec)
        for market, spec in (reg.get("market_overrides") or {}).items():
            key = self.market_keys.get(market)
            if key:
                apply(self.mkt == key, spec)
        for code, spec in (reg.get("firm_overrides") or {}).items():
            apply(self.codes == str(code), spec)
        return applicable, inactive, gates_off

    def gate_off(self, gate: str) -> np.ndarray:
        return self.gates_off.get(gate, np.zeros(self.n, dtype=bool))


# ---------- 门槛 ----------
def resolve_gates(df: pd.DataFrame, ctx: _Ctx) -> dict[str, np.ndarray]:
    """每个门槛：1=触发，0=未触发，NaN=未知（放行并列入未检查）。关闭的门槛记 0。需要当日横截面的在这里算。"""
    ev = pget(ctx.params, "evidence", {}) or {}
    gates = {g: ctx.col(g) for g in ("gate_one_off", "gate_div_cover", "gate_distress", "gate_inorganic", "gate_disclosure_delay", "gate_guidance_absent", "gate_zombie",
                                     "gate_debt_coverage", "gate_peak_earnings", "gate_tangible_book", "rev_cause_class")}
    gates["guid_remaining_growth"] = ctx.col("guid_remaining_flag")
    # 稀释：股数同比大增，或 12 个月内有增发类公告
    shares = ctx.col("gate_dilution_shares")
    recent = df["recent_dilution"].map({True: 1.0, False: 0.0}).to_numpy(dtype=float) if "recent_dilution" in df.columns else np.full(ctx.n, np.nan)
    gates["gate_dilution"] = np.where((shares == 1) | (recent == 1), 1.0, np.where((shares == 0) | (recent == 0), 0.0, np.nan))
    # 行业内安全线：EqAR 处于行业内最低 share 且低于 abs 才算
    eq = ctx.col("equity_ratio")
    share = float(ev.get("sector_equity_bottom_share", 0.10))
    abs_line = float(ev.get("sector_equity_abs_max", 0.30))
    eq_pct, _, _ = _peer_pct(eq, False, ctx.s33, ctx.s17, ctx.universe, ctx.min_group)
    # eq_pct 越小 = EqAR 越高；最低 share 对应 pct > 1 - share
    with np.errstate(invalid="ignore"):
        trig = (eq_pct > 1 - share) & (eq < abs_line)
    gates["gate_sector_equity"] = np.where(np.isnan(eq) | np.isnan(eq_pct), np.nan, trig.astype(float))
    # 未达后强气：上期没完成期初目标，新目标增速 > max(2×行业中位, 下限)
    growth = ctx.col("f_op_growth")
    prior = ctx.col("gate_miss_prior")
    med = pd.Series(growth).where(ctx.universe).groupby(ctx.s33, dropna=False).transform("median").to_numpy()
    floor = float(ev.get("miss_bullish_growth_floor", 0.10))
    mult = float(ev.get("miss_bullish_median_mult", 2.0))
    with np.errstate(invalid="ignore"):
        bullish = growth > np.maximum(mult * np.nan_to_num(med, nan=0.0), floor)
    gates["gate_miss_then_bullish"] = np.where(np.isnan(prior) | np.isnan(growth), np.nan, ((prior == 1) & bullish).astype(float))
    for g, arr in gates.items():
        off = ctx.gate_off(g)
        if off.any():
            arr = arr.copy()
            arr[off] = 0.0
            gates[g] = arr
    return gates


# ---------- 指标 → 是否命中 ----------
def _metric_rule(ctx: _Ctx, family: str, rule: dict, gates: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """按规则类型算一个指标。返回 hit（1/0/NaN）、pct（越小越好）、value、grp、n、basis、low_conf。"""
    metric = rule["metric"]
    kind = rule["rule"]
    meta = ctx.reg["metrics_by_id"].get(metric, {})
    higher = meta.get("direction", "higher_better") != "lower_better"
    th = rule.get("value")
    n = ctx.n
    value = ctx.col(metric)
    app = ctx.applicable[family][metric]
    hit = np.full(n, np.nan)
    pct = np.full(n, np.nan)
    grp = np.full(n, None, dtype=object)
    cnt = np.full(n, np.nan)
    basis = np.full(n, "", dtype=object)
    low_conf = np.zeros(n, dtype=bool)

    if kind == "peer_best_pct":
        pct, grp, cnt = _peer_pct(value, higher, ctx.s33, ctx.s17, ctx.universe & app, ctx.min_group)
        basis = np.where(~np.isnan(value), "primary", "")
        fb = FALLBACK_COLUMN.get(metric)
        if fb is not None and fb in ctx.df.columns:
            alt = ctx.col(fb)
            alt_pct, alt_grp, alt_n = _peer_pct(alt, higher, ctx.s33, ctx.s17, ctx.universe & app, ctx.min_group)
            use = np.isnan(value) & ~np.isnan(alt_pct)
            pct = np.where(use, alt_pct, pct)
            grp = np.where(use, alt_grp, grp)
            cnt = np.where(use, alt_n, cnt)
            value = np.where(use, alt, value)
            basis = np.where(use, "fallback", basis)
            low_conf = use
        with np.errstate(invalid="ignore"):
            hit = np.where(np.isnan(pct), np.nan, (pct <= float(th)).astype(float))
    elif kind == "core_best_pct":
        mask = ctx.core & app
        pct, grp, cnt = _peer_pct(value, higher, np.full(n, "流动性核心池", dtype=object), np.full(n, "流动性核心池", dtype=object), mask, 1)
        with np.errstate(invalid="ignore"):
            hit = np.where(np.isnan(value), np.nan, ((pct <= float(th)) & ctx.core).astype(float))
    elif kind == "group_best_pct":
        pct = ctx.col(f"{metric}_pct")
        with np.errstate(invalid="ignore"):
            hit = np.where(np.isnan(pct), np.nan, (pct <= float(th)).astype(float))
        hit = np.where(np.isnan(value), 0.0, hit)     # 窗口内没有事件 = 没有信号
        basis = ctx.df["guid_innovation_basis"].astype(object).to_numpy() if metric == "guid_forecast_innovation" and "guid_innovation_basis" in ctx.df.columns else basis
        grp = np.where(pd.notna(basis) & (basis != ""), "同口径公司", None)
    elif kind == "event_up_top_pct":
        pct = ctx.col(f"{metric}_pct")
        n_up = ctx.col("guid_revision_n_up") if metric == "guid_revision_event" else ctx.col(f"{metric}_n_up")
        confirm = ctx.col("guid_revision_confirm") if metric == "guid_revision_event" else ctx.col(f"{metric}_confirm")
        min_abs = float(rule.get("min_abs", 0.0))
        min_events = int(rule.get("min_events", 0))
        with np.errstate(invalid="ignore"):
            big = value >= min_abs
            top = (n_up < min_events) | (pct <= float(th))
            ok = big & top & (confirm != 0)
        hit = np.where(np.isnan(value), 0.0, ok.astype(float))
        low_conf = (hit == 1) & np.isnan(confirm)
        grp = np.where(~np.isnan(value), "窗口内上调的公司", None)
        cnt = n_up
    elif kind == "min_value":
        with np.errstate(invalid="ignore"):
            hit = np.where(np.isnan(value), np.nan, (value >= float(th)).astype(float))
    elif kind == "any_up_event":
        hit = np.where(np.isnan(value), np.nan, (value > 0).astype(float))
    else:
        raise ValueError(f"未知的规则类型 {kind}（指标 {metric}）")

    gate_name = rule.get("unless_gate")
    blocked = np.zeros(n, dtype=bool)
    if gate_name:
        g = gates.get(gate_name)
        if g is not None:
            blocked = (g == 1) & (hit == 1)
            if metric in FALLBACK_COLUMN:     # TTM 回退不适用 gate_one_off（一次性利润门看的是预想）
                blocked &= basis != "fallback"
            hit = np.where(blocked, 0.0, hit)
    hit = np.where(app, hit, np.nan)
    return {"hit": hit, "pct": pct, "value": value, "grp": grp, "n": cnt, "basis": basis, "low_conf": low_conf, "blocked": blocked}


def evaluate_metrics(df: pd.DataFrame, ctx: _Ctx, gates: dict[str, np.ndarray]) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    out: dict[str, dict] = {}
    for fam, cfg in ctx.reg["families"].items():
        out[fam] = {r["metric"]: {**_metric_rule(ctx, fam, r, gates), "rule": r} for r in cfg["threshold"]["hit_rules"]}
    return out


def _fail_mask(ctx: _Ctx, family: str, drop: str | None) -> tuple[np.ndarray, np.ndarray]:
    """registry 的 fail_rule 是文字，这里绑定实现；阈值在 params.evidence。返回 (fail, 低置信)。"""
    zeros = np.zeros(ctx.n, dtype=bool)
    if family == "business_quality" and drop != "qual_fscore7":
        f = ctx.col("qual_fscore7")
        with np.errstate(invalid="ignore"):
            return f <= float(pget(ctx.params, "evidence.fscore_fail_max", 2)), zeros
    if family == "guidance_change" and drop != "guid_revision_event" and "guid_fail" in ctx.df.columns:
        fail = ctx.df["guid_fail"].fillna(False).astype(bool).to_numpy()
        low = ctx.df["guid_fail_low_conf"].fillna(False).astype(bool).to_numpy() if "guid_fail_low_conf" in ctx.df.columns else zeros
        return fail, low & fail
    return zeros, zeros


def family_status(ctx: _Ctx, metrics: dict, drop: str | None = None) -> dict[str, np.ndarray]:
    """由指标命中算家族状态。drop 不为空时把这个指标当作不存在（留一法用）。"""
    out = {}
    for fam, cfg in ctx.reg["families"].items():
        th = cfg["threshold"]
        min_known, min_hits = int(th["min_known"]), int(th["min_hits"])
        names = [m for m in metrics[fam] if m != drop]
        n = ctx.n
        if names:
            hits = np.column_stack([metrics[fam][m]["hit"] for m in names])
            apps = np.column_stack([ctx.applicable[fam][m] for m in names])
            n_app = apps.sum(axis=1)
            known = (~np.isnan(hits)).sum(axis=1)
            n_hit = np.nansum(hits, axis=1)
        else:
            n_app = known = n_hit = np.zeros(n)
        no_fc = cfg.get("no_forecast_means") == "unknown"
        status = np.where(n_hit >= min_hits, HIT, MISS).astype(np.int8)
        status[known < min_known] = UNKNOWN
        cov = cfg.get("coverage_metric")
        if cov and cov != drop and cov in metrics[fam]:
            status[np.isnan(metrics[fam][cov]["hit"])] = UNKNOWN
        if no_fc and "has_forecast" in ctx.df.columns:
            status[~ctx.df["has_forecast"].fillna(False).astype(bool).to_numpy()] = UNKNOWN
        fail, _ = _fail_mask(ctx, fam, drop)
        status[fail] = FAIL
        status[(n_app < min_known) | ctx.inactive[fam]] = INACTIVE
        out[fam] = status
    return out


def coverage_by_market(ctx: _Ctx, metrics: dict, status: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """每个家族、每个市场的已知比例（inactive 不计入分母）。设了 coverage_metric 的只看那个指标。"""
    res: dict[str, dict[str, float]] = {}
    for fam, cfg in ctx.reg["families"].items():
        cov_metric = cfg.get("coverage_metric")
        res[fam] = {}
        for mkt in np.unique(ctx.mkt[ctx.universe]) if ctx.universe.any() else []:
            mask = ctx.universe & (ctx.mkt == mkt) & (status[fam] != INACTIVE)
            if not mask.any():
                res[fam][str(mkt)] = 0.0
                continue
            if cov_metric and cov_metric in metrics[fam]:
                res[fam][str(mkt)] = float((~np.isnan(metrics[fam][cov_metric]["hit"][mask])).mean())
            else:
                res[fam][str(mkt)] = float((status[fam][mask] != UNKNOWN).mean())
    return res


# ---------- 篮子 ----------
def bucket_entry(ctx: _Ctx, status: dict[str, np.ndarray], coverage: dict, metrics: dict, gates: dict[str, np.ndarray], eligible: np.ndarray) -> dict[str, dict]:
    """按 registry.buckets 判定入池。返回 {bucket: {"entry": bool 数组, "fails": 每行的未通过原因码列表, "low_conf": 每行的低置信说明}}。"""
    reg = ctx.reg
    floor = float(pget(reg, "family_policy.coverage_floor", 0.60))
    min_families = int(pget(ctx.params, "evidence.min_families", 3))
    n_hit = np.sum([status[f] == HIT for f in status], axis=0)
    cyc = np.isin(ctx.s33, list(reg.get("cyclical_sectors") or [])) | np.isin(ctx.codes, [str(c) for c in (reg.get("cyclical_codes") or [])])
    df = ctx.df
    out = {}

    def covered(fam: str) -> np.ndarray:
        cov = coverage.get(fam, {})
        return np.array([cov.get(m, 0.0) >= floor for m in ctx.mkt], dtype=bool)

    for bk, b in reg["buckets"].items():
        anchor = b.get("anchor_family")
        if not anchor:
            continue    # B0 不评分，事件触发，单独呈现
        fails: list[list[str]] = [[] for _ in range(ctx.n)]
        lows: list[list[str]] = [[] for _ in range(ctx.n)]

        def mark(mask: np.ndarray, code: str, target=fails) -> None:
            for i in np.flatnonzero(mask):
                target[i].append(code)

        mark(~ctx.universe, "not_universe")
        mark(~ctx.core, "not_core")
        mark(~eligible, "hard_excluded")
        mark(np.isin(ctx.s33, list(b.get("excluded_sectors") or [])), "excluded_sector")
        ex_mkts = [reg.get("market_keys", {}).get(m) for m in (b.get("excluded_markets") or [])]
        mark(np.isin(ctx.mkt, ex_mkts), "excluded_market")
        # anchor 必须 hit；该市场覆盖不足时暂停，不能跳过
        anchor_cov = covered(anchor)
        mark(~anchor_cov, f"anchor_suspended:{anchor}")
        mark(anchor_cov & (status[anchor] != HIT), f"anchor_miss:{anchor}")
        for fam in b.get("required_when_covered") or []:
            cov = covered(fam)
            mark(cov & (status[fam] != HIT), f"required_miss:{fam}")
            mark(~cov, f"low_conf_uncovered:{fam}", lows)
        one_of = list(b.get("at_least_one_of") or [])
        if one_of:
            cov_any = np.any([covered(f) for f in one_of], axis=0)
            hit_any = np.any([status[f] == HIT for f in one_of], axis=0)
            mark(cov_any & ~hit_any, "confirm_miss")
            mark(~cov_any, "low_conf_confirm_uncovered", lows)
            if b.get("cyclical_rule") and "shareholder_yield" in one_of:
                others = [f for f in one_of if f != "shareholder_yield"]
                other_hit = np.any([status[f] == HIT for f in others], axis=0) if others else np.zeros(ctx.n, dtype=bool)
                mark(cyc & ~other_hit & hit_any, "cyclical_yield_only")
                mark(cyc, "low_conf_cycle_unchecked", lows)
        for fam in b.get("must_not_fail") or []:
            mark(status[fam] == FAIL, f"family_fail:{fam}")
            mark(status[fam] == UNKNOWN, f"low_conf_unknown:{fam}", lows)
        cap = b.get("valuation_cap")
        if cap:
            cap_pct = float(pget(ctx.params, "evidence.valuation_cap_worst_share", 0.30))
            m = cap["metric"]
            fam = next(f for f in metrics if m in metrics[f])
            p = metrics[fam][m]["pct"]
            with np.errstate(invalid="ignore"):
                mark(p > 1 - cap_pct, f"valuation_cap:{m}")
            mark(np.isnan(p), f"low_conf_unknown_metric:{m}", lows)
        for g in b.get("extra_gates") or []:
            if g == "rev_intact_test":
                st = df["intact_status"].astype(object).to_numpy() if "intact_status" in df.columns else np.full(ctx.n, "unknown", dtype=object)
                mark(st == "fail", "intact_fail")
                mark(st == "unknown", "intact_unknown")     # 一项都不知道时不能算完好
            elif g == "rev_cause_class":
                w = df["cause_wait"].fillna(False).astype(bool).to_numpy() if "cause_wait" in df.columns else np.zeros(ctx.n, dtype=bool)
                mark(w, "cause_wait")
                if "hard_fail" in df.columns:
                    mark(df["hard_fail"].fillna(False).astype(bool).to_numpy(), "cause_hard_fail")
        # 硬规则：至少 min_families 个家族 hit（unknown 不算）
        mark(n_hit < min_families, f"min_families:{min_families}")
        ok = np.array([len(x) == 0 for x in fails], dtype=bool)
        out[bk] = {"entry": ok, "fails": fails, "low_conf": lows, "anchor": anchor}
    return out


def anchor_strength(ctx: _Ctx, anchor: str, metrics: dict, drop: str | None = None) -> np.ndarray:
    """anchor 家族连续分位（0–1，越大越好）：B1 取已命中估值指标分位的平均，B2 取较好的分位，B3 就是 rev_1m。"""
    agg = ANCHOR_AGG.get(anchor, "mean_hit")
    names = [m for m in metrics[anchor] if m != drop]
    if not names:
        return np.zeros(ctx.n)
    pct = np.column_stack([metrics[anchor][m]["pct"] for m in names])
    hit = np.column_stack([metrics[anchor][m]["hit"] == 1 for m in names])
    best = 1 - pct
    with np.errstate(invalid="ignore"):
        if agg == "best":
            arr = np.where(np.isnan(best), -np.inf, best).max(axis=1)
            res = np.where(np.isneginf(arr), 0.0, arr)
        else:
            val = np.where(hit & ~np.isnan(best), best, np.nan)
            cnt = (~np.isnan(val)).sum(axis=1)
            res = np.where(cnt > 0, np.nansum(val, axis=1) / np.where(cnt > 0, cnt, 1), 0.0)
    return np.clip(res, 0.0, 0.999999)


def composite(ctx: _Ctx, status: dict[str, np.ndarray], anchor: str, metrics: dict, downgraded: np.ndarray, drop: str | None = None) -> np.ndarray:
    """排序键（registry family_policy.ranking）：命中家族数优先，再看 anchor 家族的分位；触发降级门槛的减 evidence.downgrade_penalty 个家族。

    写成一个数是为了留一法能统一比较：整数部分是家族数，小数部分是 anchor 分位，不做跨家族平均。
    """
    n_hit = np.sum([status[f] == HIT for f in status], axis=0).astype(float)
    penalty = float(pget(ctx.params, "evidence.downgrade_penalty", 1.0))
    return n_hit - penalty * downgraded.astype(float) + anchor_strength(ctx, anchor, metrics, drop)


def _rank_of(score: np.ndarray, mask: np.ndarray, tie: np.ndarray) -> np.ndarray:
    """mask 内按 score 降序排名（1 起）；同分按 tie 降序，保证结果确定。"""
    out = np.full(len(score), np.nan)
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return out
    order = idx[np.lexsort((-np.nan_to_num(tie[idx], nan=0.0), -score[idx]))]
    out[order] = np.arange(1, len(order) + 1)
    return out


def leave_one_out(ctx: _Ctx, metrics: dict, entry: np.ndarray, anchor: str, downgraded: np.ndarray, tie: np.ndarray, top_n: int, mult: float, base_rank: np.ndarray) -> dict[str, np.ndarray]:
    """单指标依赖检验：对每个 score 指标（一视同仁）去掉它后重算家族和排序键，看候选排名掉到哪里。

    入池集合不变（入池规则已保证不靠单一指标）；只看排序对单一指标的依赖。超过 top_n × mult 的移出清单。
    """
    n = ctx.n
    limit = top_n * mult
    all_metrics = [m for fam in metrics for m in metrics[fam]]
    worst = np.array(base_rank, copy=True)
    worst_metric = np.full(n, None, dtype=object)
    critical: list[list[str]] = [[] for _ in range(n)]
    min_fam = int(pget(ctx.params, "evidence.min_families", 3))
    detail: dict[str, np.ndarray] = {}
    for m in all_metrics:
        st = family_status(ctx, metrics, drop=m)
        score = composite(ctx, st, anchor, metrics, downgraded, drop=m)
        rank = _rank_of(score, entry, tie)
        detail[m] = rank
        with np.errstate(invalid="ignore"):
            worse = rank > worst
        worst = np.where(worse, rank, worst)
        worst_metric = np.where(worse, m, worst_metric)
        n_hit = np.sum([st[f] == HIT for f in st], axis=0)
        for i in np.flatnonzero(entry & (n_hit < min_fam)):
            critical[i].append(m)
    with np.errstate(invalid="ignore"):
        dependent = entry & (base_rank <= limit) & (worst > limit)
    return {"worst_rank": worst, "worst_metric": worst_metric, "dependent": dependent, "critical": critical, "detail": detail}


def concentration(ctx: _Ctx, status: dict[str, np.ndarray], metrics: dict) -> tuple[np.ndarray, np.ndarray]:
    """证据集中度：每个命中家族超过门槛的幅度占总和的比例。排序键不是加权平均，所以「占比」按这个幅度算。

    分位规则：幅度 = (门槛 − 分位) / 门槛；是非类规则（F-Score、配当上调）没有连续幅度，记 0.5。
    """
    fams = list(status)
    strength = np.zeros((ctx.n, len(fams)))
    for j, fam in enumerate(fams):
        vals = []
        for m, info in metrics[fam].items():
            th = info["rule"].get("value")
            is_pct = info["rule"]["rule"] in ("peer_best_pct", "group_best_pct", "core_best_pct", "event_up_top_pct") and th
            with np.errstate(invalid="ignore", divide="ignore"):
                margin = np.where(info["hit"] == 1, np.where(np.isnan(info["pct"]), 0.5, np.clip((float(th or 1) - info["pct"]) / float(th or 1), 0, 1)) if is_pct else 0.5, 0.0)
            vals.append(margin)
        strength[:, j] = np.max(vals, axis=0) * (status[fam] == HIT)
    total = strength.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = strength.max(axis=1) / np.where(total > 0, total, np.nan)
    dom = np.array(fams, dtype=object)[strength.argmax(axis=1)]
    return share, np.where(total > 0, dom, None)


# ---------- 汇总成 JSON（reasons 只读这个，不依赖内部数组）----------
def _num(v, nd: int = 6):
    if v is None:
        return None
    try:
        if np.isnan(v):
            return None
    except TypeError:
        return None
    return round(float(v), nd)


def _source_of(metric: str, row: dict, as_of: str) -> str:
    """每个指标的数据来源标注，方便回到原始记录。"""
    val_src = row.get("val_source") or "valuation_daily"
    if metric == "val_pbr":
        return f"{val_src} {as_of}"
    if metric == "val_fwd_ep":
        return f"{val_src} {as_of}"
    if metric in ("val_ocf_yield", "qual_fscore7"):
        return f"fins_summary DiscNo={row.get('fy_disc_no')}（FY 截至 {_d(row.get('fy_end_fy0'))}）"
    if metric == "qual_op_profitability":
        return f"fins_summary DiscNo={row.get('last_stmt_disc_no')}"
    if metric == "yield_dividend":
        return f"fins_summary DiscNo={row.get('DiscNo')}，prices {as_of}"
    if metric == "guid_forecast_innovation":
        return f"fins_summary DiscNo={row.get('guid_innovation_disc_no')}（{_d(row.get('guid_innovation_date'))}）"
    if metric == "guid_revision_event":
        return f"fins_summary DiscNo={row.get('guid_revision_disc_no')}（{_d(row.get('guid_revision_date'))}）"
    if metric == "evt_dividend_change":
        return f"fins_summary DiscNo={row.get('evt_div_disc_no')}（{_d(row.get('evt_div_date'))}）"
    if metric == "rev_1m":
        return f"prices {_d(row.get('rev_1m_start_date'))} 至 {as_of}"
    return f"features_daily {as_of}"


def _d(v) -> str | None:
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    return str(v)[:10]


def build_family_json(df: pd.DataFrame, ctx: _Ctx, metrics: dict, status: dict[str, np.ndarray], gates: dict[str, np.ndarray]) -> list[str]:
    fams = ctx.reg["families"]
    extra = ["val_source", "fy_disc_no", "fy_end_fy0", "last_stmt_disc_no", "DiscNo", "guid_innovation_disc_no", "guid_innovation_date", "guid_revision_disc_no", "guid_revision_date",
             "evt_div_disc_no", "evt_div_date", "rev_1m_start_date", "evt_div_prev", "evt_div_new", "self_pct_pbr", "self_pct_ep", "self_pct_low_conf", "guid_fail_kind", "guid_fail_date",
             "guid_fail_disc_no", "guid_fail_mag", "guid_fail_low_conf", "guid_innovation_basis", "op_prof_low_conf"]
    cols = {c: df[c].to_numpy(dtype=object) if c in df.columns else None for c in extra}
    dates = df["Date"].to_numpy(dtype=object)
    self_map = {"val_pbr": "self_pct_pbr", "val_fwd_ep": "self_pct_ep"}
    out: list[str] = []
    for i in range(ctx.n):
        row = {k: (v[i] if v is not None else None) for k, v in cols.items()}
        as_of = _d(dates[i])
        rec: dict = {}
        for fam, cfg in fams.items():
            st = int(status[fam][i])
            items = {}
            for m, info in metrics[fam].items():
                h = info["hit"][i]
                self_col = self_map.get(m)
                item = {
                    "hit": None if np.isnan(h) else int(h), "value": _num(info["value"][i]), "pct": _num(info["pct"][i], 4), "grp": info["grp"][i] if info["grp"][i] is not None else None,
                    "n": _num(info["n"][i], 0), "basis": str(info["basis"][i]) if info["basis"][i] not in (None, "") else None, "low_conf": bool(info["low_conf"][i]),
                    "blocked": bool(info["blocked"][i]), "source": _source_of(m, row, as_of),
                }
                if self_col and row.get(self_col) is not None:
                    item["self_pct"] = _num(row[self_col], 4)
                    item["self_low_conf"] = bool(row.get("self_pct_low_conf"))
                if m == "evt_dividend_change":
                    item["div_prev"] = _num(row.get("evt_div_prev"), 4)
                    item["div_new"] = _num(row.get("evt_div_new"), 4)
                    item["event_date"] = _d(row.get("evt_div_date"))
                if m == "guid_forecast_innovation":
                    item["event_date"] = _d(row.get("guid_innovation_date"))
                    item["basis"] = row.get("guid_innovation_basis")
                if m == "guid_revision_event":
                    item["event_date"] = _d(row.get("guid_revision_date"))
                if m == "qual_op_profitability" and bool(row.get("op_prof_low_conf") or False):
                    item["low_conf"] = True
                items[m] = item
            rec[fam] = {"status": STATUS_NAME[st], "metrics": items}
            if fam == "guidance_change" and st == FAIL:
                rec[fam]["fail"] = {"kind": row.get("guid_fail_kind"), "date": _d(row.get("guid_fail_date")), "disc_no": row.get("guid_fail_disc_no"), "mag": _num(row.get("guid_fail_mag")),
                                    "low_conf": bool(row.get("guid_fail_low_conf"))}
        out.append(json.dumps(rec, ensure_ascii=False, separators=(",", ":"), default=str))
    return out


def evaluate(df: pd.DataFrame, params: dict, reg: dict | None = None) -> pd.DataFrame:
    """主入口：输入当日特征（已有 excluded 列），写回家族状态、篮子、排序、留一法结果和 JSON 列。"""
    reg = reg or load_registry(params)
    ctx = _Ctx(df, params, reg)
    gates = resolve_gates(df, ctx)
    metrics = evaluate_metrics(df, ctx, gates)
    status = family_status(ctx, metrics)
    coverage = coverage_by_market(ctx, metrics, status)
    eligible = ~df["excluded"].fillna(False).astype(bool).to_numpy() if "excluded" in df.columns else np.ones(ctx.n, dtype=bool)
    entries = bucket_entry(ctx, status, coverage, metrics, gates, eligible)
    # 降级门槛
    down_list = list(pget(params, "evidence.downgrade_gates", ["gate_distress", "gate_sector_equity", "guid_remaining_growth", "gate_inorganic", "gate_dilution"]))
    downgraded = np.any([gates[g] == 1 for g in down_list if g in gates], axis=0) if down_list else np.zeros(ctx.n, dtype=bool)
    top_n = int(pget(params, "bucket.top_n", 40))
    mult = float(pget(params, "evidence.loo_rank_mult", 2))
    tie = ctx.col("mcap")
    loo_records: dict[str, dict] = {}
    df = df.copy()
    n_hit = np.sum([status[f] == HIT for f in status], axis=0)
    df["family_pass_count"] = n_hit
    for fam, st in status.items():
        df[f"family_{fam}"] = pd.Series(st, index=df.index).map(STATUS_NAME)
    for g, arr in gates.items():
        df[f"gate_status_{g}"] = arr
    df["downgraded"] = downgraded
    for bk, e in entries.items():
        comp = composite(ctx, status, e["anchor"], metrics, downgraded)
        base_rank = _rank_of(comp, e["entry"], tie)
        loo = leave_one_out(ctx, metrics, e["entry"], e["anchor"], downgraded, tie, top_n, mult, base_rank)
        keep = e["entry"] & ~loo["dependent"]
        df[f"entry_{bk}"] = e["entry"]
        df[f"in_{bk}"] = keep
        df[f"composite_{bk}"] = comp
        df[f"rank_{bk}"] = _rank_of(comp, keep, tie)
        df[f"loo_removed_{bk}"] = loo["dependent"]
        df[f"loo_worst_rank_{bk}"] = np.where(e["entry"], loo["worst_rank"], np.nan)
        df[f"loo_worst_metric_{bk}"] = np.where(e["entry"], loo["worst_metric"], None)
        loo_records[bk] = {**loo, "base_rank": base_rank, "fails": e["fails"], "low_conf": e["low_conf"]}
    bks = list(entries)
    in_any = df[[f"in_{b}" for b in bks]].to_numpy()
    ranks = df[[f"rank_{b}" for b in bks]].to_numpy(dtype=float)
    comps = df[[f"composite_{b}" for b in bks]].to_numpy(dtype=float)
    r_filled = np.where(in_any & ~np.isnan(ranks), ranks, np.inf)
    best = r_filled.argmin(axis=1)
    has = np.isfinite(r_filled.min(axis=1))
    rows = np.arange(ctx.n)
    df["bucket"] = np.where(has, np.array(bks, dtype=object)[best], None)
    df["bucket_rank"] = np.where(has, ranks[rows, best], np.nan)
    df["composite_score"] = np.where(has, comps[rows, best], np.nan)
    df["in_top_n"] = df["bucket_rank"] <= top_n
    df["buckets"] = [",".join(np.array(bks)[m]) for m in in_any]
    df["resonance"] = in_any.sum(axis=1) >= 2
    share, dom = concentration(ctx, status, metrics)
    max_share = float(pget(params, "evidence.max_dimension_share", 0.5))
    df["max_family_share"] = share
    df["dominant_family"] = dom
    df["concentration_flag"] = share > max_share
    # JSON 列
    df["family_status"] = build_family_json(df, ctx, metrics, status, gates)
    cov_json = json.dumps(coverage, ensure_ascii=False, sort_keys=True)
    df["family_coverage"] = cov_json
    checks = []
    for i in range(ctx.n):
        rec = {}
        for bk, r in loo_records.items():
            base = r["base_rank"][i]
            rec[bk] = {"fails": r["fails"][i], "low_conf": r["low_conf"][i], "base_rank": _num(base, 0), "worst_rank": _num(r["worst_rank"][i], 0),
                       "worst_metric": r["worst_metric"][i], "critical": r["critical"][i], "loo_removed": bool(r["dependent"][i]),
                       "ranks": {m: _num(a[i], 0) for m, a in r["detail"].items()} if not np.isnan(base) else {}}
        checks.append(json.dumps(rec, ensure_ascii=False, separators=(",", ":"), default=str))
    df["bucket_check"] = checks
    return df


def coverage_table(df: pd.DataFrame) -> dict:
    return json.loads(df["family_coverage"].iloc[0]) if len(df) and "family_coverage" in df.columns else {}
