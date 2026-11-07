"""特征构建：价格面板 + 财务状态 + 估值 + 主数据 → mart.features_daily。

只存原始指标值和事件窗口结果。同业分位不进这张表：它只在筛选当日才用得上，存进去两年就是 200 多万行 × 几十列的字符串和小数；
screener/evidence.py 在当日的横截面上现算。

列名约定：registry 里 status=score 或 gate 的指标，特征列与指标 id 同名（val_pbr、qual_fscore7、gate_one_off ...）；
旧名字（pbr、per_fwd ...）保留给报告、组合模块展示用。

只增量重算时用 --from：技术指标向前多读 hist_days 个自然日，财务状态和分割因子永远用全历史，结果只写 >= start 的行。
"""
from __future__ import annotations

import logging
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..config import pget
from .fundamentals import build_states_and_events, fscore7, load_statements
from .panel import cum_after_at, load_adj_events, load_price_panel, technical_features, trading_dates, worst_day_in_window
from .registry import hit_rule, load_registry

log = logging.getLogger("jpscan.features")

METRIC_VERSION = "F.v2"
HIST_DAYS = 420     # 技术指标需要的自然日历史（250 个交易日窗口 + 余量）
FINANCIAL_S33 = {"銀行業", "保険業", "証券･商品先物取引業", "その他金融業"}
MASTER_COLS = ["CoName", "S17", "S17Nm", "S33", "S33Nm", "Mkt", "MktNm", "ScaleCat", "ProdCat", "Mrgn"]
VAL_COLS = ["Date", "Code", "EPS", "FwdEPS", "BPS", "ROE", "FwdROE", "PER", "FwdPER", "PBR", "MktCap"]


# ---------- 读取 ----------
def load_master_snapshots(store) -> pd.DataFrame:
    m = store.read("staging", "security_master_snapshots")
    if m.empty:
        return m
    m["Date"] = pd.to_datetime(m["Date"]).astype("datetime64[us]")
    m["Code"] = m["Code"].astype(str)
    return m.sort_values(["Code", "Date"])


def _asof_positions(codes: pd.Series, when: pd.Series, r_codes: pd.Series, r_when: pd.Series) -> np.ndarray:
    """as-of 查找：对每个 (code, when) 找 r_when <= when 的最后一行在右表的行号；找不到为 -1。只合并键，不复制数据列。"""
    left = pd.DataFrame({"Code": codes.to_numpy(dtype=object), "_d": pd.to_datetime(when).astype("datetime64[us]").to_numpy(), "_i": np.arange(len(codes))}).sort_values("_d")
    right = pd.DataFrame({"Code": r_codes.to_numpy(dtype=object), "_d": pd.to_datetime(r_when).astype("datetime64[us]").to_numpy(), "_pos": np.arange(len(r_codes))})
    right = right.dropna(subset=["_d"]).sort_values("_d", kind="stable")
    m = pd.merge_asof(left, right, on="_d", by="Code", direction="backward")
    out = np.full(len(codes), -1, dtype=np.int64)
    ok = m["_pos"].notna().to_numpy()
    out[m["_i"].to_numpy()[ok]] = m["_pos"].to_numpy()[ok].astype(np.int64)
    return out


def _take(right: pd.DataFrame, pos: np.ndarray, cols, index) -> dict[str, pd.Series]:
    """按行号取右表的列；-1 的位置是空。"""
    r = right.reset_index(drop=True)
    target = pd.Index(np.where(pos >= 0, pos, -1))
    return {c: r[c].reindex(target).set_axis(index) for c in cols}


def _assign(feat: pd.DataFrame, cols: dict[str, pd.Series]) -> pd.DataFrame:
    """一次性并入多列，避免逐列插入造成碎片。"""
    return pd.concat([feat.drop(columns=[c for c in cols if c in feat.columns]), pd.DataFrame(cols, index=feat.index)], axis=1)


def _left_join_keys(feat: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """按 (Code, Date) 左连接，只算行号再取列。frame 的键必须唯一。"""
    f = frame.reset_index(drop=True)
    keys = pd.DataFrame({"Code": feat["Code"].to_numpy(dtype=object), "Date": pd.to_datetime(feat["Date"]).astype("datetime64[us]").to_numpy()})
    rk = pd.DataFrame({"Code": f["Code"].astype(str).to_numpy(dtype=object), "Date": pd.to_datetime(f["Date"]).astype("datetime64[us]").to_numpy(), "_pos": np.arange(len(f))})
    pos = keys.merge(rk, on=["Code", "Date"], how="left")["_pos"].to_numpy()
    pos = np.where(np.isnan(pos), -1, pos).astype(np.int64)
    cols = [c for c in f.columns if c not in ("Code", "Date")]
    return _assign(feat, _take(f, pos, cols, feat.index))


def attach_master(panel: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """按日期 as-of 取主数据；早于首个快照的行用最早快照（前视，回测须披露，registry validation.warnings）。"""
    if master.empty:
        return _assign(panel, {c: pd.Series(None, index=panel.index, dtype=object) for c in MASTER_COLS})
    cols = [c for c in MASTER_COLS if c in master.columns]
    pos = _asof_positions(panel["Code"], panel["Date"], master["Code"], master["Date"])
    if (pos < 0).any():
        first_row = pd.Series(np.arange(len(master)), index=master["Code"].to_numpy(dtype=object))
        first_row = first_row[~first_row.index.duplicated(keep="first")]
        miss = pos < 0
        pos[miss] = pd.Series(panel["Code"].to_numpy(dtype=object)[miss]).map(first_row).fillna(-1).to_numpy(dtype=np.int64)
    return _assign(panel, _take(master, pos, cols, panel.index))


def load_valuation(store) -> pd.DataFrame:
    if not store.exists("staging", "valuation_daily"):
        return pd.DataFrame(columns=VAL_COLS)
    v = store.read("staging", "valuation_daily", columns=VAL_COLS)
    v["Date"] = pd.to_datetime(v["Date"]).astype("datetime64[us]")
    v["Code"] = v["Code"].astype(str)
    return v


def load_edinet_items(store) -> pd.DataFrame:
    """edinet_key_items → 每期一行（有息负债、折旧、利息等）。按 disc_date 的次日生效，流量项按期间天数折算成全年。"""
    if not store.exists("staging", "edinet_key_items"):
        return pd.DataFrame()
    it = store.read("staging", "edinet_key_items")
    if it.empty:
        return it
    it["Code"] = it["Code"].astype(str)
    it["consolidated"] = it["consolidated"].fillna(False).astype(bool)
    # 同一期同一项目：优先連結，其次最新文档
    it = it.sort_values(["Code", "period_end", "item", "consolidated", "disc_date"]).drop_duplicates(["Code", "period_end", "item"], keep="last")
    wide = it.pivot(index=["Code", "period_end"], columns="item", values="value")
    disc = it.groupby(["Code", "period_end"])[["disc_date", "docID"]].last()
    wide = wide.join(disc).reset_index()
    for c in ("interest_bearing_debt", "short_term_loans", "current_lt_loans", "long_term_loans", "bonds", "current_bonds", "commercial_paper", "depreciation",
              "interest_expense", "cash_and_deposits", "cash_equivalents", "short_term_securities", "operating_income"):
        if c not in wide.columns:
            wide[c] = np.nan
    parts = wide[["short_term_loans", "current_lt_loans", "long_term_loans", "bonds", "current_bonds", "commercial_paper"]]
    wide["debt"] = wide["interest_bearing_debt"].where(wide["interest_bearing_debt"].notna(), parts.sum(axis=1, min_count=1))
    wide["ed_cash"] = wide["cash_and_deposits"].where(wide["cash_and_deposits"].notna(), wide["cash_equivalents"])
    ann = pd.Series(1.0, index=wide.index)
    if store.exists("staging", "edinet_documents"):
        docs = store.read("staging", "edinet_documents", columns=["docID", "period_start_date", "period_end_date"]).drop_duplicates("docID")
        m = wide[["docID"]].merge(docs, on="docID", how="left")
        days = (pd.to_datetime(m["period_end_date"]) - pd.to_datetime(m["period_start_date"])).dt.days + 1
        ann = (365.0 / days.where(days > 0)).fillna(1.0).clip(upper=4.0).set_axis(wide.index)
    wide["dep_annual"] = wide["depreciation"] * ann
    wide["interest_annual"] = wide["interest_expense"] * ann
    wide["ed_op_annual"] = wide["operating_income"] * ann
    wide["ed_effective"] = (pd.to_datetime(wide["disc_date"]) + pd.Timedelta(days=1)).astype("datetime64[us]")
    wide = wide.sort_values(["Code", "period_end"])
    # 僵尸判定（OP/利息 < 1 连续 3 期）：期数不够时为空，不算通过也不算不通过
    icr = wide["ed_op_annual"] / wide["interest_annual"].where(wide["interest_annual"] > 0)
    lt1 = (icr < 1).astype(float).where(icr.notna())
    wide["gate_zombie"] = lt1.groupby(wide["Code"]).transform(lambda s: s.rolling(3, min_periods=3).sum() == 3).astype(float).where(lt1.groupby(wide["Code"]).transform(lambda s: s.rolling(3, min_periods=3).count()) == 3)
    return wide[["Code", "ed_effective", "period_end", "debt", "ed_cash", "short_term_securities", "dep_annual", "interest_annual", "gate_zombie"]].rename(columns={"period_end": "ed_period_end"})


def _jgb10y(store) -> pd.Series | None:
    if not store.exists("staging", "macro_series"):
        return None
    m = store.read("staging", "macro_series", where="series = 'jgb_10y'")
    if m.empty:
        return None
    s = m.sort_values("date").set_index(pd.to_datetime(m["date"]))["value"].astype(float)
    return s[~s.index.duplicated(keep="last")]


# ---------- 指标（逐行向量化）----------
def compute_fundamental_metrics(d: pd.DataFrame, params: dict, adj_events: pd.DataFrame | None, jgb10y: pd.Series | None = None) -> pd.DataFrame:
    """把状态、估值、价格合成 registry 的指标列。缺什么就是 NaN，不填 0 也不填行业中位。"""
    ev = pget(params, "evidence", {}) or {}
    codes = d["Code"].to_numpy(dtype=object)
    cum_t = d["cum_after"].to_numpy(dtype=float)

    def to_today(col: str) -> np.ndarray:
        """把 col 日期生效的每股值换算到当日口径的系数（B[r]/B[t]）；as-of 查找，不要求 r 在面板里。"""
        return cum_after_at(adj_events, codes, d[col]) / cum_t

    adj_shares = to_today("shares_eff")
    adj_bps = to_today("bps_eff")
    adj_eps = to_today("f_eps_eff")
    adj_div = to_today("f_div_eff")
    d["adj_ps"] = adj_shares
    d["val_source"] = np.where(d["MktCap_val"].notna(), "valuation_daily", "prices+fins_summary")
    with np.errstate(invalid="ignore", divide="ignore"):
        d["shares_net"] = (d["shares_out"] - d["treasury"].fillna(0.0)) / adj_shares
        mcap_own = d["C"] * d["shares_net"]
        mcap_val = d["MktCap_val"] * 1e6
        mcap_px = d["MktCap"] * 1e6
        d["mcap"] = mcap_val.where(mcap_val > 0, mcap_own.where(mcap_own > 0, mcap_px))
        # ---- 估值 ----
        bps_rep = d["bps_reported"] * adj_bps
        d["bps"] = d["BPS_val"].where(d["BPS_val"].notna(), bps_rep)
        pbr = d["PBR"].where(d["PBR"].notna(), d["C"] / d["bps"].where(d["bps"] > 0))
        d["val_pbr"] = pbr.where(pbr > 0)       # BPS<=0 无效，归入困境检查
        d["pbr"] = d["val_pbr"]
        d["eps_fwd_adj"] = d["FwdEPS"].where(d["FwdEPS"].notna(), d["f_eps"] * adj_eps)
        d["val_fwd_ep"] = d["eps_fwd_adj"] / d["C"]
        np_ttm_ep = (d["np_ttm"] / d["mcap"]).where(d["ttm_quarters"] == 4)
        d["ep_ttm"] = (d["EPS"] / d["C"]).where(d["EPS"].notna(), np_ttm_ep)
        d["per_fwd"] = d["FwdPER"].where(d["FwdPER"].notna(), (d["C"] / d["eps_fwd_adj"]).where(d["eps_fwd_adj"] > 0))
        d["per_ttm"] = d["PER"].where(d["PER"].notna(), (d["mcap"] / d["np_ttm"]).where((d["np_ttm"] > 0) & (d["ttm_quarters"] == 4)))
        ann0 = (365.0 / d["fy_days_fy0"].where(d["fy_days_fy0"] > 0)).fillna(1.0)
        ann1 = (365.0 / d["fy_days_fy1"].where(d["fy_days_fy1"] > 0)).fillna(1.0)
        d["val_ocf_yield"] = d["cfo_fy0"] * ann0 / d["mcap"]
        d["fcf_yield"] = (d["cfo_fy0"] + d["cfi_fy0"]) * ann0 / d["mcap"]
        d["div_ps_today"] = d["f_div_ann"] * adj_div
        d["yield_dividend"] = d["div_ps_today"] / d["C"]
        d["div_yield_fwd"] = d["yield_dividend"]
        d["cash_to_mcap"] = d["cash_latest"] / d["mcap"]
        d["val_cash_minus_liabilities"] = (d["cash_latest"] - (d["ta_latest"] - d["eq_latest"])) / d["mcap"]
        d["earnings_yield_fwd"] = d["val_fwd_ep"]
        # ---- EDINET 才有的项：没有数据时全部为空，家族和门槛按未知处理 ----
        for c in ("debt", "ed_cash", "short_term_securities", "dep_annual", "interest_annual", "gate_zombie"):
            if c not in d.columns:
                d[c] = np.nan
        cash_for_ev = d["cash_latest"].where(d["cash_latest"].notna(), d["ed_cash"])
        op_base = d["op_ttm"].where(d["ttm_quarters"] == 4, d["op_fy0"] * ann0)
        ebitda = op_base + d["dep_annual"]
        d["ev_ebitda"] = ((d["mcap"] + d["debt"] - cash_for_ev) / ebitda).where(ebitda > 0)
        d["net_cash_ratio"] = (cash_for_ev + d["short_term_securities"].fillna(0.0) - d["debt"]) / d["mcap"]
        d["net_debt_ebitda"] = ((d["debt"] - cash_for_ev) / ebitda).where(ebitda > 0)
        d["interest_coverage"] = (op_base / d["interest_annual"]).where(d["interest_annual"] > 0)
        # 偿债能力（registry 现为 display，EDINET 到位后转 gate）：净负债/EBITDA 过高或利润不够付息
        bad_nd = float(pget(params, "quality.net_debt_ebitda_bad", 4.0))
        d["gate_debt_coverage"] = ((d["net_debt_ebitda"] > bad_nd) | (d["interest_coverage"] < 1.0)).astype(float).where(d["net_debt_ebitda"].notna() | d["interest_coverage"].notna())
        # ---- 质量 ----
        ta_prev = d["ta_fy1"].where(d["doc_kind_last_stmt_is_fy"], d["ta_fy0"])
        ta_avg = (d["ta_latest"] + ta_prev) / 2
        d["op_prof_low_conf"] = ta_avg.isna() & d["ta_latest"].notna()
        ta_den = ta_avg.where(ta_avg.notna(), d["ta_latest"])
        d["qual_op_profitability"] = (op_base / ta_den).where(ta_den > 0)
        d["op_prof_basis"] = np.where(d["ttm_quarters"] == 4, "ttm", np.where(d["op_fy0"].notna(), "fy", ""))
        eq0 = d["ta_fy0"] * d["eqar_fy0"]
        eq1 = d["ta_fy1"] * d["eqar_fy1"]
        d["roe_fy0"] = d["np_fy0"] * ann0 / pd.concat([eq0, eq1], axis=1).mean(axis=1, skipna=True)
        d["roe_avg"] = d["ROE"].where(d["ROE"].notna(), d["roe_fy0"])
        d["equity_ratio"] = d["eqar_latest"]
        d["accrual_avg"] = (d["np_fy0"] - d["cfo_fy0"]) * ann0 / d["ta_fy0"]
        d["ocf_positive_years"] = (d[["cfo_fy0", "cfo_fy1"]] > 0).sum(axis=1)
        d["ocf_years_available"] = d[["cfo_fy0", "cfo_fy1"]].notna().sum(axis=1)
        d["ocf_to_op"] = (d["cfo_fy0"] / d["op_fy0"]).where(d["op_fy0"] > 0)
        d["dividend_cut"] = (d["div_fy0"] < d["div_fy1"]) & d["div_fy0"].notna() & d["div_fy1"].notna()
        d["negative_equity"] = (d["eq_latest"] <= 0) | (d["eqar_latest"] < 0)
        # 股数变化（还原分割）：把上一财年股数换到 fy0 的生效日口径
        b0 = cum_after_at(adj_events, codes, d["fy_eff_fy0"])
        b1 = cum_after_at(adj_events, codes, d["fy_eff_fy1"])
        d["share_growth_fy"] = d["net_shares_fy0"] / (d["net_shares_fy1"] / (b1 / b0)) - 1
        real_estate = d["S33Nm"].astype(str).eq("不動産業")
        fs = fscore7(d, cfo_items_off=real_estate, share_tol=float(ev.get("fscore_share_tolerance", 0.005)))
        d["qual_fscore7"] = fs["qual_fscore7"]
        d["fscore_n"] = fs["fscore_n"]
        # ---- 成长（只展示）：决算期不是 12 个月时按天数折算并标记；记录实际用的期限 ----
        lo, hi = ev.get("fy_days_normal", [350, 380])
        d["fy_irregular"] = ~d["fy_days_fy0"].between(lo, hi) & d["fy_days_fy0"].notna()
        adjacent = ((d["fy_end_fy0"] - d["fy_end_fy1"]).dt.days).between(250, 500)
        d["rev_growth_1y"] = (d["sales_fy0"] * ann0 / (d["sales_fy1"] * ann1) - 1).where(adjacent)
        d["op_growth_1y"] = (d["op_fy0"] * ann0 / (d["op_fy1"] * ann1) - 1).where(adjacent & (d["op_fy1"] > 0))
        three_ok = ((d["fy_end_fy0"] - d["fy_end_fy3"]).dt.days).between(1000, 1190)
        d["rev_cagr_3y"] = ((d["sales_fy0"] / d["sales_fy3"]) ** (1 / 3) - 1).where(three_ok & (d["sales_fy3"] > 0))
        d["op_cagr_3y"] = ((d["op_fy0"] / d["op_fy3"]) ** (1 / 3) - 1).where(three_ok & (d["op_fy3"] > 0))
        d["growth_horizon"] = np.where(d["rev_cagr_3y"].notna(), "3y", np.where(d["rev_growth_1y"].notna(), "1y", ""))
        d["op_margin_fy0"] = d["op_fy0"] / d["sales_fy0"]
        d["op_margin_chg"] = d["op_margin_fy0"] - d["op_fy1"] / d["sales_fy1"]
        d["f_op_growth"] = (d["f_op"] / (d["op_fy0"] * ann0) - 1).where(d["op_fy0"] > 0)
        d["progress_dev"] = d["progress_rate"] - d["progress_rate_prev_year"]
        d["actual_vs_initial"] = (d["op_fy0"] / d["f_op_initial_fy0"]).where(d["f_op_initial_fy0"] > 0)
        # ---- 门槛（1=触发，0=未触发，NaN=未知）----
        d["gate_one_off"], d["gate_one_off_loss"] = _gate_one_off(d, ev)
        d["gate_div_cover"] = _gate_div_cover(d)
        cond0 = (d["op_fy0"] < 0) & (d["cfo_fy0"] < 0)
        cond1 = (d["op_fy1"] < 0) & (d["cfo_fy1"] < 0)
        known0 = d["op_fy0"].notna() & d["cfo_fy0"].notna()
        known1 = d["op_fy1"].notna() & d["cfo_fy1"].notna()
        d["gate_distress"] = (cond0 & cond1).astype(float).where((known0 & known1) | (known0 & ~cond0))
        d["guid_remaining_growth"], d["guid_remaining_flag"] = _remaining_growth(d, ev)
        d["gate_miss_prior"] = (d["op_fy0"] < d["f_op_initial_fy0"]).astype(float).where(d["op_fy0"].notna() & d["f_op_initial_fy0"].notna() & (d["op_fy0"] > 0))
        ta_yoy = (d["ta_fy0"] / d["ta_fy1"] - 1).abs()
        inorg = d["fy_irregular"] | (d["sig_chg_fy0"].fillna(False).astype(bool) & (ta_yoy >= float(ev.get("inorganic_ta_change", 0.20))))
        d["gate_inorganic"] = inorg.astype(float).where(d["fy_days_fy0"].notna())
        d["gate_dilution_shares"] = (d["share_growth_fy"] >= float(ev.get("dilution_share_growth", 0.10))).astype(float).where(d["share_growth_fy"].notna())
        d["gate_disclosure_delay"] = (d["disc_lag_days"] > float(ev.get("disclosure_lag_days", 45))).astype(float).where(d["disc_lag_days"].notna())
        d["has_forecast"] = d[["f_eps", "f_op", "f_np", "FwdEPS"]].notna().any(axis=1)
        d["gate_guidance_absent"] = (~d["has_forecast"]).astype(float).where(d["effective_date"].notna())
    # ---- 数据质量与宇宙 ----
    d["days_since_report"] = (pd.to_datetime(d["Date"]) - pd.to_datetime(d["last_stmt_disc_date"])).dt.days
    d["stale_financials"] = d["days_since_report"] > float(pget(params, "universe.stale_financials_days", 135))
    d["days_since_any_doc"] = (pd.to_datetime(d["Date"]) - pd.to_datetime(d["DiscDate"])).dt.days
    d["is_financial"] = d["S33Nm"].astype(str).isin(FINANCIAL_S33)
    d["is_common_stock"] = d["ProdCat"].astype(str).eq("011") & d["Code"].astype(str).str.endswith("0")
    d["in_tse_main"] = d["Mkt"].astype(str).isin(["0111", "0112", "0113"])
    liq = params.get("liquidity", {})
    d["in_core"] = (d["adv20_turnover"] >= float(pget(liq, "core.min_adv20_turnover_jpy", 5e7))) & (d["mcap"] >= float(pget(liq, "core.min_mcap_jpy", 5e9)))
    if jgb10y is not None:
        d["jgb10y"] = pd.to_datetime(d["Date"]).map(jgb10y)
        d["jgb10y"] = d["jgb10y"].astype(float)
    else:
        d["jgb10y"] = np.nan
    return d


def _gate_one_off(d: pd.DataFrame, ev: dict) -> tuple[pd.Series, pd.Series]:
    """registry gate_one_off。分母优先经常利益预想，没有时用营业利润预想。"""
    fnp = d["f_np"]
    base = d["f_odp"].where(d["f_odp"].notna(), d["f_op"])
    ratio = fnp / base.where(base != 0)
    th = float(ev.get("one_off_ratio", 0.85))
    flag = (fnp > 0) & ((base <= 0) | (ratio >= th))
    known = fnp.notna() & base.notna()
    gate = flag.astype(float).where(known)
    gate = gate.where(~(fnp <= 0), 0.0)     # 预想亏损不算一次性利润
    low = (d["f_np"] / d["f_odp"].where(d["f_odp"] > 0) <= float(ev.get("one_off_loss_ratio", 0.40))).astype(float).where(d["f_np"].notna() & (d["f_odp"] > 0))
    return gate, low


def _gate_div_cover(d: pd.DataFrame) -> pd.Series:
    """预计利润不够付分红：DPS/EPS>1，或 EPS<=0 仍有分红。不分红算未触发，字段未知算未知。"""
    div, eps = d["f_div_ann"], d["f_eps"]
    flag = (div > 0) & ((eps <= 0) | (div / eps.where(eps > 0) > 1))
    out = flag.astype(float).where(div.notna() & eps.notna())
    return out.where(~(div == 0), 0.0)


def _remaining_growth(d: pd.DataFrame, ev: dict) -> tuple[pd.Series, pd.Series]:
    """registry guid_remaining_growth：只在 2Q、3Q 判定；分母<=0 时看日元缺口；1Q 和 FY 后不判定（未知）。"""
    in_scope = d["latest_q_period"].isin([2, 3])
    gap = d["rem_fc"] - d["rem_ytd"]
    den = d["rem_prev_fy"] - d["rem_prev_ytd"]
    val = (gap / den.where(den > 0) - 1).where(in_scope)
    gap_ratio = (gap / d["rem_fc"].abs().where(d["rem_fc"] != 0)).where(in_scope & (den <= 0))
    th = float(ev.get("remaining_growth_flag", 0.50))
    th_gap = float(ev.get("remaining_gap_flag", 0.50))
    flag = pd.Series(np.nan, index=d.index, dtype=float)
    flag = flag.where(~val.notna(), (val >= th).astype(float))
    flag = flag.where(~gap_ratio.notna(), (gap_ratio > th_gap).astype(float))
    return val, flag


# ---------- 事件 → 日频 ----------
def _rows_of(dates: pd.DatetimeIndex, values, side: str = "left") -> np.ndarray:
    return np.searchsorted(dates.values, pd.to_datetime(values).astype("datetime64[ns]").to_numpy(dtype="datetime64[ns]"), side=side)


def _peer_of(events: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """给事件表补当时的 S33Nm、S17Nm。"""
    if master.empty:
        return events.assign(S33Nm=None, S17Nm=None)
    e = events.copy()
    e["_d"] = pd.to_datetime(e["DiscDate"]).astype("datetime64[us]")
    m = master[["Code", "Date", "S33Nm", "S17Nm"]].rename(columns={"Date": "_d"}).sort_values("_d")
    out = pd.merge_asof(e.sort_values("_d"), m, on="_d", by="Code", direction="backward")
    miss = out["S33Nm"].isna().to_numpy()
    if miss.any():
        first = master.drop_duplicates("Code").set_index("Code")
        out.loc[miss, "S33Nm"] = out.loc[miss, "Code"].map(first["S33Nm"])
        out.loc[miss, "S17Nm"] = out.loc[miss, "Code"].map(first["S17Nm"])
    return out.drop(columns=["_d"])


def forecast_innovation_daily(fy_ev: pd.DataFrame, dates: pd.DatetimeIndex, mcap_w: pd.DataFrame, reg: dict, params: dict, target_rows: np.ndarray, master: pd.DataFrame) -> pd.DataFrame:
    """registry guid_forecast_innovation：相对市值、减同业中位、同口径内排名；排名池 rank_pool_days，信号窗口 DiscDate 后第 2 至 window_days 个交易日。"""
    cols = ["Code", "Date", "guid_forecast_innovation", "guid_forecast_innovation_pct", "guid_forecast_innovation_raw", "guid_innovation_basis", "guid_innovation_disc_no", "guid_innovation_date"]
    if fy_ev.empty:
        return pd.DataFrame(columns=cols)
    rule = hit_rule(reg, "guidance_change", "guid_forecast_innovation")
    pool_n = int(rule.get("rank_pool_days", 250))
    win = int(rule.get("window_days", 60))
    min_group = int(pget(params, "peer_group.min_group_size", 20))
    e = _peer_of(fy_ev, master)
    e = e.sort_values(["Code", "effective_date", "DiscNo"]).drop_duplicates(["Code", "fy_end"], keep="last")   # 同一财年的更正只留最新
    e["disc_row"] = _rows_of(dates, e["DiscDate"])
    e["eff_row"] = _rows_of(dates, e["effective_date"])
    pre = e["disc_row"].to_numpy() - 1
    ci = mcap_w.columns.get_indexer(e["Code"])
    ok = (pre >= 0) & (pre < len(dates)) & (ci >= 0)
    mv = mcap_w.to_numpy()
    cap = np.full(len(e), np.nan)
    cap[ok] = mv[pre[ok], ci[ok]]
    e["raw"] = e["diff"] / np.where(cap > 0, cap, np.nan)
    e = e.dropna(subset=["raw"]).sort_values("eff_row").reset_index(drop=True)
    if e.empty:
        return pd.DataFrame(columns=cols)
    eff = e["eff_row"].to_numpy()
    out: list[pd.DataFrame] = []
    for t in target_rows:
        hi = np.searchsorted(eff, t, side="right")
        lo = np.searchsorted(eff, t - pool_n + 1, side="left")
        if hi <= lo:
            continue
        p = e.iloc[lo:hi]
        active = (p["disc_row"].to_numpy() + 2 <= t) & (t <= p["disc_row"].to_numpy() + win)
        if not active.any():
            continue
        s33 = p.groupby("S33Nm", dropna=False)["raw"]
        s17 = p.groupby("S17Nm", dropna=False)["raw"]
        n33, m33 = s33.transform("count").to_numpy(), s33.transform("median").to_numpy()
        n17, m17 = s17.transform("count").to_numpy(), s17.transform("median").to_numpy()
        med_all = float(p["raw"].median())
        med = np.where(n33 >= min_group, m33, np.where(n17 >= min_group, m17, med_all))
        adj = p["raw"].to_numpy() - med
        tmp = pd.DataFrame({"adj": adj, "basis": p["basis"].to_numpy()})
        g = tmp.groupby("basis")["adj"]
        pct = g.rank(ascending=False, pct=True).where(g.transform("count") >= min_group).to_numpy()
        a = p[active]
        out.append(pd.DataFrame({"Code": a["Code"].to_numpy(), "Date": dates[t], "guid_forecast_innovation": adj[active], "guid_forecast_innovation_pct": pct[active],
                                 "guid_forecast_innovation_raw": a["raw"].to_numpy(), "guid_innovation_basis": a["basis"].to_numpy(),
                                 "guid_innovation_disc_no": a["DiscNo"].to_numpy(), "guid_innovation_date": a["DiscDate"].to_numpy()}))
    if not out:
        return pd.DataFrame(columns=cols)
    res = pd.concat(out, ignore_index=True)
    return res.sort_values(["Code", "Date", "guid_innovation_date"]).drop_duplicates(["Code", "Date"], keep="last")


def revision_events(rev_ev: pd.DataFrame, dates: pd.DatetimeIndex, implied_w: pd.DataFrame, mcap_w: pd.DataFrame, reg: dict, params: dict) -> pd.DataFrame:
    """registry guid_revision_event：优先用隐含净利 FwdEPS*市值/股价 的前后差（已排除回购影响）；估值表还没回填到的日期退回文档 FNP 的差。

    known_row 是这个事件能被看到的最早行：隐含值要等 D+1 更新。
    """
    if rev_ev.empty:
        return rev_ev
    rule = hit_rule(reg, "guidance_change", "guid_revision_event")
    floor = float(rule.get("noise_floor", 0.01))
    half = float(pget(params, "evidence.revision_main_min_share", 0.5))
    e = rev_ev.copy()
    ci = implied_w.columns.get_indexer(e["Code"])
    before = _rows_of(dates, e["DiscDate"], "left") - 1
    after = _rows_of(dates, e["DiscDate"], "right")
    eff = _rows_of(dates, e["effective_date"], "left")
    iw = implied_w.to_numpy()
    mw = mcap_w.to_numpy()
    n_t = len(dates)
    i_before, i_after, cap = (np.full(len(e), np.nan) for _ in range(3))
    ok_b = (before >= 0) & (before < n_t) & (ci >= 0)
    ok_a = (after < n_t) & (ci >= 0)
    i_before[ok_b] = iw[before[ok_b], ci[ok_b]]
    i_after[ok_a] = iw[after[ok_a], ci[ok_a]]
    cap[ok_b] = mw[before[ok_b], ci[ok_b]]
    use_implied = ~np.isnan(i_before) & ~np.isnan(i_after)
    prev = np.where(use_implied, i_before, e["fnp_prev"].to_numpy(dtype=float))
    new = np.where(use_implied, i_after, e["fnp_new"].to_numpy(dtype=float))
    delta = new - prev
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = delta / np.abs(prev)
        mag = delta / np.where(cap > 0, cap, np.nan)
    e["delta_fnp"], e["rel"], e["mag"] = delta, rel, mag
    e["known_row"] = np.where(use_implied, np.maximum(after, eff), eff)
    e["source_kind"] = np.where(use_implied, "implied", "doc")
    e["up"] = rel >= floor
    e["down"] = rel <= -floor
    main_delta = e["main_new"].to_numpy(dtype=float) - e["main_prev"].to_numpy(dtype=float)
    e["main_delta"] = main_delta
    known_main = ~np.isnan(main_delta)
    # 上调确认：主业同向且变动>=净利变动的一半；主业未知时 NaN（命中但标低置信）
    confirm = np.full(len(e), np.nan)
    with np.errstate(invalid="ignore"):
        confirm[known_main] = (main_delta[known_main] > 0) & (main_delta[known_main] >= half * delta[known_main])
    e["confirm"] = confirm
    # 下调才算 fail 的条件：主业也下调，或主业未知（标低置信）
    e["fail_event"] = e["withdrawn"].astype(bool) | (e["down"] & (~known_main | (main_delta < 0)))
    e["fail_low_conf"] = e["down"] & ~known_main
    e["net_only_down"] = e["down"] & known_main & (main_delta >= 0)
    return e[e["up"] | e["fail_event"] | e["net_only_down"]].sort_values("known_row").reset_index(drop=True)


def revision_daily(e: pd.DataFrame, dates: pd.DatetimeIndex, reg: dict, target_rows: np.ndarray) -> pd.DataFrame:
    cols = ["Code", "Date", "guid_revision_event", "guid_revision_event_pct", "guid_revision_n_up", "guid_revision_confirm", "guid_revision_disc_no", "guid_revision_date",
            "guid_fail", "guid_fail_low_conf", "guid_fail_kind", "guid_fail_date", "guid_fail_disc_no", "guid_net_only_down"]
    if e is None or e.empty:
        return pd.DataFrame(columns=cols)
    win = int(hit_rule(reg, "guidance_change", "guid_revision_event").get("window_days", 30))
    known = e["known_row"].to_numpy()
    out: list[pd.DataFrame] = []
    for t in target_rows:
        lo = np.searchsorted(known, t - win + 1, side="left")
        hi = np.searchsorted(known, t, side="right")
        if hi <= lo:
            continue
        w = e.iloc[lo:hi]
        ups = w[w["up"] & w["mag"].notna()]
        frames = {}
        if len(ups):
            pct = ups["mag"].rank(ascending=False, pct=True)
            u = ups.assign(_pct=pct).sort_values("mag").drop_duplicates("Code", keep="last")
            frames["up"] = pd.DataFrame({"Code": u["Code"].to_numpy(), "guid_revision_event": u["mag"].to_numpy(), "guid_revision_event_pct": u["_pct"].to_numpy(),
                                         "guid_revision_n_up": len(ups), "guid_revision_confirm": u["confirm"].to_numpy(), "guid_revision_disc_no": u["DiscNo"].to_numpy(),
                                         "guid_revision_date": u["DiscDate"].to_numpy()})
        fails = w[w["fail_event"]]
        if len(fails):
            f = fails.drop_duplicates("Code", keep="last")
            frames["fail"] = pd.DataFrame({"Code": f["Code"].to_numpy(), "guid_fail": True, "guid_fail_low_conf": f["fail_low_conf"].to_numpy(dtype=bool),
                                           "guid_fail_kind": np.where(f["withdrawn"].to_numpy(dtype=bool), "withdrawn", "down"), "guid_fail_date": f["DiscDate"].to_numpy(),
                                           "guid_fail_disc_no": f["DiscNo"].to_numpy(), "guid_fail_mag": f["mag"].to_numpy()})
        no = w.loc[w["net_only_down"], "Code"].drop_duplicates()
        if len(no):
            frames["net"] = pd.DataFrame({"Code": no.to_numpy(), "guid_net_only_down": True})
        if not frames:
            continue
        cur = None
        for fr in frames.values():
            cur = fr if cur is None else cur.merge(fr, on="Code", how="outer")
        cur["Date"] = dates[t]
        out.append(cur)
    if not out:
        return pd.DataFrame(columns=cols)
    return pd.concat(out, ignore_index=True)


def dividend_daily(div_ev: pd.DataFrame, dates: pd.DatetimeIndex, adj_events: pd.DataFrame | None, reg: dict, target_rows: np.ndarray) -> pd.DataFrame:
    """registry evt_dividend_change：相邻文档、同一目标财年比较（已还原分割）；窗口内有上调/下调。"""
    cols = ["Code", "Date", "evt_div_up", "evt_div_down", "evt_div_disc_no", "evt_div_date", "evt_div_prev", "evt_div_new"]
    if div_ev.empty:
        return pd.DataFrame(columns=cols)
    win = int(hit_rule(reg, "capital_action", "evt_dividend_change").get("window_days", 30))
    e = div_ev.copy()
    # 分割前的旧预想换到新文档的股数口径
    codes = e["Code"].to_numpy(dtype=object)
    factor = cum_after_at(adj_events, codes, e["prev_eff"]) / cum_after_at(adj_events, codes, e["effective_date"])
    e["prev_adj"] = e["div_prev"] * np.nan_to_num(factor, nan=1.0)
    tol = 1e-9
    e["up"] = e["div_new"] > e["prev_adj"] * (1 + tol) + tol
    e["down"] = e["div_new"] < e["prev_adj"] * (1 - tol) - tol
    e = e[e["up"] | e["down"]].copy()
    if e.empty:
        return pd.DataFrame(columns=cols)
    e["known_row"] = _rows_of(dates, e["effective_date"])
    e = e[e["known_row"] < len(dates)]
    tset = np.zeros(len(dates) + 1, dtype=bool)
    tset[target_rows] = True
    reps = np.full(len(e), win)
    idx = np.repeat(np.arange(len(e)), reps)
    rows = e["known_row"].to_numpy()[idx] + np.tile(np.arange(win), len(e))
    keep = (rows < len(dates))
    idx, rows = idx[keep], rows[keep]
    keep = tset[rows]
    idx, rows = idx[keep], rows[keep]
    x = e.iloc[idx]
    long = pd.DataFrame({"Code": x["Code"].to_numpy(), "_row": rows, "up": x["up"].to_numpy(dtype=bool), "down": x["down"].to_numpy(dtype=bool), "evt_div_disc_no": x["DiscNo"].to_numpy(),
                         "evt_div_date": x["DiscDate"].to_numpy(), "evt_div_prev": x["prev_adj"].to_numpy(), "evt_div_new": x["div_new"].to_numpy(), "_k": x["known_row"].to_numpy()})
    long = long.sort_values(["Code", "_row", "_k"])
    g = long.groupby(["Code", "_row"], sort=False)
    res = g.tail(1).copy()
    flags = g.agg(evt_div_up=("up", "max"), evt_div_down=("down", "max")).reset_index()
    res = res.drop(columns=["up", "down", "_k"]).merge(flags, on=["Code", "_row"])
    res["Date"] = dates[res["_row"].to_numpy()]
    return res.drop(columns=["_row"])


# ---------- 价格错位相关（宽矩阵）----------
def ex_dividend_add_back(states: pd.DataFrame, feat_codes: pd.Index, dates: pd.DatetimeIndex, adj_events: pd.DataFrame | None, cal_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """估算的除权行：Free 没有除息日，按期末前第 1 个交易日估算（registry 文件头），金额用此前已知的预想 FDiv2Q、FDivFY。返回后复权口径的「每行分红」宽表。"""
    out = pd.DataFrame(0.0, index=dates, columns=feat_codes)
    if states.empty or len(dates) == 0:
        return out
    s = states[["Code", "effective_date", "f_fy_end", "f_div_2q", "f_div_fy", "f_div_eff"]].dropna(subset=["f_fy_end", "effective_date"]).copy()
    s["f_fy_end"] = pd.to_datetime(s["f_fy_end"])
    rows: list[dict] = []
    # 对每个目标财年、每种分红，取最后一条在除权日之前生效的预想
    for (code, fy_end), g in s.groupby(["Code", "f_fy_end"], sort=False):
        for label, ex_period, col in (("FY", fy_end, "f_div_fy"), ("2Q", fy_end - pd.DateOffset(months=6), "f_div_2q")):
            r = np.searchsorted(cal_dates.values, np.datetime64(ex_period), side="right") - 1    # 期末或之前最后一个交易日 = 权利确定日
            ex_row = r - 1                                                                       # 权利落日：再往前 1 个交易日
            if ex_row < 0 or r >= len(cal_dates):
                continue
            ex_date = cal_dates[ex_row]
            if ex_date < dates[0] or ex_date > dates[-1]:
                continue
            known = g[pd.to_datetime(g["effective_date"]) < ex_date]
            known = known[known[col].notna()]
            if known.empty:
                continue
            last = known.iloc[-1]
            if last[col] > 0:
                rows.append({"Code": code, "ex_date": ex_date, "dps": float(last[col]), "eff": last["f_div_eff"] if pd.notna(last["f_div_eff"]) else last["effective_date"]})
    if not rows:
        return out
    x = pd.DataFrame(rows)
    x = x[x["Code"].isin(feat_codes)]
    if x.empty:
        return out
    # 后复权口径：DPS（当时股数）× 其生效日之后的 AdjFactor 之积，与 adj_c 同口径
    b = cum_after_at(adj_events, x["Code"].to_numpy(dtype=object), x["eff"])
    x["dps_adj"] = x["dps"] * b
    ci = feat_codes.get_indexer(x["Code"])
    ri = dates.get_indexer(x["ex_date"])
    ok = (ci >= 0) & (ri >= 0)
    arr = out.to_numpy(copy=True)
    np.add.at(arr, (ri[ok], ci[ok]), x["dps_adj"].to_numpy()[ok])
    return pd.DataFrame(arr, index=dates, columns=feat_codes)


def add_rev_1m(d: pd.DataFrame, states: pd.DataFrame, adj_events: pd.DataFrame | None, st_docs: pd.DataFrame, dates: pd.DatetimeIndex, params: dict) -> pd.DataFrame:
    """rev_1m（加回估算分红、减同业中位）与 rev_cause_class 的业绩型代理。"""
    ev = pget(params, "evidence", {}) or {}
    window = int(ev.get("rev_window_days", 21))
    min_group = int(pget(params, "peer_group.min_group_size", 20))
    codes = pd.Index(sorted(d["Code"].unique()))
    dates = pd.DatetimeIndex(dates)
    idx_r = dates.get_indexer(pd.to_datetime(d["Date"]))
    idx_c = codes.get_indexer(d["Code"])
    adj = np.full((len(dates), len(codes)), np.nan)
    adj[idx_r, idx_c] = d["adj_c"].to_numpy(dtype=float)
    adj_w = pd.DataFrame(adj, index=dates, columns=codes).ffill(limit=10)
    div = ex_dividend_add_back(states, codes, dates, adj_events, dates)
    add_back = div.rolling(window, min_periods=1).sum() / adj_w.shift(window)
    d["rev_div_add_back"] = add_back.to_numpy()[idx_r, idx_c]
    d["rev_ret_21d"] = d["ret_21d"] + pd.Series(d["rev_div_add_back"], index=d.index).fillna(0.0)
    d["rev_1m_start_date"] = dates[np.maximum(idx_r - window, 0)]
    # 同业中位（S33，不足 min_group 回退 S17，再回退全市场）；只用普通株主市场
    u = d["is_common_stock"].to_numpy(dtype=bool) & d["in_tse_main"].to_numpy(dtype=bool)
    sub = d.loc[u, ["Date", "S33Nm", "S17Nm", "rev_ret_21d"]]
    g33 = sub.groupby(["Date", "S33Nm"])["rev_ret_21d"]
    g17 = sub.groupby(["Date", "S17Nm"])["rev_ret_21d"]
    gm = sub.groupby("Date")["rev_ret_21d"]
    med = g33.transform("median").where(g33.transform("count") >= min_group, g17.transform("median").where(g17.transform("count") >= min_group, gm.transform("median")))
    d["rev_1m"] = np.nan
    d.loc[u, "rev_1m"] = (sub["rev_ret_21d"] - med).to_numpy()

    # ---- 下跌归因代理：窗口内最大单日相对跌幅发生在决算文档 −1 至 +3 自然日 ----
    r1 = np.full((len(dates), len(codes)), np.nan)
    r1[idx_r, idx_c] = d["ret_1d"].to_numpy(dtype=float)
    rel = pd.DataFrame(r1, index=dates, columns=codes)
    rel = rel.sub(rel.median(axis=1), axis=0)
    near = np.full(r1.shape, np.iinfo(np.int32).max, dtype=np.int32)
    if not st_docs.empty:
        dd = st_docs[st_docs["Code"].isin(codes)]
        cc = codes.get_indexer(dd["Code"])
        dv = pd.to_datetime(dd["DiscDate"]).astype("datetime64[ns]").to_numpy(dtype="datetime64[ns]")
        lo = np.searchsorted(dates.values, dv - np.timedelta64(1, "D"), side="left")
        hi = np.searchsorted(dates.values, dv + np.timedelta64(3, "D"), side="right")
        known = np.searchsorted(dates.values, dv, side="left")
        for c, a, b, k in zip(cc, lo, hi, known):
            if b > a:
                near[a:b, c] = np.minimum(near[a:b, c], k)
    th = float(ev.get("cause_proxy_drop", -0.05))
    t_rows = np.arange(len(dates))[:, None]
    c_cols = np.arange(len(codes))[None, :]
    wait21, wait60 = int(ev.get("cause_wait_days", 21)), int(ev.get("cause_wait_days_negative_guidance", 60))
    res = {}
    for w in (wait21, wait60):
        k, v = worst_day_in_window(rel, w)
        safe_k = np.where(k >= 0, k, 0)
        with np.errstate(invalid="ignore"):
            hit = (v <= th) & (near[safe_k, c_cols] <= t_rows) & (k >= 0)
        res[w] = (hit, safe_k)
    hit21, k21 = res[wait21]
    hit60, _ = res[wait60]
    d["cause_proxy_earnings"] = hit21[idx_r, idx_c]
    d["cause_proxy_date"] = dates[k21[idx_r, idx_c]]
    d["cause_proxy_earnings_60"] = hit60[idx_r, idx_c]
    d["cause_worst_rel_ret"] = worst_day_in_window(rel, window)[1][idx_r, idx_c]
    return d


def self_history_percentile(wide: pd.DataFrame, window: int = 1250, min_periods: int = 250) -> pd.DataFrame:
    """自身历史分位（当前值在过去 window 个交易日中的位置）。"""
    return wide.rolling(window, min_periods=min_periods).rank(pct=True, method="max")


def add_self_history(d: pd.DataFrame, val: pd.DataFrame, dates: pd.DatetimeIndex, params: dict) -> pd.DataFrame:
    """PBR、E/P 相对自身历史的分位；不足 self_history_min_days 的只算低置信（用不用由注册表决定，现在都只展示）。"""
    win = int(pget(params, "peer_group.self_history_window_days", 1250))
    min_days = int(pget(params, "peer_group.self_history_min_days", 500))
    d["self_pct_pbr"] = np.nan
    d["self_pct_ep"] = np.nan
    d["self_hist_days"] = 0
    if val.empty:
        d["self_pct_low_conf"] = True
        return d
    codes = pd.Index(sorted(d["Code"].unique()))
    v = val[val["Code"].isin(codes)]
    all_dates = pd.DatetimeIndex(sorted(set(v["Date"].unique()) | set(dates)))
    px = d[["Code", "Date", "C"]]
    # PBR 直接用估值表；E/P 需要当日收盘
    pbr_w = v.pivot(index="Date", columns="Code", values="PBR").reindex(index=all_dates, columns=codes)
    pbr_w = pbr_w.where(pbr_w > 0)
    eps_w = v.pivot(index="Date", columns="Code", values="FwdEPS").reindex(index=all_dates, columns=codes)
    c_w = px.pivot(index="Date", columns="Code", values="C").reindex(index=all_dates, columns=codes)
    ep_w = eps_w / c_w
    ri = all_dates.get_indexer(pd.to_datetime(d["Date"]))
    ci = codes.get_indexer(d["Code"])
    mp = max(60, min_days // 2)
    d["self_pct_pbr"] = self_history_percentile(pbr_w, win, mp).to_numpy()[ri, ci]
    d["self_pct_ep"] = self_history_percentile(ep_w, win, mp).to_numpy()[ri, ci]
    d["self_hist_days"] = pbr_w.notna().cumsum().to_numpy()[ri, ci]
    d["self_pct_low_conf"] = d["self_hist_days"] < min_days
    return d


# ---------- 主流程 ----------
KEEP_DROP = {
    "O", "H", "L", "adj_vo", "rem_fc", "rem_ytd", "rem_prev_fy", "rem_prev_ytd", "cfi_fy0", "fy_eff_fy0", "fy_eff_fy1", "tr_mixed", "shares_eff", "bps_eff", "f_eps_eff", "f_div_eff",
    "EPS", "FwdEPS", "BPS_val", "ROE", "FwdROE", "PER", "FwdPER", "PBR", "MktCap_val", "doc_kind_last_stmt_is_fy", "cash_eff", "doc_kind", "DocType", "cur_per_type", "cur_fy_end",
    "net_shares_fy0", "net_shares_fy1", "sales_fy2", "op_fy2", "ta_fy1", "np_fy1", "cfo_fy1", "avg_shares", "is_revision_doc", "chg_acc_est", "fy_end_fy3", "cum_after", "S17",
    "sales_ttm", "cfo_ttm", "np_ttm", "op_ytd", "f_sales", "f_payout", "f_div_2q", "f_div_fy", "eps_fy0", "div_ps_today", "earnings_yield_fwd", "adv120_turnover", "sheq_latest",
    "consolidated", "f_op_initial", "ed_period_end", "ed_cash", "short_term_securities", "dep_annual", "interest_annual", "ma25_slope20", "ma75_slope20", "shares_net", "adj_ps",
}


def build_features(store, params: dict, start: date | None = None, end: date | None = None, write: bool = True) -> pd.DataFrame:
    """start 给定时只重算并写入 >= start 的行（增量）；不给时重算全部。返回写入的行（Date 为 datetime64）。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        return _build_features(store, params, start, end, write)


def _build_features(store, params: dict, start: date | None, end: date | None, write: bool) -> pd.DataFrame:
    reg = load_registry(params)
    panel_start = (start - timedelta(days=HIST_DAYS)) if start else None
    panel = load_price_panel(store, panel_start, end)
    if panel.empty:
        log.warning("prices_daily is empty")
        return pd.DataFrame()
    adj_events = load_adj_events(store)
    tech = technical_features(panel, params, adj_events, store)
    dates = trading_dates(store, panel["Date"])
    del panel
    st = load_statements(store)
    states, events = build_states_and_events(st) if not st.empty else (pd.DataFrame(), {"revision": pd.DataFrame(), "dividend": pd.DataFrame(), "fy": pd.DataFrame()})
    log.info("states: %d rows; events rev=%d div=%d fy=%d", len(states), len(events["revision"]), len(events["dividend"]), len(events["fy"]))
    if states.empty:
        raise ValueError("financial_statements 为空，无法构建财务特征")
    states = states.copy()
    states["doc_kind_last_stmt_is_fy"] = states["last_stmt_disc_no"].astype(str).eq(states["fy_disc_no"].astype(str))
    state_cols = [c for c in states.columns if c not in ("Code", "DocType", "doc_kind", "cur_per_type", "cur_fy_end", "avg_shares", "is_revision_doc", "chg_acc_est", "sales_ttm", "cfo_ttm", "op_ytd", "f_sales", "f_payout", "f_div_2q", "f_div_fy")]
    feat = tech
    del tech
    feat["Date"] = pd.to_datetime(feat["Date"]).astype("datetime64[us]")
    feat = _assign(feat, _take(states, _asof_positions(feat["Code"], feat["Date"], states["Code"], states["effective_date"]), state_cols, feat.index))
    feat["doc_kind_last_stmt_is_fy"] = feat["doc_kind_last_stmt_is_fy"].fillna(False).astype(bool)
    # 估值
    val = load_valuation(store)
    if not val.empty:
        feat = _left_join_keys(feat, val.rename(columns={"BPS": "BPS_val", "MktCap": "MktCap_val"}))
    else:
        for c in ("EPS", "FwdEPS", "BPS_val", "ROE", "FwdROE", "PER", "FwdPER", "PBR", "MktCap_val"):
            feat[c] = np.nan
    # EDINET 明细（有就用）；J-Quants financial_details 字段未实测，存在时只记日志
    ed = load_edinet_items(store)
    if not ed.empty:
        ed_cols = [c for c in ed.columns if c not in ("Code", "ed_effective")]
        feat = _assign(feat, _take(ed, _asof_positions(feat["Code"], feat["Date"], ed["Code"], ed["ed_effective"]), ed_cols, feat.index))
    if store.exists("staging", "financial_details"):
        log.warning("staging.financial_details 存在但还没有字段映射（Free 套餐没有实测样本），EV/EBITDA 只用 edinet_key_items")
    master = load_master_snapshots(store)
    feat = attach_master(feat, master)
    jgb = _jgb10y(store)
    feat = compute_fundamental_metrics(feat, params, adj_events, jgb)

    # 宽表：市值与隐含净利，用于事件的「公告前」取值
    dates = pd.DatetimeIndex(dates)
    code_idx = pd.Index(sorted(feat["Code"].unique()))
    ri = dates.get_indexer(feat["Date"])
    ci = code_idx.get_indexer(feat["Code"])
    cap = np.full((len(dates), len(code_idx)), np.nan)
    cap[ri, ci] = feat["mcap"].to_numpy(dtype=float)
    mcap_w = pd.DataFrame(cap, index=dates, columns=code_idx).ffill(limit=5)
    imp = np.full(cap.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        imp[ri, ci] = (feat["FwdEPS"] * feat["MktCap_val"] * 1e6 / feat["C"]).to_numpy(dtype=float)
    implied_w = pd.DataFrame(imp, index=dates, columns=code_idx)
    # 中间列用完就丢，比率和展示金额转 float32，后面的连接才不会放大内存
    feat = feat.drop(columns=[c for c in KEEP_DROP if c in feat.columns and c != "C"])
    feat = _shrink(feat)
    del cap, imp
    start_ts = pd.Timestamp(start) if start else dates[0]
    end_ts = pd.Timestamp(end) if end else dates[-1]
    target_rows = np.nonzero((dates >= start_ts) & (dates <= end_ts))[0]

    innov = forecast_innovation_daily(events["fy"], dates, mcap_w, reg, params, target_rows, master)
    rev_tbl = revision_events(events["revision"], dates, implied_w, mcap_w, reg, params)
    rev_d = revision_daily(rev_tbl, dates, reg, target_rows)
    div_d = dividend_daily(events["dividend"], dates, adj_events, reg, target_rows)
    log.info("event daily rows: innovation=%d revision=%d dividend=%d", len(innov), len(rev_d), len(div_d))
    for frame in (innov, rev_d, div_d):
        if not frame.empty:
            feat = _left_join_keys(feat, frame)
    for c in ("guid_forecast_innovation", "guid_forecast_innovation_pct", "guid_forecast_innovation_raw", "guid_revision_event", "guid_revision_event_pct", "guid_revision_n_up",
              "guid_revision_confirm"):
        if c not in feat.columns:
            feat[c] = np.nan
    for c in ("guid_fail", "guid_fail_low_conf", "guid_net_only_down", "evt_div_up", "evt_div_down"):
        feat[c] = feat[c].fillna(False).astype(bool) if c in feat.columns else False
    for c in ("guid_innovation_basis", "guid_innovation_disc_no", "guid_revision_disc_no", "guid_fail_kind", "guid_fail_disc_no", "evt_div_disc_no"):
        if c not in feat.columns:
            feat[c] = None
    for c in ("guid_innovation_date", "guid_revision_date", "guid_fail_date", "evt_div_date"):
        if c not in feat.columns:
            feat[c] = pd.NaT
    # capital_action：100 日内有财务文档才算知道，否则未知
    recent_doc_days = int(reg["families"]["capital_action"].get("absence_requires_days", pget(params, "evidence.capital_action_recent_doc_days", 100)))
    change = np.where(feat["evt_div_up"], 1.0, np.where(feat["evt_div_down"], -1.0, 0.0))
    feat["evt_dividend_change"] = pd.Series(change, index=feat.index).where(feat["days_since_any_doc"] <= recent_doc_days)
    feat = add_rev_1m(feat, states, adj_events, st, dates, params)
    feat = add_self_history(feat, val, dates, params)
    if start:
        feat = feat[feat["Date"] >= pd.Timestamp(start)]
    if end:
        feat = feat[feat["Date"] <= pd.Timestamp(end)]
    feat = feat.drop(columns=[c for c in (*KEEP_DROP, "adj_c") if c in feat.columns]).reset_index(drop=True)
    feat["metric_version"] = f"{METRIC_VERSION}+registry.{reg.get('version')}"
    if write and not feat.empty:
        write_features(store, feat)
    return feat


KEEP_FLOAT64 = {"C", "adj_c", "cum_after", "Vo", "Va", "MktCap", "mcap", "adv20_turnover"}


def _shrink(feat: pd.DataFrame) -> pd.DataFrame:
    """float64 → float32（金额和价格除外）。显示精度是 2–4 位小数，比率不受影响。"""
    conv = {c: "float32" for c in feat.columns if c not in KEEP_FLOAT64 and feat[c].dtype == "float64"}
    return feat.astype(conv) if conv else feat


def _dates_to_python(feat: pd.DataFrame) -> pd.DataFrame:
    """日期列统一为 Python date（staging 约定）；parquet 中是 date32。"""
    out = feat.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.date
    return out


def write_features(store, feat: pd.DataFrame) -> None:
    """按月写。同一分区：新行替换同日旧行；口径版本不同的旧行丢弃（列变化后不留残列）。"""
    part = pd.to_datetime(feat["Date"]).dt.strftime("%Y-%m")
    version = feat["metric_version"].iloc[0]
    for p, sub in feat.groupby(part, sort=True):
        old_parts = store.partitions("mart", "features_daily")
        if str(p) in old_parts:
            old = store.read("mart", "features_daily", partitions=[str(p)])
            if "metric_version" in old.columns:
                old = old[old["metric_version"] == version]
            if not old.empty:
                keep_dates = ~pd.to_datetime(old["Date"]).isin(pd.to_datetime(sub["Date"]).unique())
                old = old[keep_dates]
            if not old.empty:
                sub = pd.concat([_dates_to_python(old), _dates_to_python(sub)], ignore_index=True)
                store.write("mart", "features_daily", sub, str(p))
                continue
        store.write("mart", "features_daily", _dates_to_python(sub), str(p))
    log.info("features_daily written: %d rows, %d partitions", len(feat), part.nunique())
