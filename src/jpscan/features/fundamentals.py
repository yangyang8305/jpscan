"""财务状态（时点）：由 staging.financial_statements 生成「每次披露后的公司状态」，再按生效日 as-of 合并到日频面板。

状态包含：最新年度实绩序列（增长率）、最新累计季度（進捗率、TTM）、当前通期预想（F*/NxF*）、每股数据、股数、
修正履歴、会计标志。字段口径见附录 B。
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

FY_FIELDS = ["Sales", "OP", "OdP", "NP", "EPS", "TA", "Eq", "EqAR", "BPS", "CFO", "CFI", "CFF", "CashEq", "DivAnn", "ShOutFY", "TrShFY", "AvgSh"]
PERIOD_IDX = {"1Q": 1, "2Q": 2, "3Q": 3, "4Q": 4, "FY": 4, "5Q": 5}


def load_statements(store) -> pd.DataFrame:
    df = store.read("staging", "financial_statements")
    if df.empty:
        return df
    for c in ("DiscDate", "CurPerSt", "CurPerEn", "CurFYSt", "CurFYEn", "effective_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
    df["DiscNo"] = df["DiscNo"].astype(str)
    return df.sort_values(["Code", "effective_date", "DiscDate", "DiscNo"]).reset_index(drop=True)


def _current_forecast(row: pd.Series) -> dict:
    """当前通期预想：非 FY 行用 F*；FY 行（已进入新年度）用 NxF*。"""
    use_next = str(row.get("CurPerType")) == "FY" and not bool(row.get("is_revision", False))
    pick = (lambda f, nf: row.get(nf) if use_next else row.get(f))
    out = {
        "f_sales": pick("FSales", "NxFSales"), "f_op": pick("FOP", "NxFOP"), "f_odp": pick("FOdP", "NxFOdP"),
        "f_np": pick("FNP", "NxFNp"), "f_eps": pick("FEPS", "NxFEPS"), "f_div_ann": pick("FDivAnn", "NxFDivAnn"),
        "f_payout": pick("FPayoutRatioAnn", "NxFPayoutRatioAnn"),
    }
    # 预想所属年度
    if use_next:
        out["f_fy_end"] = row.get("NxtFYEn") if pd.notna(row.get("NxtFYEn")) else None
    else:
        out["f_fy_end"] = row.get("CurFYEn")
    if use_next and (out["f_fy_end"] is None or pd.isna(out["f_fy_end"])):
        fe = row.get("CurFYEn")
        if fe is not None and not pd.isna(fe):
            out["f_fy_end"] = date(fe.year + 1, fe.month, fe.day) if not (fe.month == 2 and fe.day == 29) else date(fe.year + 1, 2, 28)
    return out


def build_statement_states(st: pd.DataFrame) -> pd.DataFrame:
    """对每个 Code 按披露顺序滚动，生成每次披露后的状态行（PIT）。"""
    states: list[dict] = []
    consolidated_pref = st.copy()
    for code, g in consolidated_pref.groupby("Code", sort=False):
        g = g.sort_values(["effective_date", "DiscDate", "DiscNo"])
        fy_rows: dict[date, pd.Series] = {}          # CurFYEn -> 最新 FY 实绩行
        ytd: dict[tuple[date, int], pd.Series] = {}  # (CurFYEn, period) -> 累计行
        n_up = 0
        n_down = 0
        revisions: list[tuple[date, float]] = []
        initial_f_op: dict[date, float] = {}         # 各年度期初预想 OP
        last_f_op: float | None = None
        last_f_fy: date | None = None
        for _, r in g.iterrows():
            per = str(r.get("CurPerType") or "")
            fy_end = r.get("CurFYEn")
            is_rev = bool(r.get("is_revision"))
            if not is_rev and per in PERIOD_IDX and fy_end is not None and not pd.isna(fy_end):
                if per == "FY":
                    fy_rows[fy_end] = r
                    ytd[(fy_end, 4)] = r
                else:
                    ytd[(fy_end, PERIOD_IDX[per])] = r
            fc = _current_forecast(r)
            f_op = fc["f_op"]
            f_fy = fc["f_fy_end"]
            if f_op is not None and not pd.isna(f_op) and f_fy is not None and not pd.isna(f_fy):
                if f_fy not in initial_f_op:
                    initial_f_op[f_fy] = float(f_op)
                if last_f_op is not None and last_f_fy == f_fy and last_f_op not in (0, None):
                    chg = float(f_op) / last_f_op - 1
                    if chg > 0.005:
                        n_up += 1
                        revisions.append((r["effective_date"], chg))
                    elif chg < -0.005:
                        n_down += 1
                        revisions.append((r["effective_date"], chg))
                last_f_op, last_f_fy = float(f_op), f_fy
            # ---- 年度序列 ----
            fy_sorted = sorted(fy_rows.keys())
            latest_fy = fy_rows[fy_sorted[-1]] if fy_sorted else None
            def fyv(k: int, field: str):
                if len(fy_sorted) > k:
                    v = fy_rows[fy_sorted[-1 - k]].get(field)
                    return float(v) if v is not None and not pd.isna(v) else np.nan
                return np.nan
            # ---- 累计季度 / TTM ----
            cur_q = None
            if per in ("1Q", "2Q", "3Q") and not is_rev:
                cur_q = r
            elif not is_rev and per == "FY":
                cur_q = None
            # 最近累计季度行（可能是本次或之前）
            qkeys = sorted(ytd.keys())
            latest_q_key = qkeys[-1] if qkeys else None
            latest_q = ytd[latest_q_key] if latest_q_key else None
            ttm = _ttm(ytd, latest_q_key) if latest_q_key else {}
            # 進捗率
            prog = np.nan
            prog_hist = np.nan
            if latest_q is not None and latest_q_key[1] in (1, 2, 3):
                fop = fc["f_op"] if fc["f_fy_end"] == latest_q_key[0] else np.nan
                if fop is not None and not pd.isna(fop) and fop != 0:
                    prog = float(latest_q.get("OP") or np.nan) / float(fop)
                # 去年同季进度（用当时的通期实绩近似分母不可知 → 用去年同季累计 / 去年通期实绩）
                prev_fy = date(latest_q_key[0].year - 1, latest_q_key[0].month, min(latest_q_key[0].day, 28))
                pk = [k for k in ytd if k[1] == latest_q_key[1] and abs((k[0] - prev_fy).days) <= 5]
                if pk and prev_fy in [d for d in fy_rows] or (pk and any(abs((d - prev_fy).days) <= 5 for d in fy_rows)):
                    prev_q = ytd[pk[0]]
                    prev_fy_row = next((fy_rows[d] for d in fy_rows if abs((d - prev_fy).days) <= 5), None)
                    if prev_fy_row is not None and prev_fy_row.get("OP") not in (None, 0) and not pd.isna(prev_fy_row.get("OP")):
                        prog_hist = float(prev_q.get("OP") or np.nan) / float(prev_fy_row.get("OP"))
            eq_sh = np.nan
            if latest_fy is not None:
                ta, ear = latest_fy.get("TA"), latest_fy.get("EqAR")
                if ta is not None and ear is not None and not pd.isna(ta) and not pd.isna(ear):
                    eq_sh = float(ta) * float(ear)
            # 最新资产负债（用最近一期任何报告）
            ta_l, ear_l, cash_l, sh_out, tr_sh, bps_l = (r.get("TA"), r.get("EqAR"), r.get("CashEq"), r.get("ShOutFY"), r.get("TrShFY"), r.get("BPS"))
            states.append({
                "Code": code, "effective_date": r["effective_date"], "DiscNo": r["DiscNo"], "DocType": r.get("DocType"),
                "cur_per_type": per, "cur_fy_end": fy_end, "gaap": r.get("gaap"), "consolidated": r.get("consolidated"),
                "fy_months": r.get("fy_months"),
                # 年度实绩
                "sales_fy0": fyv(0, "Sales"), "sales_fy1": fyv(1, "Sales"), "sales_fy2": fyv(2, "Sales"), "sales_fy3": fyv(3, "Sales"),
                "op_fy0": fyv(0, "OP"), "op_fy1": fyv(1, "OP"), "op_fy2": fyv(2, "OP"), "op_fy3": fyv(3, "OP"),
                "np_fy0": fyv(0, "NP"), "np_fy1": fyv(1, "NP"), "np_fy2": fyv(2, "NP"),
                "cfo_fy0": fyv(0, "CFO"), "cfo_fy1": fyv(1, "CFO"), "cfo_fy2": fyv(2, "CFO"),
                "cfi_fy0": fyv(0, "CFI"), "cfi_fy1": fyv(1, "CFI"), "cfi_fy2": fyv(2, "CFI"),
                "ta_fy0": fyv(0, "TA"), "ta_fy1": fyv(1, "TA"), "eqar_fy0": fyv(0, "EqAR"), "eqar_fy1": fyv(1, "EqAR"),
                "div_fy0": fyv(0, "DivAnn"), "div_fy1": fyv(1, "DivAnn"), "div_fy2": fyv(2, "DivAnn"),
                "eps_fy0": fyv(0, "EPS"), "n_fy": len(fy_sorted), "latest_fy_end": fy_sorted[-1] if fy_sorted else None,
                "equity_sh_fy0": eq_sh,
                # 最新一期（任何报告）
                "ta_latest": _f(ta_l), "eqar_latest": _f(ear_l), "cash_latest": _f(cash_l), "shares_out": _f(sh_out), "treasury": _f(tr_sh),
                "bps_reported": _f(bps_l), "avg_shares": _f(r.get("AvgSh")),
                # TTM 与季度
                "sales_ttm": ttm.get("Sales", np.nan), "op_ttm": ttm.get("OP", np.nan), "np_ttm": ttm.get("NP", np.nan),
                "cfo_ttm": ttm.get("CFO", np.nan), "ttm_quarters": ttm.get("n", 0),
                "latest_q_period": latest_q_key[1] if latest_q_key else np.nan, "latest_q_fy_end": latest_q_key[0] if latest_q_key else None,
                "op_ytd": _f(latest_q.get("OP")) if latest_q is not None else np.nan,
                "progress_rate": prog, "progress_rate_prev_year": prog_hist,
                # 当前预想
                "f_sales": _f(fc["f_sales"]), "f_op": _f(fc["f_op"]), "f_np": _f(fc["f_np"]), "f_eps": _f(fc["f_eps"]),
                "f_div_ann": _f(fc["f_div_ann"]), "f_fy_end": fc["f_fy_end"], "f_payout": _f(fc["f_payout"]),
                "f_op_initial": initial_f_op.get(fc["f_fy_end"], np.nan) if fc["f_fy_end"] is not None else np.nan,
                "n_up_revisions": n_up, "n_down_revisions": n_down,
                "last_revision_date": revisions[-1][0] if revisions else None, "last_revision_chg": revisions[-1][1] if revisions else np.nan,
                "is_revision_doc": is_rev, "chg_acc_est": r.get("ChgAcEst"), "retro_restate": r.get("RetroRst"),
            })
    out = pd.DataFrame(states)
    if out.empty:
        return out
    out = out.sort_values(["Code", "effective_date", "DiscNo"]).drop_duplicates(["Code", "effective_date"], keep="last")
    return out.reset_index(drop=True)


def _f(v):
    try:
        return float(v) if v is not None and not pd.isna(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _ttm(ytd: dict, latest_key: tuple) -> dict:
    """由累计季度值差分出单季，再取最近 4 个连续单季之和。"""
    fy_end, p = latest_key
    out = {"n": 0}
    fields = ["Sales", "OP", "NP", "CFO"]
    singles: list[dict] = []
    # 构造按时间排序的 (fy_end, period) 列表
    keys = sorted(ytd.keys())
    idx = keys.index(latest_key)
    for k in reversed(keys[: idx + 1]):
        fy, per = k
        row = ytd[k]
        prev = ytd.get((fy, per - 1)) if per > 1 else None
        if per == 1:
            single = {f: _f(row.get(f)) for f in fields}
        elif prev is not None:
            single = {f: _f(row.get(f)) - _f(prev.get(f)) for f in fields}
        else:
            break  # 缺上一累计期 → 无法差分
        singles.append(single)
        if len(singles) == 4:
            break
    # 连续性检查：4 个单季应覆盖连续期间
    if len(singles) == 4:
        for f in fields:
            vals = [s[f] for s in singles]
            out[f] = float(np.nansum(vals)) if not all(np.isnan(vals)) else np.nan
        out["n"] = 4
    elif singles:
        out["n"] = len(singles)
    return out


def asof_join(panel: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    """按 Code 把状态行 as-of 合并到面板（状态生效日 ≤ 面板日期）。"""
    if states.empty:
        return panel
    p = panel.copy()
    p["_d"] = pd.to_datetime(p["Date"])
    s = states.copy()
    s["_d"] = pd.to_datetime(s["effective_date"])
    p = p.sort_values("_d")
    s = s.sort_values("_d")
    merged = pd.merge_asof(p, s.drop(columns=["effective_date"]), left_on="_d", right_on="_d", by="Code", direction="backward")
    return merged.drop(columns=["_d"])
