"""财务状态（时点）：由 staging.financial_statements 生成「每次披露后的公司状态」，再按生效日 as-of 合并到日频面板。

要点（对应 registry 文件头的「预想字段陷阱」和规划文档 3.3 节）：
- 预想：FY 决算行的预想在 Nx 字段里，目标财年是下一财年；1Q、2Q、3Q 行和修订文档用 F 字段，目标财年是 CurFYEn。
- 向前延续：修订文档和 1Q、3Q 短信留空的字段不覆盖旧值，逐字段延续；预想只在同一目标财年内延续；分红修订不碰利润预想。
- 补报：期间早于已知最新期的文档不覆盖最新状态。
- 事件（修正、配当变动、翌期指引）在这里顺带抽出，但窗口、排名放到 build 里按日计算。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FC_EARN = {"f_sales": ("FSales", "NxFSales"), "f_op": ("FOP", "NxFOP"), "f_odp": ("FOdP", "NxFOdP"), "f_np": ("FNP", "NxFNp"), "f_eps": ("FEPS", "NxFEPS")}
FC_DIV = {"f_div_ann": ("FDivAnn", "NxFDivAnn"), "f_div_2q": ("FDiv2Q", "NxFDiv2Q"), "f_div_fy": ("FDivFY", "NxFDivFY"),
          "f_payout": ("FPayoutRatioAnn", "NxFPayoutRatioAnn")}
FC_ALL = {**FC_EARN, **FC_DIV}
BS_FIELDS = {"ta_latest": "TA", "eq_latest": "Eq", "eqar_latest": "EqAR", "sheq_latest": "ShEq", "cash_latest": "CashEq", "bps_reported": "BPS",
             "shares_out": "ShOutFY", "treasury": "TrShFY", "avg_shares": "AvgSh"}
PERIOD_IDX = {"1Q": 1, "2Q": 2, "3Q": 3, "4Q": 4, "FY": 4}
DATE_COLS = ("DiscDate", "CurPerSt", "CurPerEn", "CurFYSt", "CurFYEn", "NxtFYSt", "NxtFYEn", "effective_date")
FY_TOL_DAYS = 5     # 财年结束日的容差（2 月末、20 日决算等）


def load_statements(store) -> pd.DataFrame:
    df = store.read("staging", "financial_statements")
    if df.empty:
        return df
    return _prep(df)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    df["DiscNo"] = df["DiscNo"].astype(str)
    df["Code"] = df["Code"].astype(str)
    return df.sort_values(["Code", "effective_date", "DiscDate", "DiscNo"]).reset_index(drop=True)


def _f(v) -> float:
    try:
        return float(v) if v is not None and not pd.isna(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _doc_kind(row: dict) -> str:
    dt = str(row.get("DocType") or "")
    if "ForecastRevision" in dt or bool(row.get("is_revision")):
        return "revision"
    if dt.startswith("FY"):
        return "fy"
    if dt[:2] in ("1Q", "2Q", "3Q"):
        return "q"
    return "other"


def _next_fy_end(fe: pd.Timestamp) -> pd.Timestamp:
    try:
        return fe + pd.DateOffset(years=1)
    except (TypeError, ValueError):
        return fe + pd.Timedelta(days=365)


def _find_key(keys, target, tol_days: int = FY_TOL_DAYS):
    """在已有财年键中找与 target 相差不超过 tol_days 的那个。"""
    for k in keys:
        if abs((k - target).days) <= tol_days:
            return k
    return None


def _fy_days(row: dict) -> float:
    a, b = row.get("CurFYSt"), row.get("CurFYEn")
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return np.nan
    return float((b - a).days + 1)


def ttm_from_ytd(ytd: dict, fy_rows: dict, latest_key: tuple, fields=("Sales", "OP", "NP", "CFO"), min_fy_days: int = 350, max_fy_days: int = 380) -> dict:
    """滚动 12 个月：最新累计 + 上一财年全年 − 上年同期累计。

    这样得到的正好是连续 4 个季度：要求上一财年紧接本财年（否则中间缺了一年）、两个财年都是 12 个月左右。
    条件不满足时 n=0、值为 NaN，不做近似。
    """
    fy_end, per = latest_key
    row = ytd[latest_key]
    out: dict = {"n": 0, **{f: np.nan for f in fields}}
    if per == 4:
        if min_fy_days <= _fy_days(row) <= max_fy_days:
            out.update({f: _f(row.get(f)) for f in fields})
            out["n"] = 4
        return out
    cur_st = row.get("CurFYSt")
    if cur_st is None or pd.isna(cur_st):
        return out
    prev_key = _find_key(fy_rows.keys(), cur_st - pd.Timedelta(days=1))
    if prev_key is None:
        return out
    prev_fy = fy_rows[prev_key]
    prior = ytd.get((prev_key, per))
    if prior is None or not (min_fy_days <= _fy_days(prev_fy) <= max_fy_days):
        return out
    for f in fields:
        out[f] = _f(row.get(f)) + _f(prev_fy.get(f)) - _f(prior.get(f))
    out["n"] = 4
    return out


def build_statement_states(st: pd.DataFrame) -> pd.DataFrame:
    return build_states_and_events(st)[0]


def build_states_and_events(st: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """对每个 Code 按披露顺序滚动。返回 (状态表, {"revision": ..., "dividend": ..., "fy": ...})。"""
    if st.empty:
        return pd.DataFrame(), {"revision": pd.DataFrame(), "dividend": pd.DataFrame(), "fy": pd.DataFrame()}
    st = _prep(st)
    states: list[dict] = []
    rev_ev: list[dict] = []
    div_ev: list[dict] = []
    fy_ev: list[dict] = []
    for code, g in st.groupby("Code", sort=False):
        _walk_company(code, g.to_dict("records"), states, rev_ev, div_ev, fy_ev)
    out = pd.DataFrame(states)
    if not out.empty:
        out = out.sort_values(["Code", "effective_date", "DiscDate", "DiscNo"]).drop_duplicates(["Code", "effective_date"], keep="last").reset_index(drop=True)
    return out, {"revision": pd.DataFrame(rev_ev), "dividend": pd.DataFrame(div_ev), "fy": pd.DataFrame(fy_ev)}


def _walk_company(code: str, rows: list[dict], states: list, rev_ev: list, div_ev: list, fy_ev: list) -> None:
    fy_rows: dict[pd.Timestamp, dict] = {}                      # CurFYEn -> FY 决算行
    ytd: dict[tuple[pd.Timestamp, int], dict] = {}              # (CurFYEn, 期) -> 累计行（含 FY）
    fc_by_target: dict[pd.Timestamp, dict] = {}                 # 目标财年 -> {字段: 值}
    fc_eff: dict[pd.Timestamp, dict] = {}                       # 目标财年 -> {字段: 设定该值的文档生效日}（每股值换算要用）
    cur_target: pd.Timestamp | None = None
    initial_f_op: dict[pd.Timestamp, float] = {}                # 目标财年 -> 上一财年 FY 行给的 NxFOP
    bs: dict[str, float] = {k: np.nan for k in BS_FIELDS}
    bs_eff: dict[str, pd.Timestamp] = {}
    last_period_end: pd.Timestamp | None = None
    last_stmt: dict | None = None
    n_up = n_down = 0
    last_rev_date = None
    last_rev_chg = np.nan
    no_guidance_fy = False
    n_stmt = 0
    tr_seen_value = False
    tr_seen_blank = False

    for r in rows:
        kind = _doc_kind(r)
        per = str(r.get("CurPerType") or "")
        fy_end = r.get("CurFYEn")
        has_fy_end = fy_end is not None and not pd.isna(fy_end)
        period_end = r.get("CurPerEn")
        eff = r["effective_date"]

        # ---- 实绩序列 ----
        is_stmt = kind in ("fy", "q", "other") and has_fy_end
        if is_stmt and per in PERIOD_IDX:
            k_fy = _find_key(fy_rows.keys(), fy_end) or fy_end
            ytd[(k_fy, PERIOD_IDX[per])] = r
            if kind == "fy":
                fy_rows[k_fy] = r
        # ---- 最新资产负债：逐字段延续；补报（期间早于已知最新期）不覆盖 ----
        late = is_stmt and last_period_end is not None and period_end is not None and not pd.isna(period_end) and period_end < last_period_end
        if is_stmt and not late:
            n_stmt += 1
            for k, field in BS_FIELDS.items():
                v = _f(r.get(field))
                if not np.isnan(v):
                    bs[k] = v
                    bs_eff[k] = eff
            if period_end is not None and not pd.isna(period_end):
                last_period_end = period_end
            last_stmt = r
            if pd.isna(r.get("TrShFY")):
                tr_seen_blank = True
            else:
                tr_seen_value = True

        # ---- 预想：目标财年 + 字段延续 ----
        target = None
        if kind == "fy":
            nx = r.get("NxtFYEn")
            target = nx if nx is not None and not pd.isna(nx) else (_next_fy_end(fy_end) if has_fy_end else None)
        elif has_fy_end:
            target = fy_end
        doc_fc: dict[str, float] = {}
        if target is not None:
            key = _find_key(fc_by_target.keys(), target)
            target = key if key is not None else target
            col_i = 1 if kind == "fy" else 0
            doc_fc = {name: _f(r.get(cols[col_i])) for name, cols in FC_ALL.items()}
            prev = dict(fc_by_target.get(target, {}))
            fc_by_target.setdefault(target, {})
            fc_eff.setdefault(target, {})
            earn_doc = {k: doc_fc[k] for k in FC_EARN}
            earn_present = any(not np.isnan(v) for v in earn_doc.values())
            had_earn = any(not np.isnan(prev.get(k, np.nan)) for k in ("f_op", "f_np"))
            withdrawn = kind == "q" and not earn_present and had_earn
            if withdrawn:
                for k in FC_EARN:
                    fc_by_target[target][k] = np.nan
            for k, v in doc_fc.items():
                if not np.isnan(v):
                    fc_by_target[target][k] = v
                    fc_eff[target][k] = eff
            if kind == "fy":
                initial_f_op[target] = doc_fc["f_op"]
            # 当前目标财年只向前走；旧财年的迟到修正只更新那一年，不回退
            if cur_target is None or target >= cur_target:
                cur_target = target
            no_guidance_fy = kind == "fy" and not earn_present
            # ---- 事件 ----
            if kind != "fy" and (earn_present or withdrawn):
                main_basis = "odp" if not np.isnan(doc_fc["f_odp"]) else ("op" if not np.isnan(doc_fc["f_op"]) else None)
                main_field = {"odp": "f_odp", "op": "f_op"}.get(main_basis or "")
                rev_ev.append({
                    "Code": code, "DiscDate": r["DiscDate"], "effective_date": eff, "DiscNo": r["DiscNo"], "DocType": r.get("DocType"), "target_fy": target,
                    "fnp_prev": _f(prev.get("f_np")), "fnp_new": doc_fc["f_np"], "main_basis": main_basis,
                    "main_prev": _f(prev.get(main_field)) if main_field else np.nan, "main_new": doc_fc[main_field] if main_field else np.nan,
                    "withdrawn": withdrawn,
                })
            new_div, old_div = doc_fc["f_div_ann"], _f(prev.get("f_div_ann"))
            if not np.isnan(new_div) and not np.isnan(old_div):
                div_ev.append({"Code": code, "DiscDate": r["DiscDate"], "effective_date": eff, "DiscNo": r["DiscNo"], "target_fy": target,
                               "div_prev": old_div, "div_new": new_div, "prev_eff": fc_eff[target].get("_div_prev_eff", eff)})
            if not np.isnan(new_div):
                fc_eff[target]["_div_prev_eff"] = eff
            # 展示用：同一目标财年内 FOP 的上下修次数（不进入判断）
            old_op, new_op = _f(prev.get("f_op")), doc_fc["f_op"]
            if not np.isnan(new_op) and not np.isnan(old_op) and old_op != 0 and kind != "fy":
                chg = new_op / old_op - 1
                if abs(chg) > 0.005:
                    n_up += int(chg > 0)
                    n_down += int(chg < 0)
                    last_rev_date, last_rev_chg = eff, chg
        cur_fc = fc_by_target.get(cur_target, {}) if cur_target is not None else {}
        cur_fc_eff = fc_eff.get(cur_target, {}) if cur_target is not None else {}

        # ---- 翌期指引事件（只看 FY 行）----
        if kind == "fy":
            gaap = str(r.get("gaap") or "")
            nx_odp, odp, nx_op, op, nx_np, np_ = (_f(r.get("NxFOdP")), _f(r.get("OdP")), _f(r.get("NxFOP")), _f(r.get("OP")), _f(r.get("NxFNp")), _f(r.get("NP")))
            basis, diff = None, np.nan
            if gaap == "JP" and not np.isnan(nx_odp) and not np.isnan(odp):
                basis, diff = "odp", nx_odp - odp
            elif not np.isnan(nx_op) and not np.isnan(op):
                basis, diff = "op", nx_op - op
            elif not np.isnan(nx_np) and not np.isnan(np_):
                basis, diff = "np", nx_np - np_
            if basis is not None:
                fy_ev.append({"Code": code, "DiscDate": r["DiscDate"], "effective_date": eff, "DiscNo": r["DiscNo"], "basis": basis, "diff": diff, "fy_end": fy_end})

        states.append(_state_row(code, r, kind, per, eff, fy_rows, ytd, cur_target, cur_fc, cur_fc_eff, initial_f_op, bs, bs_eff, last_stmt,
                                 n_up, n_down, last_rev_date, last_rev_chg, no_guidance_fy, n_stmt, tr_seen_value, tr_seen_blank))


def _state_row(code, r, kind, per, eff, fy_rows, ytd, cur_target, cur_fc, cur_fc_eff, initial_f_op, bs, bs_eff, last_stmt,
               n_up, n_down, last_rev_date, last_rev_chg, no_guidance_fy, n_stmt, tr_seen_value, tr_seen_blank) -> dict:
    fy_sorted = sorted(fy_rows.keys())

    def fyv(k: int, field: str) -> float:
        return _f(fy_rows[fy_sorted[-1 - k]].get(field)) if len(fy_sorted) > k else np.nan

    def fyd(k: int, field: str):
        return fy_rows[fy_sorted[-1 - k]].get(field) if len(fy_sorted) > k else pd.NaT

    def eqar(k: int) -> float:
        v = fyv(k, "EqAR")
        if np.isnan(v):
            ta, sh = fyv(k, "TA"), fyv(k, "ShEq")
            return sh / ta if ta and not np.isnan(ta) else np.nan
        return v

    def net_shares(k: int) -> float:
        s, t = fyv(k, "ShOutFY"), fyv(k, "TrShFY")
        if np.isnan(s):
            return np.nan
        if np.isnan(t):
            # registry：TrShFY 全部为空的公司按 0；空与非空混用的公司股数项记未知
            return s if not tr_seen_value else np.nan
        return s - t

    days0, days1 = _days_of(fy_rows, fy_sorted, 0), _days_of(fy_rows, fy_sorted, 1)
    latest_q_key = max(ytd.keys()) if ytd else None
    ttm = ttm_from_ytd(ytd, fy_rows, latest_q_key) if latest_q_key else {"n": 0}
    latest_q = ytd[latest_q_key] if latest_q_key else None

    # 进捗率与「剩余期间必要增益率」的输入
    prog = prog_prev = np.nan
    rem = {"rem_fc": np.nan, "rem_ytd": np.nan, "rem_prev_fy": np.nan, "rem_prev_ytd": np.nan, "rem_basis": None}
    if latest_q is not None and latest_q_key[1] in (1, 2, 3):
        q_fy, q_per = latest_q_key
        same_target = cur_target is not None and abs((cur_target - q_fy).days) <= FY_TOL_DAYS
        prev_key = None
        cur_st = latest_q.get("CurFYSt")
        if cur_st is not None and not pd.isna(cur_st):
            prev_key = _find_key(fy_rows.keys(), cur_st - pd.Timedelta(days=1))
        prior = ytd.get((prev_key, q_per)) if prev_key is not None else None
        prev_fy = fy_rows.get(prev_key) if prev_key is not None else None
        if same_target:
            fop = _f(cur_fc.get("f_op"))
            ytd_op = _f(latest_q.get("OP"))
            if not np.isnan(fop) and fop != 0:
                prog = ytd_op / fop
            basis = "op" if not (np.isnan(fop) or np.isnan(ytd_op)) else ("np" if not np.isnan(_f(cur_fc.get("f_np"))) else None)
            if basis:
                field, fc_name = ("OP", "f_op") if basis == "op" else ("NP", "f_np")
                rem = {"rem_fc": _f(cur_fc.get(fc_name)), "rem_ytd": _f(latest_q.get(field)), "rem_prev_fy": _f(prev_fy.get(field)) if prev_fy is not None else np.nan,
                       "rem_prev_ytd": _f(prior.get(field)) if prior is not None else np.nan, "rem_basis": basis}
        if prior is not None and prev_fy is not None and _f(prev_fy.get("OP")) not in (0.0,) and not np.isnan(_f(prev_fy.get("OP"))):
            prog_prev = _f(prior.get("OP")) / _f(prev_fy.get("OP"))

    fy0_key = fy_sorted[-1] if fy_sorted else None
    out = {
        "Code": code, "effective_date": eff, "DiscDate": r["DiscDate"], "DiscNo": r["DiscNo"], "DocType": r.get("DocType"), "doc_kind": kind,
        "cur_per_type": per, "cur_fy_end": r.get("CurFYEn"), "gaap": (last_stmt or r).get("gaap"), "consolidated": (last_stmt or r).get("consolidated"),
        # 年度实绩（fy0 最新）
        "sales_fy0": fyv(0, "Sales"), "sales_fy1": fyv(1, "Sales"), "sales_fy2": fyv(2, "Sales"), "sales_fy3": fyv(3, "Sales"),
        "op_fy0": fyv(0, "OP"), "op_fy1": fyv(1, "OP"), "op_fy2": fyv(2, "OP"), "op_fy3": fyv(3, "OP"),
        "odp_fy0": fyv(0, "OdP"), "np_fy0": fyv(0, "NP"), "np_fy1": fyv(1, "NP"),
        "cfo_fy0": fyv(0, "CFO"), "cfo_fy1": fyv(1, "CFO"), "cfi_fy0": fyv(0, "CFI"),
        "ta_fy0": fyv(0, "TA"), "ta_fy1": fyv(1, "TA"), "eqar_fy0": eqar(0), "eqar_fy1": eqar(1),
        "div_fy0": fyv(0, "DivAnn"), "div_fy1": fyv(1, "DivAnn"), "eps_fy0": fyv(0, "EPS"),
        "net_shares_fy0": net_shares(0), "net_shares_fy1": net_shares(1), "tr_mixed": bool(tr_seen_value and tr_seen_blank),
        "fy_days_fy0": days0, "fy_days_fy1": days1,
        "fy_end_fy0": fyd(0, "CurFYEn"), "fy_end_fy1": fyd(1, "CurFYEn"), "fy_end_fy3": fyd(3, "CurFYEn"),
        "fy_eff_fy0": fyd(0, "effective_date"), "fy_eff_fy1": fyd(1, "effective_date"),
        "fy_disc_no": fyd(0, "DiscNo") if fy_sorted else None, "fy_disc_date": fyd(0, "DiscDate"),
        "sig_chg_fy0": str(fyd(0, "SigChgInC")).lower() == "true" if fy_sorted else False,
        "n_fy": len(fy_sorted), "n_stmt": n_stmt,
        # 最新一期（任何报告，逐字段延续）
        **bs,
        "shares_eff": bs_eff.get("shares_out", pd.NaT), "bps_eff": bs_eff.get("bps_reported", pd.NaT), "cash_eff": bs_eff.get("cash_latest", pd.NaT),
        # TTM 与季度
        "sales_ttm": ttm.get("Sales", np.nan), "op_ttm": ttm.get("OP", np.nan), "np_ttm": ttm.get("NP", np.nan), "cfo_ttm": ttm.get("CFO", np.nan),
        "ttm_quarters": ttm.get("n", 0),
        "latest_q_period": latest_q_key[1] if latest_q_key else np.nan, "latest_q_fy_end": latest_q_key[0] if latest_q_key else pd.NaT,
        "op_ytd": _f(latest_q.get("OP")) if latest_q is not None else np.nan,
        "progress_rate": prog, "progress_rate_prev_year": prog_prev, **rem,
        "disc_lag_days": _lag(last_stmt),
        "last_stmt_disc_no": (last_stmt or {}).get("DiscNo"), "last_stmt_disc_date": (last_stmt or {}).get("DiscDate", pd.NaT),
        # 当前预想（逐字段延续后的值）
        "f_fy_end": cur_target if cur_target is not None else pd.NaT,
        **{k: _f(cur_fc.get(k)) for k in FC_ALL},
        "f_eps_eff": cur_fc_eff.get("f_eps", pd.NaT), "f_div_eff": cur_fc_eff.get("f_div_ann", pd.NaT),
        "f_no_guidance_fy": no_guidance_fy,
        "f_op_initial": initial_f_op.get(cur_target, np.nan) if cur_target is not None else np.nan,
        "f_op_initial_fy0": _initial_for(initial_f_op, fy0_key),
        "n_up_revisions": n_up, "n_down_revisions": n_down,
        "last_revision_date": last_rev_date if last_rev_date is not None else pd.NaT, "last_revision_chg": last_rev_chg,
        "is_revision_doc": kind == "revision", "chg_acc_est": r.get("ChgAcEst"), "retro_restate": str((last_stmt or {}).get("RetroRst")).lower() == "true",
    }
    return out


def _days_of(fy_rows: dict, fy_sorted: list, k: int) -> float:
    return _fy_days(fy_rows[fy_sorted[-1 - k]]) if len(fy_sorted) > k else np.nan


def _initial_for(initial_f_op: dict, fy_key) -> float:
    if fy_key is None:
        return np.nan
    k = _find_key(initial_f_op.keys(), fy_key)
    return initial_f_op.get(k, np.nan) if k is not None else np.nan


def _lag(stmt: dict | None) -> float:
    if not stmt:
        return np.nan
    a, b = stmt.get("DiscDate"), stmt.get("CurPerEn")
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return np.nan
    return float((a - b).days)


def asof_join(panel: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    """按 Code 把状态行 as-of 合并到面板（生效日 ≤ 面板日期）。effective_date 作为真实列保留。"""
    if states.empty:
        p = panel.copy()
        p["effective_date"] = pd.NaT
        return p
    p = panel.copy()
    p["_d"] = pd.to_datetime(p["Date"]).astype("datetime64[us]")
    s = states.copy()
    s["effective_date"] = pd.to_datetime(s["effective_date"]).astype("datetime64[us]")
    s["_d"] = s["effective_date"]
    p = p.sort_values("_d")
    s = s.sort_values("_d")
    merged = pd.merge_asof(p, s, on="_d", by="Code", direction="backward")
    return merged.drop(columns=["_d"])


def fscore7(d: pd.DataFrame, cfo_items_off: pd.Series | None = None, inorganic_off: pd.Series | None = None, share_tol: float = 0.005) -> pd.DataFrame:
    """简化 F-Score（registry qual_fscore7）。返回 qual_fscore7（折算到 7 分）、fscore_n（可算项数）。

    输入列：np_fy0/1, ta_fy0/1, cfo_fy0, eqar_fy0/1, sales_fy0/1, share_growth_fy（已还原分割）, fy_days_fy0/1。
    决算期变更（天数不在 350-380）时差分项按天数折算成全年再比较。可算项少于 5 项时分数为空。
    """
    ann0 = 365.0 / d["fy_days_fy0"].where(d["fy_days_fy0"] > 0)
    ann1 = 365.0 / d["fy_days_fy1"].where(d["fy_days_fy1"] > 0)
    roa0 = d["np_fy0"] * ann0.fillna(1.0) / d["ta_fy0"]
    roa1 = d["np_fy1"] * ann1.fillna(1.0) / d["ta_fy1"]
    turn0 = d["sales_fy0"] * ann0.fillna(1.0) / d["ta_fy0"]
    turn1 = d["sales_fy1"] * ann1.fillna(1.0) / d["ta_fy1"]
    cfo = d["cfo_fy0"]
    items = {
        "roa_pos": _cond(roa0, roa0 > 0),
        "cfo_pos": _cond(cfo, cfo > 0),
        "roa_up": _cond(roa0 - roa1, roa0 > roa1),
        "cfo_gt_np": _cond(cfo - d["np_fy0"], cfo > d["np_fy0"]),
        "eqar_up": _cond(d["eqar_fy0"] - d["eqar_fy1"], d["eqar_fy0"] > d["eqar_fy1"]),
        "no_dilution": _cond(d["share_growth_fy"], d["share_growth_fy"] <= share_tol),
        "turnover_up": _cond(turn0 - turn1, turn0 > turn1),
    }
    m = pd.DataFrame(items, index=d.index)
    if cfo_items_off is not None:
        off = cfo_items_off.fillna(False).to_numpy(dtype=bool)
        m.loc[off, ["cfo_pos", "cfo_gt_np"]] = np.nan
    n = m.notna().sum(axis=1)
    score = m.sum(axis=1, skipna=True)
    out = pd.DataFrame(index=d.index)
    out["fscore_n"] = n
    out["qual_fscore7"] = (score / n * 7).where(n >= 5)
    return out


def _cond(known: pd.Series, cond: pd.Series) -> pd.Series:
    """known 为 NaN 时结果为 NaN，否则 cond 的 0/1。"""
    return cond.astype(float).where(known.notna())
