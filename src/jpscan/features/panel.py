"""价格面板与技术/流动性特征（附录 B.5、B.6）。

只用未复权的 O/H/L/C/Vo 与 AdjFactor 自算后复权价：API 的 AdjC 带下载版本，会随之后的分割追溯重算，
不同时间拉取的行放在一起会出现假暴跌（3.3 节、metric_registry 文件头「收益口径」）。

全部计算在「日期 × 代码」的宽矩阵上完成，两年 4,400 只只要几十秒；行数按交易日历对齐，
回填缺口（整段没有行情的交易日）在矩阵里是空行，窗口不会把缺口两侧的价格当成相邻日。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("jpscan.features")

PRICE_COLS = ["Date", "Code", "O", "H", "L", "C", "UL", "LL", "Vo", "Va", "AdjFactor", "MktCap"]
FF_LIMIT = 10   # 停牌时收盘价最多向前延续的交易日数；更长的缺口不算连续


def load_price_panel(store, start=None, end=None) -> pd.DataFrame:
    """读 staging.prices_daily 的原始列；Date 为 datetime64，按 (Code, Date) 排序。"""
    where = []
    if start:
        where.append(f"Date >= DATE '{start}'")
    if end:
        where.append(f"Date <= DATE '{end}'")
    df = store.read("staging", "prices_daily", where=" AND ".join(where) if where else None, columns=PRICE_COLS)
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"])
    df["Code"] = df["Code"].astype(str)
    df = df.sort_values(["Code", "Date"]).reset_index(drop=True)
    df["Vo"] = df["Vo"].fillna(0.0)     # 未成交日整行为空：量与额记 0，价格留空
    df["Va"] = df["Va"].fillna(0.0)
    df["AdjFactor"] = df["AdjFactor"].fillna(1.0)
    return df


def load_adj_events(store) -> pd.DataFrame:
    """全历史的 AdjFactor != 1 行。增量构建也必须用全历史，否则报告日早于面板的公司会漏掉分割。"""
    ev = store.read("staging", "prices_daily", where="AdjFactor IS NOT NULL AND AdjFactor <> 1", columns=["Date", "Code", "AdjFactor"])
    if ev.empty:
        return pd.DataFrame(columns=["Date", "Code", "AdjFactor"])
    ev["Date"] = pd.to_datetime(ev["Date"])
    ev["Code"] = ev["Code"].astype(str)
    return ev


def trading_dates(store, price_dates) -> pd.DatetimeIndex:
    """交易日序列：trading_calendar 的营业日与实际有行情日的并集（覆盖行情范围）。"""
    px = pd.DatetimeIndex(pd.to_datetime(pd.Series(price_dates).unique())).sort_values()
    if len(px) == 0:
        return px
    cal = store.read("staging", "trading_calendar") if store is not None and store.exists("staging", "trading_calendar") else pd.DataFrame()
    if cal.empty:
        return px
    cal_d = pd.to_datetime(cal.loc[pd.to_numeric(cal["HolDiv"], errors="coerce").isin([1, 2]), "Date"])
    cal_d = cal_d[(cal_d >= px[0]) & (cal_d <= px[-1])]
    return pd.DatetimeIndex(px.union(pd.DatetimeIndex(cal_d))).sort_values()


def cum_after_matrix(adj_events: pd.DataFrame, dates: pd.DatetimeIndex, codes) -> pd.DataFrame:
    """B[t, code] = ∏ AdjFactor(s)，s 为 t 之后的日期。后复权价 = 原始价 × B；报告日 r 的每股值换算到 t 的系数 = B[r] / B[t]。

    adj_events 可以早于 dates 的范围：早于第一行的因子不影响 B（它们不在任何 t 之后）。
    """
    out = np.ones((len(dates), len(codes)), dtype=np.float64)
    if adj_events is None or adj_events.empty:
        return pd.DataFrame(out, index=dates, columns=codes)
    pos = pd.Series(np.arange(len(codes)), index=codes)
    e = adj_events[adj_events["Code"].isin(pos.index)]
    rows = np.searchsorted(dates.values, e["Date"].values, side="left")   # 因子生效行
    cols = pos.loc[e["Code"]].to_numpy()
    facs = e["AdjFactor"].to_numpy(dtype=float)
    # 因子在第 r 行生效 → 第 0..r-1 行都要乘上它
    for r, c, f in zip(rows, cols, facs):
        out[: min(r, len(dates)), c] *= f
    return pd.DataFrame(out, index=dates, columns=codes)


def back_adjust(df: pd.DataFrame, adj_events: pd.DataFrame | None = None) -> pd.DataFrame:
    """长表版后复权：给 O/H/L/C 乘、给 Vo 除以「此后 AdjFactor 之积」。主要供校验与其他模块使用。"""
    d = df.sort_values(["Code", "Date"]).reset_index(drop=True).copy()
    dates = pd.DatetimeIndex(pd.to_datetime(d["Date"]).unique()).sort_values()
    codes = pd.Index(d["Code"].astype(str).unique())
    ev = adj_events
    if ev is None:
        ev = d.loc[d["AdjFactor"].notna() & (d["AdjFactor"] != 1), ["Date", "Code", "AdjFactor"]].copy()
        ev["Date"] = pd.to_datetime(ev["Date"])
        ev["Code"] = ev["Code"].astype(str)
    b = cum_after_matrix(ev, dates, codes)
    r = dates.searchsorted(pd.to_datetime(d["Date"]).values)
    c = codes.get_indexer(d["Code"].astype(str))
    d["cum_after"] = b.to_numpy()[r, c]
    for col, out in (("O", "adj_o"), ("H", "adj_h"), ("L", "adj_l"), ("C", "adj_c")):
        if col in d.columns:
            d[out] = d[col] * d["cum_after"]
    if "Vo" in d.columns:
        d["adj_vo"] = d["Vo"] / d["cum_after"]
    return d


def _wide(df: pd.DataFrame, col: str, dates, codes) -> pd.DataFrame:
    v = df.pivot(index="Date", columns="Code", values=col)
    return v.reindex(index=dates, columns=codes)


def _roll_mean(w: pd.DataFrame, n: int, mp: int) -> pd.DataFrame:
    return w.rolling(n, min_periods=mp).mean()


def market_proxy(ret_1d: pd.DataFrame, mcap_w: pd.DataFrame) -> pd.DataFrame:
    """全市场市值加权日收益，作为 TOPIX 代理（Free 套餐没有指数端点）。权重用前一日 MktCap（百万円）。"""
    w = mcap_w.shift(1).where(ret_1d.notna()).clip(lower=0)
    sw = w.sum(axis=1)
    weighted = (ret_1d * w).sum(axis=1) / sw.where(sw > 0)
    ew = ret_1d.mean(axis=1)
    out = pd.DataFrame({"Date": ret_1d.index, "mkt_ret": weighted.fillna(ew).to_numpy(), "mkt_ret_ew": ew.to_numpy()})
    out["mkt_level"] = (1 + out["mkt_ret"].fillna(0.0)).cumprod()
    out["mkt_source"] = "proxy_mcap_weighted"
    return out


def load_market(store, dates: pd.DatetimeIndex, ret_1d: pd.DataFrame, mcap_w: pd.DataFrame) -> pd.DataFrame:
    """市场序列：staging.index_topix 存在且覆盖这些日期时用真 TOPIX，否则用市值加权代理（缺日逐日回退）。"""
    proxy = market_proxy(ret_1d, mcap_w)
    if store is None or not store.exists("staging", "index_topix"):
        return proxy
    t = store.read("staging", "index_topix")
    px = next((c for c in ("C", "Close", "close") if c in t.columns), None)
    if t.empty or px is None:
        return proxy
    t = t.assign(Date=pd.to_datetime(t["Date"])).drop_duplicates("Date").sort_values("Date").set_index("Date")[px].astype(float)
    # 指数收益在两个相邻交易日都有值时才算，避免回填缺口被当成一天
    aligned = t.reindex(dates)
    ret = aligned.pct_change(fill_method=None)
    out = proxy.copy()
    has = ret.notna().to_numpy()
    out.loc[has, "mkt_ret"] = ret.to_numpy()[has]
    out.loc[has, "mkt_source"] = "index_topix"
    out["mkt_level"] = (1 + out["mkt_ret"].fillna(0.0)).cumprod()
    return out


def technical_features(df: pd.DataFrame, params: dict, adj_events: pd.DataFrame | None = None, store=None) -> pd.DataFrame:
    """输入 load_price_panel 的长表，输出每个 (Code, Date) 一行的技术特征（只含该行情表里存在的行）。"""
    tp = params.get("trend", {})
    op = params.get("oversold", {}).get("trigger", {})
    ma_windows = [int(x) for x in tp.get("ma_windows", [25, 75, 200])]
    rs_windows = [int(x) for x in tp.get("rs_windows_days", [21, 63, 126])]
    atr_w = int(tp.get("atr_window", 14))
    vol_w = int(tp.get("realized_vol_window", 60))
    beta_w = int(tp.get("beta_window", 250))
    hi_lookback = int(op.get("high_lookback_days", 60))
    hi52 = int(tp.get("high_52w_days", 250))

    dates = trading_dates(store, df["Date"])
    codes = pd.Index(sorted(df["Code"].unique()))
    present = _wide(df.assign(_p=True), "_p", dates, codes).notna().to_numpy()
    if adj_events is None:
        adj_events = df.loc[df["AdjFactor"] != 1, ["Date", "Code", "AdjFactor"]]
    b = cum_after_matrix(adj_events, dates, codes)

    close_raw = _wide(df, "C", dates, codes)
    adj_c_raw = close_raw * b
    adj_c = adj_c_raw.ffill(limit=FF_LIMIT)
    adj_h = _wide(df, "H", dates, codes) * b
    adj_l = _wide(df, "L", dates, codes) * b
    adj_vo = _wide(df, "Vo", dates, codes) / b
    va = _wide(df, "Va", dates, codes)
    mcap_prices = _wide(df, "MktCap", dates, codes)

    ret_1d = adj_c.pct_change(1, fill_method=None)
    feats: dict[str, pd.DataFrame] = {
        "adj_c": adj_c_raw,
        "ret_1d": ret_1d,
        "ret_5d": adj_c.pct_change(5, fill_method=None),
        "ret_21d": adj_c.pct_change(21, fill_method=None),
    }
    for w in ma_windows:
        ma = _roll_mean(adj_c, w, max(5, w // 2))
        feats[f"ma{w}"] = ma / b               # 均线等价位类列换回当日股数口径，增量构建的不同分区才可比
        feats[f"px_vs_ma{w}"] = adj_c / ma - 1
        if w in (25, 75):
            feats[f"ma{w}_slope20"] = ma.pct_change(20, fill_method=None)
    high52 = adj_c.rolling(hi52, min_periods=60).max()
    feats["high_52w"] = high52 / b
    feats["dist_52w_high"] = adj_c / high52 - 1
    feats["drawdown_60d"] = adj_c / adj_c.rolling(hi_lookback, min_periods=10).max() - 1
    feats["low_since_20d"] = adj_c.rolling(20, min_periods=5).min() / b
    adv20 = _roll_mean(va, 20, 10)
    adv120 = _roll_mean(va, 120, 40)
    feats["adv20_turnover"] = adv20
    feats["adv120_turnover"] = adv120
    feats["vol_ratio_20_120"] = adv20 / adv120
    feats["zero_volume_ratio_60d"] = (va == 0).where(va.notna()).astype(float).rolling(60, min_periods=20).mean()
    adv20_vo = _roll_mean(adj_vo, 20, 10)
    feats["volume_mult"] = adj_vo / adv20_vo
    prev_c = adj_c.shift(1)
    tr_a = (adj_h - adj_l).abs().to_numpy()
    tr_b = (adj_h - prev_c).abs().to_numpy()
    tr_c = (adj_l - prev_c).abs().to_numpy()
    tr = pd.DataFrame(np.fmax(np.fmax(tr_a, tr_b), tr_c), index=dates, columns=codes)
    atr14 = _roll_mean(tr, atr_w, 5)
    atr5 = _roll_mean(tr, 5, 3)
    feats["atr14"] = atr14 / b
    feats["atr5"] = atr5 / b
    feats["atr_pct"] = atr14 / adj_c
    feats["rvol60"] = ret_1d.rolling(vol_w, min_periods=20).std() * np.sqrt(250)

    mkt = load_market(store, dates, ret_1d, mcap_prices)
    mkt_ret = mkt.set_index("Date")["mkt_ret"].reindex(dates)
    mkt_level = mkt.set_index("Date")["mkt_level"].reindex(dates)
    for w in rs_windows:
        feats[f"rs_{w}d"] = adj_c.pct_change(w, fill_method=None).sub(mkt_level.pct_change(w, fill_method=None), axis=0)
    feats["beta"] = _rolling_beta(ret_1d, mkt_ret, beta_w, 120)
    feats["stage"] = _stage(adj_c, {w: _roll_mean(adj_c, w, max(5, w // 2)) for w in ma_windows}, feats.get("ma25_slope20"))

    # 累计因子：报告期每股值 → 当日口径（配合 fundamentals 的 cum_after_at_report）
    feats["cum_after"] = b
    # 宽表 → 长表（只保留行情表里存在的行）
    rr, cc = np.nonzero(present)
    out: dict[str, np.ndarray] = {"Code": codes.to_numpy()[cc], "Date": dates.to_numpy()[rr]}
    for k, w in feats.items():
        arr = w.to_numpy()[rr, cc]
        out[k] = arr
    res = pd.DataFrame(out)
    base = df[["Code", "Date", "C", "O", "H", "L", "Vo", "Va", "AdjFactor", "MktCap", "UL", "LL"]].copy()
    res = res.merge(base, on=["Code", "Date"], how="left")
    res["adj_vo"] = res["Vo"] / res["cum_after"]
    res["limit_up"] = res["UL"].astype(str).isin(["1", "1.0"])
    res["limit_down"] = res["LL"].astype(str).isin(["1", "1.0"])
    res = res.drop(columns=["UL", "LL"])
    res = res.merge(mkt[["Date", "mkt_ret", "mkt_source"]], on="Date", how="left")
    return res.sort_values(["Code", "Date"]).reset_index(drop=True)


def _rolling_beta(ret: pd.DataFrame, mkt_ret: pd.Series, window: int, min_periods: int) -> pd.DataFrame:
    """rolling cov(r, m) / var(m)；每只股票只用自身有收益的日子。"""
    m = pd.DataFrame(np.repeat(mkt_ret.to_numpy()[:, None], ret.shape[1], axis=1), index=ret.index, columns=ret.columns)
    m = m.where(ret.notna())
    r = ret.where(m.notna())
    mean_r = r.rolling(window, min_periods=min_periods).mean()
    mean_m = m.rolling(window, min_periods=min_periods).mean()
    cov = (r * m).rolling(window, min_periods=min_periods).mean() - mean_r * mean_m
    var = (m * m).rolling(window, min_periods=min_periods).mean() - mean_m * mean_m
    return cov / var.where(var > 0)


def _stage(px: pd.DataFrame, mas: dict[int, pd.DataFrame], s25) -> pd.DataFrame:
    """阶段分类（只作展示，registry trend_ma_52w）。"""
    if not {25, 75, 200} <= set(mas):
        return pd.DataFrame("unknown", index=px.index, columns=px.columns)
    a, m25, m75, m200 = px.to_numpy(), mas[25].to_numpy(), mas[75].to_numpy(), mas[200].to_numpy()
    sl = s25.to_numpy() if s25 is not None else np.full(a.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        up = (a > m25) & (m25 > m75) & (m75 > m200)
        down = (a < m75) & (m75 < m200)
        top = (a < m25) & (sl < 0) & (a > m200)
        base = np.abs(a / m200 - 1) < 0.05
    val = np.select([np.isnan(m200), up, down, top, base], ["unknown", "uptrend", "downtrend", "top", "base"], default="transition")
    return pd.DataFrame(val, index=px.index, columns=px.columns)


def worst_day_in_window(rel_ret: pd.DataFrame, window: int) -> tuple[np.ndarray, np.ndarray]:
    """每个 (t, code)：窗口内相对跌幅最大的那一行的行号与当日相对收益。全空窗口行号为 -1。"""
    a = rel_ret.to_numpy(dtype=float)
    t_n, c_n = a.shape
    idx = np.full((t_n, c_n), -1, dtype=np.int32)
    val = np.full((t_n, c_n), np.nan, dtype=np.float64)
    filled = np.where(np.isnan(a), np.inf, a)
    if t_n == 0:
        return idx, val
    for t in range(t_n):
        lo = max(0, t - window + 1)
        win = filled[lo: t + 1]
        k = win.argmin(axis=0)
        v = win[k, np.arange(c_n)]
        ok = np.isfinite(v)
        idx[t, ok] = lo + k[ok]
        val[t, ok] = v[ok]
    return idx, val
