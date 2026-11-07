"""披露事件汇总（规划文档 4.7 节，registry 把手工总分 news_handcrafted_score 定为 excluded：不参与判断，只展示）。

事件强度 × 来源权重 × 半衰期衰减 → 0–100，并输出红旗、催化剂、稀释与负面事件数；红旗和稀释会进入排除与门槛。

覆盖范围诚实：TDnet 免费渠道只有最近 31 天，Free 行情又晚 12 周，两者常常对不上。日期不在披露数据覆盖范围内时不输出行，
由 coverage_status 给出 news_status，screener 按「中性分 + unknown」处理：没有数据不是没有坏消息。

输入：disclosures（TDnet，L1）、news_items（RSS 标题，只作关注度）。输出：mart.event_scores_daily。
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger("jpscan.news")

RED_FLAG_CATS = {"E10": "ms_warrant", "E17": "improper_accounting", "E20": "supervision", "E21": "delisting", "E22": "earnings_release_delay",
                 "E23": "auditor_change", "E24": "going_concern"}
CATALYST_CATS = {"E05": "buyback", "E29": "capital_cost_disclosure", "E31": "payout_policy", "E32": "activist", "E11": "tob", "E12": "mbo"}
DILUTION_CATS = {"E07", "E08", "E09"}
RED_FLAG_DAYS = 120
CATALYST_DAYS = 365
NEUTRAL = 50.0


def load_taxonomy(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    return y.get("events", y)


def coverage_status(disc: pd.DataFrame | None, d: date, full_lookback_days: int = 30) -> str:
    """某日的披露数据状态：unknown（不覆盖）/ partial（只覆盖了窗口的一部分）/ ok。"""
    if disc is None or disc.empty:
        return "unknown"
    lo, hi = pd.to_datetime(disc["disclosed_date"]).min(), pd.to_datetime(disc["disclosed_date"]).max()
    ts = pd.Timestamp(d)
    if ts < lo or ts > hi + pd.Timedelta(days=3):
        return "unknown"
    return "ok" if (ts - lo).days >= full_lookback_days else "partial"


def build_event_scores(disc: pd.DataFrame | None, news: pd.DataFrame | None, dates: list[date], params: dict, taxonomy: dict | None = None) -> pd.DataFrame:
    """只对披露数据覆盖到的日期计算，并把每个事件只展开到它还起作用的日子（向量化，500 个日期也不会爆）。"""
    np_ = params.get("news", {})
    w = np_.get("source_weights", {"L1": 1.0, "L2": 0.6, "L3": 0.3, "L4": 0.0})
    clip = float(np_.get("raw_score_clip", 100))
    cap_rule = np_.get("negative_event_cap_rule", {"window_days": 30, "min_negative_l1_events": 2, "news_score_cap": 40})
    empty = pd.DataFrame(columns=["Code", "Date", "news_score", "news_status"])
    if disc is None or disc.empty or not len(dates):
        return empty
    ev = disc.copy()
    ev["disclosed_date"] = pd.to_datetime(ev["disclosed_date"])
    lo_cov, hi_cov = ev["disclosed_date"].min(), ev["disclosed_date"].max()
    if "lang" in ev.columns:
        ev = ev[ev["lang"].fillna("ja") != "en"]
    ev = ev[ev["Code"].notna() & ev["category"].notna()]
    if ev.empty:
        return empty
    ev["strength"] = pd.to_numeric(ev.get("strength", 0), errors="coerce").fillna(0.0)
    ev["half_life_days"] = pd.to_numeric(ev.get("half_life_days", 20), errors="coerce").fillna(20.0).replace(0, 1)
    dts = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(dates)))).sort_values().unique()
    dts = dts[(dts >= lo_cov) & (dts <= hi_cov + pd.Timedelta(days=3))]
    if len(dts) == 0:
        log.info("disclosures 覆盖 %s 至 %s，与请求日期没有重叠", lo_cov.date(), hi_cov.date())
        return empty
    # 每个事件起作用的天数：衰减窗口、红旗窗口、催化剂窗口取最大
    window_days = np.maximum(ev["half_life_days"].to_numpy() * 5, CATALYST_DAYS)
    start = np.searchsorted(dts.values, ev["disclosed_date"].to_numpy(dtype="datetime64[ns]"), side="left")
    stop = np.searchsorted(dts.values, (ev["disclosed_date"] + pd.to_timedelta(window_days, unit="D")).to_numpy(dtype="datetime64[ns]"), side="right")
    counts = np.maximum(stop - start, 0)
    idx = np.repeat(np.arange(len(ev)), counts)
    if len(idx) == 0:
        return empty
    offs = np.concatenate([np.arange(a, b) for a, b in zip(start, stop) if b > a])
    long = pd.DataFrame({
        "Code": ev["Code"].to_numpy(dtype=object)[idx], "Date": dts.values[offs], "cat": ev["category"].to_numpy(dtype=object)[idx],
        "strength": ev["strength"].to_numpy()[idx], "half": ev["half_life_days"].to_numpy()[idx], "ddate": ev["disclosed_date"].to_numpy()[idx],
    })
    long["age"] = (long["Date"] - long["ddate"]).dt.days.clip(lower=0)
    decay = 0.5 ** (long["age"] / long["half"])
    long["contrib"] = (long["strength"] * float(w.get("L1", 1.0)) * decay).where(long["age"] <= long["half"] * 5, 0.0)
    keys = ["Code", "Date"]
    g = long.groupby(keys, sort=False)
    res = g["contrib"].sum().clip(-clip, clip).reset_index()
    res["news_score"] = (res["contrib"] / clip + 1) * NEUTRAL
    neg = long[(long["age"] <= int(cap_rule.get("window_days", 30))) & (long["strength"] < 0)].groupby(keys).size().rename("n_neg_30d").reset_index()
    res = res.merge(neg, on=keys, how="left")
    res["n_neg_30d"] = res["n_neg_30d"].fillna(0).astype(int)
    capped = res["n_neg_30d"] >= int(cap_rule.get("min_negative_l1_events", 2))
    res.loc[capped, "news_score"] = np.minimum(res.loc[capped, "news_score"], float(cap_rule.get("news_score_cap", 40)))

    def flag_table(cats, age_max, prefix: str) -> None:
        nonlocal res
        for cat, name in cats.items():
            hit = long.loc[(long["cat"] == cat) & (long["age"] <= age_max), keys].drop_duplicates()
            col = f"{prefix}{name}"
            res = res.merge(hit.assign(**{col: True}), on=keys, how="left")
            res[col] = res[col].fillna(False).astype(bool)

    flag_table(RED_FLAG_CATS, RED_FLAG_DAYS, "flag_")
    flag_table(CATALYST_CATS, CATALYST_DAYS, "catalyst_")
    dil = long.loc[long["cat"].isin(DILUTION_CATS) & (long["age"] <= CATALYST_DAYS), keys].drop_duplicates()
    res = res.merge(dil.assign(recent_dilution=True), on=keys, how="left")
    res["recent_dilution"] = res["recent_dilution"].fillna(False).astype(bool)
    last_ev = long.sort_values("age").drop_duplicates(keys)[[*keys, "cat", "age"]].rename(columns={"cat": "last_event_cat", "age": "last_event_age"})
    res = res.merge(last_ev, on=keys, how="left").drop(columns=["contrib"])
    res["news_status"] = np.where((res["Date"] - lo_cov).dt.days >= 30, "ok", "partial")
    res["Date"] = res["Date"].dt.date
    if news is not None and not news.empty and "matched_codes" in news.columns:
        res = _attach_attention(res, news)
    return res


def _attach_attention(out: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    n = news[["published_date", "matched_codes"]].dropna().copy()
    n = n[n["matched_codes"].astype(str).str.len() > 0]
    n = n.assign(Code=n["matched_codes"].astype(str).str.split(",")).explode("Code")
    n["published_date"] = pd.to_datetime(n["published_date"])
    if n.empty:
        out["headlines_7d"] = 0
        return out
    per_day = n.groupby(["Code", "published_date"]).size().rename("c").reset_index()
    dates = pd.DataFrame({"Date": pd.to_datetime(pd.Series(out["Date"].unique()))}).sort_values("Date")
    idx = np.searchsorted(np.sort(per_day["published_date"].unique()), dates["Date"].to_numpy(dtype="datetime64[ns]"), side="right")
    frames = []
    pub = np.sort(per_day["published_date"].unique())
    for d, hi in zip(dates["Date"], idx):
        days = pub[max(0, hi - 8):hi]
        days = days[(days <= np.datetime64(d)) & (days > np.datetime64(d - pd.Timedelta(days=7)))]
        c = per_day[per_day["published_date"].isin(days)].groupby("Code")["c"].sum()
        frames.append(pd.DataFrame({"Code": c.index, "Date": d.date(), "headlines_7d": c.to_numpy()}))
    att = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["Code", "Date", "headlines_7d"])
    out = out.merge(att, on=["Code", "Date"], how="left")
    out["headlines_7d"] = out["headlines_7d"].fillna(0).astype(int)
    return out
