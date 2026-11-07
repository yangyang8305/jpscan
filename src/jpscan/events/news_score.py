"""新闻/披露分（规划文档 4.7 节）：事件强度 × 来源权重 × 半衰期衰减 → 0–100；并输出红旗、催化剂与软标记。

输入：disclosures（TDnet，L1）、news_items（RSS 标题，L2/L3，仅关注度）、事件分类表 config/events.yaml。
输出：mart.event_scores_daily（Code, Date, news_score, red_flag_*, catalyst_*, n_neg_30d ...）。
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger("jpscan.news")

RED_FLAG_CATS = {"E10": "ms_warrant", "E17": "improper_accounting", "E20": "supervision", "E21": "delisting", "E22": "earnings_release_delay",
                 "E23": "auditor_change", "E24": "going_concern"}
CATALYST_CATS = {"E05": "buyback", "E29": "capital_cost_disclosure", "E31": "payout_policy", "E32": "activist", "E11": "tob", "E12": "mbo"}
DILUTION_CATS = {"E07", "E08", "E09"}


def load_taxonomy(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    return y.get("events", y)


def build_event_scores(disc: pd.DataFrame | None, news: pd.DataFrame | None, dates: list[date], params: dict,
                       taxonomy: dict | None = None) -> pd.DataFrame:
    np_ = params.get("news", {})
    w = np_.get("source_weights", {"L1": 1.0, "L2": 0.6, "L3": 0.3, "L4": 0.0})
    clip = float(np_.get("raw_score_clip", 100))
    cap_rule = np_.get("negative_event_cap_rule", {"window_days": 30, "min_negative_l1_events": 2, "news_score_cap": 40})
    rows: list[dict] = []
    if disc is None or disc.empty:
        return pd.DataFrame(columns=["Code", "Date", "news_score"])
    ev = disc.copy()
    ev["disclosed_date"] = pd.to_datetime(ev["disclosed_date"])
    ev = ev[ev["lang"].fillna("ja") != "en"] if "lang" in ev.columns else ev
    ev["strength"] = pd.to_numeric(ev.get("strength", 0), errors="coerce").fillna(0.0)
    ev["half_life_days"] = pd.to_numeric(ev.get("half_life_days", 20), errors="coerce").fillna(20.0)
    ev["weight"] = float(w.get("L1", 1.0))
    ev = ev[ev["Code"].notna()]
    codes = ev["Code"].unique()
    dts = pd.to_datetime(pd.Series(dates))
    # 为控制计算量：每个事件只影响 ≤ 5 个半衰期
    out_frames = []
    for d in dts:
        win_lo = d - pd.Timedelta(days=600)
        e = ev[(ev["disclosed_date"] <= d) & (ev["disclosed_date"] >= win_lo)]
        if e.empty:
            continue
        age = (d - e["disclosed_date"]).dt.days.clip(lower=0)
        decay = 0.5 ** (age / e["half_life_days"].replace(0, 1))
        contrib = e["strength"] * e["weight"] * decay
        contrib = contrib.where(age <= e["half_life_days"] * 5, 0.0)
        tmp = pd.DataFrame({"Code": e["Code"].values, "contrib": contrib.values, "cat": e["category"].values,
                            "age": age.values, "strength": e["strength"].values})
        agg = tmp.groupby("Code")["contrib"].sum().clip(-clip, clip)
        score = (agg / clip + 1) * 50.0
        neg30 = tmp[(tmp["age"] <= int(cap_rule.get("window_days", 30))) & (tmp["strength"] < 0)].groupby("Code").size()
        capped = neg30[neg30 >= int(cap_rule.get("min_negative_l1_events", 2))].index
        score.loc[score.index.isin(capped)] = np.minimum(score.loc[score.index.isin(capped)], float(cap_rule.get("news_score_cap", 40)))
        frame = pd.DataFrame({"Code": score.index, "news_score": score.values})
        frame["Date"] = d.date()
        # 红旗（120 天内）/ 催化剂（365 天内）/ 稀释（365 天）/ 负面事件数
        recent = tmp[tmp["age"] <= 120]
        for cat, name in RED_FLAG_CATS.items():
            hit = recent[recent["cat"] == cat]["Code"].unique()
            frame[f"flag_{name}"] = frame["Code"].isin(hit)
        yr = tmp[tmp["age"] <= 365]
        for cat, name in CATALYST_CATS.items():
            hit = yr[yr["cat"] == cat]["Code"].unique()
            frame[f"catalyst_{name}"] = frame["Code"].isin(hit)
        frame["recent_dilution"] = frame["Code"].isin(yr[yr["cat"].isin(DILUTION_CATS)]["Code"].unique())
        frame["n_neg_30d"] = frame["Code"].map(neg30).fillna(0).astype(int)
        last_ev = tmp.sort_values("age").drop_duplicates("Code")
        frame["last_event_cat"] = frame["Code"].map(last_ev.set_index("Code")["cat"])
        frame["last_event_age"] = frame["Code"].map(last_ev.set_index("Code")["age"])
        out_frames.append(frame)
    if not out_frames:
        return pd.DataFrame(columns=["Code", "Date", "news_score"])
    out = pd.concat(out_frames, ignore_index=True)
    if news is not None and not news.empty and "matched_codes" in news.columns:
        out = _attach_attention(out, news)
    return out


def _attach_attention(out: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    n = news[["published_date", "matched_codes"]].dropna().copy()
    n = n[n["matched_codes"].astype(str).str.len() > 0]
    n = n.assign(Code=n["matched_codes"].astype(str).str.split(",")).explode("Code")
    n["published_date"] = pd.to_datetime(n["published_date"])
    cnt = []
    for d, grp in out.groupby("Date"):
        dd = pd.to_datetime(d)
        recent = n[(n["published_date"] <= dd) & (n["published_date"] > dd - pd.Timedelta(days=7))]
        c = recent.groupby("Code").size()
        cnt.append(pd.DataFrame({"Code": grp["Code"].values, "Date": d, "headlines_7d": grp["Code"].map(c).fillna(0).astype(int).values}))
    if cnt:
        out = out.merge(pd.concat(cnt), on=["Code", "Date"], how="left")
    return out
