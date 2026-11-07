"""证据包组装（规划文档 5.3、16.1 节）：把结构化指标、财报记录、披露事件、错杀信息打包成带 evidence_id 的片段，并登记到账本。"""
from __future__ import annotations

import json
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from ..ledger import Ledger
from ..util.common import sha256_bytes, stable_json

METRIC_FIELDS = [
    "C", "mcap", "pbr", "per_fwd", "per_ttm", "div_yield_fwd", "cash_to_mcap", "fcf_yield", "ey_spread", "roe_avg", "equity_ratio",
    "accrual_avg", "rev_growth_1y", "op_growth_1y", "rev_cagr_3y", "op_cagr_3y", "f_op_growth", "progress_rate", "progress_dev",
    "n_up_revisions", "n_down_revisions", "last_revision_chg", "drawdown_60d", "rs_63d", "stage", "rvol60", "pct_pbr", "pct_per_fwd",
    "pct_fcf_yield", "pct_div_yield_fwd", "self_pct_pbr", "score_value", "score_growth", "score_quality", "score_oversold",
    "score_trend", "score_news", "bucket", "bucket_rank", "value_trap", "quality_gate", "exclusion_codes", "soft_flags",
]
STATEMENT_FIELDS = ["DiscDate", "DiscTime", "DocType", "CurPerType", "CurFYEn", "Sales", "OP", "OdP", "NP", "EPS", "EqAR", "BPS", "CFO", "CFI",
                    "CashEq", "DivAnn", "FSales", "FOP", "FNP", "FEPS", "FDivAnn", "NxFSales", "NxFOP", "NxFEPS", "NxFDivAnn"]


def _clean(v: Any) -> Any:
    if isinstance(v, (np.floating, float)):
        return None if (v is None or np.isnan(v)) else round(float(v), 6)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (pd.Timestamp, date)):
        return str(v)[:10]
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return v


def build_pack(ledger: Ledger, score_row: pd.Series, statements: pd.DataFrame | None, disclosures: pd.DataFrame | None,
               news: pd.DataFrame | None, oversold_row: pd.Series | None, as_of: date) -> dict[str, Any]:
    code = str(score_row["Code"])
    items: list[dict[str, Any]] = []
    # 1. 指标
    metrics = {k: _clean(score_row.get(k)) for k in METRIC_FIELDS if k in score_row.index}
    eid = f"EV-metric-{code}-{as_of.isoformat()}"
    snippet = stable_json(metrics)
    ledger.add_evidence(eid, "metric", f"features_daily:{code}:{as_of}", None, as_of.isoformat(), {"table": "scores_daily"}, snippet[:4000], "zh",
                        sha256_bytes(snippet.encode()), code)
    items.append({"evidence_id": eid, "type": "metrics", "content": metrics})
    # 2. 财报记录（最近 8 条）
    if statements is not None and not statements.empty:
        st = statements[statements["Code"] == code].sort_values(["DiscDate", "DiscNo"]).tail(8)
        for _, r in st.iterrows():
            rec = {k: _clean(r.get(k)) for k in STATEMENT_FIELDS if k in r.index}
            sid = f"EV-jquants-fins_summary-{code}-{r['DiscNo']}"
            sn = stable_json(rec)
            ledger.add_evidence(sid, "jquants", f"fins_summary:{r['DiscNo']}", None, str(r.get("DiscDate"))[:10], {"DiscNo": str(r["DiscNo"])},
                                sn[:4000], "ja", sha256_bytes(sn.encode()), code)
            items.append({"evidence_id": sid, "type": "statement", "content": rec})
    # 3. 披露事件（最近 180 天，最多 25 条）
    if disclosures is not None and not disclosures.empty:
        dd = disclosures[(disclosures["Code"] == code)].copy()
        dd["disclosed_date"] = pd.to_datetime(dd["disclosed_date"])
        dd = dd[(dd["disclosed_date"] <= pd.Timestamp(as_of)) & (dd["disclosed_date"] >= pd.Timestamp(as_of) - pd.Timedelta(days=180))]
        dd = dd[dd.get("lang", "ja") != "en"].sort_values("disclosed_date", ascending=False).head(25)
        for _, r in dd.iterrows():
            did = f"EV-tdnet-{r['disclosure_id']}"
            items.append({"evidence_id": did, "type": "disclosure", "content": {
                "date": str(r["disclosed_date"])[:10], "time": r.get("disclosed_time"), "title": r.get("title"), "category": r.get("category"),
                "direction": _clean(r.get("direction")), "strength": _clean(r.get("strength")), "url": r.get("pdf_url")}})
    # 4. 新闻标题（7 天）
    if news is not None and not news.empty and "matched_codes" in news.columns:
        nn = news[news["matched_codes"].astype(str).str.contains(code, na=False)].copy()
        nn["published_date"] = pd.to_datetime(nn["published_date"])
        nn = nn[(nn["published_date"] <= pd.Timestamp(as_of)) & (nn["published_date"] >= pd.Timestamp(as_of) - pd.Timedelta(days=7))].head(10)
        for _, r in nn.iterrows():
            nid = f"EV-news-{r['news_id']}"
            ledger.add_evidence(nid, "news", str(r["news_id"]), r.get("link"), str(r.get("published_at")), {}, str(r.get("title"))[:500], "ja",
                                sha256_bytes(str(r.get("title")).encode()), code)
            items.append({"evidence_id": nid, "type": "headline", "content": {"date": str(r["published_date"])[:10], "source": r.get("source"),
                                                                            "title": r.get("title")}})
    # 5. 错杀信息
    if oversold_row is not None:
        ov = {k: _clean(oversold_row.get(k)) for k in ("trigger_date", "trigger_type", "attr_type", "attr_category", "event_move", "intact_pass",
                                                        "reaction_ratio", "stab_signals", "stabilized", "score_oversold") if k in oversold_row.index}
        oid = f"EV-metric-{code}-{as_of.isoformat()}-oversold"
        sn = stable_json(ov)
        ledger.add_evidence(oid, "metric", f"oversold_daily:{code}:{as_of}", None, as_of.isoformat(), {"table": "oversold_daily"}, sn[:2000], "zh",
                            sha256_bytes(sn.encode()), code)
        items.append({"evidence_id": oid, "type": "oversold", "content": ov})
    return {"code": code, "name": score_row.get("CoName"), "as_of": as_of.isoformat(), "sector33": score_row.get("S33"),
            "market": score_row.get("Mkt"), "items": items, "evidence_ids": [it["evidence_id"] for it in items]}


def validate_citations(output: dict[str, Any], pack: dict[str, Any]) -> dict[str, Any]:
    """检查输出中引用的 evidence_id 是否都在证据包内；返回统计并标记无效引用。"""
    valid = set(pack["evidence_ids"])
    cited: list[str] = []

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("evidence_ids", "evidence_id"):
                    if isinstance(v, list):
                        cited.extend(str(x) for x in v)
                    elif isinstance(v, str):
                        cited.append(v)
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(output)
    bad = sorted({c for c in cited if c not in valid})
    return {"n_cited": len(cited), "n_invalid": len(bad), "invalid": bad, "hit_rate": (1 - len(bad) / len(cited)) if cited else None}


def pack_to_text(pack: dict[str, Any]) -> str:
    lines = [f"# 证据包：{pack['code']} {pack.get('name') or ''}（基准日 {pack['as_of']}，業種 {pack.get('sector33')}，市場 {pack.get('market')}）", ""]
    for it in pack["items"]:
        lines.append(f"[{it['evidence_id']}] ({it['type']}) {json.dumps(it['content'], ensure_ascii=False)}")
    return "\n".join(lines)
