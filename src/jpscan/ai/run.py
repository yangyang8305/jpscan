"""AI 分析运行：对候选池逐股组装证据包 → 调用 → 校验引用 → 写入 mart.ai_analysis（规划文档 5.5、13 章）。"""
from __future__ import annotations

import json
import logging
from datetime import date

import pandas as pd

from ..config import pget
from .client import ClaudeClient
from .evidence import build_pack, pack_to_text, validate_citations
from .tasks import PROMPT_VERSION, STOCK_CARD_SCHEMA, SYSTEM_ANALYST, stock_card_prompt

log = logging.getLogger("jpscan.ai.run")


def rule_summary(row: pd.Series) -> str:
    parts = [
        f"篮子={row.get('bucket')} 排名={row.get('bucket_rank')} 复合分={_r(row.get('composite_score'))}",
        f"六维分：低估 {_r(row.get('score_value'))} / 成长 {_r(row.get('score_growth'))} / 质量 {_r(row.get('score_quality'))} / 错杀 {_r(row.get('score_oversold'))} / 趋势 {_r(row.get('score_trend'))} / 新闻 {_r(row.get('score_news'))}",
        f"候选标志：低估={bool(row.get('value_candidate'))}（核心命中 {row.get('value_core_hits')}，陷阱={bool(row.get('value_trap'))}），成长={bool(row.get('growth_candidate'))}，质量门槛={bool(row.get('quality_gate'))}，错杀={bool(row.get('oversold_candidate'))}（归因 {row.get('attr_type')}）",
        f"排除码={row.get('exclusion_codes') or '无'}；软标记={row.get('soft_flags') or '无'}；数据：最新财报生效日 {row.get('effective_date')}，有预想={bool(row.get('has_forecast'))}",
    ]
    return "\n".join(parts)


def _r(v) -> str:
    try:
        return f"{float(v):.1f}"
    except (TypeError, ValueError):
        return "NA"


def analyze_day(store, ledger, params: dict, d: date, run_id: str, max_per_bucket: int | None = None, model: str | None = None,
                effort: str = "high") -> pd.DataFrame:
    client = ClaudeClient(ledger, store.paths.ai_cache)
    model = model or str(pget(params, "ai.models.deep_analysis", "claude-opus-5-5"))
    cap = float(pget(params, "ai.adjustment_cap", 10))
    part = d.strftime("%Y-%m")
    scores = store.read("mart", "scores_daily", where=f"Date = DATE '{d.isoformat()}'", partitions=[part])
    if scores.empty:
        return pd.DataFrame()
    top_n = max_per_bucket or int(pget(params, "bucket.top_n", 40))
    pool = scores[scores["bucket"].notna() & (scores["bucket_rank"] <= top_n)].sort_values(["bucket", "bucket_rank"])
    statements = store.read("staging", "financial_statements") if store.exists("staging", "financial_statements") else None
    disclosures = store.read("staging", "disclosures") if store.exists("staging", "disclosures") else None
    news = store.read("staging", "news_items") if store.exists("staging", "news_items") else None
    ov = store.read("mart", "oversold_daily", where=f"Date = DATE '{d.isoformat()}'", partitions=[part]) if store.exists("mart", "oversold_daily") else pd.DataFrame()
    ov = ov.set_index("Code") if not ov.empty else ov
    out_rows = []
    for _, row in pool.iterrows():
        code = str(row["Code"])
        pack = build_pack(ledger, row, statements, disclosures, news, ov.loc[code] if (not ov.empty and code in ov.index) else None, d)
        res = client.structured(task="stock_card", prompt_version=PROMPT_VERSION, model=model, system=SYSTEM_ANALYST,
                                user=stock_card_prompt(pack_to_text(pack), rule_summary(row)), schema=STOCK_CARD_SCHEMA,
                                payload_for_key={"code": code, "as_of": d.isoformat(), "evidence_ids": pack["evidence_ids"], "rules": rule_summary(row)},
                                effort=effort, run_id=run_id, internal_id=code)
        if res is None:
            out_rows.append({"Code": code, "Date": d, "run_id": run_id, "status": "unavailable" if not client.available else "failed",
                             "model": model, "prompt_version": PROMPT_VERSION, "n_evidence": len(pack["evidence_ids"]), "analysis_json": None,
                             "ai_adjustment_total": 0.0, "citation_hit_rate": None, "n_invalid_citations": None})
            continue
        val = validate_citations(res, pack)
        adj = 0.0
        clean_adj = []
        for a in res.get("adjustments", []):
            delta = max(-cap, min(cap, float(a.get("delta", 0))))
            ok = all(e in pack["evidence_ids"] for e in a.get("evidence_ids", [])) and len(a.get("evidence_ids", [])) > 0
            if ok:
                clean_adj.append({**a, "delta": delta})
                adj += delta
        res["adjustments_applied"] = clean_adj
        res["citation_check"] = val
        out_rows.append({"Code": code, "Date": d, "run_id": run_id, "status": "ok" if val["n_invalid"] == 0 else "invalid_citations",
                         "model": model, "prompt_version": PROMPT_VERSION, "n_evidence": len(pack["evidence_ids"]),
                         "analysis_json": json.dumps(res, ensure_ascii=False), "ai_adjustment_total": max(-cap * 2, min(cap * 2, adj)),
                         "citation_hit_rate": val["hit_rate"], "n_invalid_citations": val["n_invalid"]})
    out = pd.DataFrame(out_rows)
    if not out.empty:
        store.write_merge("mart", "ai_analysis", out, part, ["Code", "Date"])
    return out
