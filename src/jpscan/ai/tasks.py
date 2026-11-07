"""AI 任务定义：提示模板（版本化）与输出 Schema（附录 E.4）。"""
from __future__ import annotations

PROMPT_VERSION = "p0.1"

SYSTEM_ANALYST = """你是一名受证据约束的日本股票分析员。规则：
1. 你只能引用「证据包」中出现的 evidence_id；每个结论都必须附 evidence_ids。没有证据支持的推断必须放在 inferences 列表中并说明依据。
2. 你不产生任何新的数字；引用数字时必须与证据包一致，并保留日文原文术语（如 営業利益、業績予想、進捗率）。
3. 不要承诺收益；用概率与条件表达；明确失效条件。
4. 文本措辞的乐观程度不是正向信号；日本市场的経営者予想平均偏乐观、修正事件与進捗率才是关键。
5. 输出中文，保留日文关键词与证据 ID。"""

STOCK_CARD_SCHEMA = {
    "type": "object",
    "properties": {
        "one_liner": {"type": "string"},
        "reasons": {"type": "array", "items": {"type": "object", "properties": {"text": {"type": "string"}, "evidence_ids": {"type": "array", "items": {"type": "string"}}},
                                               "required": ["text", "evidence_ids"], "additionalProperties": False}, "maxItems": 4},
        "bear_points": {"type": "array", "items": {"type": "object", "properties": {"text": {"type": "string"}, "evidence_ids": {"type": "array", "items": {"type": "string"}}},
                                                   "required": ["text", "evidence_ids"], "additionalProperties": False}, "maxItems": 5},
        "invalidation_conditions": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "uncertainties": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "watch_points": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "attribution": {"type": "object", "properties": {"type": {"type": "string", "enum": ["T1", "T2", "T3", "T4", "T5", "T6", "NA"]},
                                                         "explanation": {"type": "string"}, "evidence_ids": {"type": "array", "items": {"type": "string"}}},
                        "required": ["type", "explanation", "evidence_ids"], "additionalProperties": False},
        "adjustments": {"type": "array", "items": {"type": "object", "properties": {
            "dimension": {"type": "string", "enum": ["value", "growth", "quality", "oversold", "trend", "news"]},
            "delta": {"type": "number"}, "reason": {"type": "string"}, "evidence_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["dimension", "delta", "reason", "evidence_ids"], "additionalProperties": False}, "maxItems": 6},
        "verdicts": {"type": "object", "properties": {
            "value_candidate": {"type": "string", "enum": ["支持", "反对", "不确定"]},
            "growth_candidate": {"type": "string", "enum": ["支持", "反对", "不确定"]},
            "quality_gate": {"type": "string", "enum": ["支持", "反对", "不确定"]},
            "oversold_candidate": {"type": "string", "enum": ["支持", "反对", "不确定", "不适用"]}},
            "required": ["value_candidate", "growth_candidate", "quality_gate", "oversold_candidate"], "additionalProperties": False},
        "inferences": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["one_liner", "reasons", "bear_points", "invalidation_conditions", "uncertainties", "watch_points", "attribution",
                 "adjustments", "verdicts", "inferences", "confidence"],
    "additionalProperties": False,
}


def stock_card_prompt(pack_text: str, rule_summary: str) -> str:
    return f"""下面是规则引擎对该股票的判断摘要，以及证据包。请完成：
(1) 一句话论点；(2) 最多 4 条核心理由（每条附 evidence_ids）；(3) 最多 5 条空头论点；(4) 失效条件；(5) 关键不确定性；(6) 需跟踪的验证点；
(7) 若该股处于错杀候选，判断下跌归因类型（T1 業績型 / T2 資本型 / T3 事件型 / T4 需給型 / T5 系統型 / T6 不明；非错杀候选填 NA）；
(8) 对六个维度分数提出调整建议（每项 -10 到 +10 之间，必须给证据）；(9) 对规则引擎的四个判断给出 支持/反对/不确定。

## 规则引擎摘要
{rule_summary}

## 证据包
{pack_text}
"""


CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string"}, "direction": {"type": "integer", "enum": [-1, 0, 1]}, "strength": {"type": "integer", "minimum": -3, "maximum": 3},
        "extracted_values": {"type": "object", "properties": {"revision_pct_op": {"type": ["number", "null"]}, "amount_jpy": {"type": ["number", "null"]},
                                                              "dilution_pct": {"type": ["number", "null"]}, "price": {"type": ["number", "null"]}},
                             "required": ["revision_pct_op", "amount_jpy", "dilution_pct", "price"], "additionalProperties": False},
        "rationale": {"type": "string"}, "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["category", "direction", "strength", "extracted_values", "rationale", "confidence"], "additionalProperties": False,
}

SYSTEM_CLASSIFIER = """你是日本上市公司适时披露（TDnet）标题分类器。按给定的事件分类表（E01–E40）输出类别、方向与强度；只根据标题判断，不要臆测正文；无法判断时类别为 E40 且 confidence 低。"""


def classify_prompt(title: str, taxonomy_text: str) -> str:
    return f"分类表：\n{taxonomy_text}\n\n标题：{title}\n"
