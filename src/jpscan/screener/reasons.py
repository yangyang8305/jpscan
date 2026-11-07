"""筛选理由（规划文档 4.11 节）：确定性生成，不用 AI。

使用者不是金融专业人士，所以每条理由都带注册表里那句白话（plain_zh），并写出实际数字。

稳定接口（报告模块使用）：
- build_reasons(row, params) -> dict：reasons、counter_reasons、unknowns、selection_summary_zh
- exclusion_reason(row, params) -> str：把每个 X 码和每条未满足的篮子要求翻成一句中文

row 是 mart.scores_daily 的一行（Series 或 dict）。这两个函数只读 family_status、bucket_check 等存储列，
所以可以脱离筛选过程单独调用。text_zh 里的数字都由存储值格式化得到，不再次计算。
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ..config import pget
from ..features.registry import load_registry

# 指标的展示单位（公式绑定，新指标缺省按百分比之外的原值显示）
UNITS: dict[str, tuple[str, float, int, str]] = {
    # metric: (单位后缀, 乘数, 小数位, 比较方式)
    "val_pbr": ("倍", 1.0, 2, "lower"),
    "val_fwd_ep": ("%", 100.0, 2, "higher"),
    "val_ocf_yield": ("%", 100.0, 2, "higher"),
    "yield_dividend": ("%", 100.0, 2, "higher"),
    "qual_fscore7": ("分", 1.0, 1, "higher"),
    "qual_op_profitability": ("%", 100.0, 2, "higher"),
    "guid_forecast_innovation": ("%", 100.0, 2, "higher"),
    "guid_revision_event": ("%", 100.0, 2, "higher"),
    "evt_dividend_change": ("", 1.0, 0, "higher"),
    "rev_1m": ("%", 100.0, 2, "lower"),
}
BASIS_ZH = {"odp": "经常利益", "op": "营业利润", "np": "净利润"}
EXCLUSION_ZH = {
    "X01": "不是东证三个市场的普通股（ETF、REIT、优先股等不在筛选范围）。",
    "X02": "被交易所列入监理或整理名单，或已决定退市。",
    "X03": "公告里有持续经营存疑的注记。",
    "X04": "公告里有不当会计或第三者委员会调查尚未结束。",
    "X05": "审计意见有问题，或财报发布被推迟。",
    "X06": "发行了会持续压低股价的 MS 权证，尚未行使完。",
    "X07": "资不抵债：净资产小于等于零。",
    "X08": "成交额或市值太小，低于观察门槛，买卖不动。",
    "X09": "财务数据不足：还没有一份完整的年度财报。",
    "X10": "未达到上市维持标准，正处于改善期间。",
    "X11": "已有收购要约且价格接近收购价，没有上涨空间。",
}
SOFT_FLAG_ZH = {
    "S01": ("流通盘很小或母公司持股很高", "买卖时可能更难成交。"),
    "S03": ("几天内要发财报", "财报前不确定性大，暂缓新买入。"),
    "S04": ("最近一年有增发或可转债", "老股东的份额可能被稀释。"),
    "S05": ("30 天内有多条负面公告", "需要先读公告内容。"),
    "S06": ("金融业", "金融业用专门的尺子，部分通用指标不适用。"),
    "S07": ("股价低于 100 円", "低价股波动和交易成本都偏大。"),
    "STALE": ("财务数据偏旧", "最新一份财报距今已超过 135 天。"),
    "NOFCST": ("公司没有公布全年预测", "依赖预测的指标算不出，不算坏事也不当作 0。"),
    "RETRO": ("财报被追溯修正过", "请人工核对修正内容。"),
}
COUNTER_GATES = {
    # 门槛：触发时写入反面理由
    "gate_one_off": "预计净利润接近或超过主业利润，低 PER 可能是一次性收益造成的假便宜。",
    "gate_div_cover": "预计利润不够付承诺的分红，股息率不算命中。",
    "gate_distress": "最近两年主业都亏损、现金也在流出。",
    "gate_sector_equity": "自有资本比率处于行业最低的一成，而且低于 30%。",
    "gate_miss_then_bullish": "去年没完成目标，今年又定了很激进的增长目标。",
    "guid_remaining_growth": "要完成全年目标，剩余期间必须比去年多赚很多，目标可能不现实。",
    "gate_inorganic": "决算期变更或合并范围变化较大，同比数字可能不可比。",
    "gate_dilution": "股数同比大幅增加，或最近有增发。",
    "gate_disclosure_delay": "财报发布比规定晚，请人工复核。",
    "gate_zombie": "连续三期主业利润不够付利息。",
    "gate_debt_coverage": "净负债相对 EBITDA 偏高，或主业利润不够付利息。",
}
FAIL_CODE_ZH = {
    "not_universe": "不在筛选范围内。",
    "not_core": "成交额或市值低于流动性核心池门槛（日均成交额 5,000 万円、市值 50 億円），篮子只取核心池。",
    "hard_excluded": "已被硬排除（见上）。",
    "excluded_sector": "这个行业不进这个篮子（通用指标对它不适用）。",
    "excluded_market": "这个市场不启用这个篮子。",
    "confirm_miss": "只有估值便宜，没有另一类证据（经营质量、股东回报、指引变化、资本行动都没有命中）。",
    "cyclical_yield_only": "周期行业只靠股息确认不够：周期顶部时估值和股息率会同时偏好。",
    "intact_fail": "基本面完好检验没通过：预测下修、减配、全年目标不现实或财务困境，至少出现一项。",
    "intact_unknown": "基本面完好检验的各项数据都缺失，无法判断。",
    "cause_wait": "下跌发生在财报附近，属于业绩型下跌，先等待 21 个交易日（翌期指引差时最长 60 日）。",
    "cause_hard_fail": "下跌原因是增发、会计问题这类硬伤。",
}


def _loads(v: Any) -> dict:
    if isinstance(v, dict):
        return v
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return {}
    return json.loads(v)


def _get(row, key, default=None):
    v = row.get(key, default) if hasattr(row, "get") else default
    return default if v is None or (isinstance(v, float) and np.isnan(v)) else v


def fmt_value(metric: str, value: float | None) -> str | None:
    """格式化指标值。text_zh 只用这个函数，所以文字里的数字一定来自存储值。"""
    if value is None:
        return None
    unit, mult, nd, _ = UNITS.get(metric, ("", 1.0, 4, "higher"))
    return f"{value * mult:.{nd}f}{unit}"


def round_value(metric: str, value: float | None) -> float | None:
    """存储值按展示精度取整，保证 value 与 text_zh 里的数字完全一致。"""
    if value is None:
        return None
    _, mult, nd, _ = UNITS.get(metric, ("", 1.0, 4, "higher"))
    return round(round(value * mult, nd) / mult, nd + 4)


def _pct_str(p: float | None) -> str | None:
    return None if p is None else f"{p * 100:.1f}%"


def _rule_text(metric: str, rule: dict, item: dict, label: str, plain: str, as_of: str, val_str: str) -> tuple[str, Any, str]:
    """返回 (text_zh, threshold, comparison)。"""
    kind = rule["rule"]
    th = rule.get("value")
    pct = item.get("pct")
    grp = item.get("grp")
    n = item.get("n")
    if kind in ("peer_best_pct", "core_best_pct", "group_best_pct"):
        where = "流动性核心池" if kind == "core_best_pct" else f"同业（{grp}）" if grp else "同业"
        n_str = f"共 {int(n)} 只，" if n else ""
        extra = ""
        if kind == "group_best_pct" and item.get("event_date"):
            extra = f"依据 {item['event_date']} 的年报（{BASIS_ZH.get(item.get('basis'), '')}口径，已减同业中位）。"
            where = "同口径公司"
        if item.get("basis") == "fallback":
            extra = "公司没有全年预测，改用过去 12 个月的实际盈利，与同业的同口径比较，可信度较低。"
        text = f"{label}：{plain}这只股票是 {val_str}，在{where}中{n_str}排在最好的 {_pct_str(pct)}，门槛是最好的 {th * 100:.0f}%。{extra}（截至 {as_of}）"
        return text, th, "pct_le"
    if kind == "min_value":
        text = f"{label}：{plain}这只股票 {val_str}，门槛是至少 {float(th):.1f} 分。（截至 {as_of}）"
        return text, th, "ge"
    if kind == "event_up_top_pct":
        ev = item.get("event_date")
        rank = f"在窗口内上调的公司中（共 {int(n)} 家）排在最好的 {_pct_str(pct)}，" if n else ""
        text = f"{label}：{plain}{ev} 上调，幅度相当于市值的 {val_str}，{rank}门槛是最好的 {th * 100:.0f}% 且不低于市值的 {float(rule.get('min_abs', 0)) * 100:.1f}%。（截至 {as_of}）"
        return text, th, "event_top_pct"
    if kind == "any_up_event":
        ev = item.get("event_date")
        prev, new = item.get("div_prev"), item.get("div_new")
        amounts = f"从每股 {prev:.2f} 円上调到 {new:.2f} 円" if prev is not None and new is not None else "上调"
        text = f"{label}：{plain}{ev} {amounts}，在最近 {int(rule.get('window_days', 30))} 个交易日内。（截至 {as_of}）"
        return text, rule.get("window_days"), "event_within"
    return f"{label}：{plain}这只股票是 {val_str}。（截至 {as_of}）", th, "value"


def _reason(fam: str, metric: str, item: dict, rule: dict, meta: dict, as_of: str) -> dict:
    label, plain = meta.get("name_zh", metric), meta.get("plain_zh", "")
    if plain and not plain.endswith(("。", "！", "？")):
        plain += "。"
    value = round_value(metric, item.get("value"))
    val_str = fmt_value(metric, value)
    text, th, comp = _rule_text(metric, rule, item, label, plain, as_of, val_str)
    if item.get("self_pct") is not None:
        note = "（历史还不长，只作参考）" if item.get("self_low_conf") else ""
        text = text.replace("（截至", f"相对它自己过去的水平排在 {_pct_str(item['self_pct'])}{note}。（截至", 1)
    return {
        "family": fam, "rule_id": f"{fam}.{metric}.{rule['rule']}", "metric": metric, "label_zh": label, "plain_zh": plain, "value": value, "threshold": th, "comparison": comp,
        "peer_pct": item.get("pct"), "self_pct": item.get("self_pct"), "self_low_confidence": item.get("self_low_conf"), "peer_group": item.get("grp"), "peer_n": item.get("n"),
        "as_of": as_of, "source": item.get("source"), "low_confidence": bool(item.get("low_conf")), "text_zh": text,
        "detail": {k: v for k, v in {"event_date": item.get("event_date"), "div_prev": item.get("div_prev"), "div_new": item.get("div_new"), "basis": item.get("basis"),
                                     "min_abs": rule.get("min_abs"), "window_days": rule.get("window_days")}.items() if v is not None},
    }


def _counter(rule_id: str, family: str | None, metric: str | None, label: str, plain: str, as_of: str, text: str, value=None, threshold=None, comparison=None, source=None, peer_pct=None, peer_group=None) -> dict:
    return {"family": family, "rule_id": rule_id, "metric": metric, "label_zh": label, "plain_zh": plain, "value": value, "threshold": threshold, "comparison": comparison,
            "peer_pct": peer_pct, "self_pct": None, "peer_group": peer_group, "as_of": as_of, "source": source, "text_zh": text}


def _as_of(row) -> str:
    return str(_get(row, "Date", ""))[:10]


def build_reasons(row, params: dict, reg: dict | None = None) -> dict:
    """生成单只股票的理由。未入选的股票也能调用：reasons 列出已命中的家族，selection_summary_zh 为空。"""
    reg = reg or load_registry(params)
    fam_rec = _loads(_get(row, "family_status"))
    checks = _loads(_get(row, "bucket_check"))
    metrics_meta = reg["metrics_by_id"]
    fam_cfg = reg["families"]
    as_of = _as_of(row)
    reasons, counters, unknowns = [], [], []

    for fam, rec in fam_rec.items():
        cfg = fam_cfg.get(fam, {})
        rules = {r["metric"]: r for r in cfg.get("threshold", {}).get("hit_rules", [])}
        st = rec.get("status")
        fam_name = cfg.get("name_zh", fam)
        if st == "hit":
            for m, item in rec.get("metrics", {}).items():
                if item.get("hit") == 1 and m in rules:
                    reasons.append(_reason(fam, m, item, rules[m], metrics_meta.get(m, {}), as_of))
        elif st == "fail":
            counters.append(_family_fail(fam, fam_name, rec, as_of, metrics_meta))
        elif st == "miss":
            counters.append(_family_miss(fam, fam_name, cfg, rec, rules, metrics_meta, as_of))
        elif st in ("unknown", "inactive"):
            unknowns.append(_family_unknown(fam, fam_name, cfg, rec, st, metrics_meta))
        # 被门槛拦下的指标（值很好但不算命中）
        for m, item in rec.get("metrics", {}).items():
            if item.get("blocked"):
                gate = rules.get(m, {}).get("unless_gate")
                meta = metrics_meta.get(m, {})
                val_str = fmt_value(m, round_value(m, item.get("value")))
                text = f"{meta.get('name_zh', m)} 是 {val_str}，本来够好，但没有计入：{COUNTER_GATES.get(gate, gate)}（截至 {as_of}）"
                counters.append(_counter(f"gate.{gate}.blocks.{m}", fam, m, metrics_meta.get(gate, {}).get("name_zh", gate), metrics_meta.get(gate, {}).get("plain_zh", ""), as_of, text,
                                         value=round_value(m, item.get("value")), source=item.get("source"), peer_pct=item.get("pct"), peer_group=item.get("grp")))

    # 门槛触发（降级、人工复核）
    for gate, desc in COUNTER_GATES.items():
        v = _get(row, f"gate_status_{gate}")
        if v == 1:
            meta = metrics_meta.get(gate, {})
            effect = "因此排序降一级。" if gate in pget(params, "evidence.downgrade_gates", []) else ""
            counters.append(_counter(f"gate.{gate}", None, gate, meta.get("name_zh", gate), meta.get("plain_zh", ""), as_of, f"{meta.get('name_zh', gate)}：{desc}{effect}（截至 {as_of}）", value=1))
    if _get(row, "gate_status_gate_one_off_loss") == 1 or _get(row, "gate_one_off_loss") == 1:
        counters.append(_counter("gate.gate_one_off.loss", None, "gate_one_off", "一次性亏损提示", "预计净利润明显低于主业利润，可能有一次性亏损压低利润。", as_of,
                                 f"一次性亏损提示：预计净利润明显低于主业利润，当前估值可能被一次性亏损扭曲，请人工核对。（截至 {as_of}）"))
    for code in str(_get(row, "soft_flags", "") or "").split(","):
        if code in SOFT_FLAG_ZH:
            lab, plain = SOFT_FLAG_ZH[code]
            counters.append(_counter(f"soft.{code}", None, None, lab, plain, as_of, f"{lab}：{plain}（截至 {as_of}）"))
    # 依赖度
    if bool(_get(row, "concentration_flag", False)):
        dom = _get(row, "dominant_family")
        share = _get(row, "max_family_share")
        counters.append(_counter("evidence.concentration", dom, None, "证据集中", "这只股票的证据主要来自一个家族，其他家族只是刚过线。", as_of,
                                 f"证据集中：{fam_cfg.get(dom, {}).get('name_zh', dom)} 占全部证据强度的 {share * 100:.1f}%，超过 {float(pget(params, 'evidence.max_dimension_share', 0.5)) * 100:.1f}%。（截至 {as_of}）",
                                 value=round(float(share), 4), threshold=float(pget(params, "evidence.max_dimension_share", 0.5)), comparison="share_gt"))
    bucket = _get(row, "bucket")
    for bk, ch in checks.items():
        if bk != bucket and bucket is not None:
            continue
        crit = ch.get("critical") or []
        if crit and (bucket == bk):
            names = "、".join(metrics_meta.get(m, {}).get("name_zh", m) for m in crit)
            counters.append(_counter("evidence.critical_metrics", None, None, "少了一项就不够", "命中的家族数刚好够门槛。", as_of,
                                     f"少了一项就不够：{names} 中去掉任何一项，命中的家族数就不到 {int(pget(params, 'evidence.min_families', 3))} 个。（截至 {as_of}）"))
        for low in ch.get("low_conf") or []:
            txt = _low_conf_text(low, fam_cfg, reg)
            fam_of = low.split(":")[-1] if ":" in low else None
            if txt and not (low.startswith("low_conf_unknown:") and fam_rec.get(fam_of, {}).get("status") == "unknown"):
                unknowns.append({"family": low.split(":")[-1] if ":" in low else None, "rule_id": f"low_conf.{low}", "text_zh": txt, "needed_zh": None})

    # 明确列出未检查的门槛：不能显示为通过
    for g in pget(reg, "gates.unchecked_on_current_data", []) or []:
        v = _get(row, f"gate_status_{g}")
        if v is None:
            meta = metrics_meta.get(g, {})
            unknowns.append({"family": None, "rule_id": f"gate_unchecked.{g}", "metric": g, "label_zh": meta.get("name_zh", g),
                             "text_zh": f"未检查：{meta.get('name_zh', g)}。{meta.get('plain_zh', '')}", "needed_zh": meta.get("unlocks_with")})
    if str(_get(row, "news_status", "unknown")) == "unknown":
        unknowns.append({"family": None, "rule_id": "news.unknown", "label_zh": "公告事件", "text_zh": "这一天的公告数据没有覆盖：回购、增发、丑闻和下跌原因都没有检查。没有数据不等于没有坏消息。", "needed_zh": "TDnet 每日采集"})
    if bool(_get(row, "supervision_unknown", False)):
        unknowns.append({"family": None, "rule_id": "gate.supervision_unknown", "label_zh": "监理名单", "text_zh": "历史名单没有写解除日期，这一天是否还在监理名单里判断不了。", "needed_zh": "JPX 解除日期"})
    plan = str(pget(params, "data_sources.jquants.plan", "Free"))
    if plan == "Free":
        unknowns.append({"family": None, "rule_id": "data.free_plan_delay", "label_zh": "数据延迟", "text_zh": "当前用免费数据，价格和财务都晚 12 周，最近 12 周的变化看不到，结果只能用于研究和回放。", "needed_zh": "J-Quants Light 以上"})

    summary = _selection_summary(row, reasons, fam_rec, fam_cfg, reg) if bucket else None
    return {"reasons": reasons, "counter_reasons": counters, "unknowns": unknowns, "selection_summary_zh": summary, "families_passed": sorted({r["family"] for r in reasons}),
            "context": _context(row, as_of)}


def _context(row, as_of: str) -> list[dict]:
    """只展示、不参与判断的背景数字，写明实际用的期限：Free 只有 2 年，3 年增长常常算不出。"""
    out = []
    horizon = _get(row, "growth_horizon", "") or ""
    g3, g1 = _get(row, "rev_cagr_3y"), _get(row, "rev_growth_1y")
    if horizon == "3y" and g3 is not None:
        out.append({"metric": "rev_cagr_3y", "value": round(float(g3), 4), "horizon": "3y", "text_zh": f"过去 3 年收入年均增长 {g3 * 100:.1f}%（只作背景，不参与筛选）。"})
    elif horizon == "1y" and g1 is not None:
        extra = "这一年决算期不是 12 个月，已按天数折算。" if bool(_get(row, "fy_irregular", False)) else ""
        out.append({"metric": "rev_growth_1y", "value": round(float(g1), 4), "horizon": "1y", "text_zh": f"最近 1 年收入增长 {g1 * 100:.1f}%（只有 2 年数据，算不出 3 年增长；只作背景，不参与筛选）。{extra}"})
    return out


def _family_fail(fam, fam_name, rec, as_of, metrics_meta) -> dict:
    if fam == "guidance_change":
        info = rec.get("fail") or {}
        if info.get("kind") == "withdrawn":
            text = f"{fam_name}：公司在 {info.get('date')} 撤回了全年预测。（截至 {as_of}）"
        else:
            mag = info.get("mag")
            size = f"，幅度相当于市值的 {abs(mag) * 100:.2f}%" if mag is not None else ""
            low = "（主业预测未知，这一判断可信度较低）" if info.get("low_conf") else ""
            text = f"{fam_name}：公司在 {info.get('date')} 下调了全年利润预测{size}{low}。（截至 {as_of}）"
        return _counter(f"family_fail.{fam}", fam, "guid_revision_event", fam_name, "公司下调或撤回了全年预测。", as_of, text, value=info.get("mag"), source=f"fins_summary DiscNo={info.get('disc_no')}")
    item = rec.get("metrics", {}).get("qual_fscore7", {})
    val = fmt_value("qual_fscore7", round_value("qual_fscore7", item.get("value")))
    meta = metrics_meta.get("qual_fscore7", {})
    return _counter(f"family_fail.{fam}", fam, "qual_fscore7", fam_name, meta.get("plain_zh", ""), as_of, f"{fam_name}：财务打分只有 {val}，属于明显偏差。（截至 {as_of}）",
                    value=round_value("qual_fscore7", item.get("value")), source=item.get("source"))


def _family_miss(fam, fam_name, cfg, rec, rules, metrics_meta, as_of) -> dict:
    parts = []
    for m, item in rec.get("metrics", {}).items():
        meta = metrics_meta.get(m, {})
        if item.get("pct") is not None:
            where = "流动性核心池" if rules.get(m, {}).get("rule") == "core_best_pct" else (item.get("grp") or "同业")
            parts.append(f"{meta.get('name_zh', m)} {fmt_value(m, round_value(m, item.get('value')))}，在{where}中从好到差排在 {_pct_str(item['pct'])} 的位置")
        elif item.get("value") is not None and m in ("qual_fscore7",):
            parts.append(f"{meta.get('name_zh', m)} {fmt_value(m, round_value(m, item.get('value')))}")
    detail = "；".join(parts) if parts else "最近没有相关事件"
    text = f"{fam_name}没有命中（{cfg.get('question_zh', '')}）：{detail}。（截至 {as_of}）"
    return _counter(f"family_miss.{fam}", fam, None, fam_name, cfg.get("question_zh", ""), as_of, text)


def _family_unknown(fam, fam_name, cfg, rec, st, metrics_meta) -> dict:
    missing = [m for m, item in rec.get("metrics", {}).items() if item.get("hit") is None]
    needed = []
    for m in missing or cfg.get("metrics", []):
        meta = metrics_meta.get(m, {})
        needed.append(f"{meta.get('name_zh', m)}：{meta.get('data_needed', '')}" + (f"（解锁：{meta.get('unlocks_with')}）" if meta.get("unlocks_with") else ""))
    if st == "inactive":
        text = f"{fam_name}：这个行业或市场不适用这个家族，不计入判断。"
        needed = []
    else:
        text = f"{fam_name}：数据不够，判断不了（不算命中，也不算没命中）。{cfg.get('question_zh', '')}"
    return {"family": fam, "rule_id": f"family_{st}.{fam}", "label_zh": fam_name, "status": st, "text_zh": text, "needed_zh": "；".join(needed) if needed else None}


def _low_conf_text(code: str, fam_cfg: dict, reg: dict) -> str | None:
    if code.startswith("low_conf_uncovered:"):
        f = code.split(":")[1]
        return f"{fam_cfg.get(f, {}).get('name_zh', f)} 在这个市场能判断的公司不到六成，暂时不作必要条件，这一项没有检查，结论可信度较低。"
    if code == "low_conf_confirm_uncovered":
        return "确认用的家族在这个市场都判断不了，结论可信度较低。"
    if code == "low_conf_cycle_unchecked":
        return "周期行业：周期位置还没法检查（需要 5 年利润），低估值可能只是周期顶部的错觉，可信度较低。"
    if code.startswith("low_conf_unknown:"):
        f = code.split(":")[1]
        return f"{fam_cfg.get(f, {}).get('name_zh', f)} 未知，是否恶化没有检查。"
    if code.startswith("low_conf_unknown_metric:"):
        m = code.split(":")[1]
        return f"{reg['metrics_by_id'].get(m, {}).get('name_zh', m)} 未知，贵不贵没有检查。"
    return None


def _selection_summary(row, reasons: list[dict], fam_rec: dict, fam_cfg: dict, reg: dict) -> str:
    bucket = _get(row, "bucket")
    bk_name = reg["buckets"].get(bucket, {}).get("name_zh", bucket)
    rank = _get(row, "bucket_rank")
    n_fam = len({r["family"] for r in reasons})
    # 最强的两个家族：同业分位最靠前的理由优先
    best: dict[str, dict] = {}
    for r in reasons:
        cur = best.get(r["family"])
        if cur is None or (r.get("peer_pct") if r.get("peer_pct") is not None else 1.0) < (cur.get("peer_pct") if cur.get("peer_pct") is not None else 1.0):
            best[r["family"]] = r
    top = sorted(best.values(), key=lambda r: r.get("peer_pct") if r.get("peer_pct") is not None else 1.0)[:2]
    parts = []
    for r in top:
        val = fmt_value(r["metric"], r["value"])
        fname = fam_cfg.get(r["family"], {}).get("name_zh", r["family"])
        if r.get("peer_pct") is not None:
            parts.append(f"{fname}（{r['label_zh']} {val}，排在最好的 {_pct_str(r['peer_pct'])}）")
        elif val and r["comparison"] != "event_within":
            parts.append(f"{fname}（{r['label_zh']} {val}）")
        else:
            parts.append(f"{fname}（{r['label_zh']}）")
    return f"进入「{bucket} {bk_name}」，篮内排第 {int(rank)}，有 {n_fam} 个相互独立的证据家族命中。最强的是{'、'.join(parts)}。"


def exclusion_reason(row, params: dict, reg: dict | None = None) -> str:
    """没有进清单的原因。硬排除逐码解释；否则逐个篮子说明差在哪里。已入选返回空字符串。"""
    reg = reg or load_registry(params)
    fam_cfg = reg["families"]
    lines: list[str] = []
    codes = [c for c in str(_get(row, "exclusion_codes", "") or "").split(",") if c]
    for c in codes:
        lines.append(f"{c}：{EXCLUSION_ZH.get(c, c)}")
    if codes:
        return " ".join(lines)
    if bool(_get(row, "in_top_n", False)):
        return ""
    top_n = int(pget(params, "bucket.top_n", 40))
    checks = _loads(_get(row, "bucket_check"))
    for bk, ch in checks.items():
        name = reg["buckets"].get(bk, {}).get("name_zh", bk)
        msgs = []
        for code in ch.get("fails") or []:
            msgs.append(_fail_text(code, fam_cfg, reg, row))
        if ch.get("loo_removed"):
            m = ch.get("worst_metric")
            mname = reg["metrics_by_id"].get(m, {}).get("name_zh", m)
            msgs.append(f"过度依赖单一指标：去掉 {mname} 后排名从第 {int(ch.get('base_rank') or 0)} 掉到第 {int(ch.get('worst_rank') or 0)}，超过前 {int(top_n * float(pget(params, 'evidence.loo_rank_mult', 2)))} 名。")
        rank = _get(row, f"rank_{bk}")
        if not msgs and rank is not None and rank > top_n:
            msgs.append(f"符合条件，但排第 {int(rank)}，在前 {top_n} 名之外。")
        if msgs:
            lines.append(f"{bk} {name}：" + " ".join(dict.fromkeys(msgs)))
    return " ".join(lines)


def _fail_text(code: str, fam_cfg: dict, reg: dict, row) -> str:
    if code in FAIL_CODE_ZH:
        return FAIL_CODE_ZH[code]
    kind, _, arg = code.partition(":")
    name = fam_cfg.get(arg, {}).get("name_zh", arg)
    if kind == "anchor_miss":
        return f"这个篮子必须有「{name}」，但它没有命中（{fam_cfg.get(arg, {}).get('question_zh', '')}）。"
    if kind == "anchor_suspended":
        return f"「{name}」在这个市场能判断的公司不到六成，篮子暂停。"
    if kind == "required_miss":
        return f"还需要「{name}」命中，但它没有命中或判断不了。"
    if kind == "family_fail":
        return f"「{name}」明确恶化，不能进这个篮子。"
    if kind == "valuation_cap":
        mname = reg["metrics_by_id"].get(arg, {}).get("name_zh", arg)
        return f"估值太贵：{mname} 落在同业最差的三成。"
    if kind == "min_families":
        n = int(_get(row, "family_pass_count", 0))
        return f"命中的独立证据家族只有 {n} 个，至少要 {arg} 个；一两个指标好不足以入选。"
    return code


def attach_reasons(df: pd.DataFrame, params: dict, reg: dict | None = None) -> pd.DataFrame:
    """写入 JSON 字符串列。候选股（任一篮子已入池）生成完整理由；其他股票只生成未入选原因，避免 4,000 多行全部展开。"""
    reg = reg or load_registry(params)
    in_any = df[[c for c in df.columns if c.startswith("in_B")]].any(axis=1).to_numpy()
    n = len(df)
    reasons, counters, unknowns, summaries = [None] * n, [None] * n, [None] * n, [None] * n
    excl = [""] * n
    records = df.to_dict("records")
    for i, row in enumerate(records):
        excl[i] = exclusion_reason(row, params, reg)
        if not in_any[i]:
            continue
        r = build_reasons(row, params, reg)
        min_fam = int(pget(params, "evidence.min_families", 3))
        if len(r["families_passed"]) < min_fam:
            raise ValueError(f"{row.get('Code')} 入池但理由只覆盖 {len(r['families_passed'])} 个家族，少于 {min_fam}：入池规则和理由不一致")
        reasons[i] = json.dumps(r["reasons"], ensure_ascii=False, default=str)
        counters[i] = json.dumps(r["counter_reasons"], ensure_ascii=False, default=str)
        unknowns[i] = json.dumps(r["unknowns"], ensure_ascii=False, default=str)
        summaries[i] = r["selection_summary_zh"]
    df["reasons"] = reasons
    df["counter_reasons"] = counters
    df["unknowns"] = unknowns
    df["selection_summary_zh"] = summaries
    df["exclusion_reason_zh"] = excl
    return df
