"""报告渲染（规划文档 6 章）：筛后清单、个股卡片、组合报告 → Markdown（+ CSV）。"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

DISCLAIMER = "> 本报告由规则引擎与 AI 辅助生成，仅供个人研究；所有区间为基于明确假设的情景估计，不构成投资建议，不承诺收益。"


def _f(v, fmt="{:.2f}", na="—"):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return na
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return na


def _pct(v):
    return _f(v, "{:+.1%}") if v is not None and not (isinstance(v, float) and np.isnan(v)) else "—"


def render_candidates(d: date, cand: pd.DataFrame, scores: pd.DataFrame, regime: dict | None, run_id: str, param_version: str) -> str:
    lines = [f"# 筛后清单 {d.isoformat()}", "", DISCLAIMER, "",
             f"- 运行 ID：`{run_id}`；参数版本：`{param_version}`；全市场 {len(scores)} 只，可投 {int((~scores['excluded']).sum())} 只",
             f"- 篮子规模：B1 低估反转 {int(scores['in_B1'].sum())} / B2 质量成长 {int(scores['in_B2'].sum())} / B3 错杀修复 {int(scores['in_B3'].sum())}"]
    if regime:
        lines.append(f"- 市场环境：{regime.get('regime')}（市场代理相对 200 日均线 {_pct(regime.get('mkt_vs_ma200'))}，广度 {_pct(regime.get('breadth'))}，现金缓冲 {_pct(regime.get('cash_buffer'))}）")
    lines.append("")
    names = {"B1": "B1 低估反转", "B2": "B2 质量成长", "B3": "B3 错杀修复"}
    for bk in ("B1", "B2", "B3"):
        sub = cand[cand["list_bucket"] == bk].sort_values("list_rank")
        lines.append(f"## {names[bk]}（前 {len(sub)}）")
        lines.append("")
        lines.append("| 排名 | 代码 | 名称 | 業種 | 株価 | 時価総額(億円) | PBR | 予想PER | 配当利回り | ROE | 低估 | 成长 | 质量 | 错杀 | 趋势 | 新闻 | 复合 | 共振 | 标记 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            lines.append(f"| {int(r['list_rank'])} | {r['Code'][:4]} | {r.get('CoName')} | {r.get('S33')} | {_f(r['C'], '{:,.0f}')} | {_f(r['mcap'] / 1e8, '{:,.0f}')} | "
                         f"{_f(r['pbr'])} | {_f(r['per_fwd'], '{:.1f}')} | {_pct(r['div_yield_fwd'])} | {_pct(r['roe_avg'])} | {_f(r['score_value'], '{:.0f}')} | "
                         f"{_f(r['score_growth'], '{:.0f}')} | {_f(r['score_quality'], '{:.0f}')} | {_f(r['score_oversold'], '{:.0f}')} | {_f(r['score_trend'], '{:.0f}')} | "
                         f"{_f(r['score_news'], '{:.0f}')} | {_f(r['list_score'], '{:.1f}')} | {'✓' if r.get('resonance') else ''} | {r.get('soft_flags') or ''} |")
        lines.append("")
    return "\n".join(lines)


def render_stock_card(row: pd.Series, signal: dict | None, ai: dict | None, evidence_lookup=None) -> str:
    code = str(row["Code"])
    L = [f"# {code[:4]} {row.get('CoName')}（{row.get('S33')}，{row.get('Mkt')}）— 基准日 {row['Date']}", "", DISCLAIMER, ""]
    L += [f"**篮子** {row.get('bucket')} 排名 {_f(row.get('bucket_rank'), '{:.0f}')}，复合分 {_f(row.get('composite_score'), '{:.1f}')}；共振：{'是' if row.get('resonance') else '否'}", ""]
    if ai and ai.get("one_liner"):
        L += ["## 论点", "", f"**{ai['one_liner']}**", ""]
        for r in ai.get("reasons", []):
            L.append(f"- {r['text']}  （证据：{', '.join(r.get('evidence_ids', []))}）")
        L.append("")
    L += ["## 关键指标", "", "| 指标 | 值 | 行业百分位 | 自身历史百分位 |", "|---|---|---|---|"]
    for k, lab, pk, sk, fmt in (("pbr", "PBR", "pct_pbr", "self_pct_pbr", "{:.2f}"), ("per_fwd", "予想 PER", "pct_per_fwd", "self_pct_per_fwd", "{:.1f}"),
                                ("div_yield_fwd", "予想配当利回り", "pct_div_yield_fwd", None, "{:.2%}"), ("fcf_yield", "FCF 利回り", "pct_fcf_yield", None, "{:.2%}"),
                                ("cash_to_mcap", "現預金/時価総額", None, None, "{:.2f}"), ("roe_avg", "ROE", None, None, "{:.1%}"),
                                ("equity_ratio", "自己資本比率", None, None, "{:.1%}"), ("rev_growth_1y", "売上成長(1Y)", None, None, "{:+.1%}"),
                                ("op_growth_1y", "営業利益成長(1Y)", None, None, "{:+.1%}"), ("f_op_growth", "予想営業利益成長", None, None, "{:+.1%}"),
                                ("progress_rate", "進捗率", None, None, "{:.1%}"), ("progress_dev", "進捗率偏差", None, None, "{:+.1%}"),
                                ("drawdown_60d", "60 日回撤", None, None, "{:+.1%}"), ("rs_63d", "63 日相对强弱", None, None, "{:+.1%}")):
        L.append(f"| {lab} | {_f(row.get(k), fmt)} | {_f(row.get(pk), '{:.0%}') if pk else '—'} | {_f(row.get(sk), '{:.0%}') if sk else '—'} |")
    L.append("")
    L += ["## 六维分", "", f"低估 {_f(row.get('score_value'), '{:.0f}')} · 成长 {_f(row.get('score_growth'), '{:.0f}')} · 质量 {_f(row.get('score_quality'), '{:.0f}')} · "
          f"错杀 {_f(row.get('score_oversold'), '{:.0f}')} · 趋势 {_f(row.get('score_trend'), '{:.0f}')} · 新闻 {_f(row.get('score_news'), '{:.0f}')}",
          f"标记：陷阱={bool(row.get('value_trap'))}，质量门槛={bool(row.get('quality_gate'))}，排除码={row.get('exclusion_codes') or '无'}，软标记={row.get('soft_flags') or '无'}", ""]
    if signal:
        sc = signal.get("scenarios", {})
        L += ["## 估值区间与买卖区间", "", "| 情景 | 价值 | 相对现价 |", "|---|---|---|",
              f"| 悲观 | {_f(sc.get('bear'), '{:,.0f}')} | {_pct(signal.get('downside_bear'))} |",
              f"| 基准 | {_f(sc.get('base'), '{:,.0f}')} | {_pct(signal.get('upside_base'))} |",
              f"| 乐观 | {_f(sc.get('bull'), '{:,.0f}')} | {_pct(signal.get('upside_bull'))} |", ""]
        for m in signal.get("methods", []):
            L.append(f"- {m['method']}：目标倍数 {_f(m['target_multiple'])} → {_f(m['fair_value'], '{:,.0f}')}（{m['assumptions']}）")
        L += ["", f"- 买入区间：{_f(signal.get('zone_low'), '{:,.0f}')} – {_f(signal.get('zone_high'), '{:,.0f}')}，分批 {signal.get('tranches')}；条件：{'；'.join(signal.get('conditions') or ['无'])}",
              f"- 禁止建仓：{signal.get('blocked_reason') or '无'}",
              f"- 减仓/清仓：达到基准价值减半（{_f(signal['exit'].get('trim_at'), '{:,.0f}')}）、达到乐观价值清仓（{_f(signal['exit'].get('exit_at'), '{:,.0f}')}）；技术止损 {_f(signal['exit'].get('technical_stop'), '{:,.0f}')}；时间止损 {signal['exit'].get('time_stop_months')} 个月",
              f"- 基本面止损：{'；'.join(signal['exit'].get('fundamental_stops', []))}", ""]
    if ai:
        if ai.get("bear_points"):
            L += ["## 空头论点与风险", ""]
            for b in ai["bear_points"]:
                L.append(f"- {b['text']}  （证据：{', '.join(b.get('evidence_ids', []))}）")
            L.append("")
        if ai.get("invalidation_conditions"):
            L += ["**失效条件**：" + "；".join(ai["invalidation_conditions"]), ""]
        if ai.get("watch_points"):
            L += ["**跟踪点**：" + "；".join(ai["watch_points"]), ""]
        att = ai.get("attribution") or {}
        if att.get("type") and att["type"] != "NA":
            L += [f"**下跌归因（AI）**：{att['type']} — {att.get('explanation')}（证据：{', '.join(att.get('evidence_ids', []))}）", ""]
        if ai.get("adjustments_applied"):
            L += ["**AI 调整分**：" + "；".join(f"{a['dimension']} {a['delta']:+.0f}（{a['reason']}）" for a in ai["adjustments_applied"]), ""]
        v = ai.get("verdicts") or {}
        L += [f"**复核结论（AI）**：低估 {v.get('value_candidate')} / 成长 {v.get('growth_candidate')} / 质量 {v.get('quality_gate')} / 错杀 {v.get('oversold_candidate')}；置信度 {_f(ai.get('confidence'))}",
              f"引用校验：命中率 {_f((ai.get('citation_check') or {}).get('hit_rate'), '{:.0%}')}，无效引用 {(ai.get('citation_check') or {}).get('n_invalid')}", ""]
        if ai.get("inferences"):
            L += ["**无证据推断（显式标注）**：" + "；".join(ai["inferences"]), ""]
    L += ["## 元数据", "", f"- 最新财报生效日：{row.get('effective_date')}；下一决算发表预定：{row.get('next_earnings_date')}",
          f"- 参数版本：{row.get('param_version')}；运行 ID：{row.get('run_id')}；指标口径：{row.get('metric_version')}"]
    if ai and ai.get("_meta"):
        L.append(f"- 模型：{ai['_meta'].get('model')}；提示版本：{ai['_meta'].get('prompt_version')}")
    return "\n".join(L)


def render_portfolio(d: date, port: dict, regime: dict | None, signals: pd.DataFrame | None) -> str:
    pos = port.get("positions")
    L = [f"# 组合报告 {d.isoformat()}", "", DISCLAIMER, ""]
    if pos is None or len(pos) == 0:
        L.append(f"无可建仓候选：{port.get('note')}")
        return "\n".join(L)
    L += [f"- 持仓 {len(pos)} 只，现金缓冲 {_pct(port.get('cash'))}，篮子配置 {json.dumps(port.get('allocation'), ensure_ascii=False)}"]
    if regime:
        L.append(f"- 市场环境：{regime.get('regime')}")
    ex = port.get("exposures", {})
    L += [f"- 行业(17)暴露：{json.dumps(ex.get('sector17'), ensure_ascii=False)}", f"- 市场区分暴露：{json.dumps(ex.get('market'), ensure_ascii=False)}",
          f"- 篮子暴露：{json.dumps(ex.get('bucket'), ensure_ascii=False)}；HHI {_f(ex.get('hhi'), '{:.3f}')}", "",
          "| 代码 | 名称 | 篮子 | 排名 | 权重 | 株価 | 60 日波动率 | 流动性上限(円) | 买入区间 | 基准价值 |", "|---|---|---|---|---|---|---|---|---|---|"]
    sig = signals.set_index("Code") if signals is not None and len(signals) else None
    for _, r in pos.sort_values("weight", ascending=False).iterrows():
        zone = base = "—"
        if sig is not None and r["Code"] in sig.index:
            s = sig.loc[r["Code"]]
            zone = f"{_f(s.get('zone_low'), '{:,.0f}')}–{_f(s.get('zone_high'), '{:,.0f}')}"
            base = _f((s.get("scenarios") or {}).get("base"), "{:,.0f}") if isinstance(s.get("scenarios"), dict) else "—"
        L.append(f"| {str(r['Code'])[:4]} | {r.get('CoName')} | {r['bucket']} | {int(r['rank'])} | {_pct(r['weight'])} | {_f(r['price'], '{:,.0f}')} | "
                 f"{_pct(r.get('rvol60'))} | {_f(r.get('liquidity_cap_jpy'), '{:,.0f}')} | {zone} | {base} |")
    if port.get("changes"):
        L += ["", "## 变动建议", ""] + [f"- {c['action']} {c['Code'][:4]}：{c['reason']}" for c in port["changes"]]
    return "\n".join(L)


def write_reports(out_dir: Path, d: date, candidates_md: str, cards: dict[str, str], portfolio_md: str | None, cand_df: pd.DataFrame) -> dict:
    day_dir = out_dir / d.isoformat()
    (day_dir / "cards").mkdir(parents=True, exist_ok=True)
    (day_dir / "candidates.md").write_text(candidates_md, encoding="utf-8")
    cand_df.to_csv(day_dir / "candidates.csv", index=False, encoding="utf-8-sig")
    for code, md in cards.items():
        (day_dir / "cards" / f"{code}.md").write_text(md, encoding="utf-8")
    if portfolio_md:
        (day_dir / "portfolio.md").write_text(portfolio_md, encoding="utf-8")
    return {"dir": str(day_dir), "cards": len(cards)}
