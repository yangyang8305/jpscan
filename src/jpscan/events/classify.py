"""披露/新闻标题分类（规划文档 附录 D、4.7 节）：标题 → 类别 / 方向 / 强度 / 半衰期。

规则见 config/events.yaml 文件头。规则层零成本；needs_ai=true 或 E40 的标题才交给 AI（A3）。
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

_BRACKETS = str.maketrans({"〔": "[", "〕": "]", "［": "[", "］": "]", "【": "[", "】": "]"})
_SHORT_ASCII = re.compile(r"^[a-z0-9]{1,5}$")
REVISION_CODES = ("E01", "E02", "E03", "E04")
CORRECTION = "E38"
OTHER = "E40"


def normalize_title(title: str | None) -> str:
    """NFKC（全角→半角）、括号统一、去掉所有空白、ASCII 小写。标题与关键词走同一函数。"""
    return "".join(_spaced(title).split())


def _spaced(title: str | None) -> str:
    return unicodedata.normalize("NFKC", title or "").translate(_BRACKETS).lower()


@dataclass(frozen=True)
class _Text:
    norm: str     # 去空白后的标题，用于子串匹配
    spaced: str   # 保留空白，只用于短英文词的词边界判断


def _make_text(title: str | None) -> _Text:
    return _Text(normalize_title(title), _spaced(title))


def _in(text: _Text, kw: str) -> bool:
    """精确子串匹配。≤5 字符的纯 ASCII 关键词（mbo、msci…）额外要求词边界，否则 embodying 会命中 MBO。"""
    if kw not in text.norm:
        return False
    if _SHORT_ASCII.match(kw):
        return re.search(rf"(?<![a-z]){re.escape(kw)}(?![a-z])", text.spaced) is not None
    return True


def _pairs(keywords: Iterable[str] | None) -> list[tuple[str, str]]:
    """[(归一化关键词, 原文)]，同一归一化结果只留一个。"""
    out: dict[str, str] = {}
    for kw in keywords or []:
        out.setdefault(normalize_title(kw), kw)
    return list(out.items())


def _norms(keywords: Iterable[str] | None) -> list[str]:
    return [n for n, _ in _pairs(keywords)]


def load_taxonomy(path: Path | str) -> dict[str, Any]:
    """读取 events.yaml，并预先归一化全部关键词（放在 _compiled 下，原始字段保持不变）。"""
    with open(path, "r", encoding="utf-8") as f:
        tax = yaml.safe_load(f) or {}
    cats = {}
    for code, c in (tax.get("categories") or {}).items():
        cats[code] = {
            "keywords": _pairs(c.get("keywords")),
            "exclude": _norms(c.get("exclude")),
            "prefix": _norms(c.get("prefix_keywords")),
            "cue_pos": _norms((c.get("direction_cues") or {}).get("positive")),
            "cue_neg": _norms((c.get("direction_cues") or {}).get("negative")),
            "keyword_strength": {normalize_title(k): v for k, v in (c.get("keyword_strength") or {}).items()},
        }
    families = []
    for name in ("earnings_revision", "dividend_revision"):
        f = tax["families"][name]
        families.append({
            "positive": f["positive"], "negative": f["negative"], "undetermined": f["undetermined"],
            "keywords": _pairs(f.get("keywords")),
            "all_of": [(_norms(group), "+".join(group)) for group in f.get("all_of") or []],
            "all_of_unless": _norms(f.get("all_of_unless")),
            "cue_pos": _norms(f.get("cues_positive")),
            "cue_neg": _norms(f.get("cues_negative")),
        })
    tax["_compiled"] = {"cats": cats, "families": families, "markers": _norms(tax.get("lang_markers"))}
    return tax


def detect_lang(title: str | None, taxonomy: dict[str, Any]) -> str:
    """命中英文标记，或 ASCII 字母占字母类字符 ≥ 70%（且至少 5 个字母）→ en。"""
    text = normalize_title(title)
    if any(m in text for m in taxonomy["_compiled"]["markers"]):
        return "en"
    ascii_letters = sum(1 for ch in text if "a" <= ch <= "z")
    others = sum(1 for ch in text if ch.isalpha() and not ("a" <= ch <= "z"))
    return "en" if ascii_letters >= 5 and ascii_letters / (ascii_letters + others) >= 0.7 else "ja"


def _category_hit(text: _Text, compiled: dict[str, Any]) -> str | None:
    """返回最长命中关键词的原文；category.exclude 命中则该类别不参与。"""
    if any(_in(text, ex) for ex in compiled["exclude"]):
        return None
    hits = [(norm, orig) for norm, orig in compiled["keywords"] if _in(text, norm)]
    return max(hits, key=lambda h: len(h[0]))[1] if hits else None


def _all_hits(text: _Text, taxonomy: dict[str, Any]) -> dict[str, str]:
    out = {}
    for code, compiled in taxonomy["_compiled"]["cats"].items():
        if code in (CORRECTION, OTHER):
            continue
        kw = _category_hit(text, compiled)
        if kw is not None:
            out[code] = kw
    return out


def _pick_longest(hits: dict[str, str]) -> tuple[str, str]:
    """最长关键词优先，等长取代码小者。"""
    return min(hits.items(), key=lambda kv: (-len(normalize_title(kv[1])), kv[0]))


def _spec(code: str, taxonomy: dict[str, Any]) -> dict[str, Any]:
    return taxonomy["categories"].get(code) or taxonomy["undetermined"][code]


def _direction_of(code: str, text: _Text, taxonomy: dict[str, Any]) -> int:
    """类别方向；E34 这类带 direction_cues 的类别按 cue 判定，正负同现或无 cue → 0。"""
    compiled = taxonomy["_compiled"]["cats"].get(code) or {}
    if compiled.get("cue_pos") or compiled.get("cue_neg"):
        pos = any(_in(text, c) for c in compiled["cue_pos"])
        neg = any(_in(text, c) for c in compiled["cue_neg"])
        return 0 if pos == neg else (1 if pos else -1)
    return int(_spec(code, taxonomy)["direction"])


def _conflict(code: str, direction: int, hits: dict[str, str], text: _Text, taxonomy: dict[str, Any]) -> bool:
    """主类别与其他命中类别方向相反。"""
    if direction == 0:
        return False
    return any(_direction_of(other, text, taxonomy) * direction < 0 for other in hits if other != code)


def _result(code: str, keyword: str | None, lang: str, *, direction: int | None = None, strength: int | None = None,
            needs_ai: bool | None = None, underlying: str | None = None, taxonomy: dict[str, Any]) -> dict[str, Any]:
    spec = _spec(code, taxonomy)
    return {
        "category": code,
        "direction": int(spec["direction"]) if direction is None else direction,
        "strength": int(spec["strength"]) if strength is None else strength,
        "half_life_days": spec.get("half_life_days"),
        "hard_exclude": bool(spec.get("hard_exclude", False)),
        "special": bool(spec.get("special", False)),
        "needs_ai": bool(spec.get("needs_ai", False)) if needs_ai is None else needs_ai,
        "matched_keyword": keyword,
        "lang": lang,
        "underlying": underlying,
    }


def _family_hit(text: _Text, family: dict[str, Any], hits: dict[str, str]) -> dict[str, Any] | None:
    """修正家族命中：返回 {code, keyword}。正负类别的关键词本身也算相应方向的 cue。"""
    direct = [(n, o) for n, o in family["keywords"] if _in(text, n)]
    groups = []
    if not any(_in(text, u) for u in family["all_of_unless"]):
        groups = [label for group, label in family["all_of"] if all(_in(text, g) for g in group)]
    pos_kw, neg_kw = hits.get(family["positive"]), hits.get(family["negative"])
    if not (direct or groups or pos_kw or neg_kw):
        return None
    pos = bool(pos_kw) or any(_in(text, c) for c in family["cue_pos"])
    neg = bool(neg_kw) or any(_in(text, c) for c in family["cue_neg"])
    code = family["undetermined"] if pos == neg else (family["positive"] if pos else family["negative"])
    candidates = [o for _, o in direct] + [k for k in (pos_kw if pos else None, neg_kw if neg else None) if k]
    keyword = max(candidates, key=lambda o: len(normalize_title(o))) if candidates else groups[0]
    return {"code": code, "keyword": keyword, "determined": pos != neg}


def _classify_soft(text: _Text, hits: dict[str, str], lang: str, taxonomy: dict[str, Any]) -> dict[str, Any]:
    """优先级 ③ 修正类 → ④ 其余类别 → ⑤ E40（不含硬排除与 E38，二者已在外层处理）。"""
    fam_hits = [h for h in (_family_hit(text, f, hits) for f in taxonomy["_compiled"]["families"]) if h]
    others = {c: k for c, k in hits.items() if c not in REVISION_CODES}
    if fam_hits:
        primary = next((h for h in fam_hits if h["determined"]), fam_hits[0])
        code = primary["code"]
        direction = int(_spec(code, taxonomy)["direction"])
        strength = int(_spec(code, taxonomy)["strength"])
        overrides = taxonomy["_compiled"]["cats"].get(code, {}).get("keyword_strength", {})
        for kw, value in overrides.items():
            if _in(text, kw):
                strength = int(value)
                break
        signs = {int(_spec(h["code"], taxonomy)["direction"]) for h in fam_hits}
        needs_ai = (not all(h["determined"] for h in fam_hits)  # 修正方向未定
                    or {1, -1} <= signs                         # 业绩与配当方向相反
                    or _conflict(code, direction, others, text, taxonomy))
        return _result(code, primary["keyword"], lang, direction=direction, strength=strength, needs_ai=needs_ai, taxonomy=taxonomy)
    if others:
        code, keyword = _pick_longest(others)
        direction = _direction_of(code, text, taxonomy)
        spec = _spec(code, taxonomy)
        needs_ai = bool(spec.get("needs_ai", False)) or _conflict(code, direction, others, text, taxonomy)
        strength = int(spec["strength"])
        if "direction_cues" in spec:      # E34：强度随 cue 取正负；无 cue → 0 并交 AI
            strength = abs(strength) * direction
            needs_ai = needs_ai or direction == 0
        return _result(code, keyword, lang, direction=direction, strength=strength, needs_ai=needs_ai, taxonomy=taxonomy)
    return _result(OTHER, None, lang, taxonomy=taxonomy)


def _correction_keyword(text: _Text, taxonomy: dict[str, Any]) -> str | None:
    compiled = taxonomy["_compiled"]["cats"][CORRECTION]
    if any(_in(text, ex) for ex in compiled["exclude"]):
        return None
    for prefix in compiled["prefix"]:
        if text.norm.startswith(prefix):
            return prefix
    return _category_hit(text, compiled)


def classify_title(title: str | None, taxonomy: dict[str, Any]) -> dict[str, Any]:
    """按 events.yaml 文件头的优先级分类。

    返回 category, direction, strength, half_life_days, hard_exclude, special, needs_ai, matched_keyword, lang,
    以及 underlying（仅 E38：被订正内容的类别，且其 needs_ai 会传给 E38）。
    """
    text = _make_text(title)
    lang = detect_lang(title, taxonomy)
    hits = _all_hits(text, taxonomy)
    # ① 硬排除
    hard = {c: k for c, k in hits.items() if _spec(c, taxonomy).get("hard_exclude")}
    if hard:
        code, keyword = _pick_longest(hard)
        direction = _direction_of(code, text, taxonomy)
        needs_ai = bool(_spec(code, taxonomy).get("needs_ai", False)) or _conflict(code, direction, hits, text, taxonomy)
        return _result(code, keyword, lang, needs_ai=needs_ai, taxonomy=taxonomy)
    # ② 訂正：方向固定，方向冲突只在被订正内容内部判断
    correction = _correction_keyword(text, taxonomy)
    soft = _classify_soft(text, hits, lang, taxonomy)
    if correction is not None:
        return _result(CORRECTION, correction, lang, needs_ai=soft["needs_ai"], underlying=soft["category"], taxonomy=taxonomy)
    return soft


def is_correction_title(title: str | None, taxonomy: dict[str, Any]) -> bool:
    """标题本身是否表示订正（即使类别因硬排除优先而不是 E38）。"""
    return _correction_keyword(_make_text(title), taxonomy) is not None


def download_policy(result: dict[str, Any], taxonomy: dict[str, Any]) -> dict[str, bool]:
    """按分类结果决定下载：PDF 看类别 download 白名单；XBRL 看 xbrl_always（E38 也看被订正内容）。"""
    code = result["category"]
    always = set(taxonomy.get("xbrl_always") or [])
    return {"pdf": bool(_spec(code, taxonomy).get("download", False)), "xbrl": code in always or result.get("underlying") in always}


def _strip_lang_markers(title: str | None, taxonomy: dict[str, Any]) -> str:
    text = normalize_title(title)
    for marker in taxonomy["_compiled"]["markers"]:
        text = text.replace(marker, "")
    return text


def dedupe_ja_en(rows: list[dict[str, Any]], taxonomy: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """日英去重（10.5 节：コード + 時刻 + 表題）：英文行与同代码同日同时刻的日文行重复时去掉英文行。

    重复 = 去掉语言标记后标题相同，或类别相同（E40 除外：其无分值且不可判断是否同一披露）。
    行需含 Code, disclosed_date, disclosed_time, title, lang, category。返回 (保留, 去掉)。
    """
    ja: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = {}
    for r in rows:
        if r["lang"] != "en":
            ja.setdefault((r["Code"], r["disclosed_date"], r["disclosed_time"]), []).append(r)
    kept, dropped = [], []
    for r in rows:
        peers = ja.get((r["Code"], r["disclosed_date"], r["disclosed_time"]), []) if r["lang"] == "en" else []
        title = _strip_lang_markers(r["title"], taxonomy)
        duplicate = any(_strip_lang_markers(p["title"], taxonomy) == title or (r["category"] != OTHER and p["category"] == r["category"])
                        for p in peers)
        (dropped if duplicate else kept).append(r)
    return kept, dropped
