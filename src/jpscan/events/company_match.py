"""公司名 ↔ 证券代码匹配（规划文档 4.7、10.5 节 `news.rss`）：用于把新闻标题关联到股票。

别名来自 security_master_snapshots（每个 Code 取最新一行），规范化后去掉法人形态与「ホールディングス」变体。
宁可漏配也不错配：标题级关注度只是线索，错配会污染 headlines_7d 与 AI 证据包。
- 别名至少 3 个字符；长别名优先，命中后遮盖，避免「日本精工」的片段再命中更短的别名。
- 拉丁字母/数字别名要求前后不接拉丁字母/数字；片假名别名要求前后不接片假名。
- 与普通词相同的公司名（アクセス、ギフト、IPS ……）进入 STOP_ALIASES，只有带法人形态才算命中。
- ETF、ETN 与日本銀行（ProdCat 014/023/012）不生成别名：标题里的「日本銀行」几乎都指央行。
- 显式代码（（7203）、<7203>、(東証:7203)）必须是主数据里存在的代码；形如年份的代码（1990–2035）
  还要求紧贴在该公司别名之后，否则「決算(2026)」会被当成代码 2026。
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

MIN_ALIAS_CHARS = 3
EXCLUDED_PRODCATS = {"012", "014", "023"}   # 日本銀行、ETF/ETN、外国 ETF
YEAR_LIKE = range(1990, 2036)

_LEGAL_FORMS = ("株式会社", "(株)", "(有)", "有限会社", "合同会社", "投資法人")
_HOLDING_SUFFIX = re.compile(r"(ホールディングス|ホールディング|ホールデイングス|HOLDINGS|HOLDING|HLDGS|HDGS|HDS|HD)$")
_HYPHENS = str.maketrans({"−": "-", "‐": "-", "‑": "-", "–": "-", "—": "-"})
_TRAILING_PUNCT = re.compile(r"[・.,&\-]+$")

_EXPLICIT_CODE = re.compile(
    r"[(<\[【]"
    r"(?:(?:証券コード|コード|東証[^:)>\]】]{0,8})[:]?)?"
    r"(\d{3}[0-9A-Z])(?:\.T)?"
    r"[)>\]】]"
)
_ASCII_ALNUM = re.compile(r"[A-Z0-9]")
_NIHON_PREFIX_BLOCK = set("東西南北全新")   # 「東日本…」「全日本…」不是「日本…」开头的公司


def _is_katakana(ch: str) -> bool:
    return "ァ" <= ch <= "ヺ" or ch == "ー"


# 规范化后的别名。这些公司名同时是普通词、通用缩写或常见人名，单独出现不足以判定为该公司。
STOP_ALIASES = frozenset("""
アイズ アイル アクシス アクセス アクセル アスタリスク アルファ アンサー イシン インデックス インテグラル
ウィル ウエスト エージェント エーアイ オープンドア オリジン カバー カラダノート カワサキ ガーデン ガーラ キャリア キューブ
キング ギフト クイック クエスト クリエイト クルーズ ケアサービス ケイブ コロナ サポート シード シンカ シンシア ジェイ
ショーケース スカラ スクロール スタメン ステップ ストリーム スペース ズーム セキュア ゼット ソフィア タスキ
チェンジ ティア デジタリフト トップス トライアル トラスト トルク ドーン ナイス ハイパー ハートアップ ハンズ ハンモック
パルマ ヒガシ ヒット ヒップ ヒューマン ビジョン ビーイング ピープル ファースト フィーチャ フライヤー フリー
フューチャー ブラス ブリーチ プラザ プラネット プラン・ドゥ フロンティア プロジェクト ベーシック ベース ホープ ボルテージ ポート
ポーターズ ポプラ マーキュリー マックス マニー マルシェ メンバーズ ユナイテッド ライク ライトアップ ランド リード
リズム リップス リベロ ルック ルネサンス レント ローカル ロイヤル ロココ ロゴス ワイズ ワールド
いい生活 いちご いつも うるる かっこ じげん じもと はてな ひかり みらい あさひ あらた 手間いらず 孫の手 力の源 三ッ星
光通信 電源開発 表示灯 情報企画 応用技術 空港施設 東海運 進学会 夢展望 日本銀行
ABC ACCESS AIRMAN AKIBA ANDDO ARE BASE BIRDMAN BRUNO CAC CAPITA CASA CDS CSS DTS ELEMENTS ERI FFF FIG FUJI G-7 GAIA GLP
INCLUSIVE IPS MIC MIT MRT MTG NANO NO.1 NOTE NSW ORCHESTRA PCI PEGASUS PILLAR PLANT PRO REVOLUTION RISE SCREEN SHIFT SMILE
SRS SYS TONE TRE ZETA
""".split())


def normalize_text(text: str) -> str:
    """NFKC、统一连字符、去空白、拉丁字母转大写。匹配两侧都用这个形式。"""
    s = unicodedata.normalize("NFKC", str(text)).translate(_HYPHENS)
    return re.sub(r"\s+", "", s).upper()


def _strip_legal(name: str) -> str:
    for form in _LEGAL_FORMS:
        name = name.replace(form, "")
    return _TRAILING_PUNCT.sub("", name)


def _strip_holding(name: str) -> str:
    return _TRAILING_PUNCT.sub("", _HOLDING_SUFFIX.sub("", name))


def aliases_for(co_name: str) -> list[str]:
    """由正式公司名生成别名：去法人形态后的全名、再去持股公司后缀。普通词名只保留带法人形态的写法。"""
    base = _strip_legal(normalize_text(co_name))
    out: list[str] = []
    for cand in dict.fromkeys([base, _strip_holding(base)]):
        if len(cand) < MIN_ALIAS_CHARS:
            continue
        if cand in STOP_ALIASES:
            out.extend([f"株式会社{cand}", f"{cand}株式会社", f"(株){cand}", f"{cand}(株)"])
        else:
            out.append(cand)
    return out


class CompanyMatcher:
    def __init__(self, alias_to_code: dict[str, str], known_codes: set[str], code_aliases: dict[str, list[str]] | None = None):
        self.alias_to_code = alias_to_code
        self.known_codes = known_codes
        self.code_aliases = code_aliases or {}
        self._ordered = sorted(alias_to_code, key=lambda a: (-len(a), a))

    @classmethod
    def from_master(cls, df: pd.DataFrame) -> "CompanyMatcher":
        """df 含 Code、CoName，可选 Date、ProdCat。同一别名被多个 Code 使用时：最新快照优先，其次普通股（末位 0）；
        仍分不出就丢弃该别名。"""
        m = df.dropna(subset=["Code", "CoName"]).copy()
        m["Code"] = m["Code"].astype(str)
        known = {c[:4] for c in m["Code"]}
        if "Date" in m.columns:
            m["_d"] = pd.to_datetime(m["Date"])
            m = m.sort_values("_d")
        else:
            m["_d"] = pd.Timestamp(0)
        m = m.drop_duplicates("Code", keep="last")
        if "ProdCat" in m.columns:
            m = m[~m["ProdCat"].astype(str).isin(EXCLUDED_PRODCATS)]

        best: dict[str, tuple[pd.Timestamp, bool, str]] = {}
        ambiguous: set[str] = set()
        for code, name, d in zip(m["Code"], m["CoName"], m["_d"]):
            common = code.endswith("0")
            for alias in aliases_for(name):
                cur = best.get(alias)
                if cur is None or (d, common) > cur[:2]:
                    best[alias] = (d, common, code)
                    ambiguous.discard(alias)
                elif (d, common) == cur[:2] and code != cur[2]:
                    ambiguous.add(alias)
        alias_to_code = {a: v[2] for a, v in best.items() if a not in ambiguous}
        code_aliases: dict[str, list[str]] = {}
        for alias, code in alias_to_code.items():
            code_aliases.setdefault(code, []).append(alias)
        return cls(alias_to_code, known, code_aliases)

    # ---------- 匹配 ----------
    def match(self, text: str) -> list[str]:
        return self.match_detail(text)[0]

    def match_detail(self, text: str) -> tuple[list[str], str]:
        """返回 (按出现顺序去重的 5 位代码, 方法：code / alias / code+alias / 空串)。"""
        s = normalize_text(text)
        hits: list[tuple[int, str]] = []
        by_code = self._explicit_codes(s)
        hits.extend(by_code)
        by_alias = self._alias_hits(s)
        hits.extend(by_alias)
        codes = list(dict.fromkeys(c for _, c in sorted(hits, key=lambda h: h[0])))
        method = "+".join(name for name, found in (("code", by_code), ("alias", by_alias)) if found)
        return codes, method

    def match_name(self, name: str) -> str | None:
        """公司名完全一致才算命中（PR TIMES 的 dc:corp）。普通词公司名不接受，同名非上市公司太多。"""
        n = _strip_legal(normalize_text(name))
        return self.alias_to_code.get(n) or self.alias_to_code.get(_strip_holding(n))

    def _explicit_codes(self, s: str) -> list[tuple[int, str]]:
        out: list[tuple[int, str]] = []
        for m in _EXPLICIT_CODE.finditer(s):
            code4 = m.group(1)
            if code4 not in self.known_codes:
                continue
            code = code4 + "0"
            if code4.isdigit() and int(code4) in YEAR_LIKE:
                before = s[: m.start()]
                if not any(before.endswith(a) for a in self.code_aliases.get(code, [])):
                    continue
            out.append((m.start(), code))
        return out

    def _alias_hits(self, s: str) -> list[tuple[int, str]]:
        out: list[tuple[int, str]] = []
        masked = s
        for alias in self._ordered:
            if alias not in masked:
                continue
            start = 0
            while True:
                pos = masked.find(alias, start)
                if pos < 0:
                    break
                end = pos + len(alias)
                if _boundary_ok(masked, alias, pos, end):
                    out.append((pos, self.alias_to_code[alias]))
                    masked = masked[:pos] + "\0" * len(alias) + masked[end:]
                start = end
        return out


def _boundary_ok(s: str, alias: str, pos: int, end: int) -> bool:
    prev = s[pos - 1] if pos > 0 else ""
    nxt = s[end] if end < len(s) else ""
    if prev and _ASCII_ALNUM.fullmatch(alias[0]) and _ASCII_ALNUM.fullmatch(prev):
        return False
    if nxt and _ASCII_ALNUM.fullmatch(alias[-1]) and _ASCII_ALNUM.fullmatch(nxt):
        return False
    if prev and _is_katakana(alias[0]) and _is_katakana(prev):
        return False
    if nxt and _is_katakana(alias[-1]) and _is_katakana(nxt):
        return False
    if alias.startswith("日本") and prev in _NIHON_PREFIX_BLOCK:
        return False
    return True
