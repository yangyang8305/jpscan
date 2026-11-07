"""TDnet 连接器测试。全部使用合成数据：公司与数字均为虚构，真实下载内容不进入仓库。"""
from pathlib import Path

import pytest

from jpscan.events.classify import classify_title, dedupe_ja_en, detect_lang, download_policy, is_correction_title, load_taxonomy, normalize_title

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy(ROOT / "config" / "events.yaml")


def test_normalize_title():
    assert normalize_title("２０２７年３月期　決算短信〔日本基準〕（連結）") == "2027年3月期決算短信[日本基準](連結)"
    assert normalize_title("ＭＢＯの実施【重要】") == "mboの実施[重要]"


@pytest.mark.parametrize("title,category,direction,strength,hard,special,needs_ai", [
    ("通期業績予想の上方修正に関するお知らせ", "E01", 1, 3, False, False, False),
    ("業績予想の下方修正に関するお知らせ", "E02", -1, -3, False, False, False),
    ("業績予想の修正（上方）に関するお知らせ", "E01", 1, 3, False, False, False),
    ("業績予想の修正に関するお知らせ", "E01/E02", 0, 0, False, False, True),
    ("業績予想と実績との差異に関するお知らせ", "E01/E02", 0, 0, False, False, True),
    ("配当予想の修正に関するお知らせ", "E03/E04", 0, 0, False, False, True),
    ("配当予想の修正（増配）に関するお知らせ", "E03", 1, 2, False, False, False),
    ("配当予想の修正（無配）に関するお知らせ", "E04", -1, -2, False, False, False),
    ("創立50周年記念配当に関するお知らせ", "E03", 1, 1, False, False, False),
    ("自己株式取得に係る事項の決定に関するお知らせ", "E05", 1, 2, False, False, False),
    ("公募による新株式発行に関するお知らせ", "E07", -1, -2, False, False, False),
    ("第三者割当による新株式発行に関するお知らせ", "E08", -1, -1, False, False, True),
    ("第三者割当による行使価額修正条項付新株予約権の発行に関するお知らせ", "E10", -1, -3, True, False, False),
    ("公開買付けの開始に関するお知らせ", "E11", 1, 3, False, True, False),
    ("ＭＢＯの実施及び応募の推奨に関するお知らせ", "E12", 1, 3, False, True, False),
    ("継続企業の前提に関する事項の注記についてのお知らせ", "E24", -1, -3, True, False, False),
    ("監理銘柄（確認中）への指定に関するお知らせ", "E20", -1, -3, True, False, False),
    ("株式分割及び定款の一部変更に関するお知らせ", "E25", 1, 1, False, False, False),
    ("2027年3月期 第2四半期（中間期）決算短信〔日本基準〕（連結）", "E37", 0, 0, False, False, False),
    ("役員の異動に関するお知らせ", "E36", 0, 0, False, False, False),
    ("月次売上高のお知らせ", "E40", 0, 0, False, False, False),
])
def test_classify_japanese(tax, title, category, direction, strength, hard, special, needs_ai):
    r = classify_title(title, tax)
    assert (r["category"], r["direction"], r["strength"], r["hard_exclude"], r["special"], r["needs_ai"]) == \
        (category, direction, strength, hard, special, needs_ai), r
    assert r["lang"] == "ja"


def test_result_fields(tax):
    r = classify_title("公開買付けの開始に関するお知らせ", tax)
    assert set(r) == {"category", "direction", "strength", "half_life_days", "hard_exclude", "special", "needs_ai",
                      "matched_keyword", "lang", "underlying"}
    assert r["matched_keyword"] == "公開買付けの開始"
    assert r["half_life_days"] is None
    assert classify_title("通期業績予想の上方修正について", tax)["half_life_days"] == 20


@pytest.mark.parametrize("title,underlying", [
    ("（訂正）2027年3月期 第2四半期決算短信〔日本基準〕（連結）", "E37"),
    ("（訂正・数値データ訂正）2027年3月期 決算短信〔IFRS〕（連結）", "E37"),
    ("【訂正】業績予想の上方修正に関するお知らせ", "E01"),
    ("「自己株式取得に係る事項の決定」の一部訂正について", "E05"),
    ("（訂正）会社説明会のご案内", "E40"),
])
def test_correction_underlying(tax, title, underlying):
    r = classify_title(title, tax)
    assert (r["category"], r["direction"], r["strength"], r["underlying"]) == ("E38", -1, -1, underlying), r
    assert is_correction_title(title, tax)


def test_correction_inherits_undetermined_revision_needs_ai(tax):
    assert classify_title("（訂正）業績予想の修正に関するお知らせ", tax)["needs_ai"] is True
    assert classify_title("（訂正）業績予想の上方修正に関するお知らせ", tax)["needs_ai"] is False


def test_hard_exclude_beats_correction(tax):
    r = classify_title("過年度決算の訂正に関するお知らせ", tax)
    assert (r["category"], r["hard_exclude"]) == ("E17", True)
    r = classify_title("（訂正）行使価額修正条項付新株予約権の発行に関するお知らせ", tax)
    assert (r["category"], r["hard_exclude"], r["underlying"]) == ("E10", True, None)
    assert is_correction_title("（訂正）行使価額修正条項付新株予約権の発行に関するお知らせ", tax)


@pytest.mark.parametrize("title,category,direction,needs_ai", [
    ("通期業績予想の上方修正及び期末配当予想の修正（増配）に関するお知らせ", "E01", 1, False),
    ("業績予想の修正及び配当予想の修正（増配）に関するお知らせ", "E03", 1, True),      # 业绩方向未定
    ("業績予想の下方修正及び配当予想の修正（増配）に関するお知らせ", "E02", -1, True),  # 同标题正负冲突
    ("業績予想の修正及び配当予想の修正に関するお知らせ", "E01/E02", 0, True),
    ("売上高の上方修正及び利益の下方修正について、業績予想の修正のお知らせ", "E01/E02", 0, True),
    ("通期業績予想の上方修正及び特別損失の計上に関するお知らせ", "E01", 1, True),      # 修正类优先，但方向冲突
])
def test_revision_family_combinations(tax, title, category, direction, needs_ai):
    r = classify_title(title, tax)
    assert (r["category"], r["direction"], r["needs_ai"]) == (category, direction, needs_ai), r


@pytest.mark.parametrize("title,category", [
    ("自己株式の取得状況に関するお知らせ", "E40"),            # 进度报告不计为新事件
    ("自己株式の公開買付けの開始に関するお知らせ", "E05"),    # E11.exclude 含 自己株式
    ("継続企業の前提に関する重要事象の解消に関するお知らせ", "E40"),
    ("監理銘柄の指定解除に関するお知らせ", "E40"),
    ("資本業務提携の解消に関するお知らせ", "E40"),
    ("決算説明資料（決算短信補足資料）", "E40"),
    ("公募による新株式発行及び株式売出しに関するお知らせ", "E07"),  # 更长关键词优先于 E33
])
def test_category_exclude_and_longest_keyword(tax, title, category):
    assert classify_title(title, tax)["category"] == category


def test_index_category_uses_direction_cues(tax):
    up = classify_title("日経平均株価の構成銘柄への採用に関するお知らせ", tax)
    down = classify_title("日経平均株価の構成銘柄からの除外に関するお知らせ", tax)
    unknown = classify_title("MSCI指数に関するお知らせ", tax)
    assert (up["category"], up["direction"], up["strength"], up["needs_ai"]) == ("E34", 1, 1, False)
    assert (down["category"], down["direction"], down["strength"], down["needs_ai"]) == ("E34", -1, -1, False)
    assert (unknown["category"], unknown["direction"], unknown["strength"], unknown["needs_ai"]) == ("E34", 0, 0, True)


@pytest.mark.parametrize("title,category,direction", [
    ("Notice Regarding Upward Revision of Earnings Forecast", "E01", 1),
    ("Notice Regarding Revision of Dividend Forecast (Dividend Increase)", "E03", 1),
    ("Notice Regarding Revision of Earnings Forecast", "E01/E02", 0),
    ("Consolidated Financial Results for the Six Months Ended September 30, 2026 [Japanese GAAP]", "E37", 0),
    ("(Correction) Consolidated Financial Results for the Fiscal Year Ended March 31, 2026", "E38", -1),
    ("Notice Regarding Commencement of Tender Offer", "E11", 1),
    ("Notice Regarding Corporate Governance Report", "E40", 0),
])
def test_classify_english(tax, title, category, direction):
    r = classify_title(title, tax)
    assert (r["category"], r["direction"], r["lang"]) == (category, direction, "en"), r


def test_short_ascii_keyword_needs_word_boundary(tax):
    assert classify_title("Embodying Our Corporate Philosophy", tax)["category"] == "E40"
    assert classify_title("Notice Regarding the MBO of a Subsidiary", tax)["category"] == "E12"


@pytest.mark.parametrize("title,lang", [
    ("2027年3月期 決算短信〔IFRS〕（連結）", "ja"),
    ("決算説明資料（英訳）", "en"),
    ("[Delayed] Notice of Annual Results", "en"),
    ("IRカレンダー", "ja"),
])
def test_detect_lang(tax, title, lang):
    assert detect_lang(title, tax) == lang


def test_download_policy(tax):
    def policy(title):
        return download_policy(classify_title(title, tax), tax)
    assert policy("2027年3月期 決算短信〔日本基準〕（連結）") == {"pdf": True, "xbrl": True}
    assert policy("業績予想の修正に関するお知らせ") == {"pdf": True, "xbrl": True}
    assert policy("公開買付けの開始に関するお知らせ") == {"pdf": True, "xbrl": False}
    assert policy("役員の異動に関するお知らせ") == {"pdf": False, "xbrl": False}
    assert policy("月次売上高のお知らせ") == {"pdf": False, "xbrl": False}
    assert policy("（訂正）2027年3月期 決算短信〔日本基準〕（連結）") == {"pdf": True, "xbrl": True}


def test_dedupe_ja_en(tax):
    def row(code, tm, title):
        r = classify_title(title, tax)
        return {"Code": code, "disclosed_date": "2026-01-05", "disclosed_time": tm, "title": title, "lang": r["lang"], "category": r["category"]}
    rows = [
        row("99910", "15:30:00", "2027年3月期 決算短信〔日本基準〕（連結）"),
        row("99910", "15:30:00", "Consolidated Financial Results for the Fiscal Year [Japanese GAAP]"),  # 同类别 → 去掉
        row("99920", "15:30:00", "株主総会資料"),
        row("99920", "15:30:00", "株主総会資料（英訳）"),                 # 去标记后同标题 → 去掉
        row("99930", "16:00:00", "配当予想の修正（増配）に関するお知らせ"),
        row("99930", "17:00:00", "Notice Regarding Dividend Increase"),   # 时刻不同 → 保留
        row("99940", "12:00:00", "Notice of Corporate Governance Report"),   # 无日文对应 → 保留
        row("99950", "12:00:00", "役員の異動に関するお知らせ"),
        row("99950", "12:00:00", "Notice Regarding a New Product"),       # E40 且标题不同：无法判断 → 保留
    ]
    kept, dropped = dedupe_ja_en(rows, tax)
    assert [r["title"] for r in dropped] == [rows[1]["title"], rows[3]["title"]]
    assert len(kept) == 7
