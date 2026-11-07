"""新闻 RSS 与公司名匹配测试（不联网；公司、代码、标题全部虚构）。"""
from datetime import date

import pandas as pd
import pytest

from jpscan.events.company_match import CompanyMatcher, aliases_for, normalize_text


def _master() -> pd.DataFrame:
    return pd.DataFrame([
        {"Date": date(2026, 8, 14), "Code": "99010", "CoName": "架空自動車", "ProdCat": "011"},
        {"Date": date(2026, 8, 14), "Code": "99020", "CoName": "株式会社モックパーツ", "ProdCat": "011"},
        {"Date": date(2026, 8, 14), "Code": "99030", "CoName": "ダミー精密ホールディングス", "ProdCat": "011"},
        {"Date": date(2026, 8, 14), "Code": "99040", "CoName": "ＸＹＺ", "ProdCat": "011"},
        {"Date": date(2026, 8, 14), "Code": "99050", "CoName": "ギフト", "ProdCat": "011"},
        {"Date": date(2026, 8, 14), "Code": "99060", "CoName": "日本架空", "ProdCat": "011"},
        {"Date": date(2026, 8, 14), "Code": "99070", "CoName": "架空", "ProdCat": "011"},          # 2 字，不生成别名
        {"Date": date(2026, 8, 14), "Code": "13060", "CoName": "架空ＥＴＦ連動投信", "ProdCat": "014"},
        {"Date": date(2026, 8, 14), "Code": "20250", "CoName": "年号商事", "ProdCat": "011"},       # 形如年份的代码
        {"Date": date(2026, 8, 14), "Code": "130A0", "CoName": "英数コード工業", "ProdCat": "011"},
        # 同一 Code 的旧名：只用最新一行
        {"Date": date(2026, 8, 3), "Code": "99080", "CoName": "旧社名工業", "ProdCat": "011"},
        {"Date": date(2026, 8, 14), "Code": "99080", "CoName": "新社名工業", "ProdCat": "011"},
        # 同名：优先最新快照（99090 已退市，99100 在最新快照里）
        {"Date": date(2020, 1, 6), "Code": "99090", "CoName": "同名電機", "ProdCat": "011"},
        {"Date": date(2026, 8, 14), "Code": "99100", "CoName": "同名電機ホールディングス", "ProdCat": "011"},
        # 普通股优先于优先股
        {"Date": date(2026, 8, 14), "Code": "99115", "CoName": "種類株銀行", "ProdCat": "011"},
        {"Date": date(2026, 8, 14), "Code": "99110", "CoName": "種類株銀行", "ProdCat": "011"},
    ])


@pytest.fixture(scope="module")
def matcher() -> CompanyMatcher:
    return CompanyMatcher.from_master(_master())


def test_normalize_text_nfkc_and_spaces():
    assert normalize_text("ＸＹＺ　工業 (株)") == "XYZ工業(株)"
    assert normalize_text("㈱架空") == "(株)架空"


def test_aliases_strip_legal_form_and_holding_suffix():
    assert aliases_for("株式会社モックパーツ") == ["モックパーツ"]
    assert aliases_for("（株）モックパーツ") == ["モックパーツ"]
    assert aliases_for("ダミー精密ホールディングス") == ["ダミー精密ホールディングス", "ダミー精密"]
    assert aliases_for("ダミー精密ＨＤ") == ["ダミー精密HD", "ダミー精密"]
    assert aliases_for("ダミー・ホールディングス") == ["ダミー・ホールディングス", "ダミー"]
    assert aliases_for("架空") == []


def test_match_by_alias(matcher):
    assert matcher.match("架空自動車、新型車を発表") == ["99010"]
    assert matcher.match("モックパーツが増配") == ["99020"]
    assert matcher.match("ダミー精密ＨＤの決算") == ["99030"]
    assert matcher.match("ダミー精密 ホールディングスの決算") == ["99030"]
    assert matcher.match("関係のない話題") == []


def test_match_returns_first_appearance_order_without_duplicates(matcher):
    assert matcher.match("モックパーツと架空自動車が提携、モックパーツは増資") == ["99020", "99010"]


def test_short_names_and_etf_are_not_aliases(matcher):
    assert matcher.match("架空の話") == []
    assert matcher.match("架空ＥＴＦ連動投信") == []


def test_ascii_alias_requires_word_boundary(matcher):
    assert matcher.match("XYZが上場") == ["99040"]
    assert matcher.match("ＸＹＺ、新工場") == ["99040"]
    assert matcher.match("ABCXYZの話") == []
    assert matcher.match("XYZ123の話") == []


def test_katakana_alias_requires_boundary(matcher):
    assert matcher.match("モックパーツセンターを開設") == []
    assert matcher.match("新モックパーツ") == ["99020"]
    assert matcher.match("ナノモックパーツの話") == []


def test_common_word_alias_needs_legal_form(matcher):
    assert matcher.match("お歳暮のギフト特集") == []
    assert matcher.match("株式会社ギフトが新店") == ["99050"]
    assert matcher.match("ギフト(株)が新店") == ["99050"]
    assert matcher.match_name("ギフト") is None


def test_nihon_prefix_not_matched_after_region_kanji(matcher):
    assert matcher.match("日本架空が増益") == ["99060"]
    assert matcher.match("西日本架空が増益") == []


def test_explicit_codes(matcher):
    assert matcher.match("○○が上方修正（9902）") == ["99020"]
    assert matcher.match("<9902>が反発") == ["99020"]
    assert matcher.match("＜9902＞が反発") == ["99020"]
    assert matcher.match("[9902]急騰") == ["99020"]
    assert matcher.match("【9902】急騰") == ["99020"]
    assert matcher.match("某社(東証プライム:9902)") == ["99020"]
    assert matcher.match("某社(証券コード:9902)") == ["99020"]
    assert matcher.match("<130A>が続伸") == ["130A0"]
    assert matcher.match("(9999)は未知の代码") == []          # 主数据里没有
    assert matcher.match("売上高(1000)") == []


def test_year_like_explicit_code_needs_company_name(matcher):
    assert matcher.match("決算短信(2025)") == []
    assert matcher.match("年号商事(2025)が増配") == ["20250"]


def test_match_detail_reports_method(matcher):
    assert matcher.match_detail("架空自動車が発表") == (["99010"], "alias")
    assert matcher.match_detail("某社<9902>が発表") == (["99020"], "code")
    assert matcher.match_detail("架空自動車<9902>が発表") == (["99010", "99020"], "code+alias")
    assert matcher.match_detail("無関係") == ([], "")


def test_latest_row_per_code_is_used(matcher):
    assert matcher.match("新社名工業が増配") == ["99080"]
    assert matcher.match("旧社名工業が増配") == []


def test_alias_collision_prefers_latest_snapshot_then_common_share(matcher):
    assert matcher.match("同名電機が増配") == ["99100"]
    assert matcher.match("種類株銀行が増配") == ["99110"]


def test_collision_without_tiebreak_drops_alias():
    m = CompanyMatcher.from_master(pd.DataFrame([
        {"Date": date(2026, 8, 14), "Code": "99010", "CoName": "分不出工業"},
        {"Date": date(2026, 8, 14), "Code": "99020", "CoName": "分不出工業"},
    ]))
    assert m.match("分不出工業が増配") == []


def test_longest_alias_wins_and_masks_shorter():
    m = CompanyMatcher.from_master(pd.DataFrame([
        {"Code": "99010", "CoName": "架空工業"},
        {"Code": "99020", "CoName": "架空工業機械"},
    ]))
    assert m.match("架空工業機械が増配") == ["99020"]
    assert m.match("架空工業が増配") == ["99010"]


# ---------- RSS 解析（feed 内容全部虚构，格式与真实 feed 相同） ----------
YAHOO_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<rss version="2.0"><channel><title>架空 - Yahoo!ニュース</title>
<item><title>架空自動車が上方修正(架空新聞)</title>
<link>https://news.example.jp/articles/aaa?source=rss</link>
<pubDate>Fri, 30 Oct 2026 15:30:00 GMT</pubDate>
<description>架空自動車は&lt;b&gt;本日&lt;/b&gt;、通期予想を  引き上げた。</description></item>
<item><title>無関係な話題(モックパーツ)</title>
<link>https://news.example.jp/articles/bbb?source=rss</link>
<pubDate>Sat, 31 Oct 2026 01:00:00 GMT</pubDate></item>
<item><title>日付なしの記事</title><link>https://news.example.jp/articles/ccc?source=rss</link></item>
<item><title></title><link>https://news.example.jp/articles/ddd?source=rss</link><pubDate>Sat, 31 Oct 2026 01:00:00 GMT</pubDate></item>
</channel></rss>""".encode("utf-8")

PRTIMES_RDF = """<?xml version="1.0" encoding="UTF-8" ?>
<rdf:RDF xml:language="ja" xmlns="http://purl.org/rss/1.0/" xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<channel rdf:about="https://prtimes.example.jp/index.rdf"><title>PR</title><link>https://prtimes.example.jp</link><description>d</description></channel>
<item rdf:about="https://prtimes.example.jp/p/1.html"><title>新製品を発売しました</title>
<link>https://prtimes.example.jp/p/1.html</link><description>[モックパーツ]
株式会社モックパーツは新製品を発売。</description><dc:corp>モックパーツ</dc:corp><dc:date>2026-11-01T09:15:00+09:00</dc:date></item>
<item rdf:about="https://prtimes.example.jp/p/2.html"><title>架空自動車と協業</title>
<link>https://prtimes.example.jp/p/2.html</link><dc:corp>非上場スタートアップ</dc:corp><dc:date>2026-11-01T10:00:00+09:00</dc:date></item>
</rdf:RDF>""".encode("utf-8")

FEED_Y = {"name": "架空新聞", "url": "https://news.example.jp/rss/a.xml", "tier": "L2", "source": "yahoo_media_a"}
FEED_C = {"name": "聚合", "url": "https://news.example.jp/rss/c.xml", "tier": "L3", "source": "yahoo_category_c"}
FEED_P = {"name": "PR", "url": "https://prtimes.example.jp/index.rdf", "tier": "L3", "source": "prtimes"}


def test_news_id_is_first_24_hex_of_sha256():
    import hashlib

    from jpscan.connectors.news_rss import news_id
    link = "https://news.example.jp/articles/aaa?source=rss"
    assert news_id(link) == hashlib.sha256(link.encode()).hexdigest()[:24]
    assert len(news_id(link)) == 24


def test_title_for_matching_strips_media_suffix_only():
    from jpscan.connectors.news_rss import title_for_matching
    assert title_for_matching("架空自動車が上方修正(架空新聞)") == "架空自動車が上方修正"
    assert title_for_matching("某社が上方修正（9902）") == "某社が上方修正（9902）"
    assert title_for_matching("普通の見出し") == "普通の見出し"


def test_parse_yahoo_feed(matcher):
    from jpscan.connectors.news_rss import COLUMNS, parse_feed
    df, n_undated = parse_feed(YAHOO_XML, FEED_Y, matcher)
    assert n_undated == 1
    assert list(df.columns) == COLUMNS
    assert len(df) == 2
    r = df.iloc[0]
    # 15:30 GMT → 次日 00:30 JST，发布日按 JST
    assert r["published_at"] == "2026-10-31T00:30:00+09:00"
    assert r["published_date"] == date(2026, 10, 31)
    assert r["tier"] == "L2" and r["source"] == "yahoo_media_a"
    assert r["summary"] == "架空自動車は 本日 、通期予想を 引き上げた。"
    assert r["matched_codes"] == "99010" and r["match_method"] == "alias"
    assert r["category"] == ""
    # 标题末尾的媒体名不参与匹配；feed 没给摘要就留空
    assert df.iloc[1]["matched_codes"] == "" and df.iloc[1]["match_method"] == ""
    assert df.iloc[1]["summary"] == ""


def test_parse_prtimes_rdf_uses_corp_field(matcher):
    from jpscan.connectors.news_rss import parse_feed
    df, _ = parse_feed(PRTIMES_RDF, FEED_P, matcher)
    assert len(df) == 2
    assert df.iloc[0]["published_at"] == "2026-11-01T09:15:00+09:00"
    assert df.iloc[0]["matched_codes"] == "99020" and df.iloc[0]["match_method"] == "corp"
    assert df.iloc[1]["matched_codes"] == "99010" and df.iloc[1]["match_method"] == "alias"


def test_parse_feed_without_matcher():
    from jpscan.connectors.news_rss import parse_feed
    df, _ = parse_feed(YAHOO_XML, FEED_Y, None)
    assert (df["matched_codes"] == "").all()


def test_config_feeds_are_well_formed():
    from pathlib import Path

    from jpscan.connectors.news_rss import load_feeds
    feeds = load_feeds(Path(__file__).resolve().parents[1] / "config" / "news_feeds.yaml")
    assert len(feeds) >= 5
    assert len({f["source"] for f in feeds}) == len(feeds)
    assert len({f["url"] for f in feeds}) == len(feeds)
    assert all(f["tier"] in {"L2", "L3"} and f["url"].startswith("https://") and f["name"] for f in feeds)
