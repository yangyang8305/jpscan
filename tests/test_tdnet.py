"""TDnet 连接器测试。全部使用合成数据：公司与数字均为虚构，真实下载内容不进入仓库。"""
import zipfile
from datetime import date
from pathlib import Path

import pytest
import requests

from jpscan.connectors.base import RawSink
from jpscan.connectors.tdnet import SOURCE, TdnetBlocked, TdnetClient, TdnetFormatError, assign_disclosure_ids, parse_list_page
from jpscan.connectors.tdnet_xbrl import parse_summary
from jpscan.events.classify import classify_title, dedupe_ja_en, detect_lang, download_policy, is_correction_title, load_taxonomy, normalize_title
from jpscan.ledger import Ledger
from jpscan.paths import DataPaths
from jpscan.util.common import RateLimiter

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
    ("自己株式立会外買付取引（ToSTNeT-3）による自己株式の買付け結果に関するお知らせ", "E40"),
    ("自己株式立会外買付取引（ToSTNeT-3）による自己株式の買付け価格確定のお知らせ", "E40"),
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


# ---------------- 列表页 ----------------
FIXTURE = ROOT / "tests" / "fixtures" / "tdnet" / "I_list_001_20260105.html"
DAY = date(2026, 1, 5)


class FakeResponse:
    def __init__(self, status_code=200, content=b""):
        self.status_code, self.content = status_code, content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class FakeSession:
    """按 URL 文件名返回内容；未登记的 URL 返回 404。记录请求序列。"""

    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []

    def get(self, url, **kw):
        name = url.rsplit("/", 1)[-1]
        self.calls.append(name)
        r = self.responses.get(name)
        return FakeResponse(404) if r is None else r if isinstance(r, FakeResponse) else FakeResponse(200, r)


def make_client(tmp_path, responses):
    paths = DataPaths(tmp_path).ensure()
    ledger = Ledger(paths.ledger_sqlite)
    session = FakeSession(responses)
    return TdnetClient(RawSink(paths, ledger, SOURCE), session=session, limiter=RateLimiter(10_000_000)), session, paths, ledger


def synthetic_page(d, rows, total, page_date=None):
    """rows: (time, code, title, pdf)。结构同真实页面。"""
    trs = "".join(
        f'<tr><td class="oddnew-L kjTime" noWrap>{t}</td><td class="oddnew-M kjCode" noWrap>{c}</td>'
        f'<td class="oddnew-M kjName" noWrap>架空{c}</td><td class="oddnew-M kjTitle" align="left">'
        + (f'<a href="{pdf}" target="_blank">{title}</a>' if pdf else title)
        + '</td><td class="oddnew-M kjXbrl" noWrap align="center"> </td><td class="oddnew-M kjPlace" noWrap align="left">東</td>'
        '<td class="oddnew-R kjHistroy" align="left">　</td></tr>'
        for t, c, title, pdf in rows)
    shown = page_date or d
    return (f'<html><body><div id="kaiji-date-1">{shown.year}年{shown.month:02d}月{shown.day:02d}日</div>'
            f'<div class="kaijiSum">全{total}件</div><table id="main-list-table">{trs}</table></body></html>').encode("utf-8")


def test_parse_list_page_fixture():
    parsed = parse_list_page(FIXTURE.read_bytes())
    assert parsed["date"] == DAY
    assert parsed["total"] == 7
    rows = parsed["rows"]
    assert len(rows) == 7
    first = rows[0]
    assert (first["time"], first["Code"], first["code4"], first["company"]) == ("18:30", "99910", "9991", "架空工業")
    assert first["title"] == "通期業績予想の上方修正に関するお知らせ"
    assert first["pdf_url"] == "https://www.release.tdnet.info/inbs/140120260105000007.pdf"
    assert first["xbrl_url"] == "https://www.release.tdnet.info/inbs/081220260105000007.zip"
    assert (first["exchanges"], first["correction_marker"]) == ("東", False)     # 全角空格不算更新履历
    assert rows[1]["exchanges"] == "東名"
    assert rows[3]["xbrl_url"] is None
    assert (rows[4]["Code"], rows[4]["code4"], rows[4]["correction_marker"]) == ("999A0", "999A", True)   # 字母代码、更新履历
    assert (rows[5]["Code"], rows[5]["code4"], rows[5]["pdf_url"]) == ("99950", "9995", None)             # 4 位补 0、无 PDF
    assert rows[6]["pdf_url"] is None                                                                     # 外部链接不跟随


def test_parse_list_page_rejects_other_pages():
    with pytest.raises(TdnetFormatError):
        parse_list_page("<html><body>maintenance</body></html>")
    assert parse_list_page("<html><table id='main-list-table'></table></html>")["rows"] == []


def test_disclosure_ids_stable_when_new_rows_arrive():
    def rows(*codes):
        return [{"Code": c, "pdf_name": None} for c in codes]
    morning = rows("99910", "99910")
    assign_disclosure_ids(morning, DAY)
    evening = rows("99910", "99920", "99910", "99910")   # 新披露排在前面
    assign_disclosure_ids(evening, DAY)
    assert [r["disclosure_id"] for r in morning] == ["20260105_99910_2", "20260105_99910_1"]
    assert [r["disclosure_id"] for r in evening[-2:]] == ["20260105_99910_2", "20260105_99910_1"]
    with_pdf = [{"Code": "99910", "pdf_name": "140120260105000001.pdf"}]
    assign_disclosure_ids(with_pdf, DAY)
    assert with_pdf[0]["disclosure_id"] == "140120260105000001"


def test_fetch_day_pages_until_declared_total_and_saves_raw(tmp_path):
    p1 = synthetic_page(DAY, [("15:30", "99910", "株主総会資料", "140120260105000002.pdf")], 2)
    p2 = synthetic_page(DAY, [("09:00", "99920", "株主総会資料", "140120260105000001.pdf")], 2)
    client, session, paths, ledger = make_client(tmp_path, {"I_list_001_20260105.html": p1, "I_list_002_20260105.html": p2})
    day = client.fetch_day(DAY)
    assert session.calls == ["I_list_001_20260105.html", "I_list_002_20260105.html"]   # 达到总件数即停，不多请求 404
    assert (len(day["rows"]), day["pages"], day["complete"]) == (2, 2, True)
    assert [r["page"] for r in day["rows"]] == [1, 2]
    raw = paths.raw_dir("tdnet", "list") / "2026"
    assert (raw / "20260105_p001.html").read_bytes() == p1
    assert (raw / "20260105_p002.html").exists()
    with ledger.conn() as c:
        assert c.execute("SELECT COUNT(*) FROM raw_files WHERE source='tdnet' AND dataset='list'").fetchone()[0] == 2


def test_fetch_day_stops_at_404_and_flags_incomplete(tmp_path):
    p1 = synthetic_page(DAY, [("15:30", "99910", "株主総会資料", "140120260105000002.pdf")], 3)
    client, session, *_ = make_client(tmp_path, {"I_list_001_20260105.html": p1})
    day = client.fetch_day(DAY)
    assert session.calls == ["I_list_001_20260105.html", "I_list_002_20260105.html"]
    assert (len(day["rows"]), day["complete"]) == (1, False)


def test_fetch_day_empty_day_and_wrong_date_and_block(tmp_path):
    client, _, *_ = make_client(tmp_path, {})
    assert client.fetch_day(DAY) == {"rows": [], "total": None, "pages": 0, "complete": True}   # 休日：首页即 404
    wrong = synthetic_page(DAY, [("15:30", "99910", "資料", None)], 1, page_date=date(2026, 1, 6))
    client, _, *_ = make_client(tmp_path / "b", {"I_list_001_20260105.html": wrong})
    with pytest.raises(TdnetFormatError):
        client.fetch_day(DAY)
    client, session, *_ = make_client(tmp_path / "c", {"I_list_001_20260105.html": FakeResponse(403)})
    with pytest.raises(TdnetBlocked):
        client.fetch_day(DAY)
    assert len(session.calls) == 1


def test_download_file_checks_content(tmp_path):
    client, _, paths, _ = make_client(tmp_path, {"a.pdf": b"%PDF-1.7 synthetic", "bad.pdf": b"<html>error</html>"})
    url = "https://www.release.tdnet.info/inbs/"
    path = client.download_file(url + "a.pdf", "pdf", "a", ".pdf", b"%PDF", "2026-01")
    assert path == paths.raw_dir("tdnet", "pdf") / "2026-01" / "a.pdf"
    assert client.download_file(url + "bad.pdf", "pdf", "bad", ".pdf", b"%PDF", "2026-01") is None
    assert client.download_file(url + "gone.pdf", "pdf", "gone", ".pdf", b"%PDF", "2026-01") is None
    assert list((paths.quarantine / "tdnet" / "pdf").glob("bad_*.bin"))


# ---------------- XBRL ----------------
IXBRL = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:ix="http://www.xbrl.org/2008/inlineXBRL"
      xmlns:xbrli="http://www.xbrl.org/2003/instance" xmlns:xbrldi="http://xbrl.org/2006/xbrldi"
      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:tse-ed-t="http://www.xbrl.tdnet.info/taxonomy/jp/tse/tdnet/ed/t/2014-01-12"
      xmlns:ixt="http://www.xbrl.org/inlineXBRL/transformation/2011-07-31">
<head><title>synthetic</title></head><body>
<ix:header><ix:resources>
<xbrli:context id="CurrentYearDuration_ConsolidatedMember_ResultMember"><xbrli:entity><xbrli:segment>
  <xbrldi:explicitMember dimension="tse-ed-t:ConsolidatedNonconsolidatedAxis">tse-ed-t:ConsolidatedMember</xbrldi:explicitMember>
  <xbrldi:explicitMember dimension="tse-ed-t:ResultForecastAxis">tse-ed-t:ResultMember</xbrldi:explicitMember>
</xbrli:segment></xbrli:entity><xbrli:period><xbrli:startDate>2026-04-01</xbrli:startDate><xbrli:endDate>2027-03-31</xbrli:endDate></xbrli:period></xbrli:context>
<xbrli:context id="NextYearDuration_ConsolidatedMember_ForecastMember"><xbrli:entity><xbrli:segment>
  <xbrldi:explicitMember dimension="tse-ed-t:ConsolidatedNonconsolidatedAxis">tse-ed-t:ConsolidatedMember</xbrldi:explicitMember>
  <xbrldi:explicitMember dimension="tse-ed-t:ResultForecastAxis">tse-ed-t:ForecastMember</xbrldi:explicitMember>
</xbrli:segment></xbrli:entity><xbrli:period><xbrli:startDate>2027-04-01</xbrli:startDate><xbrli:endDate>2028-03-31</xbrli:endDate></xbrli:period></xbrli:context>
</ix:resources></ix:header>
<ix:nonFraction name="tse-ed-t:NetSales" contextRef="CurrentYearDuration_ConsolidatedMember_ResultMember" unitRef="JPY" scale="6" decimals="-6" format="ixt:numdotdecimal">1,234</ix:nonFraction>
<ix:nonFraction name="tse-ed-t:OperatingIncome" contextRef="CurrentYearDuration_ConsolidatedMember_ResultMember" unitRef="JPY" scale="6" decimals="-6" sign="-" format="ixt:numdotdecimal">56</ix:nonFraction>
<ix:nonFraction name="tse-ed-t:NetSales" contextRef="NextYearDuration_ConsolidatedMember_ForecastMember" unitRef="JPY" scale="6" decimals="-6" format="ixt:numdotdecimal">1,500</ix:nonFraction>
<ix:nonFraction name="tse-ed-t:DividendPerShare" contextRef="NextYearDuration_ConsolidatedMember_ForecastMember" unitRef="JPY" decimals="2" format="ixt:numdotdecimal">12.50</ix:nonFraction>
<ix:nonFraction name="tse-ed-t:NetIncomePerShare" contextRef="NextYearDuration_ConsolidatedMember_ForecastMember" unitRef="JPY" xsi:nil="true"></ix:nonFraction>
<ix:nonFraction name="tse-ed-t:ChangeInNetSales" contextRef="PriorYearDuration_NonConsolidatedMember_ForecastMember" unitRef="Pure" scale="-2" decimals="3" format="ixt:numdotdecimal">７.５</ix:nonFraction>
<ix:nonNumeric name="tse-ed-t:CompanyName" contextRef="CurrentYearDuration_ConsolidatedMember_ResultMember">架空工業</ix:nonNumeric>
</body></html>
"""


def _zip(tmp_path, members):
    path = tmp_path / "synthetic.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for name, text in members.items():
            zf.writestr(name, text.encode("utf-8"))
    return path


def test_parse_summary_lists_facts_with_forecast_helpers(tmp_path):
    path = _zip(tmp_path, {
        "XBRLData/Summary/tse-acedjpsm-99910-20270331-ixbrl.htm": IXBRL,
        "XBRLData/Attachment/0101010-acbs01-tse-acedjpsm-99910-ixbrl.htm": IXBRL.replace("1,234", "999"),   # 非摘要，不读
        "XBRLData/Summary/tse-acedjpsm-99910-20270331.xsd": "<schema/>",
    })
    facts = parse_summary(path)
    assert len(facts) == 6
    assert {f["doc_type"] for f in facts} == {"acedjpsm"}
    by = {(f["local_name"], f["contextRef"]): f for f in facts}
    sales = by[("NetSales", "CurrentYearDuration_ConsolidatedMember_ResultMember")]
    assert sales["name"] == "tse-ed-t:NetSales"
    assert (sales["unitRef"], sales["scale"], sales["sign"], sales["text"], sales["value"]) == ("JPY", "6", None, "1,234", 1_234_000_000.0)
    assert (sales["is_forecast"], sales["result_forecast"], sales["period_end"]) == (False, "ResultMember", "2027-03-31")
    assert sales["dimensions"]["ConsolidatedNonconsolidatedAxis"] == "ConsolidatedMember"
    op = by[("OperatingIncome", "CurrentYearDuration_ConsolidatedMember_ResultMember")]
    assert (op["sign"], op["value"]) == ("-", -56_000_000.0)
    fc = by[("NetSales", "NextYearDuration_ConsolidatedMember_ForecastMember")]
    assert (fc["is_forecast"], fc["result_forecast"], fc["value"]) == (True, "ForecastMember", 1_500_000_000.0)
    assert by[("DividendPerShare", "NextYearDuration_ConsolidatedMember_ForecastMember")]["value"] == 12.5
    nil = by[("NetIncomePerShare", "NextYearDuration_ConsolidatedMember_ForecastMember")]
    assert (nil["is_nil"], nil["value"]) == (True, None)
    # 文件内没有 context 定义时，按 contextRef 名称判断预想
    fallback = by[("ChangeInNetSales", "PriorYearDuration_NonConsolidatedMember_ForecastMember")]
    assert (fallback["is_forecast"], fallback["dimensions"], fallback["value"]) == (True, {}, pytest.approx(0.075))


def test_parse_summary_without_summary_file(tmp_path):
    assert parse_summary(_zip(tmp_path, {"XBRLData/Attachment/report.htm": "<html/>"})) == []


def test_parse_summary_does_not_expand_external_entities(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("99999", encoding="utf-8")
    doc = (f'<?xml version="1.0"?><!DOCTYPE html [<!ENTITY x SYSTEM "{secret.as_uri()}">]>'
           '<html xmlns:ix="http://www.xbrl.org/2008/inlineXBRL"><ix:nonFraction name="a" contextRef="c">&x;</ix:nonFraction></html>')
    facts = parse_summary(_zip(tmp_path, {"tse-rvfc-99910-ixbrl.htm": doc}))
    assert [f["value"] for f in facts] == [None]
