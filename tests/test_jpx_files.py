"""JPX 连接器与作业的测试：全部使用合成数据（虚构公司与数字，格式仿真实文件），不联网。"""
from __future__ import annotations

import io
from datetime import date, datetime
from types import SimpleNamespace
from typing import Any

import openpyxl
import pandas as pd
import pytest

from jpscan.config import Settings
from jpscan.connectors import jpx_files as J
from jpscan.jobs import ingest_jpx
from jpscan.ledger import Ledger
from jpscan.paths import DataPaths
from jpscan.store import Store


def xlsx(sheets: dict[str, list[list[Any]]]) -> bytes:
    """运行时用 openpyxl 生成工作簿（仓库不存任何 xlsx）。"""
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        for r in rows:
            ws.append(r)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------------------------
# 合成页面（虚构公司）
# ---------------------------------------------------------------------------------------------
SUPERVISION_CURRENT_HTML = """<html><body><h2>現在の指定状況</h2>
<h3>監理銘柄（確認中）</h3>
<table><tr><th>指定年月日</th><th>銘柄名</th><th>コード</th><th>市場区分 （注）</th><th>詳細</th><th>備考</th></tr>
<tr><td>2026/10/01</td><td>（株）架空商事</td><td>9991</td><td>スタンダード</td><td><a href="/listing/x/a.html">詳細</a></td><td>超過計画開示会社</td></tr>
</table>
<h3>監理銘柄（審査中）</h3>
<table><tr><th>指定年月日</th><th>銘柄名</th><th>コード</th><th>市場区分 （注）</th><th>詳細</th><th>備考</th></tr>
<tr><td>2026/09/15</td><td>（株）試験製作所</td><td>999A</td><td>グロース</td><td></td><td></td></tr>
</table>
<h3>整理銘柄</h3>
<table><tr><th>指定年月日</th><th>銘柄名</th><th>コード</th><th>市場区分 （注）</th><th>詳細</th><th>備考</th></tr>
</table></body></html>"""

SUPERVISION_HISTORY_HTML = """<html><body><h2>指定履歴</h2>
<select class="backnumber" size="1"><option selected="selected" value="/listing/market-alerts/supervision/01.html">2026年</option>
<option value="/listing/market-alerts/supervision/01-archives-01.html">2025年</option></select>
<table><thead><tr><th>指定年月日</th><th>銘柄名</th><th>コード</th><th>市場区分 （注）</th><th>解除年月日</th><th>内容</th><th>備考</th></tr></thead>
<tr><td>2026/10/01</td><td>（株）架空商事</td><td>9991</td><td>スタンダード</td><td>-</td><td>監理銘柄（確認中）指定</td><td></td></tr>
<tr><td>2026/05/01</td><td>（株）既廃止工業</td><td>9992</td><td>スタンダード</td><td>-</td><td>上場廃止の決定・整理銘柄指定</td><td></td></tr>
<tr><td>2026/03/02</td><td>（株）解除電機</td><td>9993</td><td>プライム</td><td>2026/04/10</td><td>監理銘柄（確認中）指定</td><td></td></tr>
</table></body></html>"""


def test_supervision_history_dash_is_not_current():
    hist = J.parse_supervision_history(SUPERVISION_HISTORY_HTML)
    assert len(hist) == 3
    assert not hist["is_current"].any()  # "-" 不代表仍在指定中（已退市者同样是 "-"）
    row = hist[hist["Code"] == "99930"].iloc[0]
    assert row["released_date"] == date(2026, 4, 10)
    assert hist[hist["Code"] == "99920"].iloc[0]["kind"] == J.KIND_LIQUIDATION


def test_supervision_merge_marks_current_and_fills_from_current_page():
    cur = J.parse_supervision_current(SUPERVISION_CURRENT_HTML)
    assert cur["kind"].tolist() == [J.KIND_CONFIRMING, J.KIND_UNDER_REVIEW]
    assert cur["Code"].tolist() == ["99910", "999A0"]
    merged = J.merge_supervision(cur, J.parse_supervision_history(SUPERVISION_HISTORY_HTML))
    by_code = merged.set_index("Code")
    assert len(merged) == 4  # 履歴 3 行 + 仅现行页有的 999A
    assert by_code.loc["99910", "is_current"] and by_code.loc["999A0", "is_current"]
    assert not by_code.loc["99920", "is_current"] and not by_code.loc["99930", "is_current"]
    assert by_code.loc["99910", "note"] == "超過計画開示会社"   # 备考、详细链接来自现行页
    assert by_code.loc["99910", "detail_url"].endswith("/listing/x/a.html")
    assert by_code.loc["99910", "content"] == "監理銘柄（確認中）指定"
    assert pd.isna(by_code.loc["99920", "note"])


# ---------------------------------------------------------------------------------------------
# 改善期間該当銘柄（矩阵 → 长表）
# ---------------------------------------------------------------------------------------------
IMPROVEMENT_HEADER = ["コード", "銘柄名", "市場区分", "株主数", "流通株式数", "流通株式比率", None, "流通株式時価総額", None,
                      "売買代金", "売買高", "時価総額", None, "純資産の額", "備考"]


def improvement_xlsx(title: str = "改善期間該当銘柄一覧（2026年11月5日公表）") -> bytes:
    blank = [None] * 15
    def row(code, name, market, **cells):  # cells: 列号 → 值
        r = [code, name, market] + [None] * 12
        for k, v in cells.items():
            r[int(k[1:])] = v
        return r
    return xlsx({"改善期間一覧": [
        [title] + blank[1:], blank, ["【掲載対象】", "（説明文）"] + blank[2:], ["※1:（注記）"] + blank[1:], blank,
        IMPROVEMENT_HEADER,
        row(9991, "株式会社架空商事", "スタンダード", c7=datetime(2027, 5, 31), c11=datetime(2027, 3, 31), c14="（備考）"),
        row("999A", "株式会社試験製作所", "グロース", c11=datetime(2027, 4, 30), c3=""),
        row(9993, "株式会社第二列電機", "プライム", c6=datetime(2027, 6, 30)),   # 合并表头的第二格
        row(9994, "株式会社文字列工業", "スタンダード", c13="審査中"),
        row(9995, "株式会社空欄物産", "スタンダード"),
        ["※ 表外注記"] + blank[1:],
    ]})


def test_parse_improvement_period_long_form():
    df = J.parse_improvement_period(improvement_xlsx(), "Companies_ImprovementPeriod_JP_261105.xlsx")
    assert set(df["published_date"]) == {date(2026, 11, 5)}
    assert df["Code"].nunique() == 5 and set(df["code4"]) == {"9991", "999A", "9993", "9994", "9995"}
    r = df[df["Code"] == "99910"].set_index("criterion")
    assert r.loc["流通株式時価総額", "period_end_date"] == date(2027, 5, 31)
    assert r.loc["時価総額", "period_end_date"] == date(2027, 3, 31)
    assert (r["note"] == "（備考）").all()
    assert df[df["Code"] == "999A0"]["criterion"].tolist() == ["時価総額"]  # 空字符串单元格不算
    assert df[df["Code"] == "99930"]["criterion"].tolist() == ["流通株式比率"]
    t = df[df["Code"] == "99940"].iloc[0]
    assert t["criterion"] == "純資産の額" and t["period_end_date"] is None and t["period_end_text"] == "審査中"
    assert df[df["Code"] == "99950"]["criterion"].tolist() == [J.CRITERION_UNKNOWN]  # 不能漏掉銘柄


def test_parse_improvement_period_date_from_filename_or_error():
    df = J.parse_improvement_period(improvement_xlsx(title="改善期間該当銘柄一覧"), "Companies_ImprovementPeriod_JP_261001.xlsx")
    assert set(df["published_date"]) == {date(2026, 10, 1)}
    with pytest.raises(ValueError):
        J.parse_improvement_period(improvement_xlsx(title="改善期間該当銘柄一覧"), "x.xlsx")
    with pytest.raises(ValueError):
        J.parse_improvement_period(xlsx({"s": [["全く別の表"]]}), "x.xlsx")


def test_find_improvement_link_and_year_pages():
    html = ('<table><tr><th>改善期間該当銘柄一覧</th><td><a href="/listing/market-alerts/improvement-period/abc-att/'
            'Companies_ImprovementPeriod_JP_261105.xlsx"><img/></a></td></tr></table>')
    assert J.find_improvement_link(html).endswith("abc-att/Companies_ImprovementPeriod_JP_261105.xlsx")
    assert J.find_improvement_link("<p>なし</p>") is None
    pages = J.find_year_pages(SUPERVISION_HISTORY_HTML)
    assert pages == {"2026": J.BASE + "/listing/market-alerts/supervision/01.html",
                     "2025": J.BASE + "/listing/market-alerts/supervision/01-archives-01.html"}


# ---------------------------------------------------------------------------------------------
# 其余解析器（合成文件，结构仿真实格式）
# ---------------------------------------------------------------------------------------------
def listed_xlsx() -> bytes:
    return xlsx({"Sheet1": [
        ["日付", "コード", "銘柄名", "市場・商品区分", "33業種コード", "33業種区分", "17業種コード", "17業種区分", "規模コード", "規模区分"],
        [20261031, 9991, "架空商事", "プライム（内国株式）", 6050, "卸売業", 13, "商社・卸売", 7, "TOPIX Small 2"],
        [20261031, "999A", "試験製作所", "グロース（内国株式）", 5250, "情報・通信業", 10, "情報通信・サービスその他", "-", "-"],
        [20261031, 9997, "架空ＥＴＦ", "ETF・ETN", "-", "-", "-", "-", "-", "-"],
    ]})


def topix_csv() -> bytes:
    return ("日付,銘柄名,コード,業種,TOPIXに占める個別銘柄のウエイト,ニューインデックス区分\n"
            "20260930,架空商事,9991,卸売業,0.0047%,TOPIX Small 2\n"
            "20260930,試験製作所,999A,情報・通信業,1.2500%,TOPIX Core30\n"
            "・当社は、本資料の利用について責任を負いません。,,,,,\n").encode("cp932")


def perpbr_xlsx(ym: str = "2026/10") -> bytes:
    conso_header = ["年月", "市場区分名", "Section", "種別", "Industry", "会社数", "単純＿PER（倍）", "単純＿PBR（倍）",
                    "単純＿1株当たり\n当期純利益（円）", "単純＿1株当たり\n純資産（円）", "加重＿PER（倍）", "加重＿PBR（倍）",
                    "加重＿親会社株主に帰属する\n当期純利益（円）", "加重＿純資産合計（円）"]
    solo_header = ["年月", "市場区分名", "Section", "会社数", "単純＿PER（倍）", "単純＿PBR（倍）", "単純＿1株当たり\n当期純利益（円）",
                   "単純＿1株当たり\n純資産（円）", "加重＿PER（倍）", "加重＿PBR（倍）", "加重＿当期純利益（円）", "加重＿純資産（円）"]
    eng = ["Year/Month"] + [None] * 11
    return xlsx({
        "規模別・業種別（連結）": [
            ["規模別・業種別　PER・PBR（連結）", None, None, None, None, "(注)1.（注記）"], ["Average PER and PBR"],
            conso_header, eng,
            [ym, "プライム市場", "Prime", "総合", "Composite", "=100", "=11.1", "=1.1", "=10.5", "=900.5", "=12.1", "=1.2", "=5000", "=6000"],
            [ym, "プライム市場", "Prime", "総合（金融業を除く）", "Composite excl. Financials", "=90", "=11.2", "=1.2", "=10", "=800", "=12", "=1.3", "=4000", "=5000"],
            [ym, "プライム市場", "Prime", "1 水産・農林業", "1 Fishery", "=5", '="－"', "=0.8", "=-3", "=700", '="＊"', "=0.9", "=-10", "=20"],
            [ym, "－", "－", "大型株", "Large", "=50", "=15", "=2", "=100", "=1500", "=16", "=2.1", "=900", "=1000"],
        ],
        "市場別（単体）": [["Average PER and PBR"], [None], solo_header, eng,
                           [ym, "プライム市場", "Prime", "=80", "=13", "=1.4", "=9", "=500", "=14", "=1.5", "=300", "=400"]],
    })


def capital_cost_xlsx() -> bytes:
    def sheet(as_of):
        return [[None], [None, "「資本コストや株価を意識した経営」（説明）"], [None],
                [None] * 13 + [as_of, "時点"], [None], [None, "※１（注記）"], [None],
                [None, "業種\nコード", "業種", "市場区分", "証券\nコード", "銘柄名", "開示状況", None, "開示内容の\nアップデート日", "開示内容 ※１",
                 "機関投資家からのより活発なコンタクトを希望する企業", None, None, "英文開示"],
                [None, None, None, None, None, None, "要請に基づく\n開示状況", "前月からの\n開示状況の変更", None, None, "申請状況", "掲載開始日", "コンタクト先", None],
                [None, 50, "水産・農林業", "プライム", 9991, "架空商事", "開示済", "-", datetime(2026, 6, 30), "（表題）", "○", datetime(2025, 1, 15), "IR部", "有"],
                [None, 5250, "情報・通信業", "スタンダード", "999A", "試験製作所", "検討中", "変更あり", "-", "-", "-", "-", "-", "-"]]
    return xlsx({"本資料の目的": [["（説明）"]], "開示企業一覧（2026年9月末時点）  ": sheet(datetime(2026, 9, 30)),
                 "【過去分】開示企業一覧（2026年8月末時点）  ": sheet(datetime(2026, 8, 31))})


KESSAN_HEADER = ["決算発表予定日\nSchedule", "コード\nCode", "会社名", "Issue Name", "決算期末\nFiscal Year", "業種名", "Industry", "種別",
                 "Fiscal Year/Quarter", "市場区分", "Market Segment"]


def kessan_xlsx(as_of_line: str | None, rows: list[list[Any]]) -> bytes:
    return xlsx({"Sheet1": [["９月に四半期末又は期末を迎えた決算発表予定日一覧"], ["List of companies"], [as_of_line], [None], KESSAN_HEADER, *rows]})


def kessan_row(sched: Any, code: Any, quarter: str = "第２四半期") -> list[Any]:
    return [sched, code, "架空商事", "KAKU SHOJI", datetime(2027, 3, 31), "卸売業", "Wholesale Trade", quarter, "Second quarter", "プライム", "Prime"]


def price_limit_csv(day: str = "2026/11/06") -> bytes:
    return (f"日付(date),銘柄コード(code),ストップ情報種別(type of closing price),ストップ配分連続日(consecutive business date)\n"
            f"{day},9991,2,1\n{day},999A,1,3\n").encode("cp932")


def price_limit_etf_csv(day: str = "2026/11/06") -> bytes:
    return (f"日付(date),銘柄コード(code),ストップ情報種別(type of closing price)\n{day},9997,1\n").encode("cp932")


DELISTED_HTML = """<select class="backnumber" size="1"><option selected="selected" value="/listing/stocks/delisted/index.html">2026年</option>
<option value="/listing/stocks/delisted/archives-01.html">2025年</option></select>
<table><thead><tr><th>上場廃止日</th><th>銘柄名</th><th>コード</th><th>市場区分</th><th>上場廃止理由</th></tr></thead>
<tr><td>2027/04/01</td><td>（株）架空商事</td><td>9991</td><td>スタンダード</td><td>上場維持基準への不適合</td></tr>
<tr><td>2026/06/01</td><td>（株）試験製作所</td><td>999A</td><td>グロース</td><td>株式の併合</td></tr></table>"""

DELISTED_2025_HTML = """<table><thead><tr><th>上場廃止日</th><th>銘柄名</th><th>コード</th><th>市場区分</th><th>上場廃止理由</th></tr></thead>
<tr><td>2025/03/03</td><td>（株）過去工業</td><td>9981</td><td>プライム</td><td>他社による買収</td></tr></table>"""

NEW_LISTINGS_HTML = """<select class="backnumber" size="1"><option selected="selected" value="/listing/stocks/new/index.html">2026年</option></select>
<table><thead><tr><th rowspan="2">上場日 （上場承認日）</th><th rowspan="2">会社名</th><th>コード</th><th>会社概要</th><th>確認書</th><th>仮条件（円）</th><th>公募（千株）</th><th>売買 単位</th></tr>
<tr><th>市場区分</th><th>Iの部</th><th>CG 報告書</th><th>公募・売出価格 （円）</th><th>売出（千株）</th><th>決算 短信</th></tr></thead>
<tr><td rowspan="2">2026/12/09 （2026/11/06）</td><td rowspan="2"><a href="https://example.invalid/">架空商事（株）</a></td><td>999A</td>
<td><a href="/listing/stocks/new/o.pdf"><img/></a></td><td><a href="/listing/stocks/new/c.pdf"><img/></a></td><td>-</td><td>5,000.5</td><td>100</td></tr>
<tr><td>グロース</td><td><a href="/listing/stocks/new/p.pdf"><img/></a></td><td><a href="/listing/stocks/new/g.pdf"><img/></a></td><td>1,200</td><td>1,000.5(OA200.5)</td><td>-</td></tr></table>"""


def test_parse_listed_monthly():
    df = J.parse_listed_monthly(listed_xlsx())
    assert df["Code"].tolist() == ["99910", "999A0", "99970"]
    assert df["code4"].tolist() == ["9991", "999A", "9997"]
    assert set(df["date"]) == {date(2026, 10, 31)}
    assert df.loc[0, "s33"] == "6050" and df.loc[0, "scale_code"] == "7"
    assert pd.isna(df.loc[2, "s33"]) and pd.isna(df.loc[1, "scale_name"])


def test_parse_topix_weights_ignores_footer():
    df = J.parse_topix_weights(topix_csv())
    assert df["Code"].tolist() == ["99910", "999A0"]
    assert df.loc[0, "weight"] == pytest.approx(0.000047) and df.loc[1, "weight"] == pytest.approx(0.0125)
    assert df.loc[1, "new_index_category"] == "TOPIX Core30" and df.loc[0, "date"] == date(2026, 9, 30)


def test_parse_per_pbr_long_form():
    df = J.parse_per_pbr(perpbr_xlsx())
    assert set(df["month"]) == {"2026-10"} and len(df) == 5
    conso = df[df["basis"] == "連結"].set_index("segment")
    assert conso.loc["市場区分", "category"] == "プライム市場" and conso.loc["市場区分", "per_simple"] == 11.1
    assert conso.loc["業種集計", "category"] == "総合（金融業を除く）"
    sector = conso.loc["業種"]
    assert sector["category"] == "水産・農林業" and sector["sector_no"] == 1 and sector["category_en"] == "Fishery"
    assert pd.isna(sector["per_simple"]) and pd.isna(sector["per_weighted"]) and sector["pbr_simple"] == 0.8  # "－" / "＊" → NaN
    assert pd.isna(conso.loc["規模区分", "market"]) and conso.loc["規模区分", "category"] == "大型株"
    solo = df[df["basis"] == "単体"].iloc[0]
    assert solo["n_companies"] == 80 and solo["net_income_sum"] == 300 and solo["segment"] == "市場区分"


def test_parse_capital_cost_multi_sheet():
    df = J.parse_capital_cost(capital_cost_xlsx())
    assert sorted(df["as_of_month"].unique()) == ["2026-08", "2026-09"] and len(df) == 4
    a = df[(df["as_of_month"] == "2026-09") & (df["Code"] == "99910")].iloc[0]
    assert a["as_of_date"] == date(2026, 9, 30) and a["s33_code"] == "0050" and a["disclosure_status"] == "開示済"
    assert a["update_date"] == date(2026, 6, 30) and a["contact_listed_since"] == date(2025, 1, 15)
    assert a["contact_wanted"] and a["english_disclosure"] and a["contact_dept"] == "IR部"
    b = df[(df["as_of_month"] == "2026-09") & (df["Code"] == "999A0")].iloc[0]
    assert not b["contact_wanted"] and not b["english_disclosure"] and b["update_date"] is None
    assert b["change_from_prev_month"] == "変更あり"


def test_parse_earnings_schedule():
    content = kessan_xlsx("2026年11月5日 現在", [kessan_row(datetime(2026, 11, 9), 9991), kessan_row("未定_Undecided", "999A")])
    df = J.parse_earnings_schedule(content, "kessan09_1106.xlsx", date(2026, 11, 7))
    assert df["sched_status"].tolist() == ["scheduled", "undecided"]
    assert df.loc[0, "sched_date"] == date(2026, 11, 9) and df.loc[1, "sched_date"] is None
    assert set(df["as_of_date"]) == {date(2026, 11, 5)} and set(df["period_end_month"]) == {9.0}
    assert df.loc[0, "fiscal_year_end"] == date(2027, 3, 31) and df.loc[0, "quarter"] == "第２四半期"
    no_as_of = J.parse_earnings_schedule(kessan_xlsx(None, [kessan_row(datetime(2026, 11, 9), 9991)]), "kessan.xlsx", date(2026, 11, 7))
    assert set(no_as_of["as_of_date"]) == {date(2026, 11, 7)}  # kessan.xlsx 无「現在」→ 抓取日
    with pytest.raises(ValueError):
        J.parse_earnings_schedule(xlsx({"s": [["別の表"]]}))


def test_parse_price_limit():
    df = J.parse_price_limit(price_limit_csv())
    assert df["Code"].tolist() == ["99910", "999A0"] and df["stop_type_name"].tolist() == ["ストップ安", "ストップ高"]
    assert df["consecutive_days"].tolist() == [1.0, 3.0] and df.loc[0, "date"] == date(2026, 11, 6)
    etf = J.parse_price_limit(price_limit_etf_csv(), list_kind="etf")
    assert etf.loc[0, "list_kind"] == "etf" and pd.isna(etf.loc[0, "consecutive_days"])
    assert J.parse_price_limit("日付(date),銘柄コード(code),ストップ情報種別(type)\n".encode("cp932")).empty


def test_parse_delisted_and_new_listings():
    d = J.parse_delisted(DELISTED_HTML)
    assert d["Code"].tolist() == ["99910", "999A0"] and d.loc[0, "delist_date"] == date(2027, 4, 1)
    assert d.loc[1, "reason"] == "株式の併合"
    n = J.parse_new_listings(NEW_LISTINGS_HTML).iloc[0]
    assert n["Code"] == "999A0" and n["code4"] == "999A" and n["market"] == "グロース"
    assert n["listing_date"] == date(2026, 12, 9) and n["approval_date"] == date(2026, 11, 6)
    assert n["public_kshares"] == 5000.5 and n["secondary_kshares"] == 1000.5 and n["oa_kshares"] == 200.5
    assert n["offer_price"] == 1200 and n["trading_unit"] == 100 and pd.isna(n["price_range"])
    assert n["outline_url"].endswith("/o.pdf") and n["cg_report_url"].endswith("/g.pdf") and pd.isna(n["kessan_url"])


def test_find_links():
    html = """<a href="/m/a-att/perpbr202610.xlsx"></a><a href="/m/b-att/perpbr202509.xlsx"></a><a href="/m/x.html"></a>
    <a href="/f/k-att/kessan.xlsx"></a><a href="/f/k-att/kessan09_1106.xlsx"></a>
    <a href="/t/s-att/1106.csv"></a><a href="/t/s-att/1106_ETF.csv"></a><a href="/t/s-att/other.csv"></a>"""
    assert J.find_perpbr_links(html) == {"2026-10": J.BASE + "/m/a-att/perpbr202610.xlsx", "2025-09": J.BASE + "/m/b-att/perpbr202509.xlsx"}
    assert [u.rsplit("/", 1)[-1] for u in J.find_kessan_links(html)] == ["kessan.xlsx", "kessan09_1106.xlsx"]
    assert [u.rsplit("/", 1)[-1] for u in J.find_price_limit_links(html)] == ["1106.csv", "1106_ETF.csv"]
    assert J.norm_code(9991) == "99910" and J.norm_code("999A") == "999A0" and J.norm_code("99910") == "99910" and J.norm_code("-") is None


# ---------------------------------------------------------------------------------------------
# 作业（注入 fetcher，全程不联网）
# ---------------------------------------------------------------------------------------------
K_ATT = J.BASE + "/listing/event-schedules/financial-announcement/k-att/"
PL_ATT = J.BASE + "/equities/trading/domestic/p-att/"
IMP_FILE = J.BASE + "/listing/market-alerts/improvement-period/i-att/Companies_ImprovementPeriod_JP_261105.xlsx"
PERPBR_A = J.BASE + "/markets/statistics-equities/misc/a-att/perpbr202609.xlsx"
PERPBR_B = J.BASE + "/markets/statistics-equities/misc/b-att/perpbr202610.xlsx"
DELISTED_ARCHIVE = J.BASE + "/listing/stocks/delisted/archives-01.html"
TODAY = date(2026, 11, 7)

SUPERVISION_HISTORY_2025_HTML = """<table><thead><tr><th>指定年月日</th><th>銘柄名</th><th>コード</th><th>市場区分 （注）</th><th>解除年月日</th><th>内容</th><th>備考</th></tr></thead>
<tr><td>2025/12/01</td><td>（株）長期監理</td><td>9982</td><td>スタンダード</td><td>-</td><td>監理銘柄（審査中）指定</td><td></td></tr></table>"""


def site() -> dict[str, Any]:
    """虚构站点：url → bytes 或 (status, None)。"""
    return {
        J.URLS["listed_monthly"]: listed_xlsx(),
        J.URLS["topix_weights"]: topix_csv(),
        J.URLS["per_pbr_page"]: f'<a href="{PERPBR_A}"></a><a href="{PERPBR_B}"></a>'.encode(),
        PERPBR_A: perpbr_xlsx("2026/09"),
        PERPBR_B: perpbr_xlsx("2026/10"),
        J.URLS["capital_cost_disclosure"]: capital_cost_xlsx(),
        J.URLS["earnings_page"]: f'<a href="{K_ATT}kessan.xlsx"></a><a href="{K_ATT}kessan09_1106.xlsx"></a>'.encode(),
        K_ATT + "kessan.xlsx": kessan_xlsx(None, [kessan_row(datetime(2026, 11, 9), 9991)]),
        K_ATT + "kessan09_1106.xlsx": kessan_xlsx("2026年11月5日 現在", [kessan_row(datetime(2026, 11, 9), 9991),
                                                                     kessan_row(datetime(2026, 12, 1), 9992),
                                                                     kessan_row("未定_Undecided", "999A")]),
        J.URLS["price_limit_page"]: f'<a href="{PL_ATT}1106.csv"></a><a href="{PL_ATT}1106_ETF.csv"></a>'.encode(),
        PL_ATT + "1106.csv": price_limit_csv(),
        PL_ATT + "1106_ETF.csv": price_limit_etf_csv(),
        J.URLS["delisted"]: DELISTED_HTML.encode(),
        DELISTED_ARCHIVE: DELISTED_2025_HTML.encode(),
        J.URLS["supervision_current"]: SUPERVISION_CURRENT_HTML.encode(),
        J.URLS["supervision_history"]: SUPERVISION_HISTORY_HTML.encode(),
        J.BASE + "/listing/market-alerts/supervision/01-archives-01.html": SUPERVISION_HISTORY_2025_HTML.encode(),
        J.URLS["new_listings"]: NEW_LISTINGS_HTML.encode(),
        J.URLS["improvement_page"]: f'<a href="{IMP_FILE}"><img/></a>'.encode(),
        IMP_FILE: improvement_xlsx(),
    }


class Env:
    def __init__(self, tmp_path):
        self.settings = Settings(root=tmp_path, data_dir=tmp_path / "data", params_path=tmp_path / "p.yaml", jquants_api_key=None,
                                 anthropic_api_key=None, edinet_api_key=None, fred_api_key=None)
        self.paths = DataPaths(tmp_path / "data").ensure()
        self.ledger = Ledger(self.paths.ledger_sqlite)
        self.store = Store(self.paths)
        self.site = site()
        self.requests: list[str] = []

    def fetcher(self, url: str):
        self.requests.append(url)
        return self.site.get(url, (404, None))

    def run(self, datasets=None, month=None, today=TODAY) -> dict[str, Any]:
        self.requests.clear()
        client = J.JPXFiles(self.paths, self.ledger, fetcher=self.fetcher)
        return ingest_jpx.run(self.settings, self.paths, self.ledger, self.store, datasets=datasets, month=month, client=client, today=today)

    def table(self, ds: str) -> pd.DataFrame:
        return self.store.read("staging", J.TABLES[ds].table)


@pytest.fixture
def env(tmp_path) -> Env:
    return Env(tmp_path)


def test_job_all_datasets_written(env):
    res = env.run()
    assert res["status"] == "ok"
    ds = res["datasets"]
    assert set(ds) == set(J.DATASETS) and all(v["status"] == "ok" and not v["errors"] for v in ds.values())
    expected = {"listed_monthly": 3, "topix_weights": 2, "per_pbr_monthly": 10, "capital_cost_disclosure": 4, "earnings_schedule": 3,
                "price_limit_expansion": 3, "delisted": 3, "supervision": 5, "new_listings": 1, "improvement_period": 6}
    for name, n in expected.items():
        assert ds[name]["rows_table"] == n, name
        assert ds[name]["table"] == J.TABLES[name].table
        assert len(env.table(name)) == n
        assert not env.table(name).duplicated(subset=J.TABLES[name].keys).any()
    assert ds["listed_monthly"]["data_date"] == "2026-10-31" and ds["improvement_period"]["data_date"] == "2026-11-05"
    assert ds["per_pbr_monthly"]["data_date"] == "2026-10" and ds["capital_cost_disclosure"]["data_date"] == "2026-09"
    assert env.store.partitions("staging", "jpx_listed_monthly") == ["2026-10"]
    assert env.store.partitions("staging", "jpx_earnings_schedule") == ["2026-11", "2026-12", "undecided"]
    assert env.store.partitions("staging", "jpx_improvement_period") == ["all"]
    # 日期列落盘为 date32（Python date 读回）
    raw = pd.read_parquet(env.store.table_dir("staging", "jpx_improvement_period") / "all.parquet")
    assert isinstance(raw["published_date"].iloc[0], date)
    assert env.ledger.get_run(res["run_id"])["status"] == "ok"


def test_job_saves_raw_and_marks_ledger(env):
    env.run(["improvement_period"])
    root = env.paths.raw_dir("jpx", "improvement_period")
    files = sorted(p.name for p in root.rglob("*") if p.is_file())
    assert len(files) == 2 and any(f.startswith("Companies_ImprovementPeriod_JP_261105_20261107_") and f.endswith(".xlsx") for f in files)
    assert (root / "202611").is_dir()
    assert env.ledger.status_of("jpx.improvement_period", "2026-11-05") == "ok"
    with env.ledger.conn() as c:
        n = c.execute("SELECT COUNT(*) FROM raw_files WHERE source='jpx' AND dataset='improvement_period'").fetchone()[0]
    assert n == 2


def test_job_idempotent_and_skips_finished_history(env):
    first = env.run()
    assert PERPBR_A in env.requests and DELISTED_ARCHIVE in env.requests
    raw_count = len([p for p in env.paths.raw.rglob("*") if p.is_file()])
    second = env.run()
    for name, r in second["datasets"].items():
        assert r["status"] == "ok" and r["rows_table"] == first["datasets"][name]["rows_table"], name
    assert PERPBR_A not in env.requests and DELISTED_ARCHIVE not in env.requests   # 已完成的月份 / 存档年不再抓
    assert J.URLS["delisted"] in env.requests                                      # 当年页每次都抓
    assert second["datasets"]["per_pbr_monthly"]["files_skipped"] == 2
    assert second["datasets"]["delisted"]["files_skipped"] == 1
    assert len([p for p in env.paths.raw.rglob("*") if p.is_file()]) == raw_count  # 同内容不重复落盘


def test_job_http_errors_are_recorded_and_others_continue(env):
    env.site[J.URLS["listed_monthly"]] = (403, None)
    env.site[K_ATT + "kessan09_1106.xlsx"] = (404, None)
    del env.site[J.URLS["improvement_page"]]
    res = env.run()
    assert res["status"] == "partial"
    ds = res["datasets"]
    assert ds["listed_monthly"]["status"] == "error" and "403" in ds["listed_monthly"]["errors"][0]
    assert ds["listed_monthly"]["rows_table"] == 0 and ds["listed_monthly"]["data_date"] is None
    assert ds["earnings_schedule"]["status"] == "partial" and ds["earnings_schedule"]["rows_table"] == 1
    assert ds["improvement_period"]["status"] == "error"
    assert ds["topix_weights"]["status"] == "ok" and ds["delisted"]["status"] == "ok"
    assert env.ledger.status_of("jpx.listed_monthly", "data_j") == "error"
    assert env.ledger.status_of("jpx.improvement_period", "page") == "error"
    assert env.ledger.get_run(res["run_id"])["status"] == "partial"
    # 恢复后重跑可补齐
    env.site[J.URLS["listed_monthly"]] = listed_xlsx()
    assert env.run(["listed_monthly"])["datasets"]["listed_monthly"]["rows_table"] == 3


def test_job_quarantines_unparsable_files_and_retries(env):
    env.site[J.URLS["topix_weights"]] = "全く別の内容\n".encode("cp932")   # 解析出 0 行
    env.site[PERPBR_A] = b"<html>not xlsx</html>"                          # 200 但不是 xlsx
    res = env.run(["topix_weights", "per_pbr_monthly"])
    assert res["datasets"]["topix_weights"]["status"] == "error"
    pp = res["datasets"]["per_pbr_monthly"]
    assert pp["status"] == "partial" and pp["data_date"] == "2026-10"
    assert env.ledger.status_of("jpx.per_pbr_monthly", "2026-09") == "error"
    assert list((env.paths.quarantine / "jpx" / "per_pbr_monthly").glob("2026-09_*.bin"))
    # 出错的月份下次重试，已完成的不重抓
    env.site[PERPBR_A] = perpbr_xlsx("2026/09")
    again = env.run(["per_pbr_monthly"])
    assert PERPBR_A in env.requests and PERPBR_B not in env.requests
    assert again["datasets"]["per_pbr_monthly"]["rows_table"] == 10


def test_job_page_without_links_is_reported(env):
    env.site[J.URLS["improvement_page"]] = "<p>JS でのみ描画</p>".encode()
    env.site[J.URLS["per_pbr_page"]] = b"<p></p>"
    res = env.run(["improvement_period", "per_pbr_monthly"])
    assert "xlsx" in res["datasets"]["improvement_period"]["errors"][0]
    assert res["datasets"]["per_pbr_monthly"]["status"] == "error"


def test_job_month_argument(env):
    res = env.run(["per_pbr_monthly", "capital_cost_disclosure"], month="2026-09")
    assert res["month"] == "2026-09"
    assert set(env.table("per_pbr_monthly")["month"]) == {"2026-09"} and PERPBR_B not in env.requests
    assert set(env.table("capital_cost_disclosure")["as_of_month"]) == {"2026-09"}
    env.run(["per_pbr_monthly"], month="2026-09")   # 指定月强制重取
    assert PERPBR_A in env.requests
    assert env.run(["per_pbr_monthly"], month="2019-01")["datasets"]["per_pbr_monthly"]["status"] == "error"


def test_job_supervision_current_flag_recomputed(env):
    env.run(["supervision"])
    t = env.table("supervision").set_index("Code")
    assert t.loc["99910", "is_current"] and t.loc["999A0", "is_current"]
    assert not t.loc["99820", "is_current"] and not t.loc["99920", "is_current"]
    assert t.loc["99820", "kind"] == J.KIND_UNDER_REVIEW  # 存档页
    # 9991 从现行页消失，2025 年存档行（不再抓取）进入现行页 → 全表重算
    env.site[J.URLS["supervision_current"]] = """<h3>監理銘柄（審査中）</h3>
<table><tr><th>指定年月日</th><th>銘柄名</th><th>コード</th><th>市場区分 （注）</th><th>詳細</th><th>備考</th></tr>
<tr><td>2025/12/01</td><td>（株）長期監理</td><td>9982</td><td>スタンダード</td><td></td><td></td></tr>
<tr><td>2026/09/15</td><td>（株）試験製作所</td><td>999A</td><td>グロース</td><td></td><td></td></tr></table>""".encode()
    env.run(["supervision"])
    t = env.table("supervision").set_index("Code")
    assert not t.loc["99910", "is_current"] and t.loc["99820", "is_current"] and t.loc["999A0", "is_current"]
    # 现行页取失败：保留上次的状态，履歴照常写入
    env.site[J.URLS["supervision_current"]] = (403, None)
    res = env.run(["supervision"])
    assert res["datasets"]["supervision"]["status"] == "partial"
    t = env.table("supervision").set_index("Code")
    assert t.loc["99820", "is_current"] and not t.loc["99910", "is_current"]


def test_job_earnings_latest_as_of_wins_and_snapshots_accumulate(env):
    env.run(["earnings_schedule"])
    t = env.table("earnings_schedule").set_index("Code")
    assert t.loc["99910", "source_file"] == "kessan.xlsx"   # 同键保留 as_of 较新（抓取日 11/07 > 表内 11/05）
    assert t.loc["999A0", "sched_status"] == "undecided"
    # 决定日期后：undecided 旧行仍在（下游取最新 as_of），新日期入对应分区
    env.site[K_ATT + "kessan09_1106.xlsx"] = kessan_xlsx("2026年11月6日 現在", [kessan_row(datetime(2026, 12, 15), "999A")])
    env.run(["earnings_schedule"], today=date(2026, 11, 8))
    t = env.table("earnings_schedule")
    rows = t[t["Code"] == "999A0"]
    assert len(rows) == 2 and rows.sort_values("as_of_date").iloc[-1]["sched_status"] == "scheduled"


def test_job_improvement_period_keeps_each_publication(env):
    env.run(["improvement_period"])
    env.site[IMP_FILE] = improvement_xlsx(title="改善期間該当銘柄一覧（2026年12月1日公表）")
    env.run(["improvement_period"], today=date(2026, 12, 2))
    t = env.table("improvement_period")
    assert t["published_date"].nunique() == 2 and len(t) == 12


def test_job_rejects_unknown_dataset(env):
    with pytest.raises(ValueError):
        env.run(["nope"])


# ---------------------------------------------------------------------------------------------
# 抓取层（假 session，不联网）
# ---------------------------------------------------------------------------------------------
class FakeSession:
    def __init__(self, status: int = 200):
        self.status, self.calls, self.headers = status, [], {}

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(url)
        return SimpleNamespace(status_code=self.status, content=b"body")


def test_fetch_status_handling_and_rate_limit(env, monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(J.time, "sleep", sleeps.append)
    client = J.JPXFiles(env.paths, env.ledger, session=FakeSession(200))
    assert client.fetch("https://www.jpx.co.jp/a") == b"body" and client.fetch("https://www.jpx.co.jp/b") == b"body"
    assert len(sleeps) == 1 and 1.0 <= sleeps[0] <= J.MIN_INTERVAL   # 相邻请求间隔 ≥ 1 秒
    for status in (403, 404):
        with pytest.raises(J.JPXFetchError) as ei:
            J.JPXFiles(env.paths, env.ledger, session=FakeSession(status)).fetch("https://www.jpx.co.jp/c")
        assert ei.value.status == status


def test_default_client_uses_browser_user_agent(env):
    client = J.JPXFiles(env.paths, env.ledger)
    assert client.user_agent.startswith("Mozilla/5.0") and client.min_interval >= 1.0
