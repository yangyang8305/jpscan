"""JPX 连接器与作业的测试：全部使用合成数据（虚构公司与数字，格式仿真实文件），不联网。"""
from __future__ import annotations

import io
from datetime import date, datetime
from typing import Any

import openpyxl
import pandas as pd
import pytest

from jpscan.connectors import jpx_files as J


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
