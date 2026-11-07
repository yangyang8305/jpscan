"""JPX 连接器与作业的测试：全部使用合成数据（虚构公司与数字，格式仿真实文件），不联网。"""
from __future__ import annotations

from datetime import date

import pandas as pd

from jpscan.connectors import jpx_files as J


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
