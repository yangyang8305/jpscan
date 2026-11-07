"""宏观序列连接器测试（不联网；数值全部虚构，格式与真实来源相同）。"""
from datetime import date

import pytest

from jpscan.connectors.macro import MacroParseError, parse_jgb_csv, parse_wareki


@pytest.mark.parametrize("text,expected", [
    ("R1.5.1", date(2019, 5, 1)),
    ("H31.4.30", date(2019, 4, 30)),
    ("S64.1.7", date(1989, 1, 7)),
    ("R6.8.15", date(2024, 8, 15)),
    ("H1.1.9", date(1989, 1, 9)),
    ("S49.9.24", date(1974, 9, 24)),
    ("R元.5.7", date(2019, 5, 7)),
    ("Ｒ８．１１．５", date(2026, 11, 5)),
    (" R8.11.5 ", date(2026, 11, 5)),
])
def test_parse_wareki(text, expected):
    assert parse_wareki(text) == expected


@pytest.mark.parametrize("text", ["", "基準日", "2026-11-05", "T5.1.1", "R8.13.1", "R8.2.30", "※注記"])
def test_parse_wareki_rejects_non_dates(text):
    assert parse_wareki(text) is None


JGB_CSV = (
    "国債金利情報 (令和8年11月),,,,,,,,,,,,,,,(単位 : %)\n"
    "基準日,1年,2年,3年,4年,5年,6年,7年,8年,9年,10年,15年,20年,25年,30年,40年\n"
    "R8.11.2,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.5,2.0,2.5,3.0,4.0\n"
    "S49.9.24,10.327,9.362,8.83,8.515,8.348,8.29,8.24,8.2,8.17,8.15,-,-,-,-,-\n"
    "H19.11.5,-,0.7,,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.8,2.1,2.2,2.3,2.4\n"
    ",,,,,,,,,,,,,,,\n"
    "※最新のcsvデータをダウンロードできない場合、ご利用のブラウザにおいてキャッシュの削除を実施し再度ダウンロードください。,,,,,,,,,,,,,,,\n"
).encode("cp932")


def test_parse_jgb_csv_long_format_and_missing_values():
    df = parse_jgb_csv(JGB_CSV)
    assert set(df["series"]) == {"jgb_1y", "jgb_2y", "jgb_5y", "jgb_10y", "jgb_20y", "jgb_30y", "jgb_40y"}
    assert list(df.columns) == ["series", "date", "value", "source", "unit"]
    assert (df["source"] == "mof").all() and (df["unit"] == "%").all()

    def val(series, d):
        return df[(df["series"] == series) & (df["date"] == d)]["value"].tolist()
    assert val("jgb_10y", date(2026, 11, 2)) == [1.0]
    assert val("jgb_40y", date(2026, 11, 2)) == [4.0]
    assert val("jgb_10y", date(1974, 9, 24)) == [8.15]
    assert val("jgb_20y", date(1974, 9, 24)) == []        # "-" 是缺失
    assert val("jgb_1y", date(2007, 11, 5)) == []
    assert val("jgb_2y", date(2007, 11, 5)) == [0.7]
    assert len(df) == 7 + 4 + 6                            # 脚注与空行不出行


def test_parse_jgb_csv_bad_header_raises():
    with pytest.raises(MacroParseError):
        parse_jgb_csv("<html>メンテナンス中</html>".encode("cp932"))
    with pytest.raises(MacroParseError):
        parse_jgb_csv("基準日,1年,2年\nR8.1.5,0.1,0.2\n".encode("cp932"))
    with pytest.raises(MacroParseError):
        parse_jgb_csv("基準日,1年,2年,5年,10年,20年,30年,40年\n".encode("cp932"))
