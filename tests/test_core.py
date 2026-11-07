from datetime import date

import pandas as pd

from jpscan.ledger import Ledger
from jpscan.paths import DataPaths
from jpscan.staging.jquants_norm import effective_date_for, normalize, partition_of
from jpscan.store import Store
from jpscan.util.common import RateLimiter, params_hash


def test_effective_date_rule():
    cal = {date(2026, 8, 4), date(2026, 8, 5), date(2026, 8, 6)}
    assert effective_date_for(date(2026, 8, 4), "14:00:00", cal) == date(2026, 8, 4)   # 盘中披露 → 当日
    assert effective_date_for(date(2026, 8, 4), "15:30:00", cal) == date(2026, 8, 5)   # 收盘后 → 次一交易日
    assert effective_date_for(date(2026, 8, 4), None, cal) == date(2026, 8, 5)         # 时刻缺失 → 次一交易日
    assert effective_date_for(date(2026, 8, 3), "10:00:00", cal) == date(2026, 8, 4)   # 非交易日盘中 → 下一交易日


def test_normalize_fins_summary():
    rows = [{"DiscDate": "2026-08-04", "DiscTime": "14:00:00", "Code": "41890", "DiscNo": "1", "DocType": "2QFinancialStatements_Consolidated_JP",
             "CurPerType": "2Q", "CurPerSt": "2026-01-01", "CurPerEn": "2026-06-30", "CurFYSt": "2026-01-01", "CurFYEn": "2026-12-31",
             "Sales": "53299000000", "OP": "7501000000", "EPS": "150.85", "BPS": "", "ChgAcEst": "false"},
            {"DiscDate": "2026-08-04", "DiscTime": "14:00:00", "Code": "41890", "DiscNo": "1", "DocType": "2QFinancialStatements_Consolidated_JP",
             "CurPerType": "2Q", "CurPerSt": "2026-01-01", "CurPerEn": "2026-06-30", "CurFYSt": "2026-01-01", "CurFYEn": "2026-12-31",
             "Sales": "53299000000", "OP": "7501000000", "EPS": "150.85", "BPS": "", "ChgAcEst": "false"}]
    df = normalize("fins_summary", rows, {date(2026, 8, 4)})
    assert len(df) == 1  # 去重
    assert df.loc[0, "Sales"] == 53299000000.0
    assert pd.isna(df.loc[0, "BPS"])
    assert df.loc[0, "gaap"] == "JP" and df.loc[0, "consolidated"] == "consolidated"
    assert df.loc[0, "fy_months"] == 12
    assert df.loc[0, "effective_date"] == date(2026, 8, 4)
    assert df.loc[0, "code4"] == "4189"
    assert list(partition_of("fins_summary", df)) == ["2026-08"]


def test_store_and_ledger(tmp_path):
    paths = DataPaths(tmp_path / "data").ensure()
    store = Store(paths)
    df = pd.DataFrame({"Date": [date(2026, 8, 1), date(2026, 8, 2)], "Code": ["10000", "10000"], "C": [1.0, 2.0]})
    store.write("staging", "prices_daily", df, "2026-08")
    df2 = pd.DataFrame({"Date": [date(2026, 8, 2)], "Code": ["10000"], "C": [3.0]})
    store.write_merge("staging", "prices_daily", df2, "2026-08", ["Date", "Code"])
    out = store.read("staging", "prices_daily")
    assert len(out) == 2 and out.sort_values("Date")["C"].tolist() == [1.0, 3.0]
    led = Ledger(paths.ledger_sqlite)
    led.mark("bars_daily", "2026-08-01", "ok", 10)
    assert led.status_of("bars_daily", "2026-08-01") == "ok"
    assert "2026-08-01" in led.done_keys("bars_daily")
    rid = led.start_run("test", "abc", "0.1.0")
    led.finish_run(rid, "ok", "deadbeef")
    assert led.get_run(rid)["status"] == "ok"


def test_rate_limiter_and_hash():
    rl = RateLimiter(100)
    for _ in range(3):
        rl.wait()
    assert params_hash({"a": 1}) == params_hash({"a": 1})
    assert params_hash({"a": 1}) != params_hash({"a": 2})
