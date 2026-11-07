"""宏观序列连接器测试（不联网；数值全部虚构，格式与真实来源相同）。"""
from datetime import date

import pytest

from jpscan.connectors.macro import (BOJ_SERIES, MacroParseError, boj_url, parse_boj_csv, parse_fred_json, parse_jgb_csv,
                                     parse_wareki)


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


BOJ_CSV = (
    "STATUS,200\nMESSAGEID,M181000I\nMESSAGE,正常に終了しました。\nDATE,2026-11-07T11:58:10.596+09:00\n"
    "PARAMETER,FORMAT,CSV\nPARAMETER,LANG,\nPARAMETER,DB,FM01\nPARAMETER,STARTDATE,\nPARAMETER,ENDDATE,\nPARAMETER,STARTPOSITION,\n"
    "NEXTPOSITION,\n"
    "SERIES_CODE,NAME_OF_TIME_SERIES_J,UNIT_J,FREQUENCY,CATEGORY_J,LAST_UPDATE,SURVEY_DATES,VALUES\n"
    "STRDCLUCON,架空金利,年％,DAILY,架空,20261106,20261102,1.5\n"
    "STRDCLUCON,架空金利,年％,DAILY,架空,20261106,20261103,null\n"
    "STRDCLUCON,架空金利,年％,DAILY,架空,20261106,20261104,0.25\n"
    "OTHERCODE,別系列,年％,DAILY,架空,20261106,20261104,9.99\n"
).encode("shift_jis")


def test_parse_boj_csv():
    df = parse_boj_csv(BOJ_CSV, "call_rate_on", "STRDCLUCON")
    assert df["date"].tolist() == [date(2026, 11, 2), date(2026, 11, 4)]    # null 不出行
    assert df["value"].tolist() == [1.5, 0.25]
    assert (df["source"] == "boj").all() and (df["unit"] == "年%").all()
    assert (df["series"] == "call_rate_on").all()


def test_parse_boj_csv_error_status_raises():
    body = "STATUS,400\nMESSAGEID,M181005E\nMESSAGE,架空のエラー\n".encode("shift_jis")
    with pytest.raises(MacroParseError, match="M181005E"):
        parse_boj_csv(body, "call_rate_on", "STRDCLUCON")


def test_parse_boj_csv_no_rows_in_window_is_empty():
    body = "STATUS,200\nMESSAGEID,M181000I\nMESSAGE,架空\nNEXTPOSITION,\n".encode("shift_jis")
    assert parse_boj_csv(body, "call_rate_on", "STRDCLUCON").empty


def test_boj_url_keeps_at_sign_and_adds_start():
    db, code = BOJ_SERIES["boj_basic_loan_rate"]
    assert boj_url(db, code) == "https://www.stat-search.boj.or.jp/api/v1/getDataCode?db=IR01&code=MADR1Z@D&format=csv"
    assert boj_url(db, code, "202610").endswith("&format=csv&startDate=202610")


def test_parse_fred_json():
    body = b'{"observations":[{"date":"2026-11-02","value":"123.45"},{"date":"2026-11-03","value":"."}]}'
    df = parse_fred_json(body, "usd_jpy", "JPY/USD")
    assert df["date"].tolist() == [date(2026, 11, 2)] and df["value"].tolist() == [123.45]
    assert df.loc[0, "source"] == "fred"
    with pytest.raises(MacroParseError):
        parse_fred_json(b'{"error_message":"x"}', "usd_jpy", "JPY/USD")


# ---------- 作业（假 HTTP，不联网） ----------
class _Resp:
    def __init__(self, content: bytes, status: int = 200):
        self.content, self.status_code = content, status


class _Session:
    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params))
        return self.pages[url]


def _boj_body(code: str, dates: list[str]) -> bytes:
    head = "STATUS,200\nMESSAGEID,M181000I\nMESSAGE,架空\nNEXTPOSITION,\n" \
           "SERIES_CODE,NAME_OF_TIME_SERIES_J,UNIT_J,FREQUENCY,CATEGORY_J,LAST_UPDATE,SURVEY_DATES,VALUES\n"
    return (head + "".join(f"{code},架空,年％,DAILY,架空,20261106,{d},0.5\n" for d in dates)).encode("shift_jis")


def _pages(mof_all=JGB_CSV, mof_cur=None, boj_start=""):
    from jpscan.connectors.macro import BOJ_SERIES, MOF_ALL_URL, MOF_CURRENT_URL, boj_url
    pages = {MOF_ALL_URL: _Resp(mof_all),
             MOF_CURRENT_URL: _Resp(mof_cur or "基準日,1年,2年,5年,10年,20年,30年,40年\nR8.11.2,9,9,9,9.5,9,9,9\nR8.11.5,1,1,1,1.1,1,1,1\n".encode("cp932"))}
    for series, (db, code) in BOJ_SERIES.items():
        pages[boj_url(db, code, boj_start or None)] = _Resp(_boj_body(code, ["20261102", "20261104"]))
    return pages


@pytest.fixture
def env(tmp_path, monkeypatch):
    from jpscan.config import Settings
    from jpscan.jobs import ingest_macro
    from jpscan.ledger import Ledger
    from jpscan.paths import DataPaths
    from jpscan.store import Store
    monkeypatch.setattr(ingest_macro, "MIN_INTERVAL", 0)
    paths = DataPaths(tmp_path / "data").ensure()
    settings = Settings(root=tmp_path, data_dir=paths.root, params_path=tmp_path / "p.yaml", jquants_api_key=None,
                        anthropic_api_key=None, edinet_api_key=None, fred_api_key=None)
    return settings, paths, Ledger(paths.ledger_sqlite), Store(paths)


def _install(monkeypatch, pages):
    from jpscan.jobs import ingest_macro
    sess = _Session(pages)
    monkeypatch.setattr(ingest_macro, "http_session", lambda: sess)
    return sess


def test_job_writes_series_and_skips_fred_without_key(env, monkeypatch):
    from jpscan.jobs import ingest_macro
    settings, paths, ledger, store = env
    sess = _install(monkeypatch, _pages())
    out = ingest_macro.run(settings, paths, ledger, store)
    assert out["status"] == "ok"
    assert out["sources"]["fred"] == "skipped_no_key"
    assert ledger.status_of("macro", "fred") == "skipped_no_key"
    assert ledger.status_of("macro", "jgb_all") == "ok"
    assert len(sess.calls) == 4           # 全史、当月、两个日銀序列各一次

    df = store.read("staging", "macro_series")
    assert list(df.columns) == ["series", "date", "value", "source", "unit", "fetched_at"]
    assert not df.duplicated(["series", "date"]).any()
    assert store.partitions("staging", "macro_series") == ["all"]
    assert out["series"]["jgb_10y"] == {"rows": 4, "latest_date": "2026-11-05", "latest_value": 1.1}
    assert out["series"]["call_rate_on"]["rows"] == 2
    assert out["series"]["boj_basic_loan_rate"]["latest_date"] == "2026-11-04"
    # 同一日期以当月文件为准
    v = df[(df["series"] == "jgb_10y") & (df["date"] == "2026-11-02")]["value"].tolist()
    assert v == [9.5]
    assert list((paths.raw / "mof").rglob("*.csv")) and list((paths.raw / "boj").rglob("*.csv"))


def test_job_second_run_is_incremental_and_idempotent(env, monkeypatch):
    from jpscan.connectors.macro import MOF_ALL_URL
    from jpscan.jobs import ingest_macro
    settings, paths, ledger, store = env
    _install(monkeypatch, _pages())
    ingest_macro.run(settings, paths, ledger, store)
    n1 = len(store.read("staging", "macro_series"))
    raw1 = len(list(paths.raw.rglob("*.csv")))

    # 最新日期 2026-11-04，回看 40 天 → 2026-09-25 所在月
    sess = _install(monkeypatch, _pages(boj_start="202609"))
    out = ingest_macro.run(settings, paths, ledger, store)
    urls = [u for u, _ in sess.calls]
    assert out["status"] == "ok"
    assert MOF_ALL_URL not in urls        # 同月不重取全史
    assert any("startDate=202609" in u for u in urls)
    assert len(store.read("staging", "macro_series")) == n1
    assert len(list(paths.raw.rglob("*.csv"))) == raw1   # 内容没变，不重复落盘


def test_job_series_filter_limits_requests(env, monkeypatch):
    from jpscan.jobs import ingest_macro
    settings, paths, ledger, store = env
    sess = _install(monkeypatch, _pages())
    out = ingest_macro.run(settings, paths, ledger, store, series=["call_rate_on"])
    assert len(sess.calls) == 1 and "STRDCLUCON" in sess.calls[0][0]
    assert set(out["series"]) == {"call_rate_on"}
    assert "fred" not in out["sources"]
    with pytest.raises(ValueError):
        ingest_macro.run(settings, paths, ledger, store, series=["nope"])


def test_job_refetches_full_history_when_a_requested_series_has_none(env, monkeypatch):
    from jpscan.connectors.macro import MOF_ALL_URL
    from jpscan.jobs import ingest_macro
    settings, paths, ledger, store = env
    _install(monkeypatch, _pages())
    ingest_macro.run(settings, paths, ledger, store, series=["jgb_10y"])
    sess = _install(monkeypatch, _pages())
    out = ingest_macro.run(settings, paths, ledger, store, series=["jgb_10y", "jgb_40y"])
    assert MOF_ALL_URL in [u for u, _ in sess.calls]
    assert out["series"]["jgb_40y"]["rows"] > 1


def test_job_source_failure_is_recorded_and_others_continue(env, monkeypatch):
    from jpscan.connectors.macro import MOF_ALL_URL
    from jpscan.jobs import ingest_macro
    settings, paths, ledger, store = env
    pages = _pages()
    pages[MOF_ALL_URL] = _Resp("<html>メンテナンス中</html>".encode("cp932"))
    _install(monkeypatch, pages)
    out = ingest_macro.run(settings, paths, ledger, store)
    assert out["status"] == "partial"
    assert ledger.status_of("macro", "jgb_all") == "error"
    assert ledger.get_kv(ingest_macro.KV_JGB_ALL_MONTH) is None      # 下次运行重试全史
    assert list(paths.quarantine.rglob("*.bin"))
    assert out["series"]["call_rate_on"]["rows"] == 2


def test_fred_key_never_reaches_raw_ledger_or_errors(env, monkeypatch, caplog):
    import dataclasses
    import requests

    from jpscan.connectors.macro import FRED_URL
    from jpscan.jobs import ingest_macro
    settings, paths, ledger, store = env
    secret = "SYNTHETICFREDKEY0123456789"
    settings = dataclasses.replace(settings, fred_api_key=secret)
    body = b'{"observations":[{"date":"2026-11-02","value":"123.5"}]}'
    sess = _install(monkeypatch, {FRED_URL: _Resp(body)})
    out = ingest_macro.run(settings, paths, ledger, store, series=["usd_jpy"])
    assert out["series"]["usd_jpy"]["latest_value"] == 123.5
    assert sess.calls[0][1]["series_id"] == "DEXJPUS" and sess.calls[0][1]["api_key"] == secret

    # 连接失败时，requests 的异常文本带完整 URL（含 key）
    class Boom(_Session):
        def get(self, url, params=None, headers=None, timeout=None):
            raise requests.ConnectionError(f"Max retries exceeded with url: {url}?api_key={secret}")

    monkeypatch.setattr(ingest_macro, "http_session", lambda: Boom({}))
    monkeypatch.setattr("jpscan.connectors.macro.time.sleep", lambda s: None)
    out = ingest_macro.run(settings, paths, ledger, store, series=["nikkei225"])
    assert out["status"] == "partial" and "ConnectionError" in out["sources"]["fred_nikkei225"]
    assert secret not in str(out) and secret not in caplog.text

    db_bytes = b"".join(p.read_bytes() for p in paths.db.iterdir() if p.is_file())
    assert secret.encode() not in db_bytes
    for p in paths.raw.rglob("*"):
        assert secret not in str(p)
        if p.is_file():
            assert secret.encode() not in p.read_bytes()
