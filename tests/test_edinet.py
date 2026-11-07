"""EDINET 连接器测试：不联网、不需要 API Key。所有数据都是虚构的（公司、代码、金额皆为编造）。"""
from __future__ import annotations

import io
import json
import logging
import zipfile
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import requests

from jpscan.connectors.edinet import (EdinetAuthError, EdinetClient, EdinetError, EdinetKeyMissing, EdinetNotFound,
                                      doc_kind, normalize_sec_code, parse_code_list)
from jpscan.connectors.edinet_csv import decode_csv_bytes, extract_key_items, parse_csv_bytes, parse_financial_csv_zip, to_number
from jpscan.ledger import Ledger
from jpscan.paths import DataPaths
from jpscan.store import Store

FAKE_KEY = "FAKEKEY0123456789abcdef"
HEADER = ["要素ID", "項目名", "コンテキストID", "相対年度", "連結・個別", "期間・時点", "ユニットID", "単位", "値"]


# ---------------- 替身 ----------------
class FakeResponse:
    def __init__(self, content: bytes = b"", status_code: int = 200, content_type: str = "application/json; charset=utf-8",
                 headers: dict[str, str] | None = None):
        self.content = content
        self.status_code = status_code
        self.headers = {"Content-Type": content_type, **(headers or {})}


class FakeSession:
    """按顺序或按 URL 返回预设响应，并记录请求。"""

    def __init__(self, responses=None, router=None):
        self.responses = list(responses or [])
        self.router = router
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, params=None, headers=None, timeout=None, allow_redirects=True):
        self.calls.append((url, dict(params or {})))
        out = self.router(url, params) if self.router else self.responses.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


class FakeLimiter:
    def __init__(self):
        self.waits = 0
        self.penalties: list[float] = []

    def wait(self):
        self.waits += 1

    def penalize(self, seconds=61.0, grow=1.05):
        self.penalties.append(seconds)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr("jpscan.connectors.edinet.time.sleep", lambda s: None)
    paths = DataPaths(tmp_path / "data").ensure()
    return SimpleNamespace(paths=paths, ledger=Ledger(paths.ledger_sqlite), store=Store(paths), limiter=FakeLimiter(),
                           settings=SimpleNamespace(edinet_api_key=None))


def make_client(env, session, key=FAKE_KEY):
    return EdinetClient(key, env.paths, env.ledger, session=session, limiter=env.limiter)


# ---------------- 虚构数据 ----------------
def doc_row(doc_id="S100TST1", seq=1, sec="99990", dtype="120", form="030000", csv_flag="1", **kw):
    row = {
        "seqNumber": seq, "docID": doc_id, "edinetCode": "E99990", "secCode": sec, "JCN": "1234567890123", "filerName": "架空商事株式会社",
        "fundCode": None, "ordinanceCode": "010", "formCode": form, "docTypeCode": dtype, "periodStart": "2025-04-01",
        "periodEnd": "2026-03-31", "submitDateTime": "2026-06-26 15:00", "docDescription": "有価証券報告書－第1期",
        "issuerEdinetCode": None, "subjectEdinetCode": None, "subsidiaryEdinetCode": None, "currentReportReason": None, "parentDocID": None,
        "opeDateTime": None, "withdrawalStatus": "0", "docInfoEditStatus": "0", "disclosureStatus": "0", "xbrlFlag": "1", "pdfFlag": "1",
        "attachDocFlag": "0", "englishDocFlag": "0", "csvFlag": csv_flag, "legalStatus": "1",
    }
    row.update(kw)
    return row


def list_json(rows, status="200", message="OK", d="2026-06-26") -> bytes:
    return json.dumps({"metadata": {"title": "提出された書類を把握するためのAPI", "parameter": {"date": d, "type": "2"},
                                    "resultset": {"count": len(rows)}, "processDateTime": f"{d} 23:00", "status": status,
                                    "message": message}, "results": rows}, ensure_ascii=False).encode("utf-8")


def tombstone(doc_id, withdrawal="2", seq=1):
    """閲覧期間満了或取下げ后的样子：只有连番与 docID，其余为 null，区分与标志为 "0"（仕様書 3-1-4）。"""
    keys = ["edinetCode", "secCode", "JCN", "filerName", "fundCode", "ordinanceCode", "formCode", "docTypeCode", "periodStart", "periodEnd",
            "submitDateTime", "docDescription", "issuerEdinetCode", "subjectEdinetCode", "subsidiaryEdinetCode", "currentReportReason",
            "parentDocID", "opeDateTime"]
    row = {k: None for k in keys}
    row.update(seqNumber=seq, docID=doc_id, withdrawalStatus=withdrawal, docInfoEditStatus="0", disclosureStatus="0", xbrlFlag="0", pdfFlag="0",
               attachDocFlag="0", englishDocFlag="0", csvFlag="0", legalStatus="0")
    return row


FIN_ROWS = [
    ["jppfs_cor:NetSales", "売上高", "CurrentYearDuration", "当期", "連結", "期間", "JPY", "円", "1000000"],
    ["jppfs_cor:NetSales", "売上高", "Prior1YearDuration", "前期", "連結", "期間", "JPY", "円", "900000"],
    ["jppfs_cor:NetSales", "売上高", "CurrentYearDuration_NonConsolidatedMember", "当期", "個別", "期間", "JPY", "円", "400000"],
    ["jpcrp_cor:NetSalesSummaryOfBusinessResults", "売上高、経営指標等", "CurrentYearDuration", "当期", "その他", "期間", "JPY", "円", "999999"],
    ["jppfs_cor:OperatingIncome", "営業利益", "CurrentYearDuration", "当期", "連結", "期間", "JPY", "円", "120000"],
    ["jppfs_cor:InterestExpensesNOE", "支払利息", "CurrentYearDuration", "当期", "連結", "期間", "JPY", "円", "3000"],
    ["jppfs_cor:DepreciationAndAmortizationOpeCF", "減価償却費", "CurrentYearDuration", "当期", "連結", "期間", "JPY", "円", "50000"],
    ["jppfs_cor:PurchaseOfPropertyPlantAndEquipmentInvCF", "有形固定資産の取得", "CurrentYearDuration", "当期", "連結", "期間", "JPY", "円", "-70000"],
    ["jppfs_cor:CashAndDeposits", "現金及び預金", "CurrentYearInstant", "当期末", "連結", "時点", "JPY", "円", "200000"],
    ["jppfs_cor:ShortTermInvestmentSecurities", "有価証券", "CurrentYearInstant", "当期末", "連結", "時点", "JPY", "円", "-"],
    ["jppfs_cor:NotesReceivableTrade", "受取手形", "CurrentYearInstant", "当期末", "連結", "時点", "JPY", "円", "10000"],
    ["jppfs_cor:AccountsReceivableTrade", "売掛金", "CurrentYearInstant", "当期末", "連結", "時点", "JPY", "円", "20000"],
    ["jppfs_cor:MerchandiseAndFinishedGoods", "商品及び製品", "CurrentYearInstant", "当期末", "連結", "時点", "JPY", "円", "5000"],
    ["jppfs_cor:RawMaterialsAndSupplies", "原材料及び貯蔵品", "CurrentYearInstant", "当期末", "連結", "時点", "JPY", "円", "2000"],
    ["jppfs_cor:ShortTermLoansPayable", "短期借入金", "CurrentYearInstant", "当期末", "連結", "時点", "JPY", "円", "1000"],
    ["jppfs_cor:CurrentPortionOfLongTermLoansPayable", "1年内返済予定の長期借入金", "CurrentYearInstant", "当期末", "連結", "時点", "JPY", "円", "2000"],
    ["jppfs_cor:BondsPayable", "社債", "CurrentYearInstant", "当期末", "連結", "時点", "JPY", "円", "4000"],
    ["jppfs_cor:LongTermLoansPayable", "長期借入金", "CurrentYearInstant", "当期末", "連結", "時点", "JPY", "円", "8000"],
    ["jppfs_cor:LeaseObligationsNCL", "リース債務", "CurrentYearInstant", "当期末", "連結", "時点", "JPY", "円", "16000"],
    ["jppfs_cor:LongTermLoansPayable", "長期借入金", "Prior1YearInstant", "前期末", "連結", "時点", "JPY", "円", "7777"],
    ["jpcrp_cor:DescriptionOfBusinessTextBlock", "事業の内容", "FilingDateInstant", "提出日時点", "その他", "時点", "-", "-", "架空の事業を営む。\n\"引用\"を含む。"],
    ["jpdei_cor:SecurityCodeDEI", "証券コード", "FilingDateInstant", "提出日時点", "その他", "時点", "-", "-", "99990"],
    ["jpcrp030000-asr_E99990-000:CustomElement", "", "CurrentYearDuration", "当期", "その他", "期間", "JPY", "円", "123"],
]


def tsv_bytes(rows=FIN_ROWS, header=HEADER, encoding="utf-16", sep="\t") -> bytes:
    """官方格式：全部带双引号、CRLF、内部 " 转义为 ""。"""
    def fmt(row):
        return sep.join('"' + str(c).replace('"', '""') + '"' for c in row)
    return "\r\n".join(fmt(r) for r in [header, *rows]).__add__("\r\n").encode(encoding)


def make_csv_zip(rows=FIN_ROWS, folder="XBRL_TO_CSV") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(f"{folder}/jpcrp030000-asr-001_E99990-000_2026-03-31_01_2026-06-26.csv", tsv_bytes(rows))
        z.writestr(f"{folder}/jpaud-aar-cn-001_E99990-000_2026-03-31_01_2026-06-26.csv",
                   tsv_bytes([["jpaud_cor:AuditorOpinionTextBlock", "監査意見", "FilingDateInstant", "提出日時点", "その他", "時点", "-", "-", "適正"]]))
        z.writestr(f"{folder}/unrelated.csv", "foo,bar\r\n1,2\r\n".encode("utf-8"))
    return buf.getvalue()


def make_code_list_zip(as_of_line="ダウンロード実行日,2026年11月06日 現在,,,,,,,,,,,") -> bytes:
    header = "ＥＤＩＮＥＴコード,提出者種別,上場区分,連結の有無,資本金,決算日,提出者名,提出者名（英字）,提出者名（ヨミ）,所在地,提出者業種,証券コード,提出者法人番号"
    lines = [
        as_of_line, header,
        "E99990,内国法人・組合,上場,有,1000,3月31日,架空商事株式会社,Kakuu Shoji Co. Ltd.,カクウショウジ,東京都架空区1-1,卸売業,99990,1234567890123",
        "E99991,内国法人・組合,上場,無,500,12月31日,虚構工業株式会社,Kyoko Kogyo,キョコウコウギョウ,大阪府架空市2-2,機械,9998,1234567890124",
        "E99992,内国法人・組合（有価証券報告書等の提出義務者以外）,非上場,,,,架空ファンド合同会社,,カクウファンド,,,,",
        ",,,,,,,,,,,,",
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("EdinetcodeDlInfo.csv", "\r\n".join(lines).encode("cp932"))
    return buf.getvalue()


# ---------------- 基础函数 ----------------
def test_sec_code_and_doc_kind():
    assert normalize_sec_code("99990") == ("99990", "9999")
    assert normalize_sec_code("9999") == ("99990", "9999")
    assert normalize_sec_code("130A0") == ("130A0", "130A")
    assert normalize_sec_code(None) == (None, None)
    assert normalize_sec_code("") == (None, None)
    assert doc_kind("120") == "有価証券報告書"
    assert doc_kind("350", "010000") == "大量保有報告書"
    assert doc_kind("350", "010002") == "変更報告書"
    assert doc_kind("360", "010002") == "訂正大量保有報告書・変更報告書"
    assert doc_kind(None) is None


# ---------------- 客户端：书类一览 ----------------
def test_list_documents_saves_raw_without_key(env):
    session = FakeSession([FakeResponse(list_json([doc_row()]))])
    rows = make_client(env, session).list_documents(date(2026, 6, 26))
    assert [r["docID"] for r in rows] == ["S100TST1"]
    url, params = session.calls[0]
    assert url.endswith("/api/v2/documents.json")
    assert params["date"] == "2026-06-26" and params["type"] == 2 and params["Subscription-Key"] == FAKE_KEY
    with env.ledger.conn() as c:
        raw = c.execute("SELECT path, params_json, n_rows FROM raw_files").fetchall()
    assert len(raw) == 1 and raw[0][2] == 1
    assert FAKE_KEY not in raw[0][0] and FAKE_KEY not in raw[0][1]


def test_list_documents_use_cache_makes_no_request(env):
    client = make_client(env, FakeSession([FakeResponse(list_json([doc_row()]))]))
    client.list_documents("2026-06-26")
    client.session = FakeSession([])
    assert client.list_documents("2026-06-26", use_cache=True)[0]["docID"] == "S100TST1"
    assert client.session.calls == []


def test_no_key_error_is_clear_and_makes_no_request(env):
    session = FakeSession([])
    client = make_client(env, session, key=None)
    with pytest.raises(EdinetKeyMissing, match="EDINET_API_KEY"):
        client.list_documents("2026-06-26")
    with pytest.raises(EdinetKeyMissing):
        client.download("S100TST1", 5)
    assert session.calls == []


@pytest.mark.parametrize("body,exc", [
    (b'{"metadata":{"title":"t","status":"404","message":"Not Found"}}', EdinetNotFound),
    (b'{"metadata":{"title":"t","status":"400","message":"Bad Request"}}', EdinetError),
    (b'{"StatusCode":401,"message":"Access denied due to invalid subscription key."}', EdinetAuthError),
])
def test_errors_with_http_200_are_detected(env, body, exc):
    with pytest.raises(exc):
        make_client(env, FakeSession([FakeResponse(body)])).list_documents("2026-06-26")


def test_metadata_500_is_retried(env):
    bad = b'{"metadata":{"title":"t","status":"500","message":"Internal Server Error"}}'
    session = FakeSession([FakeResponse(bad), FakeResponse(list_json([doc_row()]))])
    assert len(make_client(env, session).list_documents("2026-06-26")) == 1
    assert len(session.calls) == 2


@pytest.mark.parametrize("first", [
    FakeResponse(b'{"StatusCode":429,"message":"Too Many Requests"}', status_code=200),
    FakeResponse(b'{"StatusCode":429,"message":"Too Many Requests"}', status_code=429, headers={"Retry-After": "90"}),
])
def test_429_backs_off_and_retries(env, first):
    session = FakeSession([first, FakeResponse(list_json([]))])
    assert make_client(env, session).list_documents("2026-06-26") == []
    assert len(env.limiter.penalties) == 1 and env.limiter.penalties[0] >= 60
    assert env.limiter.waits == 2  # 每次请求前都经过限速器


def test_maintenance_html_is_an_error(env):
    html = FakeResponse(b"<html>Sorry</html>", content_type="text/html")
    with pytest.raises(EdinetError, match="维护"):
        make_client(env, FakeSession([html])).list_documents("2026-06-26")


# ---------------- 密钥不外泄 ----------------
def test_key_never_leaks_on_connection_error(env, caplog):
    leaky = requests.ConnectionError(f"Max retries exceeded with url: /api/v2/documents.json?date=2026-06-26&Subscription-Key={FAKE_KEY}")
    client = make_client(env, FakeSession([leaky] * 4))
    with caplog.at_level(logging.DEBUG), pytest.raises(EdinetError) as ei:
        client.list_documents("2026-06-26")
    assert FAKE_KEY not in str(ei.value)
    assert ei.value.__cause__ is None and ei.value.__context__ is None   # 不带原始异常链
    assert FAKE_KEY not in caplog.text
    env.ledger.mark("edinet_list", "2026-06-26", "error", error=str(ei.value))
    assert_key_absent_everywhere(env)


def test_key_never_leaks_with_real_requests_exception(env, caplog, monkeypatch):
    """真实的 requests 异常文本带完整 URL（含密钥）：连本机关闭的端口触发，不访问外部网络。"""
    monkeypatch.setattr("jpscan.connectors.edinet.API_BASE", "http://127.0.0.1:9/api/v2/")
    client = EdinetClient(FAKE_KEY, env.paths, env.ledger, limiter=env.limiter, retries=1, timeout=5)
    session = requests.Session()
    session.trust_env = False
    client.session = session
    with pytest.raises(requests.ConnectionError) as raw:   # 先确认前提：直接请求时异常文本确实含密钥
        session.get("http://127.0.0.1:9/x", params={"Subscription-Key": FAKE_KEY}, timeout=5)
    assert FAKE_KEY in str(raw.value)
    with caplog.at_level(logging.DEBUG), pytest.raises(EdinetError) as ei:
        client.list_documents("2026-06-26")
    assert FAKE_KEY not in str(ei.value) and FAKE_KEY not in caplog.text and "ConnectionError" in str(ei.value)


def assert_key_absent_everywhere(env):
    for f in Path(env.paths.root).rglob("*"):
        assert FAKE_KEY not in f.name
    with env.ledger.conn() as c:
        for table in ("raw_files", "completeness"):
            for row in c.execute(f"SELECT * FROM {table}").fetchall():
                assert FAKE_KEY not in " ".join(str(v) for v in row)


def test_key_stripped_from_server_echo_and_redirects_disabled(env):
    echo = json.dumps({"metadata": {"status": "400", "message": f"bad Subscription-Key={FAKE_KEY}"}}).encode()
    seen = {}

    class Recording(FakeSession):
        def get(self, url, params=None, headers=None, timeout=None, allow_redirects=True):
            seen["allow_redirects"] = allow_redirects
            return super().get(url, params=params, headers=headers, timeout=timeout, allow_redirects=allow_redirects)

    with pytest.raises(EdinetError) as ei:
        make_client(env, Recording([FakeResponse(echo)])).list_documents("2026-06-26")
    assert FAKE_KEY not in str(ei.value)
    assert seen["allow_redirects"] is False


def test_urllib3_debug_log_is_redacted(caplog):
    logger = logging.getLogger("urllib3.connectionpool")
    with caplog.at_level(logging.DEBUG, logger="urllib3.connectionpool"):
        logger.debug('%s://%s "GET %s" %s', "https", "api.edinet-fsa.go.jp",
                     f"/api/v2/documents.json?date=2026-06-26&Subscription-Key={FAKE_KEY}", 200)
    assert "date=2026-06-26" in caplog.text and FAKE_KEY not in caplog.text


# ---------------- 客户端：书类取得 ----------------
def test_download_saves_raw_and_skips_when_present(env):
    content = make_csv_zip()
    session = FakeSession([FakeResponse(content, content_type="application/octet-stream")])
    client = make_client(env, session)
    path = client.download("S100TST1", 5, subdir="2026-06")
    assert path.read_bytes() == content and path.name == "S100TST1_5.zip" and path.parent.name == "2026-06"
    assert env.ledger.status_of("edinet_doc_type5", "S100TST1") == "ok"
    assert client.download("S100TST1", "5", subdir="2026-06") == path
    assert len(session.calls) == 1
    assert session.calls[0][0].endswith("/documents/S100TST1") and session.calls[0][1]["type"] == 5
    assert_key_absent_everywhere(env)


def test_download_error_json_raises_and_writes_nothing(env):
    body = b'{"metadata":{"title":"t","status":"404","message":"Not Found"}}'
    client = make_client(env, FakeSession([FakeResponse(body)]))
    with pytest.raises(EdinetNotFound):
        client.download("S100GONE", 5, subdir="2026-06")
    assert not client.doc_path("S100GONE", 5, "2026-06").exists()


def test_download_wrong_content_is_quarantined(env):
    client = make_client(env, FakeSession([FakeResponse(b"<html>" + FAKE_KEY.encode(), content_type="application/octet-stream")]))
    with pytest.raises(EdinetError, match="隔离"):
        client.download("S100BAD1", 5)
    quarantined = [f for f in env.paths.quarantine.rglob("*.bin")]
    assert len(quarantined) == 1 and FAKE_KEY.encode() not in quarantined[0].read_bytes()


@pytest.mark.parametrize("bad", ["../evil", "a/b", "", "S100 X"])
def test_doc_id_is_validated_before_use_in_paths(env, bad):
    with pytest.raises(ValueError):
        make_client(env, FakeSession([])).doc_path(bad, 5)


def test_doc_type_is_validated(env):
    with pytest.raises(ValueError):
        make_client(env, FakeSession([])).doc_path("S100TST1", 9)


# ---------------- CSV 解析 ----------------
@pytest.mark.parametrize("encoding", ["utf-16", "utf-16-le", "utf-8-sig", "utf-8", "cp932"])
def test_decode_detects_encoding(encoding):
    raw = tsv_bytes(FIN_ROWS[:2], encoding=encoding)
    text, _ = decode_csv_bytes(raw)
    assert "要素ID" in text and "売上高" in text


@pytest.mark.parametrize("encoding,sep", [("utf-16", "\t"), ("utf-16-le", "\t"), ("cp932", ","), ("utf-8-sig", ","), ("utf-8", "\t"), ("utf-8", ";")])
def test_parse_csv_handles_encoding_and_delimiter(encoding, sep):
    df = parse_csv_bytes(tsv_bytes(FIN_ROWS[:3], encoding=encoding, sep=sep), "D", "f.csv")
    assert list(df["element_id"]) == ["jppfs_cor:NetSales"] * 3
    assert df.loc[0, "item_name"] == "売上高" and df.loc[0, "numeric_value"] == 1000000.0
    assert df.loc[2, "consolidated_flag"] == "個別" and df.loc[0, "docID"] == "D" and df.loc[0, "file"] == "f.csv"


def test_header_heuristics_variants():
    header = ["要素 ＩＤ", "項目名称", "コンテキストＩＤ", "相対年度", "連結/個別", "期間/時点", "ユニットＩＤ", "単位", "値"]
    df = parse_csv_bytes(tsv_bytes(FIN_ROWS[:1], header=header), "D", "f.csv")
    row = df.iloc[0]
    assert row["element_id"] == "jppfs_cor:NetSales" and row["context_id"] == "CurrentYearDuration"
    assert row["consolidated_flag"] == "連結" and row["period_type"] == "期間" and row["unit_id"] == "JPY" and row["item_name"] == "売上高"


def test_unrecognized_csv_returns_none():
    assert parse_csv_bytes("foo,bar\r\n1,2\r\n".encode("utf-8")) is None
    assert parse_csv_bytes(b"") is None


def test_to_number_rules():
    assert to_number("1,234") == 1234.0
    assert to_number("-5") == -5.0
    assert to_number("△5") == -5.0
    assert to_number("-") == 0.0            # 操作ガイド：値が「-」は明示的に 0
    assert to_number("－") == 0.0
    assert np.isnan(to_number("-", has_unit=False))
    assert np.isnan(to_number("99990", has_unit=False))   # 无单位视为文本
    assert np.isnan(to_number("2026-03-31"))
    assert np.isnan(to_number(None))
    assert np.isnan(to_number("nan")) and np.isnan(to_number("inf"))


def test_parse_zip_long_table(tmp_path):
    z = tmp_path / "S100TST1_5.zip"
    z.write_bytes(make_csv_zip())
    df = parse_financial_csv_zip(z)
    assert set(df["docID"]) == {"S100TST1"}
    assert list(df.columns) == ["docID", "file", "element_id", "item_name", "context_id", "relative_year", "consolidated_flag", "period_type",
                                "unit_id", "unit", "value", "numeric_value"]
    assert df.attrs["skipped_files"] == ["unrelated.csv"]
    assert len(df) == len(FIN_ROWS) + 1
    by = df.set_index(["element_id", "context_id"])
    assert by.loc[("jppfs_cor:ShortTermInvestmentSecurities", "CurrentYearInstant"), "numeric_value"] == 0.0   # 「-」
    text = by.loc[("jpcrp_cor:DescriptionOfBusinessTextBlock", "FilingDateInstant")]
    assert text["value"] == "架空の事業を営む。\n\"引用\"を含む。"           # 引号转义与内嵌换行
    assert pd.isna(text["unit_id"])                                         # 「-」→ 空
    assert np.isnan(text["numeric_value"])
    assert np.isnan(by.loc[("jpdei_cor:SecurityCodeDEI", "FilingDateInstant"), "numeric_value"])  # 证券代码不当数值
    custom = by.loc[("jpcrp030000-asr_E99990-000:CustomElement", "CurrentYearDuration")]
    assert pd.isna(custom["item_name"])                                                           # 空白项目名


def test_parse_zip_without_folder_falls_back_and_empty_zip(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("data.csv", tsv_bytes(FIN_ROWS[:1]))
    p = tmp_path / "S100TST2_5.zip"
    p.write_bytes(buf.getvalue())
    assert len(parse_financial_csv_zip(p)) == 1
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("readme.txt", "none")
    p = tmp_path / "S100TST3_5.zip"
    p.write_bytes(buf.getvalue())
    assert parse_financial_csv_zip(p).empty


# ---------------- 关键科目 ----------------
def key_items_from(rows, tmp_path=None, code="99990"):
    df = parse_csv_bytes(tsv_bytes(rows), "S100TST1", "f.csv")
    df["Code"] = code
    df["period_end"] = date(2026, 3, 31)
    df["submit_date"] = date(2026, 6, 26)
    return extract_key_items(df)


def test_extract_key_items_mapping_and_priority():
    items = key_items_from(FIN_ROWS)
    assert list(items.columns) == ["Code", "period_end", "disc_date", "consolidated", "item", "value", "docID"]
    cons = items[items["consolidated"]].set_index("item")["value"]
    assert cons["net_sales"] == 1000000.0                  # jppfs 优先于 jpcrp 摘要；不取前期与个别
    assert cons["operating_income"] == 120000.0
    assert cons["interest_expense"] == 3000.0
    assert cons["depreciation"] == 50000.0
    assert cons["capex"] == -70000.0                       # 符号保持原文件
    assert cons["cash_and_deposits"] == 200000.0
    assert cons["short_term_securities"] == 0.0            # 「-」= 0
    assert cons["long_term_loans"] == 8000.0               # 不取 Prior1YearInstant 的 7777
    assert cons["interest_bearing_debt"] == 1000 + 2000 + 4000 + 8000 + 16000
    assert cons["receivables_trade"] == 30000.0            # 缺合计，用构成项相加
    assert cons["inventories"] == 7000.0
    single = items[~items["consolidated"]].set_index("item")["value"]
    assert single.to_dict() == {"net_sales": 400000.0}
    row = items.iloc[0]
    assert row["Code"] == "99990" and row["period_end"] == date(2026, 3, 31) and row["disc_date"] == date(2026, 6, 26)
    assert row["docID"] == "S100TST1"
    assert not items.duplicated(["docID", "consolidated", "item"]).any()


def test_extract_key_items_reported_total_wins_over_components():
    rows = FIN_ROWS + [["jppfs_cor:Inventories", "棚卸資産", "CurrentYearInstant", "当期末", "連結", "時点", "JPY", "円", "9000"]]
    cons = key_items_from(rows).query("consolidated").set_index("item")["value"]
    assert cons["inventories"] == 9000.0


def test_extract_key_items_ifrs_uses_context_for_consolidation():
    rows = [
        ["jpigp_cor:RevenueIFRS", "売上収益", "CurrentYearDuration", "当期", "その他", "期間", "JPY", "円", "500"],
        ["jpigp_cor:OperatingProfitLossIFRS", "営業利益", "CurrentYearDuration", "当期", "その他", "期間", "JPY", "円", "50"],
        ["jpigp_cor:BorrowingsNCLIFRS", "借入金", "CurrentYearInstant", "当期末", "その他", "時点", "JPY", "円", "70"],
        ["jpigp_cor:LeaseLiabilitiesCLIFRS", "リース負債", "CurrentYearInstant", "当期末", "その他", "時点", "JPY", "円", "7"],
    ]
    items = key_items_from(rows)
    assert items["consolidated"].all()
    cons = items.set_index("item")["value"]
    assert cons["net_sales"] == 500 and cons["operating_income"] == 50 and cons["interest_bearing_debt"] == 77


def test_extract_key_items_interim_contexts_and_exclusions():
    rows = [
        ["jppfs_cor:NetSales", "売上高", "InterimDuration", "当中間期", "連結", "期間", "JPY", "円", "60"],
        ["jppfs_cor:CashAndDeposits", "現金及び預金", "InterimInstant", "当中間期末", "連結", "時点", "JPY", "円", "61"],
        ["jppfs_cor:OperatingIncome", "営業利益", "CurrentQuarterDuration", "当四半期", "連結", "期間", "JPY", "円", "1"],
        ["jppfs_cor:BondsPayable", "社債", "CurrentYearInstant_ReportableSegmentsMember", "当期末", "連結", "時点", "JPY", "円", "2"],
    ]
    items = key_items_from(rows)
    assert items.set_index("item")["value"].to_dict() == {"net_sales": 60.0, "cash_and_deposits": 61.0}


def test_extract_key_items_requires_code_and_period_and_handles_empty():
    assert key_items_from(FIN_ROWS, code=None).empty
    no_period = parse_csv_bytes(tsv_bytes(FIN_ROWS), "S100TST1", "f.csv").assign(Code="99990", period_end=None, submit_date=date(2026, 6, 26))
    assert extract_key_items(no_period).empty
    assert extract_key_items(parse_csv_bytes(tsv_bytes([]), "D", "f.csv")).empty
    assert list(extract_key_items(None).columns)[0] == "Code"


# ---------------- 代码表 ----------------
def test_parse_code_list(tmp_path):
    z = tmp_path / "Edinetcode.zip"
    z.write_bytes(make_code_list_zip())
    df = parse_code_list(z)
    assert list(df["edinet_code"]) == ["E99990", "E99991", "E99992"]   # 空行被跳过
    a, b, c = (df.iloc[i] for i in range(3))
    assert a["Code"] == "99990" and a["code4"] == "9999" and a["listed"] and a["consolidated"] is True
    assert a["capital"] == 1000.0 and a["fiscal_year_end"] == "3月31日" and a["name_en"] == "Kakuu Shoji Co. Ltd."
    assert a["name_kana"] == "カクウショウジ" and a["jcn"] == "1234567890123" and a["as_of"] == date(2026, 11, 6)
    assert b["Code"] == "99980" and b["consolidated"] is False        # 4 位补 0
    assert not c["listed"] and pd.isna(c["Code"]) and pd.isna(c["consolidated"]) and pd.isna(c["capital"])


def test_parse_code_list_bad_header_raises(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("EdinetcodeDlInfo.csv", "a,b\r\n1,2\r\n".encode("cp932"))
    p = tmp_path / "bad.zip"
    p.write_bytes(buf.getvalue())
    with pytest.raises(EdinetError, match="列头"):
        parse_code_list(p)


def test_code_list_http_error(env):
    client = make_client(env, FakeSession([FakeResponse(b"nope", status_code=403, content_type="text/html")]), key=None)
    with pytest.raises(EdinetError, match="403"):
        client.fetch_code_list()


