"""TDnet XBRL zip 的摘要 inline XBRL 解析（规划文档 10.5 节）。

只读 tse-*-ixbrl.htm（決算短信サマリー acedjpsm 等、業績予想修正 rvfc、配当予想修正 rvdf，科目在 tse-ed-t 命名空间）。
预想值由 ResultForecastAxis / ForecastMember 维度区分；文件内有 xbrli:context 时按其解析，否则退回 contextRef 名称。
zip 不解压到磁盘，按成员读入内存；XML 解析关闭实体展开与网络访问。
"""
from __future__ import annotations

import logging
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Any

from lxml import etree

log = logging.getLogger("jpscan.connectors.tdnet_xbrl")

_SUMMARY_FILE = re.compile(r"(?:^|/)tse-[^/]*-ixbrl\.html?$", re.IGNORECASE)
_MAX_MEMBER_BYTES = 64 << 20
XBRLI = "http://www.xbrl.org/2003/instance"
XBRLDI = "http://xbrl.org/2006/xbrldi"
XSI_NIL = "{http://www.w3.org/2001/XMLSchema-instance}nil"
RESULT_FORECAST_AXIS = "ResultForecastAxis"
FORECAST_MEMBER = "ForecastMember"


def _local(qname: str | None) -> str:
    return (qname or "").rsplit(":", 1)[-1]


def _parse_contexts(root: etree._Element) -> dict[str, dict[str, Any]]:
    """{context id: {dimensions: {轴本地名: 成员本地名}, start, end, instant}}。"""
    out: dict[str, dict[str, Any]] = {}
    for ctx in root.iter(f"{{{XBRLI}}}context"):
        dims = {_local(m.get("dimension")): _local((m.text or "").strip()) for m in ctx.iter(f"{{{XBRLDI}}}explicitMember")}
        def period(tag: str) -> str | None:
            el = next(ctx.iter(f"{{{XBRLI}}}{tag}"), None)
            return el.text.strip() if el is not None and el.text else None
        out[ctx.get("id", "")] = {"dimensions": dims, "start": period("startDate"), "end": period("endDate"), "instant": period("instant")}
    return out


def _fact_value(el: etree._Element) -> tuple[str, float | None, bool]:
    """(原文, 换算后数值, is_nil)。数值 = 符号 × 显示值 × 10^scale；ixt 的 zerodash 格式视为 0。"""
    text = unicodedata.normalize("NFKC", "".join(el.itertext())).strip()
    if el.get(XSI_NIL) == "true":
        return text, None, True
    fmt = _local(el.get("format")).lower()
    if "zerodash" in fmt:
        return text, 0.0, False
    number = text.replace(",", "").replace(" ", "")
    if "numcommadecimal" in fmt:
        number = text.replace(".", "").replace(",", ".").replace(" ", "")
    try:
        value = float(number)
    except ValueError:
        return text, None, False
    scale = int(el.get("scale") or 0)
    value *= 10.0 ** scale
    return text, -value if el.get("sign") == "-" else value, False


def _parse_document(content: bytes, file_name: str) -> list[dict[str, Any]]:
    parser = etree.XMLParser(recover=True, resolve_entities=False, no_network=True, huge_tree=False)
    root = etree.fromstring(content, parser=parser)
    if root is None:
        log.warning("cannot parse %s", file_name)
        return []
    contexts = _parse_contexts(root)
    doc_type = Path(file_name).name.split("-")[1] if Path(file_name).name.count("-") >= 1 else None
    facts = []
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        qname = etree.QName(el)
        if qname.localname != "nonFraction" or "inlineXBRL" not in (qname.namespace or ""):
            continue
        name, ref = el.get("name"), el.get("contextRef") or ""
        ctx = contexts.get(ref)
        dims = ctx["dimensions"] if ctx else {}
        tokens = ref.split("_")
        member = dims.get(RESULT_FORECAST_AXIS)
        is_forecast = member == FORECAST_MEMBER if member else FORECAST_MEMBER in tokens
        text, value, is_nil = _fact_value(el)
        facts.append({
            "file": file_name, "doc_type": doc_type, "name": name, "local_name": _local(name),
            "contextRef": ref, "unitRef": el.get("unitRef"), "scale": el.get("scale"), "decimals": el.get("decimals"),
            "sign": el.get("sign"), "format": el.get("format"), "text": text, "value": value, "is_nil": is_nil,
            "is_forecast": is_forecast, "result_forecast": member or next((t for t in tokens if t in (FORECAST_MEMBER, "ResultMember")), None),
            "dimensions": dims, "period_start": ctx["start"] if ctx else None, "period_end": ctx["end"] if ctx else None,
            "instant": ctx["instant"] if ctx else None,
        })
    return facts


def parse_summary(zip_path: Path | str) -> list[dict[str, Any]]:
    """列出 zip 内全部摘要 inline XBRL 的 ix:nonFraction 事实（含 nil 事实，value=None）。"""
    facts: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir() or not _SUMMARY_FILE.search(info.filename):
                continue
            if info.file_size > _MAX_MEMBER_BYTES:
                log.warning("skip oversized member %s (%d bytes)", info.filename, info.file_size)
                continue
            facts.extend(_parse_document(zf.read(info), info.filename))
    return facts
