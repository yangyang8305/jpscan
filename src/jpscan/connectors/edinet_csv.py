"""EDINET 財務諸表本表 CSV（书类取得 type=5）的解析与关键科目提取。全部为纯函数，不联网。

格式依据：書類閲覧操作ガイド（2026 年 6 月版）1 章 3-2-1「XBRL からの CSV 変換ファイル」：
- zip 内 XBRL_TO_CSV 目录；扩展名 csv，实为制表符分隔、UTF-16LE、CRLF、所有项目带双引号（内部 " 转义为 ""）。
- 首行列头：要素ID / 項目名 / コンテキストID / 相対年度 / 連結・個別 / 期間・時点 / ユニットID / 単位 / 値。
- 值为「-」表示明确为 0；ユニットID 与単位都为「-」表示未设单位（文本项目）；项目名空白表示未定义日文名。
- 「連結・個別」只有 jppfs 开头的要素才是 連結/個別，其余一律「その他」，须看コンテキストID。
- 值超过 30,000 字符时被截断。

编码与分隔符仍做探测，以防格式调整；列头按同义名匹配。
"""
from __future__ import annotations

import csv
import io
import logging
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("jpscan.edinet_csv")

COLUMNS = ["docID", "file", "element_id", "item_name", "context_id", "relative_year", "consolidated_flag",
           "period_type", "unit_id", "unit", "value", "numeric_value"]
_FIELDS = COLUMNS[2:11]

_HEADER_ALIASES = {
    "element_id": ("要素id", "要素", "elementid"),
    "item_name": ("項目名", "項目名称", "科目名", "勘定科目", "ラベル", "itemname", "label"),
    "context_id": ("コンテキストid", "コンテキスト", "contextid"),
    "relative_year": ("相対年度", "relativeyear"),
    "consolidated_flag": ("連結個別", "連結/個別", "連結単体", "consolidated"),
    "period_type": ("期間時点", "期間/時点", "periodtype"),
    "unit_id": ("ユニットid", "ユニット", "unitid"),
    "unit": ("単位", "unit"),
    "value": ("値", "インスタンス値", "value"),
}
_ALIAS_TO_FIELD = {alias: field for field, aliases in _HEADER_ALIASES.items() for alias in aliases}
_NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?$")


def decode_csv_bytes(raw: bytes) -> tuple[str, str]:
    """探测编码，返回 (文本, 编码名)。BOM 优先；无 BOM 时靠 NUL 字节判断 UTF-16，再严格 UTF-8，最后 cp932。

    UTF-8 排在 cp932 之前：cp932 的字节几乎不可能是合法 UTF-8，反过来 UTF-8 日文却常能被 cp932「成功」解成乱码。
    """
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16"), "utf-16"
    if raw[:3] == b"\xef\xbb\xbf":
        return raw.decode("utf-8-sig"), "utf-8-sig"
    sample = raw[:4096]
    if sample and sample.count(0) > len(sample) // 4:   # ASCII 占多数的 UTF-16 文本，NUL 落在奇偶位之一
        enc = "utf-16-le" if sample[1::2].count(0) >= sample[0::2].count(0) else "utf-16-be"
        return raw.decode(enc), enc
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("cp932", errors="replace"), "cp932"


def detect_delimiter(text: str) -> str:
    """看首行里哪个分隔符最多；并列时取制表符（官方格式）。"""
    head = text.split("\n", 1)[0]
    return max(("\t", ",", ";"), key=lambda d: head.count(d))


def _norm_header(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).strip().strip('"').lower()
    return re.sub(r"[\s・]", "", s)


def _nil(s: str | None) -> str | None:
    """元数据列：空白与「-」→ None。「値」列不走这里，其「-」表示 0。"""
    if s is None:
        return None
    t = s.strip()
    return None if t in ("", "-", "－", "ー", "―") else t


def to_number(value: str | None, has_unit: bool = True) -> float:
    """值 → 数值。无单位（文本项目）→ NaN；有单位时「-」为明确的 0（操作ガイド）；△▲ 视为负号。"""
    if value is None or not has_unit:
        return np.nan
    s = unicodedata.normalize("NFKC", value).strip().replace(",", "")
    if s in ("-", "ー", "―"):
        return 0.0
    if s[:1] in ("△", "▲"):
        s = "-" + s[1:]
    return float(s) if _NUMBER_RE.match(s) else np.nan


def parse_csv_bytes(raw: bytes, doc_id: str | None = None, file: str | None = None) -> pd.DataFrame | None:
    """单个 CSV → 长表。认不出列头（缺要素ID或値）时返回 None。"""
    text, _ = decode_csv_bytes(raw)
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=detect_delimiter(text))
    positions: dict[str, int] | None = None
    rows: list[dict[str, Any]] = []
    for line_no, row in enumerate(reader):
        if positions is None:
            if line_no >= 5:   # 列头应在最前几行
                return None
            found = {_ALIAS_TO_FIELD[h]: i for i, h in enumerate(_norm_header(c) for c in row) if h in _ALIAS_TO_FIELD}
            if "element_id" in found and "value" in found:
                positions = found
            continue
        if not any(c.strip() for c in row):
            continue
        rec = {f: (row[i] if i < len(row) else None) for f, i in positions.items()}
        value = rec.get("value")
        value = value if value not in ("", None) else None
        out = {f: _nil(rec.get(f)) for f in _FIELDS if f != "value"}
        if out["element_id"] is None:
            continue
        out["value"] = value
        # 文件没有单位列时无法判断是否文本，只能尝试解析
        out["numeric_value"] = to_number(value, has_unit=("unit_id" not in positions) or out["unit_id"] is not None)
        rows.append(out)
    if positions is None:
        return None
    df = pd.DataFrame(rows, columns=[c for c in COLUMNS if c not in ("docID", "file")])
    df.insert(0, "file", file)
    df.insert(0, "docID", doc_id)
    df["numeric_value"] = df["numeric_value"].astype("float64")
    return df


def parse_financial_csv_zip(zip_path: Path | str, doc_id: str | None = None) -> pd.DataFrame:
    """type=5 的 zip → 长表（COLUMNS）。优先读 XBRL_TO_CSV 目录；无法识别的文件跳过并记入 df.attrs["skipped_files"]。

    docID 默认取自文件名（下载时命名为 {docID}_{type}.zip）。zip 内容只读入内存，不解压到磁盘。
    """
    zip_path = Path(zip_path)
    doc_id = doc_id or zip_path.stem.split("_")[0]
    frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX")]
        in_folder = [n for n in names if "XBRL_TO_CSV/" in n.replace("\\", "/")]
        for name in in_folder or names:
            base = name.replace("\\", "/").rsplit("/", 1)[-1]
            try:
                part = parse_csv_bytes(z.read(name), doc_id, base)
            except (csv.Error, UnicodeError) as e:
                log.warning("%s/%s: %s", doc_id, base, e)
                part = None
            if part is None:
                skipped.append(base)
            elif not part.empty:
                frames.append(part)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    df.attrs["skipped_files"] = skipped
    return df
