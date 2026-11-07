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


# ---------------- 关键科目 ----------------
# 要素名已对照 2026 年版 EDINET タクソノミ（勘定科目リスト 1f、IFRS 要素リスト 1g、要素リスト 1e，附录 G1-32）。
# 同一科目的多个要素按优先级排列，取文件中最先命中的一个。
ITEM_ELEMENTS: dict[str, tuple[str, ...]] = {
    # 损益
    "net_sales": ("jppfs_cor:NetSales", "jppfs_cor:OperatingRevenue1", "jppfs_cor:OperatingRevenue2", "jpigp_cor:RevenueIFRS",
                  "jpigp_cor:NetSalesIFRS", "jpigp_cor:Revenue2IFRS", "jpcrp_cor:NetSalesSummaryOfBusinessResults",
                  "jpcrp_cor:RevenueIFRSSummaryOfBusinessResults"),
    "operating_income": ("jppfs_cor:OperatingIncome", "jpigp_cor:OperatingProfitLossIFRS"),
    "interest_expense": ("jppfs_cor:InterestExpensesNOE", "jppfs_cor:InterestExpensesOpeCF", "jpigp_cor:InterestExpensesIFRS",
                         "jpigp_cor:InterestExpensesOpeCFIFRS", "jpigp_cor:FinanceCostsIFRS"),
    # 现金流（符号保持原文件：取得支出通常为负）
    "depreciation": ("jppfs_cor:DepreciationAndAmortizationOpeCF", "jpigp_cor:DepreciationAndAmortizationOpeCFIFRS",
                     "jpigp_cor:DepreciationExpenseOpeCFIFRS"),
    "capex": ("jppfs_cor:PurchaseOfPropertyPlantAndEquipmentInvCF", "jpigp_cor:PurchaseOfPropertyPlantAndEquipmentInvCFIFRS"),
    # 流动资产
    "cash_and_deposits": ("jppfs_cor:CashAndDeposits",),
    "cash_equivalents": ("jppfs_cor:CashAndCashEquivalents", "jpigp_cor:CashAndCashEquivalentsIFRS",
                         "jpcrp_cor:CashAndCashEquivalentsSummaryOfBusinessResults"),
    "short_term_securities": ("jppfs_cor:ShortTermInvestmentSecurities",),
    "receivables_trade": ("jppfs_cor:NotesAndAccountsReceivableTrade", "jppfs_cor:NotesAndAccountsReceivableTradeAndContractAssets",
                          "jpigp_cor:TradeAndOtherReceivablesCAIFRS"),
    "notes_receivable_trade": ("jppfs_cor:NotesReceivableTrade",),
    "accounts_receivable_trade": ("jppfs_cor:AccountsReceivableTrade",),
    "inventories": ("jppfs_cor:Inventories", "jpigp_cor:InventoriesCAIFRS"),
    "merchandise_finished_goods": ("jppfs_cor:MerchandiseAndFinishedGoods",),
    "work_in_process": ("jppfs_cor:WorkInProcess",),
    "raw_materials_supplies": ("jppfs_cor:RawMaterialsAndSupplies",),
    # 有利子負債 构成项
    "short_term_loans": ("jppfs_cor:ShortTermLoansPayable", "jpigp_cor:BorrowingsCLIFRS", "jpigp_cor:BondsAndBorrowingsCLIFRS",
                         "jpigp_cor:InterestBearingLiabilitiesCLIFRS"),
    "commercial_paper": ("jppfs_cor:CommercialPapersLiabilities",),
    "current_lt_loans": ("jppfs_cor:CurrentPortionOfLongTermLoansPayable", "jpigp_cor:CurrentPortionOfLongTermBorrowingsCLIFRS"),
    "current_bonds": ("jppfs_cor:CurrentPortionOfBonds", "jpigp_cor:BondsPayableCLIFRS"),
    "bonds": ("jppfs_cor:BondsPayable", "jpigp_cor:BondsPayableNCLIFRS"),
    "long_term_loans": ("jppfs_cor:LongTermLoansPayable", "jpigp_cor:BorrowingsNCLIFRS", "jpigp_cor:BondsAndBorrowingsNCLIFRS",
                        "jpigp_cor:InterestBearingLiabilitiesNCLIFRS"),
    "lease_cl": ("jppfs_cor:LeaseObligationsCL", "jpigp_cor:LeaseLiabilitiesCLIFRS"),
    "lease_ncl": ("jppfs_cor:LeaseObligationsNCL", "jpigp_cor:LeaseLiabilitiesNCLIFRS"),
}
DEBT_COMPONENTS = ("short_term_loans", "commercial_paper", "current_lt_loans", "current_bonds", "bonds", "long_term_loans",
                   "lease_cl", "lease_ncl")
# 合计科目在 BS 上常被拆开列示：缺合计时用构成项相加
FALLBACK_SUMS = {
    "inventories": ("merchandise_finished_goods", "work_in_process", "raw_materials_supplies"),
    "receivables_trade": ("notes_receivable_trade", "accounts_receivable_trade"),
}
KEY_ITEM_COLUMNS = ["Code", "period_end", "disc_date", "consolidated", "item", "value", "docID"]

_ELEMENT_RANK = {el: (item, rank) for item, els in ITEM_ELEMENTS.items() for rank, el in enumerate(els)}
# 当期值：有報 CurrentYear*，半期 Interim*，旧四半期累计 CurrentYTD 与期末 CurrentQuarterInstant。
# 不含 Prior*（前期值）、CurrentQuarterDuration（单季）与分部等其他 Member。
_CONTEXT_RE = re.compile(r"^(CurrentYearInstant|CurrentYearDuration|InterimInstant|InterimDuration|CurrentYTDDuration"
                         r"|CurrentQuarterInstant)(_NonConsolidatedMember)?$")


def extract_key_items(df: pd.DataFrame) -> pd.DataFrame:
    """长表 → 标准科目表 edinet_key_items（只取当期值）。

    df 需含 docID、element_id、context_id、numeric_value；Code、period_end、disc_date（或 submit_date）来自书类元数据，
    由作业附加。没有 Code（非上市提交者）或没有 period_end 的行不输出。派生科目 interest_bearing_debt = DEBT_COMPONENTS 之和。
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=KEY_ITEM_COLUMNS)
    picked = df[df["element_id"].isin(list(_ELEMENT_RANK)) & df["numeric_value"].notna()]
    picked = picked[[bool(_CONTEXT_RE.match(str(c))) for c in picked["context_id"]]]
    disc = "disc_date" if "disc_date" in picked.columns else "submit_date"
    out = pd.DataFrame({
        "Code": picked["Code"] if "Code" in picked.columns else None,
        "period_end": picked["period_end"] if "period_end" in picked.columns else None,
        "disc_date": picked[disc] if disc in picked.columns else None,
        "consolidated": [_is_consolidated(f, c) for f, c in zip(picked.get("consolidated_flag", [None] * len(picked)), picked["context_id"])],
        "item": [_ELEMENT_RANK[e][0] for e in picked["element_id"]],
        "rank": [_ELEMENT_RANK[e][1] for e in picked["element_id"]],
        "value": picked["numeric_value"].astype("float64"),
        "docID": picked["docID"],
    })
    out = out[out["Code"].notna() & out["period_end"].notna()]
    if out.empty:
        return pd.DataFrame(columns=KEY_ITEM_COLUMNS)
    group = ["docID", "consolidated"]
    out = out.sort_values("rank", kind="stable").drop_duplicates(group + ["item"], keep="first").drop(columns="rank")
    derived: list[dict[str, Any]] = []
    for _, g in out.groupby(group, sort=False):
        have = dict(zip(g["item"], g["value"]))
        base = g.iloc[0].to_dict()
        for item, parts in FALLBACK_SUMS.items():
            vals = [have[p] for p in parts if p in have]
            if item not in have and vals:
                derived.append({**base, "item": item, "value": float(sum(vals))})
        debt = [have[p] for p in DEBT_COMPONENTS if p in have]
        if debt:
            derived.append({**base, "item": "interest_bearing_debt", "value": float(sum(debt))})
    if derived:
        out = pd.concat([out, pd.DataFrame(derived)], ignore_index=True)
    return out[KEY_ITEM_COLUMNS].reset_index(drop=True)


def _is_consolidated(flag: Any, context_id: str) -> bool:
    """jppfs 看「連結・個別」列；其他要素该列为「その他」，看コンテキストID 是否带 NonConsolidatedMember。"""
    if flag == "連結":
        return True
    if flag == "個別":
        return False
    return "NonConsolidatedMember" not in str(context_id)
