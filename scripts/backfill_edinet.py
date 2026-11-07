"""EDINET 全量回填：从新到旧逐日采集（可断点续传）。

用法：python scripts/backfill_edinet.py [--from 2016-11-08] [--to 2026-11-06]
默认只下载财务类书类（有報、訂正有報、半期、訂正半期）的 CSV；其他书类只记录元数据。
EDINET 规约禁止短时间大量访问，客户端限速约 1 次/秒，全量约需数十小时，中断后重跑即可续传。
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import timedelta

from jpscan.config import load_settings
from jpscan.jobs import ingest_edinet as E
from jpscan.ledger import Ledger
from jpscan.paths import DataPaths
from jpscan.store import Store
from jpscan.util.common import setup_logging, to_date, today_jst

FINANCIAL_DOC_TYPES = ["120", "130", "160", "170"]


def main() -> None:
    ap = argparse.ArgumentParser()
    today = today_jst()
    ap.add_argument("--from", dest="start", default=(today - timedelta(days=3650 - 2)).isoformat())
    ap.add_argument("--to", dest="end", default=(today - timedelta(days=1)).isoformat())
    args = ap.parse_args()
    settings = load_settings()
    paths = DataPaths(settings.data_dir).ensure()
    setup_logging(logging.INFO, paths.root / "logs" / "edinet_backfill.log")
    ledger, store = Ledger(paths.ledger_sqlite), Store(paths)
    if not store.exists("staging", "edinet_codes"):
        E.run_code_list(settings, paths, ledger, store)
    stats = E.run(settings, paths, ledger, store, to_date(args.start), to_date(args.end),
                  doc_types=FINANCIAL_DOC_TYPES, download_types="5", newest_first=True)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
