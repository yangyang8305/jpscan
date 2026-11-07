"""数据目录布局（规划文档 10.2 节）。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def quarantine(self) -> Path:
        return self.root / "raw" / "_quarantine"

    @property
    def staging(self) -> Path:
        return self.root / "staging"

    @property
    def mart(self) -> Path:
        return self.root / "mart"

    @property
    def db(self) -> Path:
        return self.root / "db"

    @property
    def snapshots(self) -> Path:
        return self.root / "snapshots"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def ledger_sqlite(self) -> Path:
        return self.db / "ledger.sqlite"

    @property
    def warehouse_duckdb(self) -> Path:
        return self.db / "warehouse.duckdb"

    @property
    def ai_cache(self) -> Path:
        return self.root / "ai_cache"

    def raw_dir(self, source: str, dataset: str) -> Path:
        return self.raw / source / dataset

    def staging_dir(self, table: str) -> Path:
        return self.staging / table

    def mart_dir(self, table: str) -> Path:
        return self.mart / table

    def ensure(self) -> "DataPaths":
        for p in [self.raw, self.quarantine, self.staging, self.mart, self.db, self.snapshots, self.reports, self.ai_cache]:
            p.mkdir(parents=True, exist_ok=True)
        return self
