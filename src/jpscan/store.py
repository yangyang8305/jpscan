"""存储层：Parquet 文件湖（按表/分区）+ DuckDB 查询（规划文档 9.2、10.2 节）。"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Iterable

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .paths import DataPaths

_SAFE = re.compile(r"[^A-Za-z0-9_\-=]")


_DATEISH = re.compile(r"(Date|date|FYSt|FYEn|PerSt|PerEn)")


def _safe(s: str) -> str:
    return _SAFE.sub("_", str(s))


def _fix_null_types(tbl: pa.Table) -> pa.Table:
    """全空列在 Arrow 中是 null 类型，跨分区 union 会冲突：日期类转 date32，其余转 string。"""
    for i, field in enumerate(tbl.schema):
        if pa.types.is_null(field.type):
            target = pa.date32() if _DATEISH.search(field.name) else pa.string()
            tbl = tbl.set_column(i, pa.field(field.name, target), pa.nulls(len(tbl), target))
    return tbl


class Store:
    """表 = 目录；分区 = 文件名（例如 prices_daily/2026-08.parquet）。写入同一分区即覆盖（幂等）。"""

    def __init__(self, paths: DataPaths):
        self.paths = paths

    # ---------- write ----------
    def write(self, layer: str, table: str, df: pd.DataFrame, partition: str) -> Path:
        base = self.paths.staging_dir(table) if layer == "staging" else self.paths.mart_dir(table)
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"{_safe(partition)}.parquet"
        tbl = _fix_null_types(pa.Table.from_pandas(df.reset_index(drop=True), preserve_index=False))
        tmp = path.with_suffix(f".parquet.{os.getpid()}.tmp")
        pq.write_table(tbl, tmp, compression="zstd")
        # Windows：读者持有文件时 replace 会 PermissionError，短暂重试
        for attempt in range(40):
            try:
                tmp.replace(path)
                break
            except PermissionError:
                if attempt == 39:
                    raise
                time.sleep(0.5)
        return path

    def write_merge(self, layer: str, table: str, df: pd.DataFrame, partition: str, keys: list[str]) -> Path:
        """与已有分区按主键合并（新记录覆盖旧记录），用于增量写入同一分区。"""
        base = self.paths.staging_dir(table) if layer == "staging" else self.paths.mart_dir(table)
        path = base / f"{_safe(partition)}.parquet"
        if path.exists():
            old = pd.read_parquet(path)
            df = pd.concat([old, df], ignore_index=True)
            df = df.drop_duplicates(subset=keys, keep="last")
        return self.write(layer, table, df, partition)

    # ---------- read ----------
    def table_dir(self, layer: str, table: str) -> Path:
        return self.paths.staging_dir(table) if layer == "staging" else self.paths.mart_dir(table)

    def exists(self, layer: str, table: str) -> bool:
        d = self.table_dir(layer, table)
        return d.exists() and any(d.glob("*.parquet"))

    def partitions(self, layer: str, table: str) -> list[str]:
        d = self.table_dir(layer, table)
        return sorted(p.stem for p in d.glob("*.parquet")) if d.exists() else []

    def read(self, layer: str, table: str, where: str | None = None, columns: Iterable[str] | None = None,
             partitions: Iterable[str] | None = None) -> pd.DataFrame:
        d = self.table_dir(layer, table)
        if not d.exists():
            return pd.DataFrame()
        files = [d / f"{_safe(p)}.parquet" for p in partitions] if partitions else sorted(d.glob("*.parquet"))
        files = [f for f in files if f.exists()]
        if not files:
            return pd.DataFrame()
        cols = ", ".join(f'"{c}"' for c in columns) if columns else "*"
        file_list = ", ".join("'" + str(f).replace("'", "''") + "'" for f in files)
        sql = f"SELECT {cols} FROM read_parquet([{file_list}], union_by_name=true)"
        if where:
            sql += f" WHERE {where}"
        con = duckdb.connect()
        try:
            return con.execute(sql).df()
        finally:
            con.close()

    def query(self, sql: str, views: dict[str, tuple[str, str]] | None = None) -> pd.DataFrame:
        """执行任意 SQL；views = {别名: (layer, table)} 自动注册为视图。"""
        con = duckdb.connect()
        try:
            for alias, (layer, table) in (views or {}).items():
                d = self.table_dir(layer, table)
                glob = str(d / "*.parquet").replace("'", "''")
                if d.exists() and any(d.glob("*.parquet")):
                    con.execute(f"CREATE VIEW {alias} AS SELECT * FROM read_parquet('{glob}', union_by_name=true)")
            return con.execute(sql).df()
        finally:
            con.close()

    def all_views(self) -> dict[str, tuple[str, str]]:
        out: dict[str, tuple[str, str]] = {}
        for layer, root in (("staging", self.paths.staging), ("mart", self.paths.mart)):
            if root.exists():
                for d in root.iterdir():
                    if d.is_dir() and any(d.glob("*.parquet")):
                        out[d.name] = (layer, d.name)
        return out

    def manifest(self) -> list[dict[str, Any]]:
        """快照清单：所有 Parquet 文件的路径、大小、修改时间（哈希由 snapshot 模块计算）。"""
        rows = []
        for root in (self.paths.staging, self.paths.mart):
            if root.exists():
                for f in sorted(root.rglob("*.parquet")):
                    st = f.stat()
                    rows.append({"path": str(f.relative_to(self.paths.root)), "bytes": st.st_size, "mtime": st.st_mtime})
        return rows
