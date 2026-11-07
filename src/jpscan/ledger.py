"""SQLite 账本：运行记录、原始文件、完整性、证据、AI 调用、复核、参数版本（规划文档 16.1 节）。"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .util.common import iso_utc, stable_json

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, run_type TEXT, started_at TEXT, finished_at TEXT,
  code_version TEXT, param_version TEXT, model_version TEXT, snapshot_id TEXT,
  status TEXT, output_checksum TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS raw_files (
  path TEXT PRIMARY KEY, source TEXT, dataset TEXT, params_json TEXT, sha256 TEXT,
  bytes INTEGER, fetched_at TEXT, http_status INTEGER, n_rows INTEGER);
CREATE INDEX IF NOT EXISTS ix_raw_ds ON raw_files(source, dataset);
CREATE TABLE IF NOT EXISTS completeness (
  dataset TEXT, key TEXT, status TEXT, attempts INTEGER DEFAULT 0, last_error TEXT,
  n_rows INTEGER, updated_at TEXT, PRIMARY KEY(dataset, key));
CREATE TABLE IF NOT EXISTS data_snapshots (
  snapshot_id TEXT PRIMARY KEY, created_at TEXT, manifest_path TEXT, n_files INTEGER, total_bytes INTEGER, note TEXT);
CREATE TABLE IF NOT EXISTS evidence (
  evidence_id TEXT PRIMARY KEY, source_type TEXT, doc_id TEXT, url TEXT, disclosed_at TEXT, fetched_at TEXT,
  locator_json TEXT, snippet TEXT, snippet_lang TEXT, content_hash TEXT, internal_id TEXT, created_at TEXT);
CREATE INDEX IF NOT EXISTS ix_ev_internal ON evidence(internal_id);
CREATE TABLE IF NOT EXISTS ai_calls (
  call_id TEXT PRIMARY KEY, task TEXT, model TEXT, prompt_version TEXT, cache_key TEXT, input_tokens INTEGER,
  output_tokens INTEGER, cost_usd REAL, duration_ms INTEGER, status TEXT, created_at TEXT, run_id TEXT, internal_id TEXT);
CREATE TABLE IF NOT EXISTS review_log (
  review_id TEXT PRIMARY KEY, candidate_id TEXT, run_id TEXT, layer TEXT, verdict TEXT, codes TEXT,
  evidence_ids TEXT, reviewer TEXT, created_at TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS param_versions (
  version TEXT PRIMARY KEY, file_hash TEXT, released_at TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS data_fix_log (
  fix_id TEXT PRIMARY KEY, dataset TEXT, description TEXT, affected_from TEXT, affected_to TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS regime_events (
  event_id TEXT PRIMARY KEY, event_date TEXT, event_type TEXT, description TEXT, impact TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT, updated_at TEXT);
"""


class Ledger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def conn(self) -> Iterator[sqlite3.Connection]:
        c = sqlite3.connect(self.path, timeout=60)
        try:
            c.execute("PRAGMA journal_mode=WAL")
            yield c
            c.commit()
        finally:
            c.close()

    # ---- kv ----
    def set_kv(self, k: str, v: Any) -> None:
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO kv(k, v, updated_at) VALUES(?,?,?)", (k, stable_json(v), iso_utc()))

    def get_kv(self, k: str, default: Any = None) -> Any:
        with self.conn() as c:
            row = c.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else default

    # ---- runs ----
    def start_run(self, run_type: str, code_version: str, param_version: str, model_version: str | None = None,
                  note: str | None = None) -> str:
        run_id = f"{run_type}-{iso_utc().replace(':', '').replace('-', '')}-{uuid.uuid4().hex[:6]}"
        with self.conn() as c:
            c.execute("INSERT INTO runs(run_id, run_type, started_at, code_version, param_version, model_version, status, note)"
                      " VALUES(?,?,?,?,?,?,?,?)", (run_id, run_type, iso_utc(), code_version, param_version, model_version, "running", note))
        return run_id

    def finish_run(self, run_id: str, status: str = "ok", output_checksum: str | None = None,
                   snapshot_id: str | None = None, note: str | None = None) -> None:
        with self.conn() as c:
            c.execute("UPDATE runs SET finished_at=?, status=?, output_checksum=COALESCE(?, output_checksum),"
                      " snapshot_id=COALESCE(?, snapshot_id), note=COALESCE(?, note) WHERE run_id=?",
                      (iso_utc(), status, output_checksum, snapshot_id, note, run_id))

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.conn() as c:
            c.row_factory = sqlite3.Row
            row = c.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    # ---- raw files ----
    def record_raw(self, path: Path, source: str, dataset: str, params: dict[str, Any] | None, sha256: str,
                   nbytes: int, http_status: int | None = None, n_rows: int | None = None) -> None:
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO raw_files(path, source, dataset, params_json, sha256, bytes, fetched_at, http_status, n_rows)"
                      " VALUES(?,?,?,?,?,?,?,?,?)",
                      (str(path), source, dataset, stable_json(params or {}), sha256, nbytes, iso_utc(), http_status, n_rows))

    def raw_exists_hash(self, sha256: str) -> bool:
        with self.conn() as c:
            return c.execute("SELECT 1 FROM raw_files WHERE sha256=? LIMIT 1", (sha256,)).fetchone() is not None

    # ---- completeness ----
    def mark(self, dataset: str, key: str, status: str, n_rows: int | None = None, error: str | None = None) -> None:
        with self.conn() as c:
            row = c.execute("SELECT attempts FROM completeness WHERE dataset=? AND key=?", (dataset, key)).fetchone()
            attempts = (row[0] if row else 0) + 1
            c.execute("INSERT OR REPLACE INTO completeness(dataset, key, status, attempts, last_error, n_rows, updated_at)"
                      " VALUES(?,?,?,?,?,?,?)", (dataset, key, status, attempts, error, n_rows, iso_utc()))

    def status_of(self, dataset: str, key: str) -> str | None:
        with self.conn() as c:
            row = c.execute("SELECT status FROM completeness WHERE dataset=? AND key=?", (dataset, key)).fetchone()
        return row[0] if row else None

    def done_keys(self, dataset: str) -> set[str]:
        with self.conn() as c:
            rows = c.execute("SELECT key FROM completeness WHERE dataset=? AND status='ok'", (dataset,)).fetchall()
        return {r[0] for r in rows}

    def completeness_summary(self) -> list[dict[str, Any]]:
        with self.conn() as c:
            rows = c.execute("SELECT dataset, status, COUNT(*) n, SUM(COALESCE(n_rows,0)) rows FROM completeness"
                             " GROUP BY dataset, status ORDER BY dataset, status").fetchall()
        return [{"dataset": r[0], "status": r[1], "n": r[2], "rows": r[3]} for r in rows]

    # ---- evidence ----
    def add_evidence(self, evidence_id: str, source_type: str, doc_id: str | None, url: str | None, disclosed_at: str | None,
                     locator: dict[str, Any] | None, snippet: str, snippet_lang: str, content_hash: str,
                     internal_id: str | None) -> None:
        with self.conn() as c:
            c.execute("INSERT OR IGNORE INTO evidence(evidence_id, source_type, doc_id, url, disclosed_at, fetched_at, locator_json,"
                      " snippet, snippet_lang, content_hash, internal_id, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                      (evidence_id, source_type, doc_id, url, disclosed_at, iso_utc(), stable_json(locator or {}), snippet,
                       snippet_lang, content_hash, internal_id, iso_utc()))

    def evidence_exists(self, evidence_id: str) -> bool:
        with self.conn() as c:
            return c.execute("SELECT 1 FROM evidence WHERE evidence_id=?", (evidence_id,)).fetchone() is not None

    def get_evidence(self, evidence_id: str) -> dict[str, Any] | None:
        with self.conn() as c:
            c.row_factory = sqlite3.Row
            row = c.execute("SELECT * FROM evidence WHERE evidence_id=?", (evidence_id,)).fetchone()
        return dict(row) if row else None

    # ---- ai calls ----
    def add_ai_call(self, **kw: Any) -> None:
        cols = ["call_id", "task", "model", "prompt_version", "cache_key", "input_tokens", "output_tokens", "cost_usd",
                "duration_ms", "status", "created_at", "run_id", "internal_id"]
        kw.setdefault("created_at", iso_utc())
        kw.setdefault("call_id", uuid.uuid4().hex)
        with self.conn() as c:
            c.execute(f"INSERT OR REPLACE INTO ai_calls({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
                      tuple(kw.get(k) for k in cols))

    # ---- review ----
    def add_review(self, candidate_id: str, run_id: str, layer: str, verdict: str, codes: list[str],
                   evidence_ids: list[str], reviewer: str, note: str | None = None) -> None:
        with self.conn() as c:
            c.execute("INSERT INTO review_log(review_id, candidate_id, run_id, layer, verdict, codes, evidence_ids, reviewer, created_at, note)"
                      " VALUES(?,?,?,?,?,?,?,?,?,?)",
                      (uuid.uuid4().hex, candidate_id, run_id, layer, verdict, json.dumps(codes), json.dumps(evidence_ids),
                       reviewer, iso_utc(), note))

    # ---- snapshots ----
    def add_snapshot(self, snapshot_id: str, manifest_path: Path, n_files: int, total_bytes: int, note: str | None = None) -> None:
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO data_snapshots(snapshot_id, created_at, manifest_path, n_files, total_bytes, note)"
                      " VALUES(?,?,?,?,?,?)", (snapshot_id, iso_utc(), str(manifest_path), n_files, total_bytes, note))

    def add_regime_event(self, event_id: str, event_date: str, event_type: str, description: str, impact: str, source: str) -> None:
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO regime_events VALUES(?,?,?,?,?,?)",
                      (event_id, event_date, event_type, description, impact, source))
