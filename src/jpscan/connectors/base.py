"""连接器通用能力：原始层落盘（不可变 + 哈希 + 账本）、HTTP 会话。"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import requests

from ..ledger import Ledger
from ..paths import DataPaths
from ..util.common import iso_utc, params_hash, read_json_gz, sha256_bytes, write_json_gz

log = logging.getLogger("jpscan.connectors")


class RawSink:
    """把来源返回的原始内容原样落盘并登记。同内容（哈希）重复拉取不重复写文件。"""

    def __init__(self, paths: DataPaths, ledger: Ledger, source: str):
        self.paths = paths
        self.ledger = ledger
        self.source = source

    def save_json(self, dataset: str, key: str, params: dict[str, Any] | None, payload: Any,
                  http_status: int | None = None, n_rows: int | None = None, page: int = 0) -> Path:
        path = self.raw_json_path(dataset, key, params, page)
        nbytes = write_json_gz(path, payload)
        digest = sha256_bytes(open(path, "rb").read())
        self.ledger.record_raw(path, self.source, dataset, params, digest, nbytes, http_status, n_rows)
        return path

    def raw_json_path(self, dataset: str, key: str, params: dict[str, Any] | None, page: int = 0) -> Path:
        year = key[:4] if key[:4].isdigit() else "misc"
        name = f"{key}_{params_hash(params)}" + (f"_p{page}" if page else "") + ".json.gz"
        return self.paths.raw_dir(self.source, dataset) / year / name

    def load_json(self, dataset: str, key: str, params: dict[str, Any] | None, page: int = 0) -> Any | None:
        path = self.raw_json_path(dataset, key, params, page)
        if not path.exists():
            return None
        try:
            return read_json_gz(path)
        except Exception:  # noqa: BLE001
            return None

    def save_bytes(self, dataset: str, key: str, content: bytes, ext: str, params: dict[str, Any] | None = None,
                   http_status: int | None = None, subdir: str | None = None) -> Path:
        year = key[:4] if key[:4].isdigit() else "misc"
        d = self.paths.raw_dir(self.source, dataset) / (subdir or year)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{key}{ext}"
        if path.exists() and sha256_bytes(open(path, "rb").read()) == sha256_bytes(content):
            return path
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(content)
        tmp.replace(path)
        self.ledger.record_raw(path, self.source, dataset, params, sha256_bytes(content), len(content), http_status, None)
        return path

    def quarantine(self, dataset: str, key: str, content: bytes, reason: str) -> Path:
        d = self.paths.quarantine / self.source / dataset
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{key}_{iso_utc().replace(':', '')}.bin"
        path.write_bytes(content)
        (path.with_suffix(".reason.txt")).write_text(reason, encoding="utf-8")
        log.warning("quarantined %s/%s: %s", dataset, key, reason)
        return path


def http_session(user_agent: str = "jpscan/0.1 (personal research; contact: local)") -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent, "Accept": "*/*"})
    return s


def get_with_retry(session: requests.Session, url: str, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None,
                   timeout: int = 120, retries: int = 4, backoff: float = 2.0) -> requests.Response:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code >= 500 or r.status_code in (408,):
                raise requests.HTTPError(f"server error {r.status_code}")
            return r
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:  # noqa: PERF203
            last = e
            wait = backoff ** attempt
            log.warning("GET %s failed (%s); retry in %.0fs", url, e, wait)
            time.sleep(wait)
    assert last is not None
    raise last
