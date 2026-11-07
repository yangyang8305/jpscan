"""通用工具：时间（JST/UTC）、哈希、日志、限速、JSON。"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import threading
import time
from collections import deque
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

JST = timezone(timedelta(hours=9), name="JST")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_jst() -> datetime:
    return datetime.now(JST)


def today_jst() -> date:
    return now_jst().date()


def iso_utc(dt: datetime | None = None) -> str:
    return (dt or now_utc()).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_date(s: str | date | datetime | None) -> date | None:
    if s is None or s == "":
        return None
    if isinstance(s, datetime):
        return s.date()
    if isinstance(s, date):
        return s
    return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            blk = f.read(chunk)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


def stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def params_hash(params: dict[str, Any] | None) -> str:
    return sha256_bytes(stable_json(params or {}).encode("utf-8"))[:12]


def write_json_gz(path: Path, obj: Any) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with gzip.open(path, "wb") as f:
        f.write(data)
    return len(data)


def read_json_gz(path: Path) -> Any:
    with gzip.open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def setup_logging(level: int = logging.INFO, logfile: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("jpscan")
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if logfile:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class RateLimiter:
    """匀速限速：两次请求间隔 >= 60/rpm × margin；线程安全。

    J-Quants 的滑动窗口对突发敏感（实测 5 连发后频繁 429），因此用匀速而不是窗口内突发。
    收到 429 时 penalize()：暂停一段时间，并小幅放大间隔（自适应）。
    """

    def __init__(self, rpm: int, margin: float = 1.08):
        self.rpm = max(1, int(rpm))
        self.gap = 60.0 / self.rpm * margin
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                if now >= self._next:
                    self._next = now + self.gap
                    return
                sleep_for = self._next - now
            time.sleep(min(sleep_for, 5.0))

    def penalize(self, seconds: float = 61.0, grow: float = 1.05) -> None:
        with self._lock:
            self.gap *= grow
            self._next = max(self._next, time.monotonic() + seconds)


def chunks(items: Iterable[Any], n: int) -> Iterable[list[Any]]:
    buf: list[Any] = []
    for it in items:
        buf.append(it)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf
