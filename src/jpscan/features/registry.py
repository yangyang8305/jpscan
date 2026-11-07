"""读取 config/metric_registry.yaml：指标、证据家族、门槛、篮子的唯一来源。

代码只绑定「公式」（features 里与指标同名的列）；家族由哪些指标组成、阈值是多少、篮子要什么，都在运行时从这里读。
增删一个 score 指标只需要改注册表并提供同名特征列。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from ..config import find_project_root, pget


def registry_path(params: dict | None = None) -> Path:
    p = pget(params or {}, "evidence.registry_path", None)
    root = find_project_root(Path(__file__).resolve())
    if p:
        q = Path(p)
        return q if q.is_absolute() else root / q
    return root / "config" / "metric_registry.yaml"


@lru_cache(maxsize=4)
def _load(path: str, mtime: float) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    y["metrics_by_id"] = {m["id"]: m for m in y.get("metrics", [])}
    return y


def load_registry(params: dict | None = None) -> dict[str, Any]:
    path = registry_path(params)
    return _load(str(path), path.stat().st_mtime)


def hit_rule(reg: dict, family: str, metric: str) -> dict:
    for r in reg["families"][family]["threshold"]["hit_rules"]:
        if r["metric"] == metric:
            return r
    return {}


def metric_family(reg: dict, metric: str) -> str | None:
    for fam, cfg in reg["families"].items():
        if metric in cfg.get("metrics", []):
            return fam
    return None
