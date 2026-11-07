"""配置加载：.env（密钥）、params.vX.yaml（参数与阈值）、数据目录。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def find_project_root(start: Path | None = None) -> Path:
    """向上查找含 pyproject.toml 的目录；找不到则用当前目录。"""
    p = (start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "pyproject.toml").exists():
            return cand
    env_root = os.environ.get("JPSCAN_ROOT")
    return Path(env_root).resolve() if env_root else p


@dataclass
class Settings:
    root: Path
    data_dir: Path
    params_path: Path
    jquants_api_key: str | None
    anthropic_api_key: str | None
    edinet_api_key: str | None
    fred_api_key: str | None
    jquants_rpm: int = 5                      # 每分钟请求上限（按套餐自动调整）
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def params_version(self) -> str:
        return str(self.params.get("version", "unknown"))


def load_params(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_settings(root: Path | None = None, params_path: Path | None = None) -> Settings:
    root = root or find_project_root()
    load_dotenv(root / ".env", override=False)
    data_dir = Path(os.environ.get("JPSCAN_DATA_DIR", root / "data")).resolve()
    params_path = Path(params_path or os.environ.get("JPSCAN_PARAMS", root / "docs" / "params.v0.yaml"))
    params = load_params(params_path) if params_path.exists() else {}
    rpm_env = os.environ.get("JQUANTS_RPM")
    plan = os.environ.get("JQUANTS_PLAN") or (params.get("data_sources", {}).get("jquants", {}) or {}).get("plan", "Free")
    plan_rpm = {"Free": 5, "Light": 60, "Standard": 120, "Premium": 500}
    rpm = int(rpm_env) if rpm_env else plan_rpm.get(str(plan), 5)
    params.setdefault("data_sources", {}).setdefault("jquants", {})["plan"] = str(plan)
    return Settings(
        root=root,
        data_dir=data_dir,
        params_path=params_path,
        jquants_api_key=os.environ.get("JQUANTS_API_KEY"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        edinet_api_key=os.environ.get("EDINET_API_KEY"),
        fred_api_key=os.environ.get("FRED_API_KEY"),
        jquants_rpm=rpm,
        params=params,
    )


def pget(params: dict[str, Any], dotted: str, default: Any = None) -> Any:
    """按 'value.core_metric_pct' 这样的点路径取参数。"""
    cur: Any = params
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
