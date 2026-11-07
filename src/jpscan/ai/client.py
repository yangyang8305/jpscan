"""Claude API 封装（规划文档 5.4、13 章）：结构化输出（JSON Schema）、永久缓存、账本记录、成本估算。

凭据：使用 Anthropic SDK 的默认解析（环境变量 ANTHROPIC_API_KEY，或 `ant auth login` 的本地 profile）。
无凭据时 `available` 为 False，调用返回 None，管线退化为纯规则结果。
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from ..ledger import Ledger
from ..util.common import iso_utc, sha256_bytes, stable_json

log = logging.getLogger("jpscan.ai")

# 参考价格（USD / 百万 token，2026-10 缓存值；实现时以官方价格页为准）
PRICES = {
    "claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50),
    "claude-fable-5-1": (10.0, 50.0), "claude-opus-5": (5.0, 25.0),
}


class ClaudeClient:
    def __init__(self, ledger: Ledger, cache_dir: Path, enabled: bool = True):
        self.ledger = ledger
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = None
        self.available = False
        if enabled:
            try:
                import anthropic  # noqa: WPS433
                self.client = anthropic.Anthropic()
                self.available = True
            except Exception as ex:  # noqa: BLE001
                log.warning("Anthropic client unavailable: %s", ex)

    def cache_key(self, task: str, prompt_version: str, model: str, payload: Any) -> str:
        return sha256_bytes(f"{task}|{prompt_version}|{model}|{stable_json(payload)}".encode("utf-8"))[:32]

    def structured(self, *, task: str, prompt_version: str, model: str, system: str, user: str, schema: dict[str, Any],
                   payload_for_key: Any, effort: str = "high", max_tokens: int = 8000, run_id: str | None = None,
                   internal_id: str | None = None, use_cache: bool = True) -> dict[str, Any] | None:
        key = self.cache_key(task, prompt_version, model, payload_for_key)
        cpath = self.cache_dir / f"{key}.json"
        if use_cache and cpath.exists():
            try:
                return json.loads(cpath.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                pass
        if not self.available or self.client is None:
            return None
        t0 = time.time()
        status = "ok"
        data: dict[str, Any] | None = None
        usage_in = usage_out = 0
        try:
            resp = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
            )
            usage_in = getattr(resp.usage, "input_tokens", 0) or 0
            usage_out = getattr(resp.usage, "output_tokens", 0) or 0
            if getattr(resp, "stop_reason", None) == "refusal":
                status = "refusal"
            else:
                text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), None)
                if text is None:
                    status = "no_text"
                else:
                    data = json.loads(text)
                    data["_meta"] = {"model": model, "prompt_version": prompt_version, "task": task, "cache_key": key,
                                     "input_tokens": usage_in, "output_tokens": usage_out, "created_at": iso_utc()}
                    cpath.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception as ex:  # noqa: BLE001
            status = f"error: {str(ex)[:200]}"
            log.exception("AI call failed (%s)", task)
        pin, pout = PRICES.get(model, (0.0, 0.0))
        cost = usage_in / 1e6 * pin + usage_out / 1e6 * pout
        self.ledger.add_ai_call(call_id=uuid.uuid4().hex, task=task, model=model, prompt_version=prompt_version, cache_key=key,
                                input_tokens=usage_in, output_tokens=usage_out, cost_usd=cost, duration_ms=int((time.time() - t0) * 1000),
                                status=status, run_id=run_id, internal_id=internal_id)
        return data
