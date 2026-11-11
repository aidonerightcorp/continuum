"""Benchmark meter. It runs in every arm, with Continuum on or off, so all arms are measured the same way.

It writes one JSON line per model call that Hermes makes to the file named by BENCH_METER_FILE: the model that
answered, the token counts the endpoint reported, and the seconds the call took. It reads no message text.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

_lock = threading.Lock()


def _usage(usage: Any) -> dict[str, int]:
    out = {}
    for key in ("prompt_tokens", "completion_tokens", "input_tokens", "output_tokens", "cache_read_tokens"):
        value = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
        if isinstance(value, (int, float)):
            out[key] = int(value)
    return out


def _write(row: dict[str, Any]) -> None:
    path = os.environ.get("BENCH_METER_FILE")
    if not path:
        return
    row["ts"] = round(time.time(), 3)
    row["item"] = os.environ.get("BENCH_METER_ITEM", "")
    with _lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def register(ctx: Any) -> None:
    def post_api_request(model: str = "", response_model: Any = None, usage: Any = None, api_duration: float = 0.0,
                         finish_reason: str = "", assistant_tool_call_count: int = 0, **_: Any) -> None:
        _write({"kind": "turn", "ok": True, "asked": str(model or ""), "model": str(response_model or model or ""),
                "usage": _usage(usage), "seconds": round(float(api_duration or 0.0), 3),
                "finish": str(finish_reason or ""), "tool_calls": int(assistant_tool_call_count or 0)})

    def api_request_error(model: str = "", request: Any = None, api_duration: float = 0.0, **_: Any) -> None:
        sent = request.get("model") if isinstance(request, dict) else None
        _write({"kind": "turn", "ok": False, "asked": str(model or ""), "model": str(sent or model or ""),
                "usage": {}, "seconds": round(float(api_duration or 0.0), 3)})

    def post_auxiliary_call(aux_task: str = "", model: str = "", response_model: Any = None, usage: Any = None,
                            duration: float = 0.0, api_duration: float = 0.0, error: Any = None, **_: Any) -> None:
        _write({"kind": "aux", "task": str(aux_task or ""), "ok": error is None,
                "model": str(response_model or model or ""), "usage": _usage(usage),
                "seconds": round(float(duration or api_duration or 0.0), 3)})

    ctx.register_hook("post_api_request", post_api_request)
    ctx.register_hook("api_request_error", api_request_error)
    ctx.register_hook("post_auxiliary_call", post_auxiliary_call)
