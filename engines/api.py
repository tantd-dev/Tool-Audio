"""
LM Studio and OpenAI-compatible API engine for audio model benchmark.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from core.audio import prepare_api_audio
from core.exceptions import MethodError


def http_json(method: str, url: str, payload: dict | None = None, timeout: float | None = 30) -> dict:
    """Perform HTTP request sending and receiving JSON."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach LM Studio at {url} ({exc.reason}). Is the server started?") from None


def fetch_api_models(base: str) -> list[dict[str, Any]]:
    """Fetch active models from LM Studio server via /api/v0/models or /v1/models."""
    base = base.rstrip("/")
    try:
        items = http_json("GET", f"{base}/api/v0/models").get("data", [])
    except RuntimeError as exc:
        if "Cannot reach" in str(exc):
            raise
        items = http_json("GET", f"{base}/v1/models").get("data", [])
    return [
        {"id": m.get("id", ""), "state": m.get("state", "unknown"), "type": m.get("type", "")}
        for m in items if m.get("type") != "embeddings"
    ]


def run_api(base: str, api_id: str, audio: str, prompt: str, max_tokens: int) -> tuple[str, dict]:
    """
    Run audio analysis via LM Studio chat completions API.
    Raises MethodError with details if API call fails.
    """
    if not api_id:
        raise MethodError("No LM Studio API id matched this model (click 'Check API' or set one via Edit).")

    try:
        b64, fmt = prepare_api_audio(audio)
        payload = {
            "model": api_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}},
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "stream": False,
        }
        resp = http_json("POST", f"{base.rstrip('/')}/v1/chat/completions", payload, timeout=None)
    except RuntimeError as exc:
        err_msg = str(exc)
        extra: dict[str, Any] = {"api_model_id": api_id}
        if "must have a 'type' field that is either 'text' or 'image_url'" in err_msg:
            extra["hint"] = (
                "LM Studio server does not support audio inputs in /v1/chat/completions "
                "(it only accepts 'text' and 'image_url'). Falling back to CLI if configured."
            )
        raise MethodError(err_msg, extra=extra) from None

    choice = (resp.get("choices") or [{}])[0]
    raw = ((choice.get("message") or {}).get("content") or "").strip()
    extra = {
        "finish_reason": choice.get("finish_reason"),
        "usage": resp.get("usage"),
        "lmstudio_stats": resp.get("stats"),
        "audio_format_sent": fmt,
        "api_model_id": api_id,
    }
    if not raw:
        raise MethodError("LM Studio returned an empty response.", extra)
    return raw, extra
