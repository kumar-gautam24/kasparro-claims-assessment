"""Ollama adapter via the native /api/chat endpoint with tool calling (local model; a disclosed
deviation from the brief's hosted-provider requirement)."""
from __future__ import annotations

import json
import os

import httpx

from app.providers.base import ProviderError, ToolCall, Turn, classify_http_status


class OllamaProvider:
    name = "ollama"

    def __init__(self, model: str, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "16384"))

    @staticmethod
    def _messages(system: str, messages: list[dict]) -> list[dict]:
        out = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                t = m["turn"]
                out.append({
                    "role": "assistant",
                    "content": t.text or "",
                    "tool_calls": [{"function": {"name": c.name, "arguments": c.args}} for c in t.tool_calls],
                })
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_name": m["call"].name, "content": json.dumps(m["result"], ensure_ascii=False)})
        return out

    async def complete(self, system, messages, tools, force_tools=None) -> Turn:
        offered = [t for t in tools if not force_tools or t["name"] in force_tools]
        body = {
            "model": self.model,
            "stream": False,
            "messages": self._messages(system, messages),
            "tools": [{"type": "function", "function": t} for t in offered],
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                r = await client.post(f"{self.base_url}/api/chat", json=body)
        except httpx.TransportError as exc:
            raise ProviderError("transient", f"ollama connection error: {type(exc).__name__}") from exc
        if r.status_code != 200:
            raise ProviderError(classify_http_status(r.status_code), f"ollama HTTP {r.status_code}")
        try:
            data = r.json()
            msg = data["message"]
            calls = []
            for i, tc in enumerate(msg.get("tool_calls") or []):
                args = tc["function"].get("arguments") or {}
                calls.append(ToolCall(
                    id=tc.get("id") or f"call_{i}",
                    name=tc["function"]["name"],
                    args=json.loads(args) if isinstance(args, str) else args,
                ))
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            raise ProviderError("malformed", "ollama returned an unparseable response") from exc
        p_in, p_out = data.get("prompt_eval_count"), data.get("eval_count")
        usage = {"input": p_in or 0, "output": p_out or 0, "total": (p_in or 0) + (p_out or 0)} if p_in or p_out else None
        return Turn(tool_calls=calls, text=msg.get("content") or None, usage=usage)
