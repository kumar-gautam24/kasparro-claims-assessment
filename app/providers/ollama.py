"""Ollama adapter (local model; a disclosed deviation from the brief's hosted-provider requirement).

Tool calling uses Qwen's native function-calling protocol: tool signatures go in the system turn inside
<tools>, the model answers with <tool_call>{"name", "arguments"}</tool_call> blocks and results return in
<tool_response> blocks. We render that template ourselves via /api/generate (raw) because Ollama 0.40's
/api/chat parser silently dropped long but valid submit_claims calls (empty content, no tool_calls)."""
from __future__ import annotations

import json
import os
import re

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

    @staticmethod
    def render_qwen(system: str, messages: list[dict], tools: list[dict]) -> str:
        tool_lines = "\n".join(json.dumps({"type": "function", "function": t}, ensure_ascii=False) for t in tools)
        sys_block = (
            f"{system}\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
            f"{tool_lines}\n</tools>\n\nFor each function call, return a json object with function name and "
            "arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
        )
        parts = [f"<|im_start|>system\n{sys_block}<|im_end|>"]
        pending_tool: list[str] = []

        def flush_tools() -> None:
            if pending_tool:
                parts.append("<|im_start|>user\n" + "\n".join(pending_tool) + "<|im_end|>")
                pending_tool.clear()

        for m in messages:
            if m["role"] == "tool":
                pending_tool.append(f"<tool_response>\n{json.dumps(m['result'], ensure_ascii=False)}\n</tool_response>")
                continue
            flush_tools()
            if m["role"] == "user":
                parts.append(f"<|im_start|>user\n{m['content']}<|im_end|>")
            elif m["role"] == "assistant":
                t = m["turn"]
                calls = "\n".join(
                    f'<tool_call>\n{json.dumps({"name": c.name, "arguments": c.args}, ensure_ascii=False)}\n</tool_call>'
                    for c in t.tool_calls
                )
                parts.append(f"<|im_start|>assistant\n{(t.text or '') + calls}<|im_end|>")
        flush_tools()
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    @staticmethod
    def parse_tool_calls(text: str) -> tuple[list[ToolCall], str | None]:
        blocks = re.findall(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", text, flags=re.S)
        calls = []
        for i, block in enumerate(blocks):
            try:
                obj = json.loads(block)
                calls.append(ToolCall(id=f"call_{i}", name=str(obj["name"]), args=obj.get("arguments", {})))
            except (ValueError, KeyError, TypeError) as exc:
                perr = ProviderError("malformed", "model emitted an unparseable <tool_call> block")
                pos = getattr(exc, "pos", None)
                perr.raw_excerpt = block[max(0, pos - 300):pos + 120] if isinstance(pos, int) else block[:400]
                raise perr from exc
        rest = re.sub(r"<tool_call>.*?(?:</tool_call>|$)", "", text, flags=re.S).strip()
        return calls, rest or None

    async def complete(self, system, messages, tools, force_tools=None) -> Turn:
        offered = [t for t in tools if not force_tools or t["name"] in force_tools]
        body = {
            "model": self.model,
            "raw": True,
            "stream": False,
            "prompt": self.render_qwen(system, messages, offered),
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx, "stop": ["<|im_end|>"]},
        }
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                r = await client.post(f"{self.base_url}/api/generate", json=body)
        except httpx.TransportError as exc:
            raise ProviderError("transient", f"ollama connection error: {type(exc).__name__}") from exc
        if r.status_code != 200:
            raise ProviderError(classify_http_status(r.status_code), f"ollama HTTP {r.status_code}")
        try:
            data = r.json()
            text = data["response"]
        except (KeyError, ValueError) as exc:
            raise ProviderError("malformed", "ollama returned an unparseable response") from exc
        calls, rest = self.parse_tool_calls(text)
        p_in, p_out = data.get("prompt_eval_count"), data.get("eval_count")
        usage = {"input": p_in or 0, "output": p_out or 0, "total": (p_in or 0) + (p_out or 0)} if p_in or p_out else None
        return Turn(tool_calls=calls, text=rest, usage=usage)

    async def complete_chat_api(self, system, messages, tools, force_tools=None) -> Turn:
        """Original /api/chat path, kept for reference (see module docstring for why it is unused)."""
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
