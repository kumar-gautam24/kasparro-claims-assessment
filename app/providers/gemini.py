"""Google Gemini adapter (google-genai SDK) with native function calling."""
from __future__ import annotations

import os
import uuid

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.providers.base import ProviderError, ToolCall, Turn, classify_http_status


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str, temperature: float = 0.0):
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ProviderError("auth", "GEMINI_API_KEY is not set")
        self.model = model
        self.temperature = temperature
        self.thinking_budget = int(os.getenv("GEMINI_THINKING_BUDGET", "1024"))
        self._client = genai.Client(api_key=key)

    def _contents(self, messages: list[dict]) -> list[types.Content]:
        contents: list[types.Content] = []
        for m in messages:
            if m["role"] == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=m["content"])]))
            elif m["role"] == "assistant":
                contents.append(m["turn"].raw)
            elif m["role"] == "tool":
                part = types.Part.from_function_response(name=m["call"].name, response=m["result"])
                if contents and contents[-1].role == "user" and contents[-1].parts and contents[-1].parts[0].function_response:
                    contents[-1].parts.append(part)
                else:
                    contents.append(types.Content(role="user", parts=[part]))
        return contents

    async def complete(self, system, messages, tools, force_tools=None) -> Turn:
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=t["name"], description=t["description"], parameters_json_schema=t["parameters"]
                        )
                        for t in tools
                    ]
                )
            ],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY", allowed_function_names=force_tools)
            ),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
        )
        try:
            resp = await self._client.aio.models.generate_content(
                model=self.model, contents=self._contents(messages), config=config
            )
        except genai_errors.APIError as exc:
            raise ProviderError(classify_http_status(exc.code or 0), f"gemini API error {exc.code}") from exc
        except (OSError, ConnectionError) as exc:
            raise ProviderError("transient", f"gemini connection error: {type(exc).__name__}") from exc

        if not resp.candidates or resp.candidates[0].content is None:
            raise ProviderError("malformed", "gemini returned no candidate content")
        calls = [
            ToolCall(id=fc.id or uuid.uuid4().hex[:12], name=fc.name or "", args=fc.args)
            for fc in (resp.function_calls or [])
        ]
        um = resp.usage_metadata
        usage = None
        if um is not None:
            usage = {
                "input": um.prompt_token_count or 0,
                "output": (um.candidates_token_count or 0) + (um.thoughts_token_count or 0),
                "total": um.total_token_count or 0,
            }
        text = None
        try:
            text = resp.text
        except ValueError:
            pass
        return Turn(tool_calls=calls, text=text, usage=usage, raw=resp.candidates[0].content)
