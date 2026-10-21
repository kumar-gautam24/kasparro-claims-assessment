"""Scripted provider for tests: records every payload and tracks in-flight concurrency."""
from __future__ import annotations

import asyncio
import copy
from typing import Callable

from app.providers.base import ProviderError, ToolCall, Turn


def call(name: str, args, id: str | None = None) -> ToolCall:
    return ToolCall(id=id or f"{name}-1", name=name, args=args)


class FakeProvider:
    name = "fake"

    def __init__(self, script: Callable[[str, list[dict], int], object] | list, delay: float = 0.0, model: str = "fake-model"):
        """script: list of Turn/Exception consumed in order, or fn(answer_text, messages, call_index)."""
        self.model = model
        self._script = script
        self.delay = delay
        self.calls = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self.payloads: list[dict] = []

    async def complete(self, system, messages, tools, force_tools=None) -> Turn:
        idx = self.calls
        self.calls += 1
        self.payloads.append(copy.deepcopy({"system": system, "messages": messages, "tools": tools}))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if callable(self._script):
                item = self._script(messages[0]["content"], messages, idx)
            else:
                item = self._script[idx]
            if isinstance(item, BaseException):
                raise item
            if asyncio.iscoroutine(item):
                item = await item
            return item
        finally:
            self.in_flight -= 1


def turn(*calls: ToolCall, text: str | None = None) -> Turn:
    return Turn(tool_calls=list(calls), text=text, usage={"input": 10, "output": 5, "total": 15})


def err(kind: str) -> ProviderError:
    return ProviderError(kind, f"fake {kind}")
