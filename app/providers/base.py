"""Provider-neutral message/turn types and error classification."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    args: Any


@dataclass
class Turn:
    tool_calls: list[ToolCall] = field(default_factory=list)
    text: str | None = None
    usage: dict[str, int] | None = None
    raw: Any = None  # provider-native assistant content, replayed on the next turn


class ProviderError(Exception):
    """kind: timeout | rate_limit | transient | auth | permanent | malformed"""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message

    @property
    def retryable(self) -> bool:
        return self.kind in ("timeout", "rate_limit", "transient")


class Provider(Protocol):
    name: str
    model: str

    async def complete(
        self, system: str, messages: list[dict], tools: list[dict], force_tools: list[str] | None = None
    ) -> Turn:
        """messages: {"role":"user","content":str} | {"role":"assistant","turn":Turn}
        | {"role":"tool","call":ToolCall,"result":dict}. Must raise ProviderError on failure."""
        ...


def classify_http_status(status: int) -> str:
    if status == 429:
        return "rate_limit"
    if status in (401, 403):
        return "auth"
    if status in (408, 500, 502, 503, 504):
        return "transient"
    return "permanent"
