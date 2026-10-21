"""Langfuse tracing: one trace per answer, generation per model call, tool observation per lookup.
Every method is a no-op when LANGFUSE_* keys are not configured."""
from __future__ import annotations

import contextlib
import os
from typing import Any


class _NullObs:
    trace_id: str | None = None
    id: str | None = None

    def start_observation(self, **_: Any) -> "_NullObs":
        return self

    def update(self, **_: Any) -> "_NullObs":
        return self

    def end(self, **_: Any) -> None:
        pass


class Tracer:
    def __init__(self, enabled: bool | None = None):
        self._lf = None
        if enabled is False:
            return
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            from langfuse import Langfuse

            self._lf = Langfuse()

    @property
    def enabled(self) -> bool:
        return self._lf is not None

    def answer_trace(self, *, run_id: str, answer_id: str, name: str, input: Any, metadata: dict) -> Any:
        if not self._lf:
            return _NullObs()
        trace_id = self._lf.create_trace_id(seed=f"{run_id}:{answer_id}")
        return self._lf.start_observation(
            trace_context={"trace_id": trace_id},
            name=name,
            as_type="agent",
            input=input,
            metadata=metadata,
            version=metadata.get("agent_version"),
        )

    def propagate(self, **attrs: Any):
        """Trace-level attributes (session_id = run_id, tags) for observations created inside."""
        if not self._lf:
            return contextlib.nullcontext()
        from langfuse import propagate_attributes

        return propagate_attributes(**attrs)

    def score(self, *, trace_id: str | None, name: str, value: float | str, data_type: str = "NUMERIC",
              comment: str | None = None, metadata: dict | None = None, session_id: str | None = None) -> None:
        if not self._lf:
            return
        self._lf.create_score(name=name, value=value, trace_id=trace_id, session_id=session_id,
                              data_type=data_type, comment=comment, metadata=metadata)

    def trace_url(self, trace_id: str | None) -> str | None:
        if not self._lf or not trace_id:
            return None
        try:
            return self._lf.get_trace_url(trace_id=trace_id)
        except Exception:
            return None

    def flush(self) -> None:
        if self._lf:
            self._lf.flush()
