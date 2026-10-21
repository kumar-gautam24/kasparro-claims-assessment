"""Bounded tool-calling executor: <=3 model turns, <=2 attempts per turn, <=N concurrent provider requests."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

from app.config import FACT_VERSION, PROMPTS_DIR, QUESTION_DATE, Settings
from app.facts import GET_BRAND_FACTS, SUBMIT_CLAIMS, FactStore, UnknownBrandError, get_brand_facts_schema, submit_claims_schema
from app.providers.base import Provider, ProviderError, Turn
from app.tracing import Tracer, _NullObs
from app.validate import Claim, InvalidOutputError, build_claims, sentence_spans

SPAN_MODE = {"v1": "quote", "v2": "sentence"}
ALLOWED_TOOLS = frozenset({GET_BRAND_FACTS, SUBMIT_CLAIMS})


class AgentError(Exception):
    """Answer-level failure. type: provider_error | timeout | rate_limit | auth | invalid_output |
    illegal_tool | unknown_brand | turn_limit"""

    def __init__(self, type: str, message: str):
        super().__init__(message)
        self.type = type
        self.message = message

    def to_dict(self) -> dict:
        return {"type": self.type, "message": self.message}


@dataclass(frozen=True)
class AgentInput:
    answer_id: str
    brand_id: str
    question: str
    ai_answer: str


@dataclass
class AgentResult:
    answer_id: str
    status: str
    claims: list[Claim] = field(default_factory=list)
    error: dict | None = None
    latency_ms: int = 0
    model_turns: int = 0
    attempts: int = 0
    usage: dict = field(default_factory=lambda: {"input": 0, "output": 0, "total": 0})
    trace_id: str | None = None
    exchanges: list[dict] = field(default_factory=list)  # sanitized model/tool exchange log


def load_prompt(version: str) -> str:
    return (PROMPTS_DIR / f"{version}.md").read_text(encoding="utf-8")


def build_user_message(version: str, inp: AgentInput) -> str:
    payload = {
        "answer_id": inp.answer_id,
        "brand_id": inp.brand_id,
        "question_date": QUESTION_DATE,
        "question": inp.question,
        "ai_answer": inp.ai_answer,
    }
    msg = "Classify the factual claims in this answer. Input (JSON):\n" + json.dumps(payload, ensure_ascii=False, indent=1)
    if SPAN_MODE[version] == "sentence":
        numbered = "\n".join(
            f"[{i}] {inp.ai_answer[s:e]}" for i, (s, e) in enumerate(sentence_spans(inp.ai_answer), start=1)
        )
        msg += "\n\nNumbered sentences of ai_answer:\n" + numbered
    return msg


def _turn_log(turn: Turn) -> dict:
    return {"text": turn.text, "tool_calls": [{"name": c.name, "args": c.args} for c in turn.tool_calls], "usage": turn.usage}


class Executor:
    def __init__(self, provider: Provider, version: str, settings: Settings, facts: FactStore,
                 tracer: Tracer | None = None, semaphore: asyncio.Semaphore | None = None):
        if version not in SPAN_MODE:
            raise ValueError(f"unknown agent_version {version!r}")
        self.provider = provider
        self.version = version
        self.settings = settings
        self.facts = facts
        self.tracer = tracer or Tracer(enabled=False)
        # One semaphore shared by every answer: caps total in-flight provider requests, retries included.
        self.semaphore = semaphore or asyncio.Semaphore(settings.max_concurrency)
        self.system = load_prompt(version)
        self.span_mode = SPAN_MODE[version]
        self.tools = [get_brand_facts_schema(facts.brands), submit_claims_schema(self.span_mode)]

    async def _model_turn(self, messages: list[dict], force: list[str] | None, obs, turn_no: int, result: AgentResult) -> Turn:
        last_exc: ProviderError | None = None
        for attempt in range(1, self.settings.max_attempts_per_turn + 1):
            result.attempts += 1
            gen = obs.start_observation(
                name=f"model-turn-{turn_no}-attempt-{attempt}", as_type="generation",
                model=self.provider.model, input={"turn": turn_no, "n_messages": len(messages), "forced_tools": force},
                model_parameters={"temperature": self.settings.temperature, "timeout_s": self.settings.request_timeout_s},
            )
            try:
                async with self.semaphore:
                    turn = await asyncio.wait_for(
                        self.provider.complete(self.system, messages, self.tools, force),
                        timeout=self.settings.request_timeout_s,
                    )
                gen.update(output=_turn_log(turn), usage_details=turn.usage or None)
                gen.end()
                return turn
            except asyncio.TimeoutError:
                last_exc = ProviderError("timeout", f"provider request exceeded {self.settings.request_timeout_s}s")
            except ProviderError as exc:
                last_exc = exc
            except Exception as exc:  # unexpected SDK failure: not retried
                last_exc = ProviderError("permanent", f"provider failure: {type(exc).__name__}")
            gen.update(level="ERROR", status_message=f"{last_exc.kind}: {last_exc.message}")
            gen.end()
            if not last_exc.retryable or attempt == self.settings.max_attempts_per_turn:
                break
            await asyncio.sleep(self.settings.retry_delay_s)  # bounded backoff, semaphore released
        kind = last_exc.kind
        etype = {"timeout": "timeout", "rate_limit": "rate_limit", "auth": "auth", "malformed": "invalid_output"}.get(kind, "provider_error")
        raise AgentError(etype, last_exc.message)

    async def run(self, inp: AgentInput, run_id: str = "adhoc", split: str = "adhoc", run_meta: dict | None = None) -> AgentResult:
        t0 = time.perf_counter()
        result = AgentResult(answer_id=inp.answer_id, status="ok")
        meta = {
            "run_id": run_id, "answer_id": inp.answer_id, "split": split, "agent_version": self.version,
            "provider": self.provider.name, "model": self.provider.model, "fact_version": FACT_VERSION,
            **(run_meta or {}),
        }
        with self.tracer.propagate(session_id=run_id, tags=[self.version, split]):
            return await self._run_traced(inp, run_id, split, meta, result, t0)

    async def _run_traced(self, inp, run_id, split, meta, result, t0) -> AgentResult:
        root = self.tracer.answer_trace(
            run_id=run_id, answer_id=inp.answer_id, name=f"classify-{self.version}-{split}",
            input={"answer_id": inp.answer_id, "brand_id": inp.brand_id, "question": inp.question, "ai_answer": inp.ai_answer},
            metadata=meta,
        )
        result.trace_id = getattr(root, "trace_id", None)
        try:
            result.claims = await self._loop(inp, root, result)
            root.update(output={"claims": [c.to_dict() for c in result.claims]},
                        metadata={**meta, "accepted_spans": [[c.start_char, c.end_char] for c in result.claims]})
        except AgentError as exc:
            result.status = "error"
            result.claims = []
            result.error = exc.to_dict()
            root.update(level="ERROR", status_message=f"{exc.type}: {exc.message}",
                        output={"error": exc.to_dict()}, metadata={**meta, "validation_failure": exc.to_dict()})
        result.latency_ms = int((time.perf_counter() - t0) * 1000)
        root.end()
        return result

    async def _loop(self, inp: AgentInput, root, result: AgentResult) -> list[Claim]:
        if inp.brand_id not in self.facts.brands:
            raise AgentError("unknown_brand", f"unknown primary brand_id {inp.brand_id!r}")
        messages: list[dict] = [{"role": "user", "content": build_user_message(self.version, inp)}]
        result.exchanges.append({"role": "user", "content": messages[0]["content"]})
        for turn_no in range(1, self.settings.max_turns + 1):
            force = [SUBMIT_CLAIMS] if turn_no == self.settings.max_turns else None
            turn = await self._model_turn(messages, force, root, turn_no, result)
            result.model_turns = turn_no
            for k in ("input", "output", "total"):
                result.usage[k] += (turn.usage or {}).get(k, 0)
            result.exchanges.append({"role": "assistant", **_turn_log(turn)})
            if not turn.tool_calls:
                raise AgentError("invalid_output", "model turn contained no tool call")
            illegal = sorted({c.name for c in turn.tool_calls} - ALLOWED_TOOLS)
            if illegal:
                raise AgentError("illegal_tool", f"model requested unknown tool(s) {illegal}")
            submits = [c for c in turn.tool_calls if c.name == SUBMIT_CLAIMS]
            if submits:
                if len(turn.tool_calls) != 1:
                    raise AgentError("invalid_output", "submit_claims must be the only tool call in its turn")
                args = submits[0].args
                if not isinstance(args, dict) or "claims" not in args:
                    raise AgentError("invalid_output", "submit_claims arguments must contain 'claims'")
                try:
                    claims = build_claims(inp.answer_id, inp.ai_answer, args["claims"], self.facts.fact_ids, self.span_mode)
                except InvalidOutputError as exc:
                    raise AgentError("invalid_output", str(exc)) from exc
                return claims
            messages.append({"role": "assistant", "turn": turn})
            for c in turn.tool_calls:
                brand = c.args.get("brand_id") if isinstance(c.args, dict) else None
                if not isinstance(c.args, dict) or set(c.args) != {"brand_id"} or not isinstance(brand, str):
                    raise AgentError("invalid_output", "get_brand_facts requires exactly one string argument brand_id")
                tool_obs = root.start_observation(name=GET_BRAND_FACTS, as_type="tool", input={"brand_id": brand})
                try:
                    out = self.facts.get_brand_facts(brand)
                except UnknownBrandError as exc:
                    tool_obs.update(level="ERROR", status_message=str(exc))
                    tool_obs.end()
                    raise AgentError("unknown_brand", str(exc)) from exc
                tool_obs.update(output={"fact_version": out["fact_version"], "n_facts": len(out["facts"]),
                                        "fact_ids": [f["fact_id"] for f in out["facts"]]})
                tool_obs.end()
                messages.append({"role": "tool", "call": c, "result": out})
                result.exchanges.append({"role": "tool", "name": GET_BRAND_FACTS, "args": {"brand_id": brand},
                                         "result_fact_ids": [f["fact_id"] for f in out["facts"]]})
        raise AgentError("turn_limit", f"no submit_claims within {self.settings.max_turns} model turns")
