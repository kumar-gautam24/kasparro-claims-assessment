"""Thin FastAPI layer over the agent, runner and database."""
from __future__ import annotations

import uuid
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from app import db
from app.agent import AgentInput, Executor
from app.config import RUNS_DIR, Settings
from app.facts import FactStore
from app.providers import make_provider
from app.runner import run_split
from app.tracing import Tracer

app = FastAPI(title="Claim classifier", version="1.0")
app.state.settings = Settings()
app.state.facts = FactStore()
app.state.provider_factory = lambda: make_provider(app.state.settings)
app.state.tracer_factory = Tracer
app.state.database_url = None
app.state.runs_dir = RUNS_DIR

ERROR_STATUS = {"timeout": 504}  # every other agent failure is a bad upstream/model result -> 502


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # gold/claim-list fields such as "claims" or "expected_label" -> 422
    answer_id: str = Field(min_length=1, max_length=64)
    brand_id: str = Field(min_length=1, max_length=64)
    question: str = Field(max_length=5000)
    ai_answer: str = Field(min_length=1, max_length=50000)
    agent_version: Literal["v1", "v2"]


class EvaluationRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent_version: Literal["v1", "v2"]
    split: Literal["development", "check"]


def _error(status: int, type_: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"type": type_, "message": message}})


@app.exception_handler(RequestValidationError)
async def _validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    fields = sorted({".".join(str(p) for p in e["loc"][1:]) or "body" for e in exc.errors()})
    return _error(422, "validation_error", f"invalid request fields: {', '.join(fields)}")


@app.exception_handler(HTTPException)
async def _http_handler(_: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, dict) else {"type": "http_error", "message": str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content={"error": detail})


@app.post("/classify")
async def classify(req: ClassifyRequest):
    facts: FactStore = app.state.facts
    if req.brand_id not in facts.brands:
        raise HTTPException(404, {"type": "unknown_brand", "message": f"unknown brand_id {req.brand_id!r}"})
    try:
        provider = app.state.provider_factory()
    except Exception:
        raise HTTPException(502, {"type": "provider_error", "message": "provider is not configured"})
    executor = Executor(provider, req.agent_version, app.state.settings, facts, app.state.tracer_factory())
    result = await executor.run(AgentInput(req.answer_id, req.brand_id, req.question, req.ai_answer), split="api")
    if result.status == "error":
        raise HTTPException(ERROR_STATUS.get(result.error["type"], 502), result.error)
    return {"answer_id": req.answer_id, "agent_version": req.agent_version,
            "claims": [c.to_dict() for c in result.claims], "trace_id": result.trace_id}


@app.post("/evaluation-runs", status_code=201)
async def create_evaluation_run(req: EvaluationRunRequest):
    conn = await db.connect(app.state.database_url)
    try:
        await db.init_schema(conn)
        report = await run_split(req.agent_version, req.split, app.state.settings, provider=app.state.provider_factory(),
                                 facts=app.state.facts, tracer=app.state.tracer_factory(), conn=conn,
                                 out_dir=app.state.runs_dir)
    finally:
        await conn.close()
    return {"run_id": report["run"]["run_id"], "status": report["run"]["status"], "split": req.split,
            "agent_version": req.agent_version, "counts": report["counts"], "metrics": report["metrics"]}


@app.get("/evaluation-runs/{run_id}")
async def get_evaluation_run(run_id: uuid.UUID):
    conn = await db.connect(app.state.database_url)
    try:
        run = await db.get_run(conn, run_id)
    finally:
        await conn.close()
    if run is None:
        raise HTTPException(404, {"type": "not_found", "message": "evaluation run not found"})
    return run
