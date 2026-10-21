"""PostgreSQL persistence with parameterized SQL only (psycopg 3)."""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

SCHEMA = (Path(__file__).resolve().parent / "schema.sql").read_text()


def database_url() -> str:
    return os.getenv("DATABASE_URL", "postgresql://claims:claims@localhost:5433/claims")


async def connect(url: str | None = None) -> psycopg.AsyncConnection:
    conn = await psycopg.AsyncConnection.connect(url or database_url(), row_factory=dict_row, autocommit=True)
    return conn


async def init_schema(conn: psycopg.AsyncConnection) -> None:
    await conn.execute(SCHEMA)


async def create_run(conn, run_id: uuid.UUID, meta: dict) -> None:
    await conn.execute(
        """INSERT INTO runs (run_id, agent_version, split, status, prompt_version, dataset_version,
               evaluator_version, fact_version, provider, model, settings, file_hashes, source_revision)
           VALUES (%s, %s, %s, 'running', %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (run_id, meta["agent_version"], meta["split"], meta["prompt_version"], meta["dataset_version"],
         meta["evaluator_version"], meta["fact_version"], meta["provider"], meta["model"],
         Jsonb(meta["settings"]), Jsonb(meta["file_hashes"]), meta.get("source_revision")),
    )


async def save_answer(conn, run_id: uuid.UUID, row: dict, claims: list[dict]) -> None:
    async with conn.transaction():
        await conn.execute(
            """INSERT INTO answer_results (run_id, answer_id, brand_id, status, error_type, error_message,
                   n_claims, latency_ms, model_turns, attempts, usage, trace_id, scores)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (run_id, row["answer_id"], row["brand_id"], row["status"],
             (row.get("error") or {}).get("type"), (row.get("error") or {}).get("message"),
             len(claims), row["latency_ms"], row["model_turns"], row["attempts"],
             Jsonb(row.get("usage")), row.get("trace_id"), Jsonb(row.get("scores"))),
        )
        for c in claims:
            await conn.execute(
                """INSERT INTO claim_results (run_id, answer_id, claim_id, start_char, end_char, quote,
                       label, fact_ids, reason, gold_match)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (run_id, row["answer_id"], c["claim_id"], c["start_char"], c["end_char"], c["quote"],
                 c["label"], Jsonb(c["fact_ids"]), c["reason"], Jsonb(c.get("gold_match"))),
            )


async def finish_run(conn, run_id: uuid.UUID, status: str, counts: dict | None, metrics: dict | None,
                     report: dict | None, failure: str | None = None) -> None:
    await conn.execute(
        """UPDATE runs SET status = %s, counts = %s, metrics = %s, report = %s, failure = %s, finished_at = now()
           WHERE run_id = %s""",
        (status, Jsonb(counts), Jsonb(metrics), Jsonb(report), failure, run_id),
    )


async def get_run(conn, run_id: uuid.UUID) -> dict | None:
    cur = await conn.execute(
        """SELECT r.run_id, r.agent_version, r.split, r.status, r.prompt_version, r.dataset_version,
                  r.evaluator_version, r.fact_version, r.provider, r.model, r.settings, r.file_hashes,
                  r.source_revision, r.counts, r.metrics, r.failure, r.created_at, r.finished_at,
                  a.answer_id, a.brand_id, a.status AS answer_status, a.error_type, a.error_message,
                  a.n_claims, a.latency_ms, a.model_turns, a.attempts, a.usage, a.trace_id, a.scores,
                  c.claim_id, c.start_char, c.end_char, c.quote, c.label, c.fact_ids, c.reason, c.gold_match
           FROM runs r
           LEFT JOIN answer_results a ON a.run_id = r.run_id
           LEFT JOIN claim_results c ON c.run_id = a.run_id AND c.answer_id = a.answer_id
           WHERE r.run_id = %s
           ORDER BY a.answer_id, c.start_char, c.end_char""",
        (run_id,),
    )
    rows = await cur.fetchall()
    if not rows:
        return None
    first = rows[0]
    run_keys = ("run_id", "agent_version", "split", "status", "prompt_version", "dataset_version",
                "evaluator_version", "fact_version", "provider", "model", "settings", "file_hashes",
                "source_revision", "counts", "metrics", "failure", "created_at", "finished_at")
    run = {k: first[k] for k in run_keys}
    run["run_id"] = str(run["run_id"])
    answers: dict[str, dict] = {}
    for r in rows:
        if r["answer_id"] is None:
            continue
        a = answers.setdefault(r["answer_id"], {
            "answer_id": r["answer_id"], "brand_id": r["brand_id"], "status": r["answer_status"],
            "error": {"type": r["error_type"], "message": r["error_message"]} if r["error_type"] else None,
            "n_claims": r["n_claims"], "latency_ms": r["latency_ms"], "model_turns": r["model_turns"],
            "attempts": r["attempts"], "usage": r["usage"], "trace_id": r["trace_id"], "scores": r["scores"],
            "claims": [],
        })
        if r["claim_id"] is not None:
            a["claims"].append({k: r[k] for k in ("claim_id", "start_char", "end_char", "quote", "label",
                                                  "fact_ids", "reason", "gold_match")})
    run["answers"] = list(answers.values())
    return run
