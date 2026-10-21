"""Run an agent version over a fixed CSV split, score it, persist it and write a JSON report."""
from __future__ import annotations

import asyncio
import json
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app import db
from app.agent import AgentInput, AgentResult, Executor, SPAN_MODE
from app.config import (DATASET_VERSION, EVALUATOR_VERSION, FACT_VERSION, ROOT, RUNS_DIR, Settings)
from app.data import Answer, GoldClaim, file_hashes, load_answers, load_gold
from app.facts import FactStore
from app.providers import make_provider
from app.scorer import AnswerOutcome, score_development, summarize_check
from app.tracing import Tracer


def source_revision() -> str | None:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "app"], cwd=ROOT, capture_output=True, text=True, timeout=5)
        if rev.returncode != 0:
            return None
        return rev.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return None


async def _safe_run(executor: Executor, a: Answer, run_id: str, split: str, run_meta: dict) -> AgentResult:
    """Isolation: an unexpected exception for one answer becomes that answer's terminal error."""
    try:
        return await executor.run(AgentInput(a.answer_id, a.brand_id, a.question, a.ai_answer), run_id, split, run_meta)
    except Exception as exc:  # noqa: BLE001
        return AgentResult(answer_id=a.answer_id, status="error",
                           error={"type": "internal_error", "message": type(exc).__name__})


async def run_split(
    version: str,
    split: str,
    settings: Settings | None = None,
    *,
    provider=None,
    facts: FactStore | None = None,
    tracer: Tracer | None = None,
    conn=None,
    answers: list[Answer] | None = None,
    gold: list[GoldClaim] | None = None,
    out_dir: Path | None = RUNS_DIR,
    answer_ids: list[str] | None = None,
) -> dict:
    if split not in ("development", "check"):
        raise ValueError(f"unknown split {split!r}")
    settings = settings or Settings()
    facts = facts or FactStore()
    tracer = tracer or Tracer()
    provider = provider or make_provider(settings)
    answers = answers if answers is not None else load_answers()
    scheduled = sorted((a for a in answers if a.split == split and (not answer_ids or a.answer_id in answer_ids)),
                       key=lambda a: a.answer_id)
    run_id = uuid.uuid4()
    started = datetime.now(timezone.utc)
    meta = {
        "run_id": str(run_id), "agent_version": version, "split": split,
        "prompt_version": f"{version}:{SPAN_MODE[version]}", "dataset_version": DATASET_VERSION,
        "evaluator_version": EVALUATOR_VERSION, "fact_version": FACT_VERSION,
        "provider": provider.name, "model": provider.model, "settings": settings.public(),
        "file_hashes": file_hashes(), "source_revision": source_revision(),
        "started_at": started.isoformat(), "scheduled_answer_ids": [a.answer_id for a in scheduled],
    }
    if conn is not None:
        await db.create_run(conn, run_id, meta)

    executor = Executor(provider, version, settings, facts, tracer)
    trace_meta = {k: meta[k] for k in ("prompt_version", "dataset_version", "evaluator_version")}
    t0 = time.perf_counter()
    try:
        results = await asyncio.gather(*(_safe_run(executor, a, str(run_id), split, trace_meta) for a in scheduled))
        results = sorted(results, key=lambda r: r.answer_id)
        outcomes = [AnswerOutcome(r.answer_id, r.status, [c.to_dict() for c in r.claims], r.error) for r in results]
        ids = [a.answer_id for a in scheduled]
        if split == "development":
            gold = gold if gold is not None else load_gold()
            texts = {a.answer_id: a.ai_answer for a in scheduled}
            evaluation = score_development(ids, outcomes, gold, texts)
        else:
            evaluation = summarize_check(ids, outcomes)
        wall_ms = int((time.perf_counter() - t0) * 1000)

        brand = {a.answer_id: a.brand_id for a in scheduled}
        matched_by_id = {m["claim_id"]: m for m in (evaluation["matched"] or [])}
        answer_rows = []
        for r in results:
            claims = [c.to_dict() for c in r.claims]
            if split == "development":
                for c in claims:
                    m = matched_by_id.get(c["claim_id"])
                    c["gold_match"] = ({"matched": True, "expected_label": m["expected_label"], "label_correct": m["label_correct"]}
                                       if m else {"matched": False})
            pa = evaluation["per_answer"][r.answer_id]
            scores = ({"extraction_f1": pa["extraction_f1"], "joint_f1": pa["joint_f1"],
                       "G": pa["G"], "P": pa["P"], "M": pa["M"], "C": pa["C"]} if split == "development" else None)
            answer_rows.append({
                "answer_id": r.answer_id, "brand_id": brand[r.answer_id], "status": r.status, "error": r.error,
                "latency_ms": r.latency_ms, "model_turns": r.model_turns, "attempts": r.attempts,
                "usage": r.usage, "trace_id": r.trace_id, "trace_url": tracer.trace_url(r.trace_id),
                "scores": scores, "claims": claims,
            })
            if conn is not None:
                await db.save_answer(conn, run_id, answer_rows[-1], claims)

        if split == "development":
            _push_dev_scores(tracer, str(run_id), evaluation, answer_rows)
        _push_run_summary(tracer, meta, evaluation)
        tracer.flush()

        report = {
            "run": {**meta, "status": "completed", "finished_at": datetime.now(timezone.utc).isoformat(), "wall_ms": wall_ms},
            "counts": evaluation["counts"],
            "metrics": evaluation["metrics"],
            "confusion": evaluation["confusion"],
            "missed_gold_ids": evaluation["missed_gold_ids"],
            "missed_gold": evaluation.get("missed_gold"),
            "unmatched_prediction_ids": evaluation["unmatched_prediction_ids"],
            "wrong_label_ids": evaluation["wrong_label_ids"],
            "matched": evaluation["matched"],
            "answers": answer_rows,
        }
        if conn is not None:
            await db.finish_run(conn, run_id, "completed", evaluation["counts"], evaluation["metrics"],
                                {k: report[k] for k in ("confusion", "missed_gold_ids", "unmatched_prediction_ids", "wrong_label_ids")})
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = started.strftime("%Y%m%dT%H%M%SZ")
            base = out_dir / f"{version}_{split}_{stamp}"
            base.with_suffix(".json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
            with open(f"{base}_exchanges.jsonl", "w", encoding="utf-8") as fh:
                for r in results:
                    fh.write(json.dumps({"answer_id": r.answer_id, "exchanges": r.exchanges}, ensure_ascii=False, default=str) + "\n")
            report["report_path"] = str(base.with_suffix(".json"))
        return report
    except BaseException as exc:
        if conn is not None:
            await db.finish_run(conn, run_id, "failed", None, None, None, failure=type(exc).__name__)
        raise


def _push_dev_scores(tracer: Tracer, run_id: str, evaluation: dict, answer_rows: list[dict]) -> None:
    missed_by_answer: dict[str, list[str]] = {}
    for cid in evaluation["missed_gold_ids"]:
        missed_by_answer.setdefault(cid.split(":")[0], []).append(cid)
    extra_by_answer: dict[str, list[str]] = {}
    for cid in evaluation["unmatched_prediction_ids"]:
        extra_by_answer.setdefault(cid.split(":")[0], []).append(cid)
    matched = {m["claim_id"]: m for m in evaluation["matched"]}
    for row in answer_rows:
        tid, aid, s = row["trace_id"], row["answer_id"], row["scores"]
        meta = {"run_id": run_id, "answer_id": aid, "missed_gold_ids": missed_by_answer.get(aid, []),
                "extra_prediction_ids": extra_by_answer.get(aid, []), "G": s["G"], "P": s["P"], "M": s["M"], "C": s["C"],
                "status": row["status"]}
        for name in ("extraction_f1", "joint_f1"):
            if s[name] is not None:
                tracer.score(trace_id=tid, name=name, value=s[name], metadata=meta)
        for c in row["claims"]:
            m = matched.get(c["claim_id"])
            if m:
                tracer.score(trace_id=tid, name="label_correct", value=1.0 if m["label_correct"] else 0.0,
                             comment=c["claim_id"], metadata={"span_id": c["claim_id"], "expected": m["expected_label"],
                                                              "predicted": m["predicted_label"]})


def _push_run_summary(tracer: Tracer, meta: dict, evaluation: dict) -> None:
    """One summary trace per run carrying aggregate metrics, linked by session_id = run_id."""
    if not tracer.enabled:
        return
    with tracer.propagate(session_id=meta["run_id"], tags=[meta["agent_version"], meta["split"], "run-summary"]):
        obs = tracer.answer_trace(run_id=meta["run_id"], answer_id="__run__",
                                  name=f"run-summary-{meta['agent_version']}-{meta['split']}",
                                  input={k: meta[k] for k in ("agent_version", "split", "provider", "model")},
                                  metadata={k: v for k, v in meta.items() if k != "scheduled_answer_ids"})
        obs.update(output={"counts": evaluation["counts"], "metrics": evaluation["metrics"]})
        obs.end()
    m = evaluation["metrics"]
    if m:
        for name, value in (("run_joint_f1", m["joint"]["f1"]), ("run_extraction_f1", m["extraction"]["f1"]),
                            ("run_macro_f1", m["macro_f1"]), ("run_gold_coverage", m["gold_coverage"])):
            tracer.score(trace_id=obs.trace_id, name=name, value=value,
                         metadata={"run_id": meta["run_id"], "agent_version": meta["agent_version"]})
