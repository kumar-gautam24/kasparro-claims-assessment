"""Deterministic evaluator (sentence-spans-and-labels-v4): exact span matching, no fuzzy or LLM judging."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from app.config import EVALUATOR_VERSION, LABELS
from app.data import GoldClaim

ROWS = (*LABELS, "no_gold")
COLS = (*LABELS, "missing_or_error")


@dataclass
class AnswerOutcome:
    answer_id: str
    status: str  # "ok" or "error"
    claims: list[dict] = field(default_factory=list)  # accepted predictions (empty on error)
    error: dict | None = None


def _div(n: int, d: int) -> float:
    return n / d if d else 0.0


def _prf(tp: int, n_pred: int, n_gold: int) -> dict:
    return {
        "precision": _div(tp, n_pred),
        "recall": _div(tp, n_gold),
        "f1": _div(2 * tp, n_pred + n_gold),
    }


def check_outcomes(scheduled_ids: list[str], outcomes: list[AnswerOutcome]) -> dict[str, AnswerOutcome]:
    by_id: dict[str, AnswerOutcome] = {}
    for o in outcomes:
        if o.answer_id in by_id:
            raise ValueError(f"duplicate outcome for {o.answer_id}")
        if o.answer_id not in scheduled_ids:
            raise ValueError(f"outcome for unscheduled answer {o.answer_id}")
        if o.status not in ("ok", "error"):
            raise ValueError(f"invalid status {o.status!r} for {o.answer_id}")
        if o.status == "error" and o.claims:
            raise ValueError(f"error outcome {o.answer_id} must not carry predictions")
        by_id[o.answer_id] = o
    missing = sorted(set(scheduled_ids) - set(by_id))
    if missing:
        raise ValueError(f"missing outcomes for {missing}")
    return by_id


def score_development(
    scheduled_ids: list[str],
    outcomes: list[AnswerOutcome],
    gold: list[GoldClaim],
    texts: dict[str, str],
) -> dict:
    by_id = check_outcomes(scheduled_ids, outcomes)
    gold_by_span = {}
    gold_by_answer: dict[str, list[GoldClaim]] = defaultdict(list)
    for g in gold:
        if g.answer_id in by_id:
            gold_by_span[(g.answer_id, g.start_char, g.end_char)] = g
            gold_by_answer[g.answer_id].append(g)

    confusion = {r: {c: 0 for c in COLS} for r in ROWS}
    label_tp = {lb: 0 for lb in LABELS}
    label_pred = {lb: 0 for lb in LABELS}
    label_gold = {lb: 0 for lb in LABELS}
    for g in gold_by_span.values():
        label_gold[g.expected_label] += 1

    matched: set[tuple] = set()
    unmatched_pred, wrong_label, matched_records = [], [], []
    per_answer = {}
    P = M = C = 0
    for aid in sorted(scheduled_ids):
        o = by_id[aid]
        a_p = a_m = a_c = 0
        for p in o.claims:
            s, e = p["start_char"], p["end_char"]
            if texts[aid][s:e] != p["quote"]:
                raise ValueError(f"{p['claim_id']}: quote does not equal ai_answer span")
            P += 1
            a_p += 1
            label_pred[p["label"]] += 1
            key = (aid, s, e)
            g = gold_by_span.get(key)
            if g is None or key in matched:
                unmatched_pred.append(p["claim_id"])
                confusion["no_gold"][p["label"]] += 1
                continue
            matched.add(key)
            M += 1
            a_m += 1
            correct = p["label"] == g.expected_label
            confusion[g.expected_label][p["label"]] += 1
            matched_records.append(
                {"claim_id": g.claim_id, "expected_label": g.expected_label, "predicted_label": p["label"], "label_correct": correct}
            )
            if correct:
                C += 1
                a_c += 1
                label_tp[g.expected_label] += 1
            else:
                wrong_label.append(g.claim_id)
        a_g = len(gold_by_answer[aid])
        per_answer[aid] = {
            "status": o.status,
            "G": a_g, "P": a_p, "M": a_m, "C": a_c,
            "extraction_f1": _div(2 * a_m, a_p + a_g) if (a_p + a_g) else None,
            "joint_f1": _div(2 * a_c, a_p + a_g) if (a_p + a_g) else None,
        }

    missed = sorted(
        (g for k, g in gold_by_span.items() if k not in matched), key=lambda g: (g.answer_id, g.start_char)
    )
    for g in missed:
        confusion[g.expected_label]["missing_or_error"] += 1
    G = len(gold_by_span)

    per_label = {lb: _prf(label_tp[lb], label_pred[lb], label_gold[lb]) for lb in LABELS}
    if not scheduled_ids:
        quality = None
    else:
        quality = {
            "extraction": _prf(M, P, G),
            "joint": _prf(C, P, G),
            "gold_coverage": _div(C, G),
            "per_label": per_label,
            "macro_f1": sum(v["f1"] for v in per_label.values()) / len(LABELS),
        }
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "counts": {
            "answers": len(scheduled_ids),
            "answer_errors": sum(1 for o in by_id.values() if o.status == "error"),
            "G": G, "P": P, "M": M, "C": C,
            "unmatched_predictions": len(unmatched_pred),
            "confusion_total": sum(sum(r.values()) for r in confusion.values()),
        },
        "metrics": quality,
        "confusion": confusion,
        "missed_gold_ids": [g.claim_id for g in missed],
        "missed_gold": [{"claim_id": g.claim_id, "expected_label": g.expected_label, "quote": g.quote} for g in missed],
        "unmatched_prediction_ids": unmatched_pred,
        "wrong_label_ids": wrong_label,
        "matched": matched_records,
        "per_answer": per_answer,
    }


def summarize_check(scheduled_ids: list[str], outcomes: list[AnswerOutcome]) -> dict:
    """Check split: no labels are available, so every quality field stays null."""
    by_id = check_outcomes(scheduled_ids, outcomes)
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "counts": {
            "answers": len(scheduled_ids),
            "answer_errors": sum(1 for o in by_id.values() if o.status == "error"),
            "G": None, "P": sum(len(o.claims) for o in by_id.values()), "M": None, "C": None,
        },
        "metrics": None,
        "confusion": None,
        "missed_gold_ids": None,
        "unmatched_prediction_ids": None,
        "wrong_label_ids": None,
        "matched": None,
        "per_answer": {aid: {"status": by_id[aid].status, "P": len(by_id[aid].claims)} for aid in sorted(scheduled_ids)},
    }
