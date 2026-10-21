"""Turn model claim payloads into validated spans. Python owns all character offsets."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from app.config import LABELS

_SENTENCE_RE = re.compile(r"[^\s][^\n]*?(?:[.!?](?=\s|$)|$)")
_REASON_SENT_RE = re.compile(r"[^.!?]*[.!?]+(?:\s+|$)|[^.!?]+$")


class InvalidOutputError(ValueError):
    """The model's final output cannot be accepted; the whole answer becomes an error."""


@dataclass(frozen=True)
class Claim:
    claim_id: str
    start_char: int
    end_char: int
    quote: str
    label: str
    fact_ids: list[str]
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Deterministic sentence segmentation: [start, end) spans with surrounding whitespace excluded."""
    spans = []
    for m in _SENTENCE_RE.finditer(text):
        seg = m.group(0).rstrip()
        if seg:
            spans.append((m.start(), m.start() + len(seg)))
    return spans


def limit_reason(reason: str) -> str:
    parts = [p for p in _REASON_SENT_RE.findall(reason.strip()) if p.strip()]
    return "".join(parts[:2]).strip()


def _check_common(raw: dict, known_fact_ids: frozenset[str], idx: int) -> tuple[str, list[str], str]:
    label = raw.get("label")
    if label not in LABELS:
        raise InvalidOutputError(f"claim {idx}: unsupported label {label!r}")
    fact_ids = raw.get("fact_ids", [])
    if not isinstance(fact_ids, list) or not all(isinstance(f, str) for f in fact_ids):
        raise InvalidOutputError(f"claim {idx}: fact_ids must be a list of strings")
    unknown = sorted(set(fact_ids) - known_fact_ids)
    if unknown:
        raise InvalidOutputError(f"claim {idx}: unknown fact_ids {unknown}")
    reason = raw.get("reason", "")
    if not isinstance(reason, str):
        raise InvalidOutputError(f"claim {idx}: reason must be a string")
    return label, list(dict.fromkeys(fact_ids)), limit_reason(reason)


def locate_quote(text: str, quote: str, taken: list[tuple[int, int]], cursor: int) -> tuple[int, int] | None:
    """Find quote in text. Prefer the first occurrence at/after cursor (reading order), else any
    earlier occurrence that is not already taken. Repeated sentences are resolved by order."""
    def free(s: int, e: int) -> bool:
        return all(e <= ts or s >= te for ts, te in taken)

    candidates = []
    pos = text.find(quote)
    while pos != -1:
        candidates.append((pos, pos + len(quote)))
        pos = text.find(quote, pos + 1)
    for s, e in candidates:
        if s >= cursor and free(s, e):
            return s, e
    for s, e in candidates:
        if free(s, e):
            return s, e
    return None


def build_claims(
    answer_id: str,
    text: str,
    raw_claims: object,
    known_fact_ids: frozenset[str],
    span_mode: str = "quote",
) -> list[Claim]:
    if not isinstance(raw_claims, list):
        raise InvalidOutputError("claims must be a list")
    sentences = sentence_spans(text) if span_mode == "sentence" else []
    located: list[tuple[int, int, str, list[str], str]] = []
    taken: list[tuple[int, int]] = []
    cursor = 0
    for idx, raw in enumerate(raw_claims):
        if not isinstance(raw, dict):
            raise InvalidOutputError(f"claim {idx}: must be an object")
        label, fact_ids, reason = _check_common(raw, known_fact_ids, idx)
        if span_mode == "quote":
            quote = raw.get("quote")
            if not isinstance(quote, str) or not quote.strip():
                raise InvalidOutputError(f"claim {idx}: missing quote")
            quote = quote.strip()
            span = locate_quote(text, quote, taken, cursor)
            if span is None:
                raise InvalidOutputError(f"claim {idx}: quote not found verbatim in ai_answer")
        else:
            num = raw.get("sentence")
            if isinstance(num, float) and num.is_integer():
                num = int(num)
            if not isinstance(num, int) or isinstance(num, bool) or not 1 <= num <= len(sentences):
                raise InvalidOutputError(f"claim {idx}: sentence number {num!r} out of range")
            span = sentences[num - 1]
            if any(not (span[1] <= ts or span[0] >= te) for ts, te in taken):
                raise InvalidOutputError(f"claim {idx}: sentence {num} returned twice")
        taken.append(span)
        cursor = span[1]
        located.append((span[0], span[1], label, fact_ids, reason))

    located.sort(key=lambda c: c[0])
    claims = [
        Claim(f"{answer_id}:{s}:{e}", s, e, text[s:e], label, fact_ids, reason)
        for s, e, label, fact_ids, reason in located
    ]
    validate_claims(answer_id, text, claims, known_fact_ids)
    return claims


def validate_claims(answer_id: str, text: str, claims: list[Claim], known_fact_ids: frozenset[str]) -> None:
    """Final contract check applied to every accepted output."""
    seen: set[str] = set()
    prev_end = -1
    prev_start = -1
    for c in claims:
        if not (0 <= c.start_char < c.end_char <= len(text)):
            raise InvalidOutputError(f"{c.claim_id}: span out of bounds")
        if text[c.start_char:c.end_char] != c.quote:
            raise InvalidOutputError(f"{c.claim_id}: quote does not equal ai_answer[start:end]")
        if c.quote != c.quote.strip():
            raise InvalidOutputError(f"{c.claim_id}: quote has leading/trailing whitespace")
        if c.claim_id != f"{answer_id}:{c.start_char}:{c.end_char}":
            raise InvalidOutputError(f"{c.claim_id}: claim_id must be answer_id:start:end")
        if c.claim_id in seen:
            raise InvalidOutputError(f"{c.claim_id}: duplicate claim_id")
        seen.add(c.claim_id)
        if c.start_char <= prev_start:
            raise InvalidOutputError(f"{c.claim_id}: claims not in increasing start order")
        if c.start_char < prev_end:
            raise InvalidOutputError(f"{c.claim_id}: overlapping spans")
        if c.label not in LABELS:
            raise InvalidOutputError(f"{c.claim_id}: unsupported label {c.label!r}")
        if set(c.fact_ids) - known_fact_ids:
            raise InvalidOutputError(f"{c.claim_id}: unknown fact_ids")
        prev_start, prev_end = c.start_char, c.end_char
