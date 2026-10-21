"""CSV loading with a real parser; ai_answer text is never modified."""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from app.config import DATA_DIR

ANSWERS_FILE = "answers.csv"
FACTS_FILE = "brand_facts.csv"
GOLD_FILE = "development_gold.csv"


@dataclass(frozen=True)
class Answer:
    answer_id: str
    split: str
    brand_id: str
    question: str
    ai_answer: str


@dataclass(frozen=True)
class Fact:
    fact_id: str
    brand_id: str
    model_id: str
    field: str
    value: object
    value_type: str
    unit: str | None
    condition: str | None
    valid_from: str
    valid_to: str

    def as_tool_row(self) -> dict:
        return {
            "fact_id": self.fact_id,
            "model_id": self.model_id,
            "field": self.field,
            "value": self.value,
            "value_type": self.value_type,
            "unit": self.unit,
            "condition": self.condition,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
        }


@dataclass(frozen=True)
class GoldClaim:
    answer_id: str
    claim_id: str
    start_char: int
    end_char: int
    quote: str
    expected_label: str
    evidence_fact_ids: tuple[str, ...]
    explanation: str
    difficulty_tags: tuple[str, ...]


def _read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def parse_value(raw: str, value_type: str) -> object:
    if value_type == "number":
        num = float(raw)
        return int(num) if num.is_integer() else num
    if value_type == "boolean":
        if raw == "true":
            return True
        if raw == "false":
            return False
        raise ValueError(f"invalid boolean literal {raw!r}")
    if value_type == "string":
        return raw
    raise ValueError(f"unknown value_type {value_type!r}")


def load_answers(path: Path | None = None) -> list[Answer]:
    return [Answer(**row) for row in _read(path or DATA_DIR / ANSWERS_FILE)]


def load_facts(path: Path | None = None) -> list[Fact]:
    facts = []
    for row in _read(path or DATA_DIR / FACTS_FILE):
        facts.append(
            Fact(
                fact_id=row["fact_id"],
                brand_id=row["brand_id"],
                model_id=row["model_id"],
                field=row["field"],
                value=parse_value(row["value"], row["value_type"]),
                value_type=row["value_type"],
                unit=row["unit"] or None,
                condition=row["condition"] or None,
                valid_from=row["valid_from"],
                valid_to=row["valid_to"],
            )
        )
    return facts


def load_gold(path: Path | None = None) -> list[GoldClaim]:
    gold = []
    for row in _read(path or DATA_DIR / GOLD_FILE):
        gold.append(
            GoldClaim(
                answer_id=row["answer_id"],
                claim_id=row["claim_id"],
                start_char=int(row["start_char"]),
                end_char=int(row["end_char"]),
                quote=row["quote"],
                expected_label=row["expected_label"],
                evidence_fact_ids=tuple(x for x in row["evidence_fact_ids"].split("|") if x),
                explanation=row["explanation"],
                difficulty_tags=tuple(x for x in row["difficulty_tags"].split("|") if x),
            )
        )
    return gold


def file_hashes(data_dir: Path | None = None) -> dict[str, str]:
    d = data_dir or DATA_DIR
    return {
        name: hashlib.sha256((d / name).read_bytes()).hexdigest()
        for name in (ANSWERS_FILE, FACTS_FILE, GOLD_FILE)
    }
