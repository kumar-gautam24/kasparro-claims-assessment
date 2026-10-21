"""Fixed versions and runtime settings (secrets come only from the environment)."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

load_dotenv(ROOT / ".env")

DATASET_VERSION = "long-brand-answers-50-v4"
FACT_VERSION = "speaker-catalogue-v4"
EVALUATOR_VERSION = "sentence-spans-and-labels-v4"
QUESTION_DATE = "2026-10-01"
AGENT_VERSIONS = ("v1", "v2")
LABELS = ("supported", "contradicted", "unknown")


def _default_model(provider: str) -> str:
    return {"gemini": "gemini-2.5-flash", "ollama": "qwen2.5:7b", "fake": "fake-model"}.get(provider, "")


@dataclass(frozen=True)
class Settings:
    provider: str = field(default_factory=lambda: os.getenv("PROVIDER", "gemini"))
    model: str = field(default_factory=lambda: os.getenv("MODEL", ""))
    # Brief: 15 s per provider request. Overridable only for slow local models (disclosed in README).
    request_timeout_s: float = field(default_factory=lambda: float(os.getenv("REQUEST_TIMEOUT_S", "15")))
    max_concurrency: int = 3
    max_turns: int = 3
    max_attempts_per_turn: int = 2
    retry_delay_s: float = field(default_factory=lambda: float(os.getenv("RETRY_DELAY_S", "2.0")))
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not self.model:
            object.__setattr__(self, "model", _default_model(self.provider))

    def public(self) -> dict:
        return asdict(self)
