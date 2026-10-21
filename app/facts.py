"""Read-only fact lookup tool and the tool schemas exposed to the model."""
from __future__ import annotations

from app.config import FACT_VERSION, LABELS
from app.data import Fact, load_facts


class UnknownBrandError(LookupError):
    pass


class FactStore:
    def __init__(self, facts: list[Fact] | None = None):
        self.facts = facts if facts is not None else load_facts()
        self.fact_ids = frozenset(f.fact_id for f in self.facts)
        self.brands = tuple(sorted({f.brand_id for f in self.facts}))

    def get_brand_facts(self, brand_id: str) -> dict:
        if not isinstance(brand_id, str) or brand_id not in self.brands:
            raise UnknownBrandError(f"unknown brand_id {brand_id!r}")
        return {
            "fact_version": FACT_VERSION,
            "brand_id": brand_id,
            "facts": [f.as_tool_row() for f in self.facts if f.brand_id == brand_id],
        }


GET_BRAND_FACTS = "get_brand_facts"
SUBMIT_CLAIMS = "submit_claims"


def get_brand_facts_schema(brands: tuple[str, ...]) -> dict:
    return {
        "name": GET_BRAND_FACTS,
        "description": (
            "Return the official catalogue fact rows (fact_version speaker-catalogue-v4) for one brand. "
            "Call once per brand mentioned in the answer, including brands used only for comparison."
        ),
        "parameters": {
            "type": "object",
            "properties": {"brand_id": {"type": "string", "enum": list(brands)}},
            "required": ["brand_id"],
        },
    }


_CLAIM_COMMON = {
    "label": {"type": "string", "enum": list(LABELS)},
    "fact_ids": {"type": "array", "items": {"type": "string"}},
    "reason": {"type": "string", "description": "At most two sentences."},
}


def submit_claims_schema(span_mode: str) -> dict:
    """span_mode 'quote' (v1): model copies the sentence; 'sentence' (v2): model cites a sentence number."""
    if span_mode == "quote":
        locator = {"quote": {"type": "string", "description": "The full factual sentence copied exactly, with its final punctuation."}}
    else:
        locator = {"sentence": {"type": "integer", "description": "Number of the factual sentence in the numbered list."}}
    return {
        "name": SUBMIT_CLAIMS,
        "description": "Submit the final list of factual claim sentences with labels. Call exactly once, after looking up facts.",
        "parameters": {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {**locator, **_CLAIM_COMMON},
                        "required": [*locator, "label", "fact_ids", "reason"],
                    },
                }
            },
            "required": ["claims"],
        },
    }
