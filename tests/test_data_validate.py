import csv
import io

import pytest

from app.data import load_answers, load_facts, load_gold, parse_value
from app.facts import FactStore, UnknownBrandError
from app.validate import Claim, InvalidOutputError, build_claims, sentence_spans, validate_claims

FIDS = frozenset({"b.m.price_2026", "b.m.wifi"})
TEXT = "Intro advice here. The speaker costs INR 1,299, tax included.\nIt has no Wi-Fi. It has no Wi-Fi."


def test_csv_quoted_commas_and_embedded_newlines(tmp_path):
    p = tmp_path / "answers.csv"
    text = 'First, a comma.\n\nSecond paragraph,  double  space "quoted".'
    with open(p, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["answer_id", "split", "brand_id", "question", "ai_answer"])
        w.writerow(["X01", "development", "b", "Q, with comma?", text])
    [a] = load_answers(p)
    assert a.ai_answer == text
    assert a.question == "Q, with comma?"


def test_issued_csvs_parse_and_gold_offsets_match_exact_text():
    answers = {a.answer_id: a for a in load_answers()}
    assert len(answers) == 50
    assert sum(a.split == "development" for a in answers.values()) == 40
    gold = load_gold()
    assert len(gold) == 376
    for g in gold:
        assert answers[g.answer_id].ai_answer[g.start_char:g.end_char] == g.quote
    assert len(load_facts()) == 132


def test_every_gold_span_is_a_deterministic_sentence():
    answers = {a.answer_id: a for a in load_answers()}
    for g in load_gold():
        assert (g.start_char, g.end_char) in sentence_spans(answers[g.answer_id].ai_answer)


def test_boolean_parsing_is_not_python_truthiness():
    assert parse_value("false", "boolean") is False
    assert parse_value("true", "boolean") is True
    assert parse_value("12", "number") == 12
    assert parse_value("5.3", "string") == "5.3"
    with pytest.raises(ValueError):
        parse_value("False", "boolean")


def test_fact_tool_returns_version_and_rejects_unknown_brand():
    store = FactStore()
    out = store.get_brand_facts("cleartrail")
    assert out["fact_version"] == "speaker-catalogue-v4"
    assert all(f["fact_id"].startswith("cleartrail.") for f in out["facts"])
    assert any(f["value"] is False for f in out["facts"])
    with pytest.raises(UnknownBrandError):
        store.get_brand_facts("acme")


def test_quote_located_and_claim_id_derived():
    claims = build_claims("A1", TEXT, [{"quote": "The speaker costs INR 1,299, tax included.", "label": "supported",
                                        "fact_ids": ["b.m.price_2026"], "reason": "Matches."}], FIDS)
    [c] = claims
    assert TEXT[c.start_char:c.end_char] == c.quote
    assert c.claim_id == f"A1:{c.start_char}:{c.end_char}"


def test_repeated_quote_resolved_in_reading_order():
    raw = [{"quote": "It has no Wi-Fi.", "label": "supported", "fact_ids": [], "reason": "a"},
           {"quote": "It has no Wi-Fi.", "label": "supported", "fact_ids": [], "reason": "b"}]
    c1, c2 = build_claims("A1", TEXT, raw, FIDS)
    assert c1.start_char < c2.start_char and c1.end_char <= c2.start_char


@pytest.mark.parametrize("raw,msg", [
    ({"quote": "Not in the answer.", "label": "supported", "fact_ids": [], "reason": ""}, "not found"),
    ({"quote": "It has no Wi-Fi.", "label": "true", "fact_ids": [], "reason": ""}, "label"),
    ({"quote": "It has no Wi-Fi.", "label": "unknown", "fact_ids": ["invented.id"], "reason": ""}, "fact_ids"),
])
def test_invalid_claims_rejected(raw, msg):
    with pytest.raises(InvalidOutputError, match=msg):
        build_claims("A1", TEXT, [raw], FIDS)


def test_overlapping_quotes_rejected():
    raw = [{"quote": "The speaker costs INR 1,299, tax included.", "label": "unknown", "fact_ids": [], "reason": ""},
           {"quote": "INR 1,299", "label": "unknown", "fact_ids": [], "reason": ""}]
    with pytest.raises(InvalidOutputError):
        build_claims("A1", TEXT, raw, FIDS)


def test_sentence_mode_and_out_of_range():
    claims = build_claims("A1", TEXT, [{"sentence": 3, "label": "supported", "fact_ids": [], "reason": ""}], FIDS, "sentence")
    assert claims[0].quote == "It has no Wi-Fi."
    with pytest.raises(InvalidOutputError):
        build_claims("A1", TEXT, [{"sentence": 99, "label": "supported", "fact_ids": [], "reason": ""}], FIDS, "sentence")
    with pytest.raises(InvalidOutputError):
        build_claims("A1", TEXT, [{"sentence": 3, "label": "supported", "fact_ids": [], "reason": ""}] * 2, FIDS, "sentence")


def _claim(s, e, cid=None, quote=None, label="supported"):
    return Claim(cid or f"A1:{s}:{e}", s, e, quote if quote is not None else TEXT[s:e], label, [], "")


def test_validate_offsets_quote_ids_overlap():
    s, e = sentence_spans(TEXT)[1]
    validate_claims("A1", TEXT, [_claim(s, e)], FIDS)
    with pytest.raises(InvalidOutputError, match="bounds"):
        validate_claims("A1", TEXT, [_claim(0, len(TEXT) + 5)], FIDS)
    with pytest.raises(InvalidOutputError, match="quote"):
        validate_claims("A1", TEXT, [_claim(s, e, quote="wrong")], FIDS)
    with pytest.raises(InvalidOutputError, match="duplicate"):
        validate_claims("A1", TEXT, [_claim(s, e), _claim(s, e)], FIDS)
    with pytest.raises(InvalidOutputError, match="overlap"):
        validate_claims("A1", TEXT, [_claim(s, e), _claim(s + 4, e + 3)], FIDS)
    with pytest.raises(InvalidOutputError, match="whitespace"):
        validate_claims("A1", TEXT, [_claim(s - 1, e)], FIDS)
    with pytest.raises(InvalidOutputError, match="claim_id"):
        validate_claims("A1", TEXT, [_claim(s, e, cid="A1:x")], FIDS)
