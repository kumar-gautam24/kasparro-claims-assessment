import pytest

from app.data import GoldClaim
from app.scorer import AnswerOutcome, score_development, summarize_check

TEXT = " ".join(f"S{i:02d} fact sentence." for i in range(12))  # 12 sentences, each 16 chars + space


def span(i):
    s = i * 17
    return s, s + 16


def gold(i, label):
    s, e = span(i)
    return GoldClaim("A1", f"A1:{s}:{e}", s, e, TEXT[s:e], label, (), "", ())


def pred(i, label):
    s, e = span(i)
    return {"claim_id": f"A1:{s}:{e}", "start_char": s, "end_char": e, "quote": TEXT[s:e], "label": label,
            "fact_ids": [], "reason": ""}


LABELS10 = ["supported"] * 4 + ["contradicted"] * 3 + ["unknown"] * 3
GOLD10 = [gold(i, lb) for i, lb in enumerate(LABELS10)]


def run(preds, status="ok", gold_=GOLD10, ids=("A1",)):
    return score_development(list(ids), [AnswerOutcome("A1", status, preds, None if status == "ok" else {"type": "x"})],
                             gold_, {"A1": TEXT})


def test_perfect_output():
    r = run([pred(i, lb) for i, lb in enumerate(LABELS10)])
    assert r["metrics"]["joint"]["f1"] == 1.0
    assert r["metrics"]["macro_f1"] == 1.0
    assert r["missed_gold_ids"] == [] and r["unmatched_prediction_ids"] == []


def test_assessment_arithmetic_example_g10_p10_m8_c7():
    preds = [pred(i, lb) for i, lb in enumerate(LABELS10[:8])]  # 8 exact matches
    preds[0]["label"] = "contradicted"  # one wrong label -> C = 7
    preds += [pred(10, "unknown"), pred(11, "unknown")]  # 2 extra advice sentences -> P = 10
    r = run(preds)
    c = r["counts"]
    assert (c["G"], c["P"], c["M"], c["C"]) == (10, 10, 8, 7)
    m = r["metrics"]
    assert m["extraction"] == pytest.approx({"precision": 0.8, "recall": 0.8, "f1": 0.8})
    assert m["joint"] == pytest.approx({"precision": 0.7, "recall": 0.7, "f1": 0.7})
    assert m["gold_coverage"] == pytest.approx(0.7)
    assert len(r["missed_gold_ids"]) == 2 and len(r["unmatched_prediction_ids"]) == 2
    assert r["wrong_label_ids"] == [GOLD10[0].claim_id]
    assert c["confusion_total"] == 10 + 2
    assert r["confusion"]["no_gold"]["missing_or_error"] == 0


def test_missed_claims_reduce_recall_only():
    r = run([pred(i, lb) for i, lb in enumerate(LABELS10[:5])])
    assert r["metrics"]["joint"]["precision"] == 1.0
    assert r["metrics"]["joint"]["recall"] == 0.5
    assert r["confusion"]["contradicted"]["missing_or_error"] == 2


def test_extra_advice_sentence_reduces_precision():
    r = run([pred(i, lb) for i, lb in enumerate(LABELS10)] + [pred(11, "supported")])
    assert r["metrics"]["extraction"]["precision"] == pytest.approx(10 / 11)
    assert r["metrics"]["per_label"]["supported"]["precision"] == pytest.approx(4 / 5)
    assert r["confusion"]["no_gold"]["supported"] == 1


def test_wrong_labels_count_fp_and_fn_per_label():
    preds = [pred(i, lb) for i, lb in enumerate(LABELS10)]
    preds[0]["label"] = "unknown"
    r = run(preds)
    pl = r["metrics"]["per_label"]
    assert pl["supported"]["recall"] == pytest.approx(3 / 4)
    assert pl["unknown"]["precision"] == pytest.approx(3 / 4)
    assert r["metrics"]["extraction"]["f1"] == 1.0
    assert r["confusion"]["supported"]["unknown"] == 1


def test_provider_error_answer_keeps_gold_as_missed():
    r = run([], status="error")
    assert r["counts"]["answer_errors"] == 1
    assert r["counts"]["G"] == 10 and r["counts"]["P"] == 0
    assert r["metrics"]["joint"]["f1"] == 0.0
    assert len(r["missed_gold_ids"]) == 10


def test_valid_empty_predictions_get_zero_credit():
    r = run([])
    assert r["counts"]["answer_errors"] == 0
    assert r["metrics"]["extraction"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_empty_split_has_null_metrics():
    r = score_development([], [], GOLD10, {"A1": TEXT})
    assert r["metrics"] is None


def test_missing_and_duplicate_outcomes_rejected():
    with pytest.raises(ValueError, match="missing"):
        score_development(["A1", "A2"], [AnswerOutcome("A1", "ok", [])], GOLD10, {"A1": TEXT})
    with pytest.raises(ValueError, match="duplicate"):
        score_development(["A1"], [AnswerOutcome("A1", "ok", []), AnswerOutcome("A1", "ok", [])], GOLD10, {"A1": TEXT})


def test_quote_mismatch_rejected_before_matching():
    p = pred(0, "supported")
    p["quote"] = "tampered"
    with pytest.raises(ValueError, match="quote"):
        run([p])


def test_check_split_quality_stays_null():
    r = summarize_check(["A41"], [AnswerOutcome("A41", "ok", [pred(0, "supported")])])
    assert r["metrics"] is None
    assert r["counts"]["G"] is None and r["counts"]["C"] is None and r["counts"]["P"] == 1
