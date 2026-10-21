import asyncio
import json

import pytest

from app.agent import AgentInput, Executor
from app.config import Settings
from app.data import load_answers, load_gold
from app.facts import FactStore
from app.providers.fake import FakeProvider, call, err, turn
from app.runner import run_split
from app.tracing import Tracer

SETTINGS = Settings(provider="fake", retry_delay_s=0.01, request_timeout_s=0.5)
FACTS = FactStore()
TEXT = "Plan your trip. ClearTrail Mini costs INR 1,299 including tax but excluding shipping. Enjoy."
INP = AgentInput("A99", "cleartrail", "Is it good?", TEXT)
GOOD_CLAIM = {"quote": "ClearTrail Mini costs INR 1,299 including tax but excluding shipping.", "label": "supported",
              "fact_ids": ["cleartrail.mini.price_2026"], "reason": "Matches the 2026 list price."}


def lookup(brand="cleartrail"):
    return turn(call("get_brand_facts", {"brand_id": brand}))


def submit(claims):
    return turn(call("submit_claims", {"claims": claims}))


def ex(script, settings=SETTINGS, version="v1", sem=None, delay=0.0):
    p = FakeProvider(script, delay=delay)
    return p, Executor(p, version, settings, FACTS, Tracer(enabled=False), semaphore=sem)


async def test_happy_path_uses_tool_then_submits():
    p, e = ex([lookup(), submit([GOOD_CLAIM])])
    r = await e.run(INP)
    assert r.status == "ok" and r.model_turns == 2
    [c] = r.claims
    assert TEXT[c.start_char:c.end_char] == c.quote and c.claim_id == f"A99:{c.start_char}:{c.end_char}"
    tool_msgs = [m for m in p.payloads[1]["messages"] if m["role"] == "tool"]
    assert tool_msgs[0]["result"]["fact_version"] == "speaker-catalogue-v4"


async def test_valid_empty_claims_is_ok():
    _, e = ex([lookup(), submit([])])
    r = await e.run(INP)
    assert r.status == "ok" and r.claims == []


@pytest.mark.parametrize("bad,etype", [
    ({**GOOD_CLAIM, "quote": "Invented sentence."}, "invalid_output"),
    ({**GOOD_CLAIM, "label": "partially_true"}, "invalid_output"),
    ({**GOOD_CLAIM, "fact_ids": ["cleartrail.mini.price_2030"]}, "invalid_output"),
])
async def test_bad_output_is_answer_error_not_unknown(bad, etype):
    _, e = ex([lookup(), submit([bad])])
    r = await e.run(INP)
    assert r.status == "error" and r.error["type"] == etype and r.claims == []


async def test_malformed_submit_arguments():
    _, e = ex([lookup(), turn(call("submit_claims", {"items": "nope"}))])
    r = await e.run(INP)
    assert r.error["type"] == "invalid_output"


async def test_illegal_tool_is_error_and_not_executed():
    _, e = ex([turn(call("delete_database", {"brand_id": "cleartrail"}))])
    r = await e.run(INP)
    assert r.error["type"] == "illegal_tool"


async def test_unknown_requested_brand_is_error():
    _, e = ex([lookup("acme")])
    r = await e.run(INP)
    assert r.error["type"] == "unknown_brand"


async def test_unknown_primary_brand_never_calls_model():
    p, e = ex([])
    r = await e.run(AgentInput("A99", "acme", "q", TEXT))
    assert r.error["type"] == "unknown_brand" and p.calls == 0


async def test_turn_exhaustion():
    p, e = ex([lookup(), lookup("stonewave"), lookup("cloudnote")])
    r = await e.run(INP)
    assert r.error["type"] == "turn_limit" and p.calls == 3


async def test_timeout_retried_once_then_error():
    async def slow():
        await asyncio.sleep(5)
    p, e = ex(lambda text, msgs, i: slow())
    r = await e.run(INP)
    assert r.error["type"] == "timeout" and p.calls == 2 and r.attempts == 2


async def test_rate_limit_retried_then_succeeds():
    p, e = ex([err("rate_limit"), lookup(), submit([GOOD_CLAIM])])
    r = await e.run(INP)
    assert r.status == "ok" and p.calls == 3 and r.attempts == 3


@pytest.mark.parametrize("kind,etype", [("auth", "auth"), ("malformed", "invalid_output"), ("permanent", "provider_error")])
async def test_nonretryable_errors_not_retried(kind, etype):
    p, e = ex([err(kind), lookup()])
    r = await e.run(INP)
    assert r.error["type"] == etype and p.calls == 1


async def test_concurrency_capped_at_three_including_retries():
    sem = asyncio.Semaphore(3)

    def script(text, msgs, i):
        if i % 4 == 0:
            return err("transient")
        return lookup() if len(msgs) == 1 else submit([])
    p = FakeProvider(script, delay=0.02)
    e = Executor(p, "v1", SETTINGS, FACTS, Tracer(enabled=False), semaphore=sem)
    results = await asyncio.gather(*(e.run(AgentInput(f"A{i:02d}", "cleartrail", "q", TEXT)) for i in range(12)))
    assert p.max_in_flight == 3
    assert all(r.status in ("ok", "error") for r in results)


async def test_failure_isolation_and_sorted_outputs():
    answers = [a for a in load_answers() if a.split == "development"][:5]

    def script(text, msgs, i):
        if '"answer_id": "A03"' in text:
            return err("auth")
        return lookup() if len(msgs) == 1 else submit([])
    p = FakeProvider(script)
    report = await run_split("v1", "development", SETTINGS, provider=p, facts=FACTS, tracer=Tracer(enabled=False),
                             answers=answers, out_dir=None)
    ids = [a["answer_id"] for a in report["answers"]]
    assert ids == sorted(ids) == ["A01", "A02", "A03", "A04", "A05"]
    statuses = {a["answer_id"]: a["status"] for a in report["answers"]}
    assert statuses["A03"] == "error" and statuses["A01"] == "ok"
    assert report["counts"]["answer_errors"] == 1
    g = [x for x in load_gold() if x.answer_id in ids]
    assert report["counts"]["G"] == len(g) and len(report["missed_gold_ids"]) == len(g)


async def test_gold_never_enters_model_or_tool_payloads():
    answers = [a for a in load_answers() if a.split == "development"][:3]
    gold = [g for g in load_gold() if g.answer_id in {a.answer_id for a in answers}]
    p = FakeProvider(lambda text, msgs, i: lookup() if len(msgs) == 1 else submit([]))
    await run_split("v2", "development", SETTINGS, provider=p, facts=FACTS, tracer=Tracer(enabled=False),
                    answers=answers, out_dir=None)
    blob = json.dumps(p.payloads, default=lambda o: getattr(o, "__dict__", str(o)))
    for forbidden in ('"expected_label"', '"evidence_fact_ids"', '"difficulty_tags"', '"explanation"'):
        assert forbidden not in blob
    for g in gold:
        assert g.claim_id not in blob
        assert g.explanation not in blob


async def test_v2_sentence_mode_derives_spans_in_python():
    p, e = ex([lookup(), submit([{"sentence": 2, "label": "supported", "fact_ids": [], "reason": "x"}])], version="v2")
    r = await e.run(INP)
    assert r.status == "ok" and r.claims[0].quote == GOOD_CLAIM["quote"]
    assert "[2] ClearTrail Mini costs" in p.payloads[0]["messages"][0]["content"]


async def test_check_split_has_no_quality_metrics():
    answers = [a for a in load_answers() if a.split == "check"][:2]
    p = FakeProvider(lambda text, msgs, i: lookup() if len(msgs) == 1 else submit([]))
    report = await run_split("v1", "check", SETTINGS, provider=p, facts=FACTS, tracer=Tracer(enabled=False),
                             answers=answers, out_dir=None)
    assert report["metrics"] is None and report["counts"]["G"] is None
    assert all(a["scores"] is None for a in report["answers"])
