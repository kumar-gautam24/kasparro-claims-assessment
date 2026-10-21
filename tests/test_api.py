"""API + PostgreSQL tests. Need the docker 'appdb' service; skipped when it is unreachable."""
import asyncio
import uuid

import psycopg
import pytest
from fastapi.testclient import TestClient

from app import db
from app.api import app
from app.data import load_answers
from app.providers.fake import FakeProvider, call, err, turn
from app.tracing import Tracer

TEST_DB = db.database_url()


def _db_available() -> bool:
    try:
        psycopg.connect(TEST_DB, connect_timeout=2).close()
        return True
    except Exception:
        return False


needs_db = pytest.mark.skipif(not _db_available(), reason="PostgreSQL not reachable at DATABASE_URL")
TEXT = "Plan ahead. ClearTrail Mini costs INR 1,299 including tax but excluding shipping. Enjoy."
BODY = {"answer_id": "T1", "brand_id": "cleartrail", "question": "q", "ai_answer": TEXT, "agent_version": "v1"}


def lookup():
    return turn(call("get_brand_facts", {"brand_id": "cleartrail"}))


def submit(claims):
    return turn(call("submit_claims", {"claims": claims}))


CLAIM = {"quote": "ClearTrail Mini costs INR 1,299 including tax but excluding shipping.", "label": "supported",
         "fact_ids": ["cleartrail.mini.price_2026"], "reason": "Matches."}


@pytest.fixture
def client():
    app.state.tracer_factory = lambda: Tracer(enabled=False)
    app.state.runs_dir = None
    app.state.settings = app.state.settings.__class__(provider="fake", retry_delay_s=0.01, request_timeout_s=0.3)
    with TestClient(app) as c:
        yield c


def use(script):
    app.state.provider_factory = lambda: FakeProvider(script)


def test_classify_200(client):
    use([lookup(), submit([CLAIM])])
    r = client.post("/classify", json=BODY)
    assert r.status_code == 200
    [c] = r.json()["claims"]
    assert TEXT[c["start_char"]:c["end_char"]] == c["quote"]


@pytest.mark.parametrize("extra", [{"claims": []}, {"expected_label": "supported"}, {"gold_spans": []}])
def test_classify_rejects_gold_or_claim_fields_422(client, extra):
    r = client.post("/classify", json={**BODY, **extra})
    assert r.status_code == 422 and r.json()["error"]["type"] == "validation_error"


def test_classify_invalid_version_422(client):
    assert client.post("/classify", json={**BODY, "agent_version": "v3"}).status_code == 422


def test_classify_unknown_brand_404(client):
    use([])
    assert client.post("/classify", json={**BODY, "brand_id": "acme"}).status_code == 404


def test_classify_invalid_output_502(client):
    use([lookup(), submit([{**CLAIM, "quote": "not there"}])])
    r = client.post("/classify", json=BODY)
    assert r.status_code == 502 and r.json()["error"]["type"] == "invalid_output"


def test_classify_provider_error_502(client):
    use([err("auth")])
    assert client.post("/classify", json=BODY).status_code == 502


def test_classify_timeout_504(client):
    async def slow():
        await asyncio.sleep(2)
    use(lambda t, m, i: slow())
    r = client.post("/classify", json=BODY)
    assert r.status_code == 504 and r.json()["error"]["type"] == "timeout"


def test_get_run_invalid_uuid_422(client):
    assert client.get("/evaluation-runs/not-a-uuid").status_code == 422


@needs_db
def test_get_run_missing_404(client):
    asyncio.run(_init())
    assert client.get(f"/evaluation-runs/{uuid.uuid4()}").status_code == 404


async def _init():
    conn = await db.connect()
    await db.init_schema(conn)
    await conn.close()


@needs_db
def test_evaluation_run_persists_success_empty_and_error_answers(client, monkeypatch):
    """Check split through the API with a fake provider: A41 ok with a claim, A42 valid empty, A43 error."""
    check = [a for a in load_answers() if a.split == "check"]
    first_sentence = {a.answer_id: a.ai_answer.split(". ")[1] + "." for a in check}

    def script(text, msgs, i):
        aid = next(a for a in first_sentence if f'"answer_id": "{a}"' in text)
        if aid == "A43":
            return err("malformed")
        if len(msgs) == 1:
            return lookup()
        claims = [{**CLAIM, "quote": first_sentence[aid], "fact_ids": []}] if aid == "A41" else []
        return submit(claims)

    use(script)
    r = client.post("/evaluation-runs", json={"agent_version": "v1", "split": "check"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["metrics"] is None and body["counts"]["answers"] == 10
    run = client.get(f"/evaluation-runs/{body['run_id']}").json()
    assert run["status"] == "completed" and run["metrics"] is None
    answers = {a["answer_id"]: a for a in run["answers"]}
    assert list(answers) == sorted(answers) and len(answers) == 10
    assert answers["A41"]["status"] == "ok" and len(answers["A41"]["claims"]) == 1
    assert answers["A42"]["status"] == "ok" and answers["A42"]["claims"] == []
    assert answers["A43"]["status"] == "error" and answers["A43"]["error"]["type"] == "invalid_output"
    assert run["file_hashes"]["answers.csv"]


@needs_db
def test_db_constraints_reject_duplicate_answer_and_span():
    async def go():
        conn = await db.connect()
        await db.init_schema(conn)
        rid = uuid.uuid4()
        await db.create_run(conn, rid, {"agent_version": "v1", "split": "check", "prompt_version": "p",
                                        "dataset_version": "d", "evaluator_version": "e", "fact_version": "f",
                                        "provider": "fake", "model": "m", "settings": {}, "file_hashes": {}})
        row = {"answer_id": "A41", "brand_id": "b", "status": "ok", "latency_ms": 1, "model_turns": 1, "attempts": 1}
        c = {"claim_id": "A41:0:5", "start_char": 0, "end_char": 5, "quote": "abcde", "label": "unknown",
             "fact_ids": [], "reason": ""}
        with pytest.raises(psycopg.errors.UniqueViolation):
            await db.save_answer(conn, rid, row, [c, c])
        await db.save_answer(conn, rid, row, [c])
        with pytest.raises(psycopg.errors.UniqueViolation):
            await db.save_answer(conn, rid, row, [])
        await conn.close()
    asyncio.run(go())
