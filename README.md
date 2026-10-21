# Claim extraction & classification agent (Kasparro FSE I assessment)

A Python agent reads a long AI answer about speakers and finds every factual claim sentence in it. It looks up the catalogue through a **model-requested `get_brand_facts` tool** and labels each sentence `supported`, `contradicted` or `unknown`. Python computes and validates every span. A deterministic harness scores extraction and classification, and a thin FastAPI + PostgreSQL service and Langfuse tracing sit on top.

> **Results, diagnosis and the v1→v2 comparison: see [`reports/REPORT.md`](reports/REPORT.md).**

## Layout

| Path | What |
|---|---|
| `data/` | The issued CSVs, byte-identical; `SHA256SUMS` records their hashes. Every run report also stores the hashes. |
| `app/data.py` | `csv` parsing with UTF-8 and `newline=''`; typed fact values (`"false"` → `False`, never `bool("false")`). |
| `app/facts.py` | Read-only `get_brand_facts(brand_id)` (fact_version `speaker-catalogue-v4`) plus the tool schemas. |
| `app/agent.py` | Executor: at most 3 model turns, at most 2 attempts per turn, one global `asyncio.Semaphore(3)` around every provider attempt including retries, a per-request timeout, and retries only on timeout, 429 or transient errors. It validates tool names and brand arguments and forces `submit_claims` on the last turn. |
| `app/validate.py` | Locates quotes and derives offsets in Python; `claim_id = answer_id:start:end`; checks bounds, quote equality, whitespace, duplicates, overlaps, ordering, labels and fact IDs. Any violation turns the whole answer into an error. |
| `app/scorer.py` | Exact-span scorer: extraction and joint P/R/F1, coverage C/G, per-label P/R/F1, macro F1, a 4×4 confusion matrix, and missed/extra/wrong-label IDs. It rejects missing or duplicate answer outcomes. |
| `app/runner.py` | Async split runner: failures are isolated per answer and output is sorted. Writes the JSON report plus an exchange log, the Postgres rows and the Langfuse scores. |
| `app/prompts/v1.md`, `v2.md` | Frozen prompts. v2 differs only in the extraction-locator section (see REPORT). |
| `app/api.py`, `app/db.py`, `app/schema.sql` | FastAPI routes and parameterized psycopg3 SQL. |
| `app/providers/` | `ollama.py` (the one used for the live runs), `gemini.py` (implemented but **not run**, since no key was available) and `fake.py` (tests). |
| `scripts/run_eval.py`, `compare_runs.py`, `export_langfuse.py` | CLI entry points. |
| `runs/` | Live-run reports (`*.json`) and sanitized model/tool exchanges (`*_exchanges.jsonl`). At submission only v1 development had finished; see REPORT §4–5. |
| `reports/` | REPORT.md, test output, Langfuse exports, comparison. |

## Setup

Requirements: Python 3.13 with [uv](https://docs.astral.sh/uv/), Docker, and Ollama (`brew install ollama`).

```bash
cp .env.example .env                 # then edit PROVIDER/MODEL; secrets stay in .env (gitignored)
docker compose up -d                 # app Postgres on :5433 + self-hosted Langfuse on :3000
uv sync                              # installs the pinned versions from uv.lock

# local model used for the submitted runs
OLLAMA_NUM_PARALLEL=3 OLLAMA_FLASH_ATTENTION=1 ollama serve &
ollama pull qwen2.5:7b
```

`DATABASE_URL` defaults to `postgresql://claims:claims@localhost:5433/claims` (the `appdb` service). The schema is created on first use from `app/schema.sql`.
Langfuse UI: http://localhost:3000 (headless-init demo login `admin@example.com` / `localdemo123`; project keys `pk-lf-local-demo` / `sk-lf-local-demo` are local-only demo values set in `docker-compose.yml`).

Pinned key versions: fastapi 0.156.6, pydantic 2.14.1, psycopg 3.3.6, langfuse 5.0.4 (server image `langfuse:4`), google-genai 2.32.0, httpx 0.28.1, pytest 9.1.1, pytest-asyncio 1.4.0, Ollama 0.40.7 with `qwen2.5:7b` (845dbda0ea48). See `uv.lock` for the rest.

## Run

```bash
# the four live runs (each writes runs/<version>_<split>_<timestamp>.json and stores rows in Postgres)
uv run python scripts/run_eval.py --version v1 --split development
uv run python scripts/run_eval.py --version v2 --split development
uv run python scripts/run_eval.py --version v1 --split check
uv run python scripts/run_eval.py --version v2 --split check

uv run python scripts/compare_runs.py runs/v1_development_*.json runs/v2_development_*.json reports/comparison.md
uv run python scripts/export_langfuse.py runs/<report>.json reports/langfuse/<name>.json

# API
uv run uvicorn app.api:app --port 8000
curl -s localhost:8000/classify -H 'content-type: application/json' \
  -d '{"answer_id":"X1","brand_id":"cleartrail","question":"q","ai_answer":"ClearTrail Mini costs INR 1,299 including tax but excluding shipping.","agent_version":"v2"}'
curl -s -XPOST localhost:8000/evaluation-runs -H 'content-type: application/json' -d '{"agent_version":"v1","split":"check"}'
curl -s localhost:8000/evaluation-runs/<run_id>
```

## Test

```bash
uv run pytest -v                     # 57 tests; API/DB tests need `docker compose up -d appdb`, else they skip
```
The actual output is in `reports/test_output.txt`. All tests use the fake provider; none of them call a model. The four live runs are separate artifacts.
