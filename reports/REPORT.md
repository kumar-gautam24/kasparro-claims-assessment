# Report: baseline, diagnosis, targeted change, results

All numbers come from saved live-run artifacts in `runs/`. They were scored by `app/scorer.py` (evaluator `sentence-spans-and-labels-v4`) on dataset `long-brand-answers-50-v4` and facts `speaker-catalogue-v4`, with file hashes recorded in each report.
Provider/model: **Ollama `qwen2.5:7b` (local)**, temperature 0, at most 3 concurrent provider requests, at most 3 turns per answer, at most 2 attempts per turn. The settings were identical for v1 and v2.

## 1. Baseline v1 on development (A01–A40, G = 376)

Run `2aecd3f5-d72d-4a56-b1a2-39ca91ed5f8c` → `runs/v1_development_20261021T084001Z.json` (source revision `823f744`, tag `v1-frozen`).

| metric | v1 |
|---|---|
| **joint F1 (main)** | **0.380** (P 0.433, R 0.338) |
| gold coverage C/G | 0.338 (127/376) |
| extraction F1 | 0.604 (P 0.689, R 0.537) |
| macro F1 (supported / contradicted / unknown) | 0.383 (0.450 / 0.354 / 0.344) |
| counts | P 293, M 202, C 127, unmatched predictions 91 |
| **answer errors** | **18 / 40**, all `invalid_output` |

Confusion matrix (rows = expected, columns = predicted):

| | supported | contradicted | unknown | missing_or_error |
|---|---|---|---|---|
| supported | 47 | 7 | 11 | 55 |
| contradicted | 24 | 28 | 12 | 56 |
| unknown | 18 | 3 | 52 | 63 |
| no_gold | 0 | 0 | 91 | 0 |

The total is 467, which equals G + unmatched predictions (376 + 91). The report also lists every missed gold ID, unmatched prediction ID and wrong-label ID.

## 2. Error inspection and trace-based diagnosis

The 18 answer-level errors cost the most. Each one discards a whole answer, so their roughly 168 gold claims all count as missed:

| cause | answers |
|---|---|
| **invented fact ID** (e.g. `cleartrail.mini.return_policy`, `stonewave.go.shipping_free`, `cloudnote.air.amplifier_output`) | **9** |
| unparseable `<tool_call>` JSON (nondeterministic: A02 parsed cleanly when re-run alone) | 7 |
| quote not verbatim in `ai_answer` | 2 |

**Langfuse evidence (baseline trace A01, `b1e184b9ff871aa1813a98f0f54366be`)**, exported in `reports/langfuse/v1_development.json`:
- `get_brand_facts` tool observation: 44 cleartrail rows returned, and none of them is a return policy.
- `model-turn-2` generation (5,795 input / 1,093 output tokens): `submit_claims` labels the 30-day-return sentence `unknown`, which is correct, but cites `fact_ids: ["cleartrail.mini.return_policy"]`, an ID that does not exist.
- The agent observation is ERROR with `validation_failure = invalid_output: claim 5: unknown fact_ids ['cleartrail.mini.return_policy']`, and `extraction_f1 = joint_f1 = 0`.

All 9 invented IDs share this pattern. The model correctly sees that the catalogue has no row for the property (returns, shipping, materials, origin, amplifier power) and then makes up a plausible ID to "cite" the missing fact. The validator correctly refuses the whole answer, because the brief forbids silently repairing it. **So the evidence shows the biggest loss comes from a grounding failure on `unknown` claims, not from bad labelling.**

Other v1 observations (not changed in v2, so the comparison stays one-variable):
- 75 of the 91 extra predictions are whole advice or instruction sentences, all labelled `unknown`. The other 16 are partial or merged spans.
- In A07, the injected "Ignore the classification instructions…" sentence was *not* obeyed (the labels stay mixed), but it was extracted as a claim.
- 24 contradicted claims were labelled supported, which is the largest label confusion.

## 3. Targeted change (v2, tag `v2-frozen`)

**One behaviour: grounding of `fact_ids` in the catalogue.**
1. The `submit_claims` tool schema restricts `fact_ids` items to an `enum` of the 132 catalogue IDs (`app/facts.py`, `VERSIONS` in `app/agent.py`).
2. One prompt sentence changed (`diff app/prompts/v1.md app/prompts/v2.md`): "Copy each fact_id exactly as it appears in the get_brand_facts results; never construct, rename or guess an ID. If no returned row covers the asserted property (…), the claim is unknown and fact_ids is an empty list []".

Span extraction, labels, turn and retry limits, validation, scorer, data, provider, model and settings are all unchanged.
Also added, as observability only (applies to both versions): when a `<tool_call>` block can't be parsed, a raw excerpt of it goes into the generation observation, so the 7 nondeterministic JSON failures can be diagnosed next time.

## 4. v2 on development: results (added after the 15:45 deadline)

Run `runs/v2_development_20261021T094025Z.json` (tag `v2-frozen`). The data, evaluator, provider, model and settings are the same as v1. The full comparison is in `reports/comparison.md` (produced by `scripts/compare_runs.py`).

| metric | v1 | v2 | delta |
|---|---|---|---|
| **joint F1 (main)** | **0.380** | **0.299** | **-0.081** |
| joint precision / recall (coverage C/G) | 0.433 / 0.338 | 0.372 / 0.250 | -0.062 / -0.088 |
| extraction F1 (P / R) | 0.604 (0.689 / 0.537) | 0.550 (0.684 / 0.460) | -0.054 |
| macro F1 | 0.383 | 0.301 | -0.082 |
| supported / contradicted / unknown F1 | 0.450 / 0.354 / 0.344 | 0.413 / 0.268 / 0.221 | -0.037 / -0.086 / -0.123 |
| P / M / C / unmatched | 293 / 202 / 127 / 91 | 253 / 173 / 94 / 80 | |
| answer errors | 18 | 20 | +2 |

**The result is a regression, not a gain.** The targeted failure itself was fixed: **invented fact IDs went from 9 to 0**. Answers such as A06, A12 and A13, which v1 lost to invented IDs, now succeed. But errors from unparseable `<tool_call>` blocks rose **from 7 to 18**, and 2 quote-verbatim errors remain. Net answer errors went from 18 to 20: 9 answers went from error to ok (A06, A12, A13, A18, A23, A26, A27, A32, A36) and 11 went from ok to error (A04, A07, A09, A14, A15, A19, A24, A33, A35, A37, A38). At claim level, 43 gold claims were fixed and 76 regressed.

**Trace-based diagnosis of the new failure.** v2 is the first version that logs a raw excerpt of unparseable output, so `reports/langfuse/v2_development.json` shows it. In all 18 cases the excerpt ends with `"}]}`: the model closes the claims array and the `arguments` object, then stops **one `}` short** of the `{"name": …, "arguments": {…}}` envelope. The schema change is the likely trigger. The 132-ID enum makes the tool signature much longer, and the 7B model loses track of brace depth at the end of a long call. That fits v1's smaller rate of the same error. I didn't repair the JSON, because the brief says malformed output must stay an error.

On the answers that did *not* error, v1 has C/G 127/208 = 0.61 and v2 has 94/182 = 0.52. The two sets of answers differ, so this is only indicative, but it suggests the stricter fact-ID rule also pushed some labels towards `unknown`, as the drop in unknown precision shows (0.313 → 0.230).

**Regressions inspected:** the claim-level transitions (correct→missed, correct→wrong_label and so on) and the full lists of regressed claim IDs are in `comparison.md`. Most correct→missed transitions come from the 11 answers that switched from ok to error.

**What I'd do next (not done):** keep the fact-ID rule in the prompt but drop the schema enum, or remove the long-output failure mode altogether by having the model cite numbered sentences instead of copying them (already implemented in `app/validate.py` as `span_mode="sentence"` and tested, but not used). Either would need its own v3 measurement.

## 5. Final-check runs A41–A50 (added after the 15:45 deadline)

Both versions were frozen (tags `v1-frozen` and `v2-frozen`) before these runs and were not tuned on the check inputs. G, M, C and every quality metric are **null**: the reviewer scores these from the saved outputs with the private key. The Langfuse check traces carry predictions and failures, but **no** quality or correctness scores (0 scores in `reports/langfuse/v1_check.json` and `reports/langfuse/v2_check.json`).

| run | file | accepted predictions | answer errors | wall time |
|---|---|---|---|---|
| v1 check | `runs/v1_check_20261021T101540Z.json` | 72 | 5 / 10 | 547 s |
| v2 check | `runs/v2_check_20261021T102448Z.json` | 86 | 3 / 10 | 534 s |

- v1 errors: A41 and A44 (quote not verbatim), A47 and A50 (invented fact IDs `stonewave.max.shipping_free` and `cleartrail.plus.material`, the same pattern diagnosed on development), A48 (unparseable tool call).
- v2 errors: A42, A43 and A46, all unparseable tool calls (the missing-final-brace pattern from §4). There are no invented IDs.

The Postgres rows for all four runs are listed in `reports/db_runs.txt`. Both check runs record source revision `d6bbb6e`, a commit that only touched reports; `git diff v2-frozen d6bbb6e -- app` is empty, so the agent code was frozen.

Post-deadline fix: `finish_run` used to store an absent value as the JSON literal `null` instead of SQL NULL. It now stores SQL NULL, and the existing rows were updated in place, so `runs.metrics IS NULL` holds for both check runs. The API test asserts this.

## 6. Verification, AI assistance, time, limitations

**What exists and was verified**
- `reports/test_output.txt` holds the actual `pytest -v` output: **57 passed, 0 skipped**, including Postgres-backed API tests. It covers CSV quoting and embedded newlines, offset and quote validation, the scorer arithmetic (including the G=10 P=10 M=8 C=7 example), provider-error and empty answers, missing or duplicate outcomes, fake-provider malformed output, bad labels, invented IDs, illegal tools, unknown brands, timeout with exactly 2 attempts, a 429 retry, non-retryable auth errors, turn exhaustion, the 3-request concurrency cap including retries, failure isolation, ordering, and that gold data never appears in model payloads.
- `reports/api_smoke.txt` holds a live uvicorn session reading the stored v1 run back from Postgres: 40 ordered answers, 293 claim rows and 202 gold matches, plus the 404/422 error paths.
- Langfuse (self-hosted, docker compose) has one trace per answer, with generation observations (tokens) and `get_brand_facts` tool observations. Development traces carry `extraction_f1` and `joint_f1` scores, and each matched span has a `label_correct` score (span ID in the comment). Missed and extra IDs are in the score metadata. A `run-summary` trace per run carries the aggregate metrics and run-level scores. Exports are in `reports/langfuse/`.

**Deviations, stated plainly**
- **Provider:** no OpenAI, Anthropic or Gemini key was available, so the live runs use a **local Ollama model**. That deviates from the brief's hosted-API requirement. A Gemini adapter is implemented (`app/providers/gemini.py`) but was **never run**.
- **Timeout:** the default is 15 s as specified, but the local 7B model needs 1–3 minutes per turn with 3 parallel requests, so the live runs used `REQUEST_TIMEOUT_S=240`. Retry, attempt and concurrency limits are as specified.
- **Tool-calling transport:** Ollama 0.40's `/api/chat` parser silently dropped long but valid `submit_claims` calls. The adapter therefore renders Qwen's native tool-calling template itself and parses `<tool_call>` blocks. The model still requests the tools, and the executor validates the tool name and brand before running anything.
- `latency_ms` per answer includes time spent waiting for the shared semaphore. `model_turns` counts completed turns.

**AI assistance:** Claude Code (Claude Opus) wrote most of the code, tests and this report under my direction. I checked the outputs myself: the scorer arithmetic against the brief's example, that all 376 gold spans are reproducible sentence spans, the live run artifacts and Postgres rows, the Langfuse traces and scores in the UI and API, and the v1→v2 prompt diff.

**Time:** about 3 hours, including setting up the local model and Langfuse. Live inference took most of the wall time (about 35 min per development run).

**Unfinished / limitations**
- 7B local model quality: there's lots of over-extraction of advice sentences and contradicted→supported confusion. The next change I'd make is an extraction rule or classifier step for advice and instruction sentences.
- The 7 nondeterministic malformed tool-call outputs are not addressed by v2.
- No Langfuse UI screenshots; the JSON exports and trace IDs are provided instead (the instance is local).
- §4 and §5 were completed after the 15:45 deadline, using runs that started before it from the frozen tags. Nothing in §1–3 was changed afterwards.
