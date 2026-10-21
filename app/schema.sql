CREATE TABLE IF NOT EXISTS runs (
    run_id            UUID PRIMARY KEY,
    agent_version     TEXT NOT NULL CHECK (agent_version IN ('v1', 'v2')),
    split             TEXT NOT NULL CHECK (split IN ('development', 'check')),
    status            TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    prompt_version    TEXT NOT NULL,
    dataset_version   TEXT NOT NULL,
    evaluator_version TEXT NOT NULL,
    fact_version      TEXT NOT NULL,
    provider          TEXT NOT NULL,
    model             TEXT NOT NULL,
    settings          JSONB NOT NULL,
    file_hashes       JSONB NOT NULL,
    source_revision   TEXT,
    counts            JSONB,
    metrics           JSONB,
    report            JSONB,
    failure           TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at       TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS answer_results (
    run_id       UUID NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    answer_id    TEXT NOT NULL,
    brand_id     TEXT NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('ok', 'error')),
    error_type   TEXT,
    error_message TEXT,
    n_claims     INTEGER NOT NULL,
    latency_ms   INTEGER NOT NULL,
    model_turns  INTEGER NOT NULL,
    attempts     INTEGER NOT NULL,
    usage        JSONB,
    trace_id     TEXT,
    scores       JSONB,
    PRIMARY KEY (run_id, answer_id)
);

CREATE TABLE IF NOT EXISTS claim_results (
    id           BIGSERIAL PRIMARY KEY,
    run_id       UUID NOT NULL,
    answer_id    TEXT NOT NULL,
    claim_id     TEXT NOT NULL,
    start_char   INTEGER NOT NULL CHECK (start_char >= 0),
    end_char     INTEGER NOT NULL CHECK (end_char > start_char),
    quote        TEXT NOT NULL,
    label        TEXT NOT NULL CHECK (label IN ('supported', 'contradicted', 'unknown')),
    fact_ids     JSONB NOT NULL,
    reason       TEXT NOT NULL,
    gold_match   JSONB,
    FOREIGN KEY (run_id, answer_id) REFERENCES answer_results(run_id, answer_id) ON DELETE CASCADE,
    UNIQUE (run_id, answer_id, start_char, end_char)
);
