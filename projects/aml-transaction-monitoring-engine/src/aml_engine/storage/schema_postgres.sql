-- PostgreSQL DDL for AML Transaction Monitoring Engine (Educational Prototype)
-- Matches the SQLiteCaseRepository logical contract for seamless adapter portability.

CREATE TABLE IF NOT EXISTS monitoring_runs (
    run_id TEXT PRIMARY KEY,
    config_version TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    model_version TEXT,
    diagnostics_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY,
    subject_account_id TEXT NOT NULL,
    subject_entity_id TEXT,
    trigger_timestamp_iso TEXT NOT NULL,
    priority TEXT NOT NULL,
    combined_score DOUBLE PRECISION NOT NULL,
    provenance_hash TEXT NOT NULL,
    payload_json JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_alerts_subject_account ON alerts (subject_account_id);
CREATE INDEX IF NOT EXISTS idx_alerts_priority ON alerts (priority);

CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    alert_id TEXT NOT NULL UNIQUE REFERENCES alerts(alert_id),
    subject_account_id TEXT NOT NULL,
    subject_entity_id TEXT,
    status TEXT NOT NULL,
    disposition TEXT NOT NULL,
    system_priority TEXT NOT NULL,
    current_priority TEXT NOT NULL,
    assigned_analyst TEXT,
    override_count INTEGER NOT NULL DEFAULT 0,
    created_at_iso TEXT NOT NULL,
    updated_at_iso TEXT NOT NULL,
    payload_json JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cases_status ON cases (status);
CREATE INDEX IF NOT EXISTS idx_cases_priority ON cases (current_priority);

CREATE TABLE IF NOT EXISTS case_audit_events (
    event_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    sequence_num INTEGER NOT NULL,
    timestamp_iso TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    action TEXT NOT NULL,
    is_override BOOLEAN NOT NULL DEFAULT FALSE,
    override_reason_code TEXT,
    rationale TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    event_hash TEXT NOT NULL,
    payload_json JSONB NOT NULL,
    UNIQUE (case_id, sequence_num)
);
