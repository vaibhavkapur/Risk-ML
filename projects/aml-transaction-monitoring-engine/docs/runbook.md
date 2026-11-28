# Operator & Developer Runbook

## 1. Environment Setup

From the project directory (`projects/aml-transaction-monitoring-engine`):

```bash
python3 --version  # Requires Python >= 3.9
```

No virtual environment or third-party installation is required for the core engine. To optionally install FastAPI, NetworkX, and PostgreSQL extras in a network-enabled environment:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-optional.txt
pip install -e .
```

---

## 2. Standard Validation & Demo Commands

```bash
# Run full unit, integration, boundary, and acceptance test suite
PYTHONPATH=src python3 -m unittest discover -s tests -p "test_*.py" -v

# Run temporal rule-vs-ML evaluation (writes reports/baseline_vs_ml_evaluation.json & ml_prioritizer_artifact.json)
python3 scripts/run_evaluation.py

# Run reproducible end-to-end demo (writes reports/demo_run_output.json)
python3 scripts/run_demo.py

# Monitor a single JSON fixture in strict validation mode
PYTHONPATH=src python3 -m aml_engine.cli monitor --input data/fixtures/reference_hand_calc.json --strict

# Monitor boundary fixture (logs duplicate skips & missing entity mapping warnings cleanly)
PYTHONPATH=src python3 -m aml_engine.cli monitor --input data/fixtures/boundary_cases.json --verbose
```

---

## 3. Diagnostics & Structured Counters

Every monitoring batch produces a persisted `RunDiagnostics` object (`GET /runs` or `diagnostics` field in CLI/API output):
- `total_raw_events`, `valid_normalized_events`, `invalid_events_rejected`
- `duplicate_events_skipped`: duplicate `tx_id` events ignored deterministically
- `missing_entity_mappings`: count of transactions where account or entity linkage is missing (never silently masked; warnings appear in `diagnostics.warnings` and `EvidenceRecord.entity_mapping_warnings`)
- `scenario_hits_count`, `alerts_generated`, `cases_created`, `cases_reused_on_replay`
- `config_version`, `config_hash`, `feature_version`, `policy_version`, `model_version`

---

## 4. Replay, Restart, and Failure Recovery Procedures

### Missing or Corrupt Configuration
- **Symptom**: `ConfigurationError` raised at startup or CLI invocation.
- **Recovery**: Verify `--config` path points to a valid JSON file (default `config/default_config.json`) containing all required top-level keys (`fx_rates_to_usd` must include `"USD"`). Never run with implicit fallback thresholds.

### Replaying a Failed or Interrupted Batch
- **Procedure**: Re-run the exact same batch against the persistent SQLite/PostgreSQL repository (`--db path/to/cases.sqlite3`).
- **Expected Behavior**: Stable content-derived `alert_id` and `case_id` values ensure idempotent upsert (`cases_created=0`, `cases_reused_on_replay=N`). Existing analyst review state and audit trails are preserved.

### Investigating Provenance or Audit Chain Failure
- Run `GET /alerts/{alert_id}` (`provenance_verified: true`) and `GET /cases/{case_id}` (`audit_chain_verified: true`).
- If either returns `false`, halt downstream consumption, inspect the specific sequence index returned in `audit_chain_error`, and compare against immutable backup snapshots.

### Handling Reviewer Override Rejection (`CaseWorkflowError` / HTTP 400)
- Overrides (changing priority away from `system_priority`, closing a `P1_HIGH` case as benign, or escalating a `P3_LOW` case) require:
  1. Non-empty human `reviewer_id` (not `SYSTEM_MONITOR`).
  2. Valid `override_reason_code` (see `docs/contracts.md`).
  3. Substantive `rationale` of at least 10 characters.

---

## 5. Limitations & Operational Guardrails
- Educational prototype only; do not connect to live banking ledgers, real customer PII, or regulatory filing portals.
- Every alert is advisory (`PENDING_ANALYST_REVIEW`) and requires human analyst judgment.
