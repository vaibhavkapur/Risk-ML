# Acceptance & Validation Report

**Project:** AML Transaction Monitoring Engine (`projects/aml-transaction-monitoring-engine/`)  
**Validation Date:** 2026-11-28  
**Runtime Environment:** Python `3.9.6` on `macOS-26.5.1-arm64-arm-64bit` (offline stdlib environment)  
**Configuration:** `config/default_config.json` (`config_version=1.0.0`, `config_hash=a738d18172e4413b`, `seed=42`)  
**Model Artifact:** `logreg-platt-1.0.0-93b3da0d0608` (`reports/ml_prioritizer_artifact.json`)

---

## 1. Project-Specific Acceptance Criteria Mapping

| Acceptance Criterion (`plan.md` §5) | Status | Measured Evidence & Verification Command |
|---|---|---|
| **Alerts are reproducible and explainable.** | **PASS** | Verified in `tests/test_m4_ml_evaluation_and_override_audit.py::test_strict_reproducibility_and_no_automatic_accusation` and `scripts/run_demo.py`. Two independent runs produced identical `run_id` (`RUN-5EB6FAD0E7778CDB`), identical `model_version` (`logreg-platt-1.0.0-93b3da0d0608`), byte-identical serialized alerts, `all_evidence_provenance_verified=true`, and `replay_idempotent=true` (`cases_created=0`, `cases_reused_on_replay=62`). Every alert contains rule/graph `ReasonContribution` entries plus exact ML log-odds feature contributions. |
| **Report precision/recall on synthetic scenarios and review volume.** | **PASS** | Recorded in [`reports/baseline_vs_ml_evaluation.json`](baseline_vs_ml_evaluation.json) via `python3 scripts/run_evaluation.py`. On `synthetic_aml_seed42_d90` (2,280 normalized transactions, 62 rule candidates: 49 true synthetic suspicious, 13 benign false positives; overall synthetic suspicious tx coverage recall = `1.0000`), strictly out-of-time test window (`n=12`, 9 positives) showed: **Rules-Only Baseline** precision `0.7500`, recall `1.0000`, F1 `0.8571`, PR-AUC `0.8623`, review volume `12`; **Hybrid Rule+ML (`P1_HIGH` queue)** precision `1.0000`, recall `1.0000`, F1 `1.0000`, PR-AUC `1.0000`, review volume `9` (**25.0% workload reduction**; ML calibration `Brier=0.00119`, `ECE=0.015127`). |
| **No automatic accusation.** | **PASS** | Enforced in `models.py` (`ADVISORY_NOTICE`, `CaseDisposition.PENDING_ANALYST_REVIEW`) and `workflow.py` (`SYSTEM_MONITOR` blocked from analyst review transitions). Asserted across all 62 synthetic alerts & cases in `test_strict_reproducibility_and_no_automatic_accusation` (zero accusatory verdict terms). |
| **All required test scenarios pass in the supported environment, or unavailable dependencies explicitly marked not run.** | **PASS** | `PYTHONPATH=src python3 -m unittest discover -s tests -p "test_*.py" -v` executed **15/15 tests passing (`OK`)** in ~0.83s. Unavailable optional dependencies in this offline environment are explicitly documented in Section 3 below. |
| **README documents setup, demo, validation, assumptions, and known limitations; measured results distinguish observations from synthetic fixtures/targets.** | **PASS** | Documented in [`README.md`](../README.md), [`docs/runbook.md`](../docs/runbook.md), and Section 2 below. |
| **No real transactions, customer data, credentials or proprietary employer/SAAT artifacts in source, fixtures, logs or reports.** | **PASS** | 100% synthetic identifiers (`ENT-*`, `ACC-*`, `TX-*`), documented in [`data/fixtures/PROVENANCE.md`](../data/fixtures/PROVENANCE.md). |

---

## 2. Test & Validation Matrix Evidence (`plan.md` §4)

| Test Area | Required Scenario | Test Evidence | Outcome |
|---|---|---|---|
| **Domain correctness** | Legitimate high-volume activity and synthetic suspicious chains | `test_m1_foundation_and_contracts.py::test_reference_hand_calculated_oracle` (tolerance `1e-6`); `test_m2_domain_detection.py::test_legitimate_high_volume_activity_produces_zero_false_positives` (0 false positives across merchant + payroll traffic); `test_m2_domain_detection.py::test_synthetic_suspicious_chains_all_five_scenarios_detected` (5/5 scenarios detected) | **PASS** |
| **Integration and resilience** | Duplicate events and missing entity mapping | `test_m3_integration_and_case_workflow.py::test_duplicate_events_and_missing_entity_mapping_diagnostics` (`duplicate_events_skipped=1`, `missing_entity_mappings=1`, explicit `PARTIAL_OR_UNMAPPED` status & warning); `test_replay_idempotency_and_sqlite_restart_recovery`; `test_service_and_http_api_end_to_end` | **PASS** |
| **Boundary and failure behavior** | Reviewer override audit | `test_m4_ml_evaluation_and_override_audit.py::test_reviewer_override_audit_boundary_and_tamper_detection` (rejects missing reviewer, `SYSTEM_MONITOR`, illegal status transitions, missing override code, short rationale; validates hash-chained override event and detects payload tampering) | **PASS** |
| **Reproducibility** | Same versioned inputs/configuration and fixed seed | `test_m4_ml_evaluation_and_override_audit.py::test_strict_reproducibility_and_no_automatic_accusation` + `test_temporal_ml_evaluation_and_artifact_roundtrip` | **PASS** |
| **Operational review** | Missing configuration, unsupported inputs, restart/rerun | `test_m1_foundation_and_contracts.py::test_missing_config_raises_explicit_error`; `test_invalid_inputs_fixture_rejected_explicitly`; `test_m2_domain_detection.py::test_unsupported_input_propagation_strict_vs_non_strict`; `test_m3_integration_and_case_workflow.py::test_replay_idempotency_and_sqlite_restart_recovery` | **PASS** |

---

## 3. Explicitly Marked Not-Run Optional Capabilities

The following optional stack adapters from `plan.md` are implemented in source code and pinned in [`requirements-optional.txt`](../requirements-optional.txt), but were **NOT RUN** in this offline validation environment because third-party packages and live database/network sockets are unavailable in the sandbox:

| Optional Capability | Implementation Location | Status in This Environment | How to Run When Available |
|---|---|---|---|
| **PostgreSQL live database adapter** | `src/aml_engine/storage/repository.py` (`PostgresCaseRepository`) & `schema_postgres.sql` | **NOT RUN** (`psycopg` and live PostgreSQL instance not installed; identical SQL contract verified on `SQLiteCaseRepository`) | Install `psycopg[binary]==3.2.3`, provision local Postgres, instantiate `PostgresCaseRepository(dsn)` |
| **FastAPI / Uvicorn ASGI server** | `src/aml_engine/api/fastapi_app.py` (`create_fastapi_app`) | **NOT RUN** (`fastapi`/`uvicorn` not installed; identical service contract & HTTP routes verified via `dispatch_http_request`) | Install `requirements-optional.txt`, run `uvicorn` against `create_fastapi_app(service)` |
| **NetworkX graph export** | `src/aml_engine/features/graph.py` (`TransactionGraph.to_networkx`) | **NOT RUN** (`networkx` not installed; core graph algorithms verified directly on `TransactionGraph`) | Install `networkx==3.2.1`, call `graph.to_networkx()` |
| **Live TCP localhost HTTP socket bind** | `src/aml_engine/api/http_server.py` (`create_http_server`) | **NOT RUN** (sandbox denies loopback `connect()`; identical dispatcher verified in-process via `dispatch_http_request`) | Run `PYTHONPATH=src python3 -m aml_engine.cli serve --host 127.0.0.1 --port 8000` outside restricted sandbox |

---

## 4. Raw Machine-Readable Artifacts

- [`reports/baseline_vs_ml_evaluation.json`](baseline_vs_ml_evaluation.json) — Temporal split, calibration (Brier/ECE), precision/recall/PR-AUC, workload reduction, and environment metadata.
- [`reports/demo_run_output.json`](demo_run_output.json) — Full batch run diagnostics, replay idempotency proof, sample explainable alerts with provenance hashes, and audited reviewer override summary.
- [`reports/ml_prioritizer_artifact.json`](ml_prioritizer_artifact.json) — Versioned serialized ML model artifact (`logreg-platt-1.0.0-93b3da0d0608`).
