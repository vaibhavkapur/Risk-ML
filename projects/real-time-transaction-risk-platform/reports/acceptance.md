# Acceptance Report — Real-Time Transaction Risk Platform

**Generated from measured run:** `2026-11-28T05:17:21Z`  
**Environment:** Python `3.9.6` on `macOS-26.5.1-arm64-arm-64bit` (`arm`), `config/default.yaml`, `seed = 42`  
**Model Artifact:** `model.logreg_cal.v1` (`sha256: e2730775eb0d9c98ef436235f1be53e91d96d64b51886f0de5b4d86fb078bb5d`)  
**Feature Version:** `feat.v1` | **Policy Version:** `policy.v1` | **Fallback Version:** `fallback.rules.v1`

> All numerical metrics below are **actual local observations** produced by `.venv/bin/pytest -v` and `.venv/bin/python scripts/run_demo.py --benchmark-requests 2000`, stored in machine-readable JSON/JSONL files under `reports/`. External distributed brokers (Kafka/Redpanda, live Redis) are explicitly marked **Not Run** in this offline prototype.

---

## 1. Acceptance Criteria Mapping

| Criterion (from `plan.md` §5) | Status | Measured Evidence & Source |
|---|---|---|
| **Offline/online feature parity on fixtures** | **PASS** | Hand-calculated 4-event oracle (`reference_transactions.json`): `max_abs_diff = 0.0` across all 12 features (`tests/test_contracts_and_reference.py`). Held-out streaming slice: `800` events compared, `max_abs_diff = 0.0` (`reports/parity_report.json`). |
| **Report p50/p99 latency at measured throughput** | **PASS** | Local in-process benchmark (`2,000` scoring requests with active timeout executor): **34,364.38 req/s**, **p50 = 0.0280 ms**, **p95 = 0.0363 ms**, **p99 = 0.0453 ms**, **max = 0.1262 ms** (`reports/benchmark.json`). |
| **Every decision traceable to model and feature version** | **PASS** | Every `RiskDecision` and FastAPI `/v1/score` response includes `decision_id`, `model_version`, `feature_version`, `policy_version`, `feature_freshness_ms`, `feature_status`, and HTTP headers (`X-Decision-Id`, `X-Model-Version`, `X-Feature-Version`, `X-Policy-Version`). Verified by `tests/test_api_load_and_recovery.py` and `800/800` exact audit replays in `reports/audit_log.jsonl`. |
| **All required test scenarios pass, or unavailable dependencies marked not run** | **PASS** | `13 passed in 1.40s` via `pytest -v`. Optional external Kafka/Redpanda and live Redis integrations are disabled in config and explicitly marked **Not Run** below. |
| **README documents setup, demo, validation, assumptions, and known limitations** | **PASS** | Documented in [README.md](../README.md) and [docs/runbook.md](../docs/runbook.md), distinguishing measured local results from synthetic fixtures and targets. |
| **No real transactions, customer data, credentials, or proprietary artifacts** | **PASS** | All fixtures and streams use deterministic synthetic tokens (`acct_syn_*`, `card_tok_*`, `merch_syn_*`); raw 13–19 digit PANs are actively rejected (`SensitiveDataViolationError` / HTTP 422). |

---

## 2. Test & Validation Matrix Coverage (`plan.md` §4)

| Test Area | Required Scenario | Test / Artifact Evidence | Observed Result |
|---|---|---|---|
| **Domain correctness** | Duplicate events (idempotent & conflicting) | `tests/test_domain_edge_cases.py::test_duplicate_idempotency_and_conflicting_duplicate_rejection` | Exact duplicate ignored (`DUPLICATE_IGNORED`, zero double-counting); mutated duplicate raises `DuplicateConflictError` / HTTP 409. |
| **Domain correctness** | Late / out-of-order events | `tests/test_domain_edge_cases.py::test_late_events_accepted_within_window_and_dropped_beyond_window` | Events within 5m lateness window `LATE_ACCEPTED` in sorted order with exact parity; events beyond 5m `LATE_DROPPED`. |
| **Domain correctness** | Stale cache | `tests/test_domain_edge_cases.py::test_stale_cache_triggers_conservative_escalation` | 30m stale account state flagged `STALE`; `$250` tx escalated from `APPROVE` to `REVIEW` (`fallback_reason = STALE_FEATURES`). |
| **Domain correctness** | Model timeout & runtime error | `tests/test_domain_edge_cases.py::test_model_timeout_and_model_error_fallbacks` & `reports/acceptance_summary.json` | 60–80ms injected slow model triggers `MODEL_TIMEOUT` at 25ms budget (`latency_ms = 28.1868ms` in demo), serving deterministic `fallback.rules.v1`. |
| **Integration & resilience** | Training-serving skew | `tests/test_parity_and_calibration.py::test_training_serving_skew_integration_clean_vs_drifted` & `reports/skew_report.json` | Clean test slice: `status = OK` (`max_psi = 0.0270`). Drifted serving slice: `status = CRITICAL_SKEW` (`max_psi = 1.9038`, `alert_triggered = true`). |
| **Boundary & failure** | Load & concurrency | `tests/test_api_load_and_recovery.py::test_concurrent_load_scoring_and_bounded_latency` | 400 concurrent requests across 4 worker threads, zero errors, `p99 < 25ms`. |
| **Boundary & failure** | Restart recovery | `tests/test_api_load_and_recovery.py::test_checkpoint_restart_recovery_preserves_state_and_decisions` & `reports/parity_report.json` | Atomic JSON checkpoint restore reproduces identical features (`max_abs_diff = 0.0`), identical `decision_id`/scores, and preserves duplicate deduplication. |
| **Reproducibility** | Fixed seed + versioned inputs | `tests/test_parity_and_calibration.py::test_reproducible_training_calibration_and_offline_online_parity` | Repeated runs with `seed = 42` produce bit-for-bit identical weights, Platt parameters, and `artifact_sha256`. |
| **Operational review** | Missing config, invalid inputs, corrupt artifact, missing Redis | `tests/test_contracts_and_reference.py::test_operational_errors_missing_config_corrupt_artifact_and_redis_unavailable` | Explicit `ConfigurationError`, `ArtifactIntegrityError`, `FeatureStoreError`, and HTTP 422/409 responses with zero silent success. |

---

## 3. Temporal Model & Calibration Observations (Synthetic Dataset)

Evaluated on the untouched chronological test slice (`704` synthetic events, positive rate `7.53%`) after excluding `483` initial 24h warm-up events and `455` immature training labels (`label_maturity_hours = 24`):

| Metric | Raw Logistic Regression | Platt-Calibrated (`model.logreg_cal.v1`) |
|---|---|---|
| ROC-AUC | `0.887633` | `0.887633` |
| PR-AUC | `0.745241` | `0.745241` |
| Brier Score | `0.028330` | `0.029850` |
| ECE (10-bin) | `0.015190` | `0.018667` |

*Engineering note:* On this synthetic logistic-generating process, the base logistic model is already well-calibrated (`ECE = 0.0152`), and fitting 2-parameter Platt scaling on the `234`-event matured calibration slice yielded a comparable `ECE = 0.0187` (`+0.0035` finite-sample variance). Both raw and calibrated scores are recorded on every decision (`raw_score`, `risk_score`) for transparency.

---

## 4. Untested / Not-Run External Capabilities

| Capability | Status | Reason |
|---|---|---|
| Live Kafka / Redpanda consumer-producer pipeline | **Not Run** | Offline prototype (`storage.kafka_enabled: false`); no local Kafka/Redpanda broker required or provisioned. |
| Live Redis feature store backend | **Not Run** | Offline prototype (`storage.redis_enabled: false`); `RedisFeatureStoreAdapter` interface is verified to fail explicitly when `redis` is unavailable. |
| Networked HTTP load testing across multi-node cluster | **Not Run** | Latency/throughput benchmark measures in-process scoring path; HTTP API correctness is verified via FastAPI `TestClient`. |

---

## 5. Raw Machine-Readable Evidence Files

- [acceptance_summary.json](../reports/acceptance_summary.json)
- [benchmark.json](../reports/benchmark.json)
- [parity_report.json](../reports/parity_report.json)
- [skew_report.json](../reports/skew_report.json)
- [runtime_metadata.json](../reports/runtime_metadata.json)
- [audit_log.jsonl](../reports/audit_log.jsonl)
