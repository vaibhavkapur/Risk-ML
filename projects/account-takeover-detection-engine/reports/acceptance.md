# Acceptance Report — Account Takeover Detection Engine

**Generated from measured run:** `2026-11-28`  
**Environment:** Python `3.9.6` (`macOS-26.5.1-arm64`), `numpy==1.26.4`, `scikit-learn==1.5.2`, `fastapi==0.115.6`, `pydantic==2.10.4`, `pytest==8.3.4`  
**Raw machine-readable evidence:** [evaluation_results.json](evaluation_results.json), [decision_audit.jsonl](decision_audit.jsonl), [model_bundle.joblib.manifest.json](model_bundle.joblib.manifest.json)  
**Model artifact SHA-256:** `90ed6fefe1f9a3e8f20ffc1c4a57eae4aed9e6ef5bd1a83cfacf6323214b9ec9`

> **Important disclaimer:** All metrics below are **measured observations on deterministic synthetic simulation data** (`seed=42`, `2,985` synthetic events using RFC 5737 documentation IPs), **not** production performance guarantees.

---

## 1. Project-Specific Acceptance Criteria

| Acceptance Criterion | Status | Evidence |
|---|---|---|
| **Report recall at fixed false-positive rate and calibration** | **PASS** | Measured on chronological test split (`n=597`, `143` ATO, `454` benign) in `reports/evaluation_results.json` (summarized in Section 3 below). Verified by `tests/test_domain_models_and_policy.py::test_recall_at_fixed_fpr_and_calibration_reported`. |
| **No future events in features** | **PASS** | Enforced in `StateStore` (`ts < before_ts`) and `features.assert_no_future_events()`. Verified by `tests/test_contracts_and_features.py::test_no_future_events_in_features_store_and_guard` (proves pre-loading future events into the store produces identical features and direct injection raises `ContractValidationError`). |
| **Every decision logged** | **PASS** | `ATODetectionEngine.evaluate()` unconditionally writes to `DecisionAuditLogger`. Measured `597/597` (`100%`) test decisions persisted in `reports/decision_audit.jsonl`; verified across normal, outage, stale-model, and drift-breaker paths in `tests/test_integration_and_boundaries.py::test_drift_alert_store_outage_stale_model_and_audit_logging` and `tests/test_api.py`. |
| **All required test scenarios pass (or unavailable deps marked not run)** | **PASS** | `pytest -v`: **14 passed in 3.86s**. All 6 scenario suites in `scripts/run_evaluation.py` returned `PASS`. Live Redis cluster test explicitly marked **Not run** in Section 4. |
| **README documents setup, demo, validation, assumptions, and limitations** | **PASS** | Documented in `README.md`, `docs/runbook.md`, `docs/contracts.md`, `docs/architecture.md`, and `docs/integration.md`, distinguishing synthetic measurements from production targets. |
| **No real transactions, customer data, credentials, or proprietary artifacts** | **PASS** | Synthetic generator uses RFC 5737 documentation IP blocks (`192.0.2.0/24`, `198.51.100.0/24`, `203.0.113.0/24`) and synthetic IDs (`usr_*`, `dev_*`, `ses_*`). Schema forbids unknown fields (`extra="forbid"`). |

---

## 2. Test & Validation Matrix Evidence

| Test Area | Required Scenario | Automated Test / Evidence | Result |
|---|---|---|---|
| **Domain correctness** | New device vs compromised session | `tests/test_contracts_and_features.py::test_reference_oracle_hand_calculated_features_and_ordering`, `tests/test_domain_models_and_policy.py::test_new_device_vs_compromised_session`, `evaluation_results.json -> required_scenarios.new_device_vs_compromised_session` (`new_device_score=0.000007 -> ALLOW` vs `compromised_score=1.0 -> BLOCK_AND_RESET`) | **PASS** |
| **Integration & resilience** | Shared network | `tests/test_integration_and_boundaries.py::test_shared_network_scenario_dampens_false_blocks`, `required_scenarios.shared_network` (`inferred_shared_network=1.0`, `shared_network_dampening=+0.06`, `ALLOW`) | **PASS** |
| **Integration & resilience** | Missing data | `tests/test_integration_and_boundaries.py::test_missing_data_triggers_fallback_and_never_fails_open`, `required_scenarios.missing_data_fallback` (`fallback_reason=excessive_missing_features`, `CHALLENGE_OTP`) | **PASS** |
| **Integration & resilience** | Delayed label | `tests/test_domain_models_and_policy.py::test_chronological_split_enforces_delayed_label_maturity` (`134` immature rows with `label_available_at > train_cutoff` excluded from training split) | **PASS** |
| **Boundary & failure behavior** | Repeated attempts | `tests/test_integration_and_boundaries.py::test_repeated_attempts_escalation_and_block`, `required_scenarios.repeated_attempts` (`5+` fails -> WebAuthn; `10+` fails -> `BLOCK_AND_RESET`) | **PASS** |
| **Boundary & failure behavior** | Feature expiry | `tests/test_integration_and_boundaries.py::test_feature_expiry_past_ttl`, `required_scenarios.feature_expiry` (`profile_expired=True`, `user_prior_event_count_30d=0.0`, `CHALLENGE_OTP`) | **PASS** |
| **Reproducibility** | Fixed seed & deterministic replay | `tests/test_integration_and_boundaries.py::test_reproducibility_and_missing_config_error`, `evaluation_results.json -> replay_reproducibility` (`597/597` identical `decision_id`, `action`, and `calibrated_risk_score` within `1e-6`) | **PASS** |
| **Operational review** | Missing config, unsupported inputs, store outage, stale model, tampered artifact, HTTP errors | `tests/test_contracts_and_features.py`, `tests/test_domain_models_and_policy.py::test_model_artifact_persistence_and_tamper_detection`, `tests/test_integration_and_boundaries.py`, `tests/test_api.py` (422 contract / 400 unsupported) | **PASS** |

---

## 3. Measured Chronological Test Results (Synthetic Stream, `seed=42`)

- **Split:** `1,657` mature training rows (`134` immature rows excluded before `2026-01-28T04:01:39Z`), `597` calibration rows (through `2026-02-06T07:28:08Z`), `597` test rows (`454` benign, `143` ATO).

| Model / Baseline | ROC-AUC | PR-AUC | Brier Score | ECE | Recall @ 1% FPR | Recall @ 2% FPR | Recall @ 5% FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rule_oracle_baseline` (hand-calculated reference) | 0.9994 | 0.9981 | 0.047378 | 0.160949 | 0.9790 | 1.0000 | 1.0000 |
| `anomaly_isolation_forest` (unsupervised baseline) | 0.9230 | 0.7481 | 0.121605 | 0.144861 | 0.2098 | 0.2517 | 0.4685 |
| `logistic_regression` (linear supervised) | 1.0000 | 1.0000 | 0.000165 | 0.001903 | 1.0000 | 1.0000 | 1.0000 |
| `supervised_uncalibrated_blend` (before Platt calibration) | 1.0000 | 1.0000 | 0.005093 | 0.061413 | 1.0000 | 1.0000 | 1.0000 |
| **`calibrated_ensemble`** (Platt calibrated) | **1.0000** | **1.0000** | **0.000003** | **0.000491** | **1.0000** | **1.0000** | **1.0000** |

**Calibration gain observed:** Post-hoc Platt calibration reduced ECE from `0.061413` (uncalibrated blend) and `0.160949` (rule oracle) down to `0.000491`, and Brier score from `0.005093` to `0.000003`.

**Adaptive policy summary on test split (`n=597`):**
- Benign (`454`): `ALLOW=454`, benign block rate `0.00%`, benign step-up rate `0.00%`
- ATO (`143`): `BLOCK_AND_RESET=142`, `CHALLENGE_WEBAUTHN=1`, ATO intervened rate `100.00%`
- Average end-to-end evaluation latency (including causal feature lookup, scoring, evidence, and disk JSONL logging): `8.033 ms/event`

---

## 4. Explicitly Untested / Out-of-Scope Capabilities

| Capability | Status | Rationale |
|---|---|---|
| **Live Redis cluster integration (`RedisStateStore`)** | **Not run** | Optional `redis` extra and local Redis daemon are not installed/running in this offline environment. Adapter code exists in `src/ato_engine/store.py`; store-outage fallback behavior is verified via fault injection on `InMemoryStateStore`. |
| **Live production traffic / real PII / real device fingerprinting** | **Out of scope** | Excluded by project scope; requires independent security, privacy, legal, and risk reviews before any production adoption. |
| **High-concurrency load / multi-worker horizontal scaling benchmark** | **Not run** | Single-process local evaluation prototype only; production deployment requires shared external state (Redis/feature store) and async audit shipping. |
