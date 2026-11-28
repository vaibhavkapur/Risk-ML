# Acceptance Evidence Report — Payment Authorization Optimization Engine

**Model Version:** `v1.0.0-bb892ac96c`  
**Config Hash:** `1364cf25c7eb2974` (`config/default.yaml`, `seed=42`)  
**Schema / Feature Version:** `1.0.0` / `1.0.0`  
**Environment:** Python `3.9.6`, `numpy 2.0.2`, `scikit-learn 1.6.1`, `fastapi 0.119.1`, `pytest 8.4.2` (`macOS-26.5.1-arm64`)  
**Data Provenance:** 100% synthetic data generated deterministically via `payment_auth_engine.synthetic` (zero real transactions, PII, or credentials).

---

## 1. Project-Specific Acceptance Criteria

| Acceptance Criterion (`plan.md` §5) | Status | Automated & Quantitative Evidence |
|---|---|---|
| Never retry a hard-declined or uncertain payment blindly | **PASS** | All hard decline codes (`14`, `54`, `57`, `59`, `63`) map to `BLOCK_HARD_DECLINE`; all uncertain outcomes (`TIMEOUT`, `NETWORK_ERROR`, `UNKNOWN`, unmapped codes, `prior_outcome_known=False`) map to `HOLD_RECONCILE`. Verified by `tests/test_taxonomy_and_retry.py` (15 tests), `tests/test_integration_resilience.py::test_engine_enforces_hard_decline_duplicate_and_uncertain_safety_end_to_end`, and `tests/test_api_and_cli.py`. |
| Report net acceptance and expected-loss bounds | **PASS** | Seeded 95% bootstrap confidence bounds reported for all three policies in [policy_comparison.json](../reports/policy_comparison.json). Example (`constrained_risk_aware`, synthetic test fold `n=1000`): DR net acceptance `0.3388` (95% CI `[0.3031, 0.3506]`, oracle `0.3102`), DR expected loss `44.00 bps` (95% CI `[39.28, 47.45] bps`, oracle `37.78 bps`). Verified by `tests/test_policy_and_counterfactual.py`. |
| No hidden constraint breaches | **PASS** | Every constrained decision attaches explicit `ConstraintCheckResult` audits and batch evaluation runs `audit_batch_constraints`. On the test fold ([audit_report.json](../reports/audit_report.json)): `constrained_risk_aware` achieves **0 decision-level breaches**, **0 portfolio breaches** (`42.05 bps <= 45.00 bps`, `P95 latency 400.0 ms <= 420.0 ms`), while `baseline_threshold` (`209.33 bps`) and `unconstrained_ev` (`46.97 bps`, `470.0 ms`) breaches are explicitly surfaced. |
| All required test scenarios pass in the supported environment | **PASS** | `35 passed` in `tests/` + reproducible demo + deterministic replay check (`replay_verified: true` across 1,000 decisions) via `scripts/validate_all.py`. |
| README documents setup, demo, validation, assumptions, and limitations; distinguishes measured synthetic results from targets | **PASS** | Documented in [README.md](../README.md) with explicit separation of synthetic observations vs. engineering targets and out-of-scope production items. |
| No real transactions, customer data, credentials or proprietary employer/SAAT artifacts | **PASS** | All fixtures (`data/fixtures/*.json`), synthetic data (`src/payment_auth_engine/synthetic.py`), and reports (`reports/*.json`) use synthetic IDs (`att_`, `txn_`, `merch_`) with zero external credentials or proprietary code. |

---

## 2. Test & Validation Matrix Coverage (`plan.md` §4)

| Test Area | Required Scenario | Test Module(s) | Measured Outcome |
|---|---|---|---|
| Domain correctness | Hard declines, duplicate attempts, uncertain authorization outcome | `tests/test_taxonomy_and_retry.py`, `tests/test_integration_resilience.py`, `data/fixtures/reference_oracle.json` | 100% pass; hand-calculated economics oracle verified within `1e-9` |
| Integration & resilience | Processor outage (failover & total outage fail-closed) and missing features | `tests/test_integration_resilience.py`, `tests/test_api_and_cli.py` | Primary `proc_a` outage fails over to secondary (`ROUTE_SECONDARY`); total outage fails closed (`BLOCK_PROCESSOR_OUTAGE`); all-null optional features trigger explicit conservative penalty & diagnostics |
| Boundary & failure behavior | Counterfactual evaluation assumptions (positivity, NaN propensity, zero support, ESS collapse) | `tests/test_policy_and_counterfactual.py`, `tests/test_contracts_and_fixtures.py` | Explicit `CounterfactualAssumptionError` (`POSITIVITY_VIOLATION`, `INVALID_PROPENSITY`, `INSUFFICIENT_SUPPORT`, `ESS_COLLAPSE`) and hand-calculated DM/IPS/SNIPS/DR oracle verified within `1e-9` |
| Reproducibility | Versioned inputs/config and fixed seed (`seed=42`) | `tests/test_reproducibility_and_ops.py`, `cli.py replay` | Exact bit-for-bit equality of model summary, counterfactual metrics, audit reports, and 1,000/1,000 `decision_id`s across reruns |
| Operational review | Missing/malformed config, unsupported inputs, restart/rerun diagnostics | `tests/test_reproducibility_and_ops.py`, `tests/test_api_and_cli.py`, `docs/runbook.md` | Explicit `ConfigurationError` (CLI exit code `2`), HTTP `422` contract errors, HTTP `404` unknown processor, structured diagnostic counters |

---

## 3. Summary of Measured Synthetic Benchmark Results (`n_test = 1,000`)

> **Important:** These observations are measured on **synthetic data** generated with `seed=42` (`config/default.yaml`). They validate engineering correctness and estimator behavior, not real-world production causal uplift.

| Policy | DR Net Acceptance (95% CI) | Oracle Acceptance | DR Expected Loss bps (95% CI) | Oracle Loss bps | Oracle Net Value (cents/att) | P95 Latency (ms) | Constraint Audit |
|---|---:|---:|---:|---:|---:|---:|---|
| `baseline_threshold` | `0.6518` `[0.6415, 0.6859]` | `0.6291` | `173.26` `[173.26, 244.63]` | `228.21` | `-28.20` | `210.0` | **FAIL** (loss breach) |
| `unconstrained_ev` | `0.4982` `[0.4766, 0.5292]` | `0.4847` | `42.82` `[42.82, 52.19]` | `50.03` | `+55.55` | `470.0` | **FAIL** (loss + latency breach) |
| `constrained_risk_aware` | `0.3388` `[0.3031, 0.3506]` | `0.3102` | `44.00` `[39.28, 47.45]` | `37.78` | `+47.12` | `400.0` | **PASS** (0 breaches) |

**Calibration Quality (Temporal Calibration Fold):**
- Acceptance model (`n=671`): ROC-AUC `0.7125`, Brier `0.1873 -> 0.1849`, ECE `0.0527 -> 0.0294`.
- Matured fraud/loss model (`n_train_matured=1,254`, `n_cal_matured=450`): ROC-AUC `0.8685`, Brier `0.0190 -> 0.0188`, ECE `0.0134 -> 0.0065`.

---

## 4. Explicitly Untested / Out-of-Scope Capabilities

- **Live payment processor / card-network integration (ISO-8583 wire protocols, Visa/Mastercard tokenization, real 3DS ACS challenges):** Not implemented / out of scope.
- **Production causal validation (live A/B tests or online contextual bandits):** Not run; observational offline evaluation cannot establish real-world causal uplift without randomized logging propensities.
- **Distributed persistent idempotency storage (Redis/PostgreSQL) and high-concurrency load/soak benchmarking:** Not tested; prototype uses deterministic in-memory `IdempotencyLedger` and `ProcessorHealthRegistry`.
- **PCI-DSS compliance, KMS encryption, authentication/RBAC, and production security hardening:** Explicitly excluded from educational/research prototype scope.
