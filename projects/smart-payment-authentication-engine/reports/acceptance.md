# Acceptance Evidence Report — Smart Payment Authentication Engine

**Run timestamp (UTC):** `2026-11-28T06:17:25Z`  
**Config:** [`config/default.yaml`](../config/default.yaml) (`config_hash = e4ca065622f53430`, `seed = 42`)  
**Runtime:** Python `3.9.6`, `numpy==1.26.4`, `scikit-learn==1.5.2`, `fastapi==0.115.6`, `pydantic==2.10.4`, `pytest==8.3.4` (`macOS-26.5.1-arm64`)  
**Model Artifact:** `gbm_sigmoid_seed42_fv1.0.0` (`artifact_sha256 = f4b38904a1a2b7a2e04f53c61075adbc705e5a00c61b94bcab3d56da80810111`)

> **Important Note on Measured vs. Synthetic Results:** All metrics below are **actual observed measurements** from running the automated test suite and `scripts/validate_all.py` on **100% deterministic synthetic fixtures and generated datasets**. They are not production guarantees or real portfolio observations.

---

## 1. Project-Specific Acceptance Criteria Mapping

| # | Acceptance Criterion | Status | Evidence & Measured Results |
|---|---|---|---|
| 1 | **No policy bypass of mandatory challenge rules** | **PASS** | Enforced structurally in [`src/smart_auth_engine/policy.py`](../src/smart_auth_engine/policy.py). Verified by `test_mandatory_challenge_rules_cannot_be_bypassed_even_with_extreme_thresholds` (0 bypasses even at `challenge_threshold=0.98`, `p=0.0001`) and chronological holdout audit (`mandatory_challenge_bypass_count = 0` across all 752 test events under calibrated, oracle, partial-maturity, and outage runs in [`chronological_evaluation.json`](chronological_evaluation.json)). |
| 2 | **Report fraud loss, acceptance and challenge rate tradeoffs** | **PASS** | Constrained optimizer evaluated `1,570` candidate `(challenge_threshold, decline_threshold)` pairs on the chronological validation split (`n=729`) and emitted the Pareto frontier in [`policy_tradeoff_frontier.json`](policy_tradeoff_frontier.json). Selected feasible policy (`challenge >= 0.0561`, `decline >= 0.1500`): validation `fraud_loss_rate = 0.76%` (`<= 1.5%`), `challenge_rate = 19.34%` (`<= 22.0%`), `acceptance_rate = 89.18%`. Holdout test split (`n=752`): `fraud_loss_rate = 3.61%`, `challenge_rate = 19.68%`, `genuine_acceptance_rate = 92.50%`. |
| 3 | **Fallback deterministic** | **PASS** | Verified by `test_missing_critical_feature_triggers_deterministic_fallback`, `test_model_outage_and_corrupted_scores_fallback_deterministically`, and `test_toggle_outage_on_healthy_engine_preserves_determinism`. Outage drill over all 752 test events (`729` fallback decisions + `23` high-risk overrides) reproduced identically across two independent demo passes. |
| 4 | **All required test scenarios pass (or unavailable dependencies marked not run)** | **PASS** | `pytest`: **20/20 passed in 7.69s**; `scripts/validate_all.py`: **2-pass bit-for-bit reproducibility passed**. Live Redis daemon test explicitly marked **NOT RUN** below (adapter verified via in-memory fake client). |
| 5 | **README documents setup, demo, validation, assumptions, and known limitations** | **PASS** | Documented in [`README.md`](../README.md) and [`docs/runbook.md`](../docs/runbook.md), distinguishing measured prototype outputs from synthetic data assumptions and engineering targets. |
| 6 | **No real transactions, customer data, credentials, or proprietary artifacts** | **PASS** | 100% synthetic data generated with `seed=42`; strict `extra="forbid"` rejects unexpected sensitive fields (verified by `unknown_extra_field_forbidden` fixture in `data/fixtures/invalid_payments.json`). |

---

## 2. Test and Validation Matrix Evidence

| Test Area | Required Scenario | Automated Tests / Artifacts | Measured Outcome |
|---|---|---|---|
| **Domain correctness** | Missing features and model outage | `tests/test_domain_missing_features_and_outage.py`, `tests/test_contracts_and_fixtures.py` | 9/9 tests passed; reference oracle matches analytical expectations within `1e-6`; missing critical features and model outages deterministically invoke fallback (`approve`/`challenge`/`decline`/`abstain`) without bypassing mandatory rules. |
| **Integration and resilience** | High-risk action overrides | `tests/test_policy_optimization_and_overrides.py`, `reports/chronological_evaluation.json` | 3/3 tests passed; prior chargebacks, extreme IP velocity, and emulator+TOR overrides force `DECLINE` (`override_bypass_count = 0` across all 752 holdout events). |
| **Boundary and failure behavior** | Chronological policy evaluation and label delay | `tests/test_training_and_calibration.py`, `tests/test_step_up_and_chronological_eval.py` | 5/5 tests passed; `1,027` immature training labels excluded prior to `2026-01-31T00:00:00Z`; partial-maturity snapshot (`+5d`) separates `87` matured vs. `665` immature events; out-of-order streams raise `ChronologicalOrderError` and missing labels raise `LabelAlignmentError`. |
| **Reproducibility** | Same versioned inputs/configuration and fixed seed | `test_end_to_end_fixed_seed_reproducibility`, `scripts/validate_all.py` | Pass 1 and Pass 2 of `scripts/run_demo.py` produced identical `chronological_evaluation.json` outputs; 25/25 sampled decisions replayed with `exact_match = true`. |
| **Operational review** | Missing configuration, unsupported inputs, restart/rerun | `tests/test_api_and_operations.py` | 3/3 tests passed; missing/invalid YAML raises `ConfigurationError`; invalid API payloads return structured `422 VALIDATION_ERROR`; `JsonlDecisionStore` reloads cleanly across simulated process restart and replays accurately. |

---

## 3. Measured Calibration and Policy Comparison

### Calibration on Chronological Validation Split (`n = 729`, empirical fraud rate `4.94%`)
Source: [`reports/baseline_vs_calibrated_metrics.json`](baseline_vs_calibrated_metrics.json)

| Model / Calibrator | ROC-AUC | Brier Score | Log-Loss | ECE | Mean Predicted Prob |
|---|---:|---:|---:|---:|---:|
| Raw GBM (`none`) | `0.8568` | `0.03756` | `0.14353` | `0.02148` | `0.06489` |
| **Sigmoid-Calibrated GBM (`sigmoid`, selected)** | **`0.8568`** | **`0.03657`** | **`0.14158`** | **`0.02061`** | **`0.06318`** |
| Isotonic-Calibrated GBM (`isotonic`) | `0.8484` | `0.03792` | `0.17110` | `0.01644` | `0.06163` |
| Uncalibrated Reference Oracle (`reference_oracle_v1.0.0`) | `0.9078` | `0.04602` | `0.17642` | `0.07887` | `0.12825` |

### Holdout Chronological Test Split Comparison (`n = 752`, `2026-02-20` to `2026-03-01`)
Source: [`reports/chronological_evaluation.json`](chronological_evaluation.json)

| Policy Variant | Challenge Rate | Decline Rate | Genuine Acceptance | Matured Fraud Loss Rate | Mandatory Bypass | Override Bypass |
|---|---:|---:|---:|---:|---:|---:|
| **Calibrated GBM Policy (full maturity)** | **`19.68%`** | **`8.78%`** | **`92.50%`** | **`3.61%`** | **`0`** | **`0`** |
| Reference Oracle Baseline (same thresholds) | `31.91%` | `20.08%` | `80.20%` | `1.48%` | `0` | `0` |
| Model Outage Deterministic Fallback | `33.91%` | `3.06%` | `94.63%` | `4.24%` | `0` | `0` |

**Discrepancy explanation (Validation vs. Holdout Test):**
- On the validation split (`n=729`), the selected thresholds achieved an expected fraud loss rate of `0.76%` (`<= 1.5%` constraint). On the holdout chronological test split (`n=752`), realized fraud loss rate was `3.61%` while maintaining `challenge_rate = 19.68%` (`<= 22%`) and `92.50%` genuine acceptance. This gap reflects heavy-tailed synthetic transaction amounts (concentrated on `2026-02-25` and `2026-02-26`), simulated stochastic challenge escapes (`fraud_pass_prob = 0.14`), and `35` missing-feature fallback events.
- The uncalibrated Reference Oracle achieves lower holdout fraud loss (`1.48%`) only because its uncalibrated probabilities over-predict the base rate (`12.83%` vs. `4.94%`), causing it to exceed the `22%` challenge budget (`31.91%`) and slash genuine acceptance by `12.3` percentage points (`80.20%` vs. `92.50%`).

---

## 4. Explicitly Untested / Out-of-Scope Capabilities

| Capability | Status | Rationale |
|---|---|---|
| Live Redis server persistence | **Not run** | Optional `redis` daemon is not provisioned in the offline test environment. `OptionalRedisDecisionStore` is verified via an in-memory stub (`test_optional_redis_store_adapter_with_fake_client`) and fails loudly if `redis` is absent. |
| EMVCo 3DS 2.x / PSD2 SCA protocol compliance | **Out of scope** | Educational/research prototype with simulated step-up outcomes only. |
| Production load / latency SLA benchmarking | **Out of scope** | Prototype runs single-process locally (`TestClient` / `uvicorn`). |
| Live card network / issuer / real PII integration | **Excluded** | Prohibited by project scope. |
