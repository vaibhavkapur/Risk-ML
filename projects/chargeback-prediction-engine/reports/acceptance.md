# Acceptance Report — Chargeback Prediction Engine

**Generated from measured run:** `run_8c20db71476a0e` (`config/default.yaml`, `config_hash=bc6cc12f868f442e`, `seed=42`) and `run_d79762a126d3cc` (`config/cold_start.yaml`, `config_hash=c5e912cfee91b607`, `seed=108`)  
**Environment:** Python 3.9.6 (`macOS-26.5.1-arm64`), `polars==1.36.1`, `scikit-learn==1.6.1`, `numpy==2.0.2`, `pytest==8.4.2` (full snapshot in [`runtime_metadata.json`](runtime_metadata.json))  
**Data provenance:** 100% synthetic offline fixtures and deterministic synthetic datasets. All numbers below are **measured synthetic-benchmark observations**, not production metrics or guarantees.

---

## 1. Project-specific acceptance criteria mapping

| Acceptance Criterion (`plan.md` §5) | Status | Measured Evidence & Artifact Link |
|---|---|---|
| Report PR-AUC, calibrated expected loss and precision at review budget | **PASS** | Default temporal holdout (`n=1,020`, 38 disputes, 3.73% rate): Primary Calibrated **PR-AUC = 0.4237** (vs 0.1285 logistic baseline), **Calibrated Expected Loss = \$4,638.32** vs Realized Loss **\$4,273.12** (ratio `1.085`; ECE = `0.00766`), **Precision@5%Budget = 0.2549** (expected-net-savings ranked; 54.23% loss capture) / **0.3333** (probability ranked; 56.57% loss capture). Raw outputs: [`demo_summary.json`](demo_summary.json), [`baseline_vs_calibrated.json`](baseline_vs_calibrated.json), [`review_simulation.json`](review_simulation.json). |
| Exclude immature negatives | **PASS** | Verified at two levels: (1) Hand-calculated reference oracle excludes both immature negatives (`TX_005`, `TX_006`) with `label=null, is_trainable=false` across both `cohort_mature` and `exclude_immature_negatives_only` modes (`tests/test_contracts_and_maturity.py::test_immature_negatives_are_always_excluded`). (2) Default synthetic demo explicitly excludes `1,341` immature negatives (`maturity_diagnostics.immature_negative_count=1341`) and masks `3` future disputes after `as_of_ts`. |
| Reproduce metrics | **PASS** | Bit-for-bit identical logical output (`exact_match: true`, identical `run_id=run_8c20db71476a0e`, `PR-AUC=0.423724`, `Expected Loss=$4638.32`, `Precision@Budget=0.254902`) across two independent runs: [`reproducibility_check.json`](reproducibility_check.json) and `tests/test_models_and_calibration.py::test_reproducibility_bit_for_bit_identical`. |
| All required test scenarios pass in supported environment | **PASS** | `pytest` passed **17 / 17** tests covering all 5 matrix areas (see §2 below). |
| README documents setup, demo, validation, assumptions, limitations; distinguishes observations vs synthetic fixtures vs targets | **PASS** | Documented in [`README.md`](../README.md), [`docs/runbook.md`](../docs/runbook.md), [`docs/contracts.md`](../docs/contracts.md), and [`docs/architecture.md`](../docs/architecture.md). |
| No real transactions, customer data, credentials or proprietary employer/SAAT artifacts | **PASS** | All fixtures in `data/fixtures/` and generator logic in `src/chargeback_engine/synthetic.py` are synthetic and original; zero credentials or external production endpoints. |

---

## 2. Test and validation matrix evidence (`plan.md` §4)

| Test Area | Required Scenario | Test / Benchmark Evidence | Result |
|---|---|---|---|
| Domain correctness | Immature labels and missing reason codes | `tests/test_contracts_and_maturity.py` (`test_reference_fixture_matches_hand_calculated_oracle`, `test_immature_negatives_are_always_excluded`, `test_missing_and_unrecognized_reason_codes_handled_deterministically`, `test_future_transactions_excluded_at_earlier_snapshot`); `tests/test_features.py::test_zero_future_dispute_leakage_in_causal_features` | **PASS (6 tests)** |
| Integration and resilience | Class imbalance and merchant cold start | `tests/test_models_and_calibration.py` (`test_default_pipeline_calibration_and_metrics`, `test_class_imbalance_and_merchant_cold_start_integration`); `tests/test_features.py::test_cold_start_merchant_shrinks_smoothly_to_prior`; [`baseline_vs_calibrated.json`](baseline_vs_calibrated.json) | **PASS (3 tests + benchmark)** |
| Boundary and failure behavior | Temporal holdout and duplicate disputes | `tests/test_boundary_and_resilience.py` (`test_duplicate_disputes_raise_under_strict_policy`, `test_duplicate_disputes_deduplicated_deterministically_when_configured`, `test_temporal_holdout_rejects_overlapping_splits_and_null_labels`); `tests/test_policy_and_monitoring.py` (`test_review_prioritization_respects_budget_and_non_punitive_guardrails`, `test_punitive_merchant_policy_rejected_at_config_validation`, `test_delayed_outcome_monitoring_matures_monotonically`) | **PASS (6 tests)** |
| Reproducibility | Same versioned inputs/configuration and fixed seed | `tests/test_models_and_calibration.py::test_reproducibility_bit_for_bit_identical`; [`reproducibility_check.json`](reproducibility_check.json) | **PASS (`exact_match=true`)** |
| Operational review | Missing configuration, unsupported inputs, rerun diagnostics | `tests/test_boundary_and_resilience.py::test_missing_and_malformed_config_operational_errors`; `tests/test_contracts_and_maturity.py::test_invalid_contract_fixtures_rejected`; CLI exit code `2` verified on missing config | **PASS (2 tests)** |

---

## 3. Baseline vs calibrated benchmark summary (synthetic holdout)

### 3.1 Default scenario (`config/default.yaml`, `n_test=1,020`, observed rate `3.73%`, realized loss `\$4,273.12`)

| Model / Calibration | PR-AUC | ROC-AUC | Brier Score | ECE | Expected Loss (USD) | Exp / Realized Loss | Precision@5% (Value) | Precision@5% (Prob) | Loss Capture (Value) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline Logistic (Platt-calibrated) | 0.1285 | 0.6701 | 0.03489 | 0.01123 | \$4,420.27 | 1.034x | 0.0980 | 0.1765 | 28.35% |
| Primary HistGBDT (Uncalibrated, balanced) | 0.4237 | 0.8539 | 0.11747 | 0.25099 | \$39,281.41 | 9.193x | 0.0784 | 0.3333 | 32.35% |
| **Primary HistGBDT (Platt-calibrated)** | **0.4237** | **0.8539** | **0.03036** | **0.00766** | **\$4,638.32** | **1.085x** | **0.2549** | **0.3333** | **54.23%** |

**Discrepancy explanation:** Uncalibrated `class_weight="balanced"` HistGBDT preserves ranking (`PR-AUC=0.4237`), but over-predicts average probability by `7.74x` (`mean_pred=0.288` vs `0.037`), inflating expected loss by `9.19x` and causing value-ranked review selection to over-weight large-ticket low-risk transactions (`Precision@5%=0.0784`). Chronological Platt calibration on the disjoint calibration split reduces Brier score by `0.08710`, reduces ECE by `0.24333`, brings expected loss within `8.5%` of realized loss, and boosts value-ranked Precision@5% from `0.0784` to `0.2549` (realized net savings `\$1,045.08` vs `\$436.84` for baseline logistic).

### 3.2 Severe class imbalance + high cold-start scenario (`config/cold_start.yaml`, 45% cold-start merchants, `n_test=933`, observed rate `1.93%`)

| Model / Slice | PR-AUC | ROC-AUC | Brier Score | ECE | Precision@5% (Value) | Loss Capture (Value) | Realized Net Savings (USD) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline Logistic (calibrated) | 0.2038 | 0.8447 | 0.01755 | 0.00580 | 0.1304 | 39.59% | \$404.83 |
| **Primary HistGBDT (calibrated)** | **0.1785** | **0.9134** | **0.01776** | **0.00308** | **0.1739** | **62.37%** | **\$535.28** |
| Slice: Cold-Start Merchants (`n=340`, 8 pos) | 0.3440 | 0.9527 | 0.02204 | 0.01286 | 0.2941 | 85.44% | — |
| Slice: Established Merchants (`n=593`, 10 pos) | 0.2112 | 0.9130 | 0.01530 | 0.00492 | 0.1724 | 68.48% | — |

### 3.3 Delayed-outcome monitoring trajectory (`review_simulation.json`, immature cohort `n=1,404`)

| Checkpoint Offset | Mature Fraction | Mean Expected Arrival CDF | Expected Disputes | Observed Disputes | O/E Ratio | Reviewed Queue Precision | Status |
|---:|---:|---:|---:|---:|---:|---:|---|
| +0d (`2025-10-15`) | 0.00% | 89.80% | 34.66 | 63 | 1.8176 | 0.2714 | `DRIFT_ALERT` |
| +15d (`2025-10-30`) | 32.69% | 95.75% | 36.90 | 65 | 1.7615 | 0.2714 | `DRIFT_ALERT` |
| +30d (`2025-11-14`) | 68.45% | 98.06% | 37.78 | 65 | 1.7204 | 0.2714 | `DRIFT_ALERT` |
| +60d (`2025-12-14`) | 100.00% | 99.91% | 38.48 | 66 | 1.7150 | 0.2714 | `DRIFT_ALERT` |
| +90d (`2026-01-13`) | 100.00% | 100.00% | 38.52 | 66 | 1.7134 | 0.2714 | `DRIFT_ALERT` |

*Diagnostic note:* The immature cohort contains a larger share of newly onboarded cold-start merchants whose actual dispute rate (`4.70%`) is higher than the earlier calibration split (`3.04%`). The delay-adjusted monitor surfaces `DRIFT_ALERT` immediately at `+0d` (before any transactions in that cohort reach 90-day maturity), demonstrating early concept-drift detection without leaking immature negatives into training.

---

## 4. Explicitly untested / out-of-scope capabilities

- **LightGBM runtime:** Not installed or tested in this environment (scikit-learn `HistGradientBoostingClassifier` was chosen per the plan's "LightGBM or scikit-learn" stack option).
- **Live streaming / online feature store / real payment gateway integration:** Out of scope; offline batch prototype only.
- **Real cardholder / merchant datasets:** Excluded; all metrics are from synthetic fixtures and synthetic generation.
- **Automated punitive merchant actions (holds, reserves, termination):** Explicitly prohibited and blocked by config validation.
- **GitHub Actions CI remote execution:** Workflow file [`.github/workflows/chargeback-prediction-engine.yml`](../../../.github/workflows/chargeback-prediction-engine.yml) is authored and all CI steps were verified locally; remote runner execution is marked **not run** in this offline workspace.
