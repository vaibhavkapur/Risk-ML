# Acceptance Evidence — Credit Risk Scoring Platform

**Run Date:** 2026-11-28  
**Runtime Environment:** Python `3.9.6` (`macOS-26.5.1-arm64`), `numpy==1.26.4`, `scipy==1.13.1`, `scikit-learn==1.5.2`, `matplotlib==3.9.2`, `PyYAML==6.0.2`, `joblib==1.4.2`, `pytest==8.3.4`  
**Config Digest (`config/default.yaml`):** `6eca6ff09edfee4fc1513ae9ab812e1373d4ddff149f5f47937524c5697f9662` (`seed=42`)  
**Data Provenance:** 100% deterministic synthetic data (`contains_real_customer_data = false`)

---

## 1. Test and Validation Matrix Coverage

| Test Area | Required Scenario | Automated Verification | Status | Evidence |
|---|---|---|---|---|
| Domain correctness | Missing values, distribution shift, and leakage from post-origination data | `tests/test_contracts_and_leakage.py` (6 tests), `tests/test_domain_correctness.py` (3 tests) | **PASS** | `18/18` pytest pass; hand-calculated oracle matches within `1e-6`; missing fields imputed on train medians with explicit `missing_*` flags; post-origination columns & future timestamps raise `PointInTimeLeakageError`; immature labels segregated (`n=83` censored in default run); shifted cohort triggers CSI/PSI warnings/alerts |
| Integration & resilience | Calibration and threshold monotonicity | `tests/test_calibration_and_monotonicity.py` (3 tests) | **PASS** | Both `sigmoid` (Platt) and `isotonic` calibrators pass `verify_monotonicity()` on 200-point grid; anti-monotone synthetic scores raise `CalibrationMonotonicityError`; `ManualReviewPolicy.verify_threshold_monotonicity()` confirms non-decreasing band severity & non-increasing SLA across 500-point grid |
| Boundary & failure behavior | Insufficient subgroup sample | `tests/test_subgroups_and_boundaries.py` (2 tests) | **PASS** | Strict mode raises `InsufficientSubgroupSampleError` with exact group count when `n < min_subgroup_sample_size`; reporting mode marks group `INSUFFICIENT_SAMPLE`, sets `metrics=null`, and appends explicit diagnostic warning |
| Reproducibility | Same versioned inputs/config and fixed seed | `tests/test_reproducibility_and_ops.py::test_end_to_end_reproducibility_same_seed_same_outputs`, `test_bundle_reload_replay_and_explanations` | **PASS** | Identical AUROC, PR-AUC, Brier, ECE, PSI, SHA-256 `decision_id`s, and calibrated PDs across independent runs; `replay_verification.decision_id_match == true` |
| Operational review | Missing config, unsupported inputs, restart/rerun | `tests/test_reproducibility_and_ops.py::test_operational_error_handling_missing_and_invalid_configs`, `test_manual_review_store_and_html_workbench` | **PASS** | Missing/malformed YAML raises `ConfigurationError`; unsupported backend/calibrator/disposition or `automated_decision_allowed=true` raises `UnsupportedInputError`; bundle reload + replay verified; structured JSON logs & diagnostic counters recorded |

---

## 2. Project-Specific Acceptance Criteria

| Criterion | Status | Evidence & Notes |
|---|---|---|
| Report AUROC, PR-AUC, Brier score and calibration plots | **PASS** | Measured on held-out temporal test split (`n=797`, `83` immature excluded):<br>- **Logistic Regression (sigmoid):** AUROC `0.7640`, PR-AUC `0.4602`, KS `0.4170`, Brier `0.1364`, ECE `0.0367`<br>- **Boosted Trees (sigmoid):** AUROC `0.7433`, PR-AUC `0.4625`, KS `0.3713`, Brier `0.1388`, ECE `0.0299`<br>- **Plots:** [`calibration_plot.png`](calibration_plot.png), [`calibration_plot.svg`](calibration_plot.svg)<br>- **Raw JSON:** [`evaluation_report.json`](evaluation_report.json), [`baseline_comparison.json`](baseline_comparison.json) |
| Document limitations and uncertainty | **PASS** | Documented in [`model_card.md`](model_card.md) Section 7 and [`README.md`](../README.md), including synthetic-data limitations, calibration lag under temporal shift (calibration split DR `0.1551` vs shifted test DR `0.2020`, causing mean calibrated PD `0.1745` to lag observed DR by `-0.0275`), missing-field epistemic uncertainty, and small-subgroup withholding |
| No automated lending decision | **PASS** | Structurally enforced across config parser, `ManualReviewPolicy`, `ScoringDecisionOutput` (`automated_decision_allowed=false`, `manual_review_required=true`), `ManualReviewStore`, HTML Workbench banner, and regression tests (`test_policy_threshold_monotonicity_and_no_automated_lending`, `test_manual_review_store_and_html_workbench`) |
| All required test scenarios pass, or unavailable dependencies explicitly marked not run | **PASS** | `18 passed in pytest`. Optional dependencies marked explicitly **not installed / not run** below |
| README documents setup, demo, validation, assumptions, limitations; distinguishes actual observations vs synthetic fixtures | **PASS** | Verified in [`README.md`](../README.md) |
| No real transactions, customer data, credentials or proprietary artifacts | **PASS** | 100% synthetic generator & hand-crafted fixtures (`data/fixtures/`); no credentials, live endpoints, or external/employer code |

---

## 3. Baseline Comparison & Discrepancy Analysis

From [`baseline_comparison.json`](baseline_comparison.json):
- **Reference (`logistic_regression_calibrated`):** AUROC `0.7640`, PR-AUC `0.4602`, Brier `0.1364`, ECE `0.0367`
- **Candidate (`boosted_trees_calibrated`):** AUROC `0.7433`, PR-AUC `0.4625`, Brier `0.1388`, ECE `0.0299`
- **Deltas:** `auroc_delta = -0.020748`, `brier_delta = +0.002381`, `candidate_ece = 0.029857`
- **Check outcomes:** `brier_within_tolerance = true` (`<= 0.015`), `calibrated_ece_within_tolerance = true` (`<= 0.05`), `auroc_within_tolerance = false` (missed the `-0.02` target by `0.000748`).
- **Explanation of discrepancy:** On the moderate-sized synthetic training window (`n=2,730`, `469` defaults) where the underlying synthetic log-odds formula is predominantly linear with modest interaction terms, regularized logistic regression retains a slight ranking edge on AUROC (`+0.0207`), while the boosted tree model achieves marginally higher PR-AUC (`0.4625` vs `0.4602`) and better test-set calibration error (`ECE 0.0299` vs `0.0367`). In practice, this transparent comparison favors keeping Logistic Regression as the primary model or collecting more training observations before promoting a tree ensemble — demonstrating why baseline tolerance gates exist rather than masking discrepancies.

---

## 4. Explicitly Untested / Not-Run Optional Capabilities

| Capability | Status | Reason |
|---|---|---|
| Optional `xgboost` boosted-tree backend | **NOT RUN** | Optional dependency `xgboost` is not installed in the base `.venv` (`runtime_metadata.json` records `xgboost: "not_installed"`). Adapter behavior when absent is verified to raise `UnsupportedInputError`. |
| Optional `shap` TreeExplainer backend | **NOT RUN** | Optional dependency `shap` is not installed in the base `.venv` (`runtime_metadata.json` records `shap: "not_installed"`). Default deterministic `marginal_baseline_log_odds` and `exact_linear_log_odds` explanations are fully tested and active. |
| GitHub Actions CI workflow execution on remote runner | **NOT RUN LOCALLY** | [`.github/workflows/credit-risk-scoring-platform.yml`](../../../.github/workflows/credit-risk-scoring-platform.yml) is authored and every command inside it (`pytest`, `scripts/run_demo.py`, artifact existence checks) was executed and verified locally on Python `3.9.6`. |
