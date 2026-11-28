# Operator & Developer Runbook

## 1. Clean Environment Setup

From `projects/credit-risk-scoring-platform/`:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Verify installation:

```bash
.venv/bin/pytest
```

## 2. Running the End-to-End Demo

```bash
.venv/bin/python scripts/run_demo.py --config config/default.yaml
```

This single command:
1. Generates 4,500 deterministic synthetic loan applications (`seed=42`) with controlled missingness and post-`2024-01-01` distribution shift.
2. Enforces point-in-time feature contracts and 365-day label maturity windows across chronological `train` / `calibration` / `test` splits.
3. Trains logistic regression and boosted-tree baselines.
4. Fits and verifies monotonicity of Platt (`sigmoid`) and `isotonic` calibrators.
5. Computes AUROC, PR-AUC, KS, Brier score, ECE, score PSI, feature CSI, and subgroup diagnostics.
6. Writes versioned artifacts to `reports/`:
   - `reports/evaluation_report.json`
   - `reports/baseline_comparison.json`
   - `reports/runtime_metadata.json`
   - `reports/demo_output.json`
   - `reports/model_card.md`
   - `reports/calibration_plot.png` and `reports/calibration_plot.svg`
   - `reports/manual_review_queue.json`
   - `reports/artifacts/scorer_bundle.joblib`

## 3. Scoring Fixtures and Inspecting the Manual Review Workbench

Score the valid/boundary fixture set using the saved model bundle:

```bash
PYTHONPATH=src .venv/bin/python -m credit_risk.cli score \
  --bundle reports/artifacts/scorer_bundle.joblib \
  --input data/fixtures/valid_applicants.json
```

Inspect the manual review queue in the terminal:

```bash
PYTHONPATH=src .venv/bin/python -m credit_risk.cli review-list \
  --store reports/manual_review_queue.json
```

Launch the interactive local HTML Manual Review Workbench:

```bash
.venv/bin/python scripts/run_review_ui.py \
  --store reports/manual_review_queue.json \
  --host 127.0.0.1 \
  --port 8088
```

Then open `http://127.0.0.1:8088` in a local browser (or query `GET http://127.0.0.1:8088/api/cases`) to inspect calibrated PDs, explanations, and record human underwriter dispositions.

## 4. Diagnostics and Structured Logs

- Structured JSON log lines are emitted on `credit_risk.inference` for:
  - `applicant_scored` (includes `decision_id`, `application_id`, `model_id`, `calibrated_pd`, `risk_band`, `manual_review_required`)
  - `contract_validation_failed`
  - `point_in_time_leakage_blocked`
- `CreditRiskScorer.diagnostics.to_dict()` exposes live counters (`total_requests`, `total_scored`, `validation_errors`, `leakage_errors`, `records_with_missing_features`, `by_risk_band`).

## 5. Failure Modes and Recovery Procedures

| Failure / Error Code | Root Cause | Recovery Action |
|---|---|---|
| `CONFIGURATION_ERROR` | Missing/malformed YAML, inverted split dates, or invalid `risk_bands` | Validate YAML syntax and ensure `train_end_date < calibration_end_date < test_end_date` and final `max_pd == 1.0`; re-run |
| `POINT_IN_TIME_LEAKAGE_ERROR` | Upstream ETL included post-origination columns, future `feature_as_of_timestamp`, or premature `MATURED` labels | Remove post-origination columns from origination extract, fix bureau snapshot timestamps, or set immature records to `CENSORED_IMMATURE` |
| `CONTRACT_VALIDATION_ERROR` | Out-of-bounds value, NaN/Inf, invalid enum, or unexpected schema field | Inspect `details.field` / `details.unknown_fields` in the error payload and repair upstream mapping to contract `v1.0.0` |
| `INSUFFICIENT_SUBGROUP_SAMPLE_ERROR` | Subgroup evaluation ran in strict mode with `n < min_subgroup_sample_size` | Aggregate longer historical window or inspect non-strict report where small groups are explicitly marked `INSUFFICIENT_SAMPLE` |
| `CALIBRATION_MONOTONICITY_ERROR` | Calibrator slope inverted or calibration window too small/corrupted | Check label alignment on the calibration split and verify positive class prevalence before re-fitting |
| `UNSUPPORTED_INPUT_ERROR` | Requested optional `xgboost` without package installed, or attempted `automated_decision_allowed: true` | Install optional extra or switch to `sklearn_hist_gb`; automated lending cannot be enabled in this prototype |
| High Drift Alert (`PSI >= 0.25`) | Macroeconomic or origination population shift between train and test | Review `feature_csi_train_vs_test.alert_features` in `reports/evaluation_report.json`, pause reliance on calibrated PDs, and trigger model/calibrator refit & underwriting review |

## 6. Deterministic Replay / Rerun Verification

To verify a prior scoring output after a restart:
1. Reload the exact `scorer_bundle.joblib` archived for that release.
2. Call `scorer.replay_decision(applicant_payload, expected_decision_id=prior_id)`.
3. Confirm `decision_id_match == True` and identical `calibrated_pd`, `risk_band`, and `explanations`.
