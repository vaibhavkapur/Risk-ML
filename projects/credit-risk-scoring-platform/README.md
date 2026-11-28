# Credit Risk Scoring Platform

> **Educational & Research Prototype Only — No Automated Lending Decisions.**  
> This project estimates calibrated probability of default (PD) strictly on deterministic synthetic loan data to support **manual underwriting review prioritization**. Automated approval or decline decisions are structurally disabled (`automated_decision_allowed = false`). It is not a deployed credit decision system or proof of fairness/regulatory compliance.

## Overview

The Credit Risk Scoring Platform demonstrates an end-to-end, audit-ready credit risk ML lifecycle with clean architectural boundaries:

- **Point-in-time feature & label contracts (`v1.0.0`):** Enforces `feature_as_of_timestamp <= application_timestamp`, blocks post-origination columns (e.g. `days_past_due_30d`, `collections_status`, `chargeoff_amount`), and excludes right-censored immature labels (`CENSORED_IMMATURE`) whose observation window has not elapsed.
- **Temporal validation & baselines:** Chronologically splits matured loans into `train`, `calibration`, and `test` windows, training regularized **Logistic Regression** and **Gradient Boosted Trees** (`HistGradientBoostingClassifier`, with optional `xgboost` adapter) while strictly isolating protected subgroup columns (`age_band`, `region_tier`) from model inputs.
- **Probability calibration & monotonicity:** Compares **Platt (sigmoid)** and **Isotonic** calibration on a held-out temporal calibration window, verifies monotonicity across a dense score grid, and outputs reliability diagrams (`PNG` + `SVG`).
- **Stability & subgroup diagnostics:** Computes score **PSI**, feature **CSI**, and subgroup calibration/TPR/FPR/review-rate diagnostics with explicit minimum-sample enforcement (`InsufficientSubgroupSampleError` in strict mode; explicit withholding + warning in reporting mode).
- **Explainability & Manual Review Workbench:** Generates deterministic SHA-256 `decision_id`s, local log-odds feature attributions with standardized reason codes (`RC01`–`RC15`), a model card, replay verification, and a local HTML/CLI manual review queue with human disposition audit trails.

## Project Layout

```text
projects/credit-risk-scoring-platform/
├── config/                  # Pinned YAML configs (default.yaml, reference.yaml)
├── data/fixtures/           # Valid, boundary, invalid, leakage, label, and oracle JSON fixtures
├── docs/
│   ├── adr/0001-point-in-time-and-calibration-architecture.md
│   ├── architecture.md      # Mermaid architecture diagram, invariants, trade-offs
│   ├── contracts.md         # Input/output schemas, units, versions, error codes
│   ├── integration.md       # Adapter boundaries, licenses, optional extras
│   └── runbook.md           # Setup, demo, diagnostics, recovery steps
├── reports/
│   ├── acceptance.md        # Acceptance criteria mapped to measured evidence
│   ├── model_card.md        # Generated model card with limitations & uncertainty
│   ├── evaluation_report.json
│   ├── baseline_comparison.json
│   ├── runtime_metadata.json
│   ├── demo_output.json
│   ├── calibration_plot.png
│   ├── calibration_plot.svg
│   └── manual_review_queue.json
├── scripts/                 # Runnable demo and manual review workbench scripts
├── src/credit_risk/         # Typed Python library (contracts, features, models, calibration, evaluation, explainability, policy, inference, review, reporting, pipeline, cli)
├── tests/                   # 18 automated unit, integration, boundary, and ops tests
├── pyproject.toml
├── requirements.txt
└── plan.md
```

## Quickstart

Run all commands from `projects/credit-risk-scoring-platform/`:

### 1. Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2. Run Automated Validation Suite

```bash
.venv/bin/pytest
```

### 3. Run Reproducible End-to-End Demo

```bash
.venv/bin/python scripts/run_demo.py --config config/default.yaml
```

### 4. Score Applicant Fixtures & Launch Manual Review Workbench

```bash
# Score deterministic valid/boundary fixtures using saved model bundle
PYTHONPATH=src .venv/bin/python -m credit_risk.cli score \
  --bundle reports/artifacts/scorer_bundle.joblib \
  --input data/fixtures/valid_applicants.json

# List queued manual review cases in terminal
PYTHONPATH=src .venv/bin/python -m credit_risk.cli review-list \
  --store reports/manual_review_queue.json

# Start local HTML Manual Review Workbench at http://127.0.0.1:8088
.venv/bin/python scripts/run_review_ui.py \
  --store reports/manual_review_queue.json \
  --host 127.0.0.1 --port 8088
```

## Measured Results (Synthetic Demo — `config/default.yaml`, `seed=42`)

> **Distinction between observations and fixtures:** The metrics below are **actual observed outputs** produced by running `scripts/run_demo.py --config config/default.yaml` on Python `3.9.6` (`scikit-learn==1.5.2`) over the **synthetic** 4,500-applicant dataset (`train=2,730`, `calibration=890`, `test=797`, `censored_excluded=83`, `as_of_cutoff_date=2025-06-15`). They are not real borrower metrics and are not performance guarantees on external data.

| Model Variant (Temporal Test Split, `n=797`, observed DR=`0.2020`) | AUROC | PR-AUC | KS | Brier Score | ECE | Mean Pred PD |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression (Uncalibrated) | 0.7640 | 0.4602 | 0.4170 | 0.1363 | 0.0273 | 0.1945 |
| Logistic Regression (Calibrated: `sigmoid`) | 0.7640 | 0.4602 | 0.4170 | 0.1364 | 0.0367 | 0.1868 |
| Boosted Trees (Uncalibrated) | 0.7433 | 0.4625 | 0.3713 | 0.1383 | 0.0207 | 0.1863 |
| Boosted Trees (Calibrated: `sigmoid`) | 0.7433 | 0.4625 | 0.3713 | 0.1388 | 0.0299 | 0.1745 |

- **Stability (Train -> Test):** Calibrated score PSI = `0.0206` (`STABLE`); max feature CSI = `0.0845`.
- **Replay Determinism:** `replay_verification.decision_id_match = true`.
- **Baseline Comparison Note:** Boosted Trees meets Brier (`+0.0024 <= 0.015`) and ECE (`0.0299 <= 0.05`) tolerances and slightly exceeds Logistic Regression on PR-AUC (`0.4625` vs `0.4602`), while trailing Logistic Regression by `-0.0207` on AUROC (just outside the `-0.02` proposed parity target due to the predominantly linear synthetic log-odds DGP at `n=2,730`). See [`reports/acceptance.md`](reports/acceptance.md) and [`reports/baseline_comparison.json`](reports/baseline_comparison.json) for full analysis.

### Example Scoring Output (`FIX-VALID-001`)

```json
{
  "decision_id": "DEC-24DC97D135A97F89",
  "application_id": "FIX-VALID-001",
  "model_id": "gbdt_baseline_v1",
  "calibration_method": "sigmoid",
  "raw_score": 0.069962,
  "calibrated_pd": 0.059291,
  "risk_band": "MODERATE_RISK_REVIEW",
  "queue_priority": "elevated",
  "review_sla_hours": 24,
  "recommended_action": "ROUTE_TO_DETAILED_UNDERWRITER_REVIEW",
  "automated_decision_allowed": false,
  "manual_review_required": true,
  "explanations": [
    {
      "feature": "credit_score",
      "value": 760.0,
      "contribution_log_odds": -0.618492,
      "direction": "DECREASES_RISK",
      "reason_code": "RC01_CREDIT_SCORE",
      "summary": "Bureau credit score (760.0) reduces estimated default risk (delta log-odds -0.6185)."
    }
  ]
}
```

## Documentation & Evidence Links

- **Acceptance Evidence:** [`reports/acceptance.md`](reports/acceptance.md)
- **Model Card:** [`reports/model_card.md`](reports/model_card.md)
- **Calibration Plots:** [`reports/calibration_plot.png`](reports/calibration_plot.png) | [`reports/calibration_plot.svg`](reports/calibration_plot.svg)
- **Architecture & ADR:** [`docs/architecture.md`](docs/architecture.md) | [`docs/adr/0001-point-in-time-and-calibration-architecture.md`](docs/adr/0001-point-in-time-and-calibration-architecture.md)
- **Data Contracts:** [`docs/contracts.md`](docs/contracts.md)
- **Integration & Licenses:** [`docs/integration.md`](docs/integration.md)
- **Operator Runbook:** [`docs/runbook.md`](docs/runbook.md)
- **CI Workflow:** [`../../.github/workflows/credit-risk-scoring-platform.yml`](../../.github/workflows/credit-risk-scoring-platform.yml)

## Key Assumptions and Limitations

1. **Synthetic Data Only:** All training, calibration, evaluation, and fixture records are synthetically generated; results do not reflect real borrower behavior.
2. **Calibration Under Regime Shift:** Calibrators are fitted on the chronological pre-shift window (`2023-H2`, DR `0.1551`); when the test window shifts upward (`2024-H1`, DR `0.2020`), mean predicted PD lags observed default rate (`-0.0275` gap), illustrating why continuous PSI/CSI monitoring and periodic recalibration are required.
3. **Subgroup Diagnostics Are Not Compliance Proof:** Subgroup metrics are diagnostic only, withhold estimates when `n < min_subgroup_sample_size`, and do not establish legal fairness or disparate-impact compliance.
4. **Optional Extras Not Installed by Default:** `xgboost` and `shap` adapters are implemented as optional extras and marked **not run** in [`reports/acceptance.md`](reports/acceptance.md) for the base environment.
5. **No Production Integrations:** No live credit bureaus, customer PII stores, credentials, or automated underwriting engines are connected.
