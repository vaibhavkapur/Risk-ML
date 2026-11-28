# Chargeback Prediction Engine

An offline, deterministic research and engineering prototype for **delayed payment dispute risk prediction**, **calibrated expected dispute loss estimation**, **budgeted manual review prioritization**, and **delayed-outcome cohort monitoring**.

> **Scope & Safety Notice:** This is an independently developed educational/research prototype using 100% synthetic data and fixtures. **Disputes are not equivalent to fraud** — punitive automated merchant decisions (auto-suspensions, holds, terminations) are prohibited and rejected by configuration validation. No real transactions, cardholder records, live credentials, or proprietary artifacts are included.

---

## Key capabilities

- **Observation snapshots & label maturity windows** (`maturity.py`): Explicitly models per-transaction `maturity_date` (override or `settlement_ts + 90d`), masks future disputes after `as_of_ts`, and **strictly excludes immature negatives** (`label=null, is_trainable=false`). Default `cohort_mature` mode also holds out immature positives from training/calibration to preserve unbiased prevalence (see [ADR-0001](docs/adr/0001-maturity-windows-and-causal-features.md)).
- **Causal point-in-time Polars features** (`features.py`): Computes trailing 7d/30d/90d merchant transaction velocity, event-time reported dispute counts by normalized reason category (with explicit missing reason-code handling), and Empirical Bayes smoothed mature merchant dispute rates for cold-start merchants with zero future-dispute leakage.
- **Chronologically calibrated models & expected loss** (`models.py`): Separates chronological `train` / `calibration` / `test` splits, compares a calibrated Logistic Regression baseline against a chronologically Platt-calibrated `HistGradientBoostingClassifier`, computes `LGD`, calibrated expected dispute loss, expected net review savings, and per-transaction local decision explanations.
- **Review prioritization & delayed-outcome monitoring** (`policy.py`): Ranks transactions under fixed manual review capacity budgets with non-punitive merchant health advisories, and simulates cohort maturation over `+0d` to `+120d` checkpoints using an empirical dispute arrival CDF.

---

## Quickstart

```bash
cd projects/chargeback-prediction-engine
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
```

### 1. Verify hand-calculated reference fixture

```bash
PYTHONPATH=src .venv/bin/python -m chargeback_engine.cli verify-reference --fixtures-dir data/fixtures
```

### 2. Run the automated validation test suite

```bash
.venv/bin/pytest
```

### 3. Run the reproducible end-to-end demo and benchmarks

```bash
.venv/bin/python scripts/run_demo.py --config config/default.yaml --output-dir reports
.venv/bin/python scripts/run_benchmarks.py --config config/default.yaml --cold-start-config config/cold_start.yaml --output-dir reports
```

---

## Example measured output (synthetic dataset)

The values below are **actual measured observations** from `scripts/run_demo.py` and `scripts/run_benchmarks.py` on deterministic synthetic data (`seed=42`, `config/default.yaml`, `run_id=run_8c20db71476a0e`) — not production guarantees or real-world metrics:

```text
Reference fixture check: passed (6 txns)
Demo Run ID: run_8c20db71476a0e (config_hash=bc6cc12f868f442e)
Maturity: mature_pos=176, mature_neg=4920, immature_pos=63, immature_neg_excluded=1341
Primary Calibrated | PR-AUC=0.4237 | ROC-AUC=0.8539 | Brier=0.03036 | ECE=0.00766 | ExpLoss=$4638.32 vs Realized=$4273.12 | Precision@Budget=0.2549
Baseline Logistic  | PR-AUC=0.1285 | ROC-AUC=0.6701 | Brier=0.03489 | ECE=0.01123 | ExpLoss=$4420.27 | Precision@Budget=0.0980
Reproducibility check: exact_match=True (run_id=run_8c20db71476a0e)
```

Full acceptance evidence and raw machine-readable outputs:
- Acceptance report: [`reports/acceptance.md`](reports/acceptance.md)
- Raw demo output: [`reports/demo_summary.json`](reports/demo_summary.json)
- Baseline vs calibrated & cold-start/imbalance benchmark: [`reports/baseline_vs_calibrated.json`](reports/baseline_vs_calibrated.json)
- Review prioritization & delayed-outcome monitoring: [`reports/review_simulation.json`](reports/review_simulation.json)
- Bit-for-bit reproducibility check: [`reports/reproducibility_check.json`](reports/reproducibility_check.json)
- Runtime & configuration metadata: [`reports/runtime_metadata.json`](reports/runtime_metadata.json)
- CI validation workflow: [`../../.github/workflows/chargeback-prediction-engine.yml`](../../.github/workflows/chargeback-prediction-engine.yml)

---

## Project structure

```text
projects/chargeback-prediction-engine/
├── plan.md                                         # Original implementation plan
├── README.md                                       # Overview, quickstart, results, limitations
├── pyproject.toml                                  # Package metadata and pytest configuration
├── requirements.lock                               # Pinned dependency versions
├── config/
│   ├── default.yaml                                # Default synthetic scenario config
│   └── cold_start.yaml                             # Severe imbalance + 45% cold-start config
├── data/fixtures/
│   ├── reference_merchants.json                    # Deterministic reference fixture inputs
│   ├── reference_transactions.json
│   ├── reference_disputes.json
│   ├── reference_expected.json                     # Independent hand-calculated oracle
│   └── invalid_cases.json                          # Negative/invalid contract boundary fixtures
├── src/chargeback_engine/
│   ├── config.py                                   # Typed config loader & invariants
│   ├── contracts.py                                # Canonical schemas, reason codes, exceptions
│   ├── maturity.py                                 # Maturity windows & chronological splits
│   ├── features.py                                 # Polars causal point-in-time aggregates
│   ├── models.py                                   # Calibrated classifiers, expected loss, explanations
│   ├── policy.py                                   # Budgeted review policy & delayed monitoring
│   ├── synthetic.py                                # Deterministic synthetic data generator
│   ├── pipeline.py                                 # End-to-end orchestration & reference check
│   └── cli.py                                      # CLI entry point
├── scripts/
│   ├── run_demo.py                                 # Reproducible demo runner
│   └── run_benchmarks.py                           # Reproducibility & stress benchmark runner
├── tests/                                          # 17 unit, integration, boundary & repro tests
├── docs/
│   ├── architecture.md                             # Mermaid diagram, invariants, trade-offs
│   ├── contracts.md                                # Schemas, units, versions, error codes
│   ├── integration.md                              # Adapter boundaries, licenses, provenance
│   ├── runbook.md                                  # Setup, commands, diagnostics, replay
│   └── adr/0001-maturity-windows-and-causal-features.md
└── reports/                                        # Acceptance report & versioned JSON outputs
```

---

## Assumptions and known limitations

1. **Synthetic data only:** All metrics measure performance on the deterministic synthetic generator (`synthetic.py`) and hand-calculated fixtures (`data/fixtures/`), not real payment networks.
2. **Fixed economic parameters:** Chargeback fee (`$15`), recovery rate (`15%`), manual review cost (`$4.50`), and prevention effectiveness (`55%`) are configurable assumptions rather than empirically estimated operational costs.
3. **Full maturity lag in `cohort_mature` mode:** Default training waits 90 days for negative maturation to prevent censoring bias; early drift detection relies on the delay-CDF monitoring layer (`policy.py`).
4. **Offline batch execution:** Real-time inference latency, streaming feature stores, representment automation, and live PCI/security controls are out of scope.
5. **Non-punitive policy boundary:** Disputes include friendly fraud, fulfillment delays, and processing errors; automated merchant penalties are explicitly disabled.
