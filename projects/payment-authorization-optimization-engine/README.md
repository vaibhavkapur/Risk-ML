# Payment Authorization Optimization Engine

> **Research Prototype Notice:** This is an independently developed educational/research prototype operating exclusively on **offline synthetic data**. Live capital, real transactions, cardholder records, PCI-DSS workflows, and production credentials are excluded.

An end-to-end Python, scikit-learn, NumPy, and FastAPI prototype evaluating how risk-aware constrained authorization policies trade off approval rates, expected fraud losses, and latency under strict payment retry safety invariants.

---

## Key Capabilities

- **Response Taxonomy & Retry Safety (`src/payment_auth_engine/taxonomy.py`):** Deterministic ISO-8583-inspired state machine enforcing hard stops on hard declines (`14`, `54`, `57`, `59`, `63`), holding uncertain outcomes (`TIMEOUT`, `NETWORK_ERROR`, `UNKNOWN`, unmapped codes, `prior_outcome_known=False`) for reconciliation, and blocking duplicates via `IdempotencyLedger`.
- **Versioned Features & Temporal Calibration (`features.py`, `models.py`):** Strict chronological train/calibration/test splits, matured-label filtering (`>= 30 days`) for fraud/loss training, training-only imputation with explicit missingness flags, and temporal Platt probability calibration.
- **Constrained Risk-Aware Decision Policy (`policy.py`):** Compares `baseline_threshold`, `unconstrained_ev`, and Lagrangian `constrained_risk_aware` policies subject to explicit expected-loss (bps) and P95 latency (ms) budgets, conservative missing-feature handling, and processor-outage failover/fail-closed guards.
- **Audited Counterfactual Evaluation (`counterfactual.py`):** Computes Direct Method (DM), IPS, SNIPS, and Doubly Robust (DR) estimates with seeded 95% bootstrap confidence bounds, synthetic ground-truth oracle comparison, and fail-fast guards against positivity violations, zero support, and ESS collapse.
- **FastAPI Service & Reproducible CLI (`api.py`, `cli.py`):** Typed HTTP endpoints (`/health`, `/v1/authorize`, `/v1/retry-eligibility`, `/v1/diagnostics`, `/v1/processors/{id}/health`) plus reproducible demo and replay verification commands.

---

## Quickstart

### 1. Setup

```bash
cd projects/payment-authorization-optimization-engine
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2. Run Reproducible Demo

```bash
.venv/bin/python scripts/run_demo.py
```

Example output (synthetic test fold, `n=1,000`, `seed=42`, `config_hash=1364cf25c7eb2974`):

```text
=== Payment Authorization Optimization Engine — Policy Comparison (Synthetic Test Fold) ===
Model Version: v1.0.0-bb892ac96c | Config Hash: 1364cf25c7eb2974 | Seed: 42
Policy                     DR Accept                95% CI  DR Loss bps          95% CI bps  P95 Lat ms    Audit
----------------------------------------------------------------------------------------------------------------
baseline_threshold            0.6518      [0.6415, 0.6859]       173.26  [173.26, 244.63]       210.0   BREACH
unconstrained_ev              0.4982      [0.4766, 0.5292]        42.82  [ 42.82,  52.19]       470.0   BREACH
constrained_risk_aware        0.3388      [0.3031, 0.3506]        44.00  [ 39.28,  47.45]       400.0     PASS
```

### 3. Run Full Validation Suite (Tests + Demo + Replay Check)

```bash
.venv/bin/python scripts/validate_all.py
```

---

## Measured Synthetic Results vs. Engineering Targets

All measurements below are **actual observations on the synthetic dataset** (`config/default.yaml`, `n=4,000`, `seed=42`), evaluated against the configured engineering targets (`max_expected_loss_bps <= 45.0`, `max_p95_latency_ms <= 420.0`). They are **not** production performance guarantees.

| Policy | DR Net Acceptance | 95% CI (Accept) | Oracle Acceptance | DR Loss (bps) | 95% CI (bps) | Oracle Loss (bps) | Oracle Net Value (cents/att) | P95 Latency (ms) | Constraint Audit |
|---|---:|---|---:|---:|---|---:|---:|---:|---|
| `baseline_threshold` | `0.6518` | `[0.6415, 0.6859]` | `0.6291` | `173.26` | `[173.26, 244.63]` | `228.21` | `-28.20` | `210.0` | **FAIL** (loss > 45 bps) |
| `unconstrained_ev` | `0.4982` | `[0.4766, 0.5292]` | `0.4847` | `42.82` | `[42.82, 52.19]` | `50.03` | `+55.55` | `470.0` | **FAIL** (loss & P95 > budget) |
| `constrained_risk_aware` | `0.3388` | `[0.3031, 0.3506]` | `0.3102` | `44.00` | `[39.28, 47.45]` | `37.78` | `+47.12` | `400.0` | **PASS** (0 breaches) |

**Calibration Summary (Temporal Calibration Fold):**
- **Acceptance Model** (`n=671`): ROC-AUC `0.7125`, Brier `0.1873 -> 0.1849`, ECE `0.0527 -> 0.0294`.
- **Matured Fraud/Loss Model** (`n_train_matured=1,254`, `n_cal_matured=450`): ROC-AUC `0.8685`, Brier `0.0190 -> 0.0188`, ECE `0.0134 -> 0.0065`.

---

## Project Layout & Documentation

- [`plan.md`](plan.md) — Implementation plan and completion checklist
- [`docs/architecture.md`](docs/architecture.md) — Component diagrams, data flow, invariants, and trade-offs
- [`docs/contracts.md`](docs/contracts.md) — Input/output schemas, units, response taxonomy, error codes, and fixtures
- [`docs/integration.md`](docs/integration.md) — `Risk-ML` sibling boundaries and adapter recommendations
- [`docs/runbook.md`](docs/runbook.md) — Local API runbook, diagnostics, and failure recovery
- [`docs/adr/0001-authorization-policy-architecture.md`](docs/adr/0001-authorization-policy-architecture.md) — First architecture decision record
- [`reports/acceptance.md`](reports/acceptance.md) — Acceptance criteria mapped to automated test and benchmark evidence
- Raw machine-readable outputs in `reports/`: [`policy_comparison.json`](reports/policy_comparison.json), [`audit_report.json`](reports/audit_report.json), [`baseline_outputs.json`](reports/baseline_outputs.json), [`model_manifest.json`](reports/model_manifest.json), [`runtime_metadata.json`](reports/runtime_metadata.json)
- [`../../.github/workflows/payment-authorization-optimization-engine.yml`](../../.github/workflows/payment-authorization-optimization-engine.yml) — Automated CI validation workflow

---

## Assumptions, Limitations & Provenance

- **Causal uplift disclaimer:** Off-policy estimators (IPS/SNIPS/DR) require known, strictly positive logging propensities and no unobserved confounders. Observational production logs without randomized exploration cannot establish causal uplift.
- **Label maturity censoring:** In test folds covering recent dates within the 30-day maturity window, unobserved dispute outcomes rely on calibrated conditional loss expectations for DR evaluation while training strictly excludes immature labels.
- **Fixture & dependency provenance:** All fixtures in `data/fixtures/` are original synthetic records. Core dependencies use permissive open-source licenses: NumPy (BSD-3), scikit-learn (BSD-3), SciPy (BSD-3), FastAPI (MIT), Pydantic (MIT), PyYAML (MIT), Uvicorn (BSD-3), pytest (MIT), httpx (BSD-3).
