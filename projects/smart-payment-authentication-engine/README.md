# Smart Payment Authentication Engine

An independently developed educational and research prototype for **payment authentication action selection** (`approve`, `challenge`, `decline`, `abstain`) that balances fraud loss, customer step-up friction, and genuine acceptance using delayed-label calibrated risk modeling, constrained policy optimization, non-bypassable mandatory rules, and deterministic outage fallback.

> **Scope & Disclaimer:** Offline, synthetic-only prototype. Not a 3DS 2.x / PSD2 SCA compliance implementation. Zero live capital, real customer records, PANs, or production credentials.

---

## Key Capabilities

- **Strict Data & Action Contracts (`v1.0.0`):** Typed Pydantic v2 schemas for synthetic payments, device telemetry, delayed risk labels (`label_available_at`), challenge outcomes, and replayable audit decisions (`src/smart_auth_engine/contracts.py`).
- **Temporal Validation & Delayed-Label Maturity Filtering:** Strict chronological `train < cal < val < test` splits that exclude immature labels (`label_available_at > as_of`) so future chargebacks never leak into training or evaluation (`src/smart_auth_engine/training.py`, `src/smart_auth_engine/evaluation.py`).
- **Post-Hoc Probability Calibration:** Compares raw, sigmoid (Platt), and isotonic calibration via Brier score, log-loss, ECE, ROC-AUC, and reliability bins (`src/smart_auth_engine/calibration.py`).
- **Constrained Policy Optimizer & Pareto Frontier:** Sweeps `(challenge_threshold, decline_threshold)` under `max_challenge_rate` and `max_fraud_loss_rate` budgets while structurally guaranteeing **zero bypass of mandatory challenge rules** and **high-risk decline overrides** (`src/smart_auth_engine/policy.py`).
- **Deterministic Fallback & Step-Up Simulation:** Missing critical features, excessive telemetry loss (`abstain`), or model outages deterministically fall back to safe rules; step-up flows (`otp_sms`, `biometric_app`, `hardware_token`) are simulated with per-decision seeded determinism (`src/smart_auth_engine/step_up.py`, `src/smart_auth_engine/inference.py`).
- **FastAPI Service & Durable Replay:** HTTP endpoints for authentication, step-up simulation, decision lookup, deterministic replay verification, and diagnostic counters (`src/smart_auth_engine/api.py`, `src/smart_auth_engine/store.py`).

---

## Quickstart

From `projects/smart-payment-authentication-engine/`:

```bash
# 1. Create virtual environment and install pinned dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Run the complete validation suite (pytest + 2-pass reproducible demo)
.venv/bin/python scripts/validate_all.py

# 3. (Optional) Start the local FastAPI service on http://127.0.0.1:8000
.venv/bin/python scripts/serve_api.py
```

---

## Example Measured Output (Synthetic Dataset, `seed=42`)

Measured by running `.venv/bin/python scripts/validate_all.py` on Python `3.9.6` (raw artifacts saved in `reports/`):

- **Automated tests:** `20 passed in 7.69s` (domain, calibration, policy optimization, overrides, chronological delay, FastAPI, restart/replay).
- **Delayed-label maturity filter (Train split):** `1,243` matured training examples retained; `1,027` immature examples excluded as of `2026-01-31T00:00:00Z`.
- **Calibration (Validation split, `n=729`):** Sigmoid-calibrated GBM achieved `ROC-AUC = 0.8568`, `Brier = 0.03657` (vs. `0.03756` raw), `ECE = 0.02061` (vs. `0.02148` raw).
- **Constrained Policy Selection (Validation split):** `challenge >= 0.0561`, `decline >= 0.1500` (`constraints_satisfied = True`, `1,570` candidates evaluated).
- **Holdout Chronological Test Split (`n=752`, `2026-02-20` to `2026-03-01`):**

| Scenario | Challenge Rate | Decline Rate | Genuine Acceptance | Matured Fraud Loss Rate | Mandatory Bypass | Override Bypass |
|---|---:|---:|---:|---:|---:|---:|
| Calibrated GBM Policy | `19.68%` | `8.78%` | `92.50%` | `3.61%` | `0` | `0` |
| Uncalibrated Reference Oracle | `31.91%` | `20.08%` | `80.20%` | `1.48%` | `0` | `0` |
| Model Outage Deterministic Fallback | `33.91%` | `3.06%` | `94.63%` | `4.24%` | `0` | `0` |

- **Replay & Reproducibility:** `25/25` sampled decisions replayed with `exact_match = True`; Pass 1 and Pass 2 `reports/chronological_evaluation.json` matched bit-for-bit.

---

## Project Layout and Evidence Links

- [`plan.md`](plan.md) — Implementation plan and acceptance criteria
- [`config/default.yaml`](config/default.yaml) — Pinned configuration, seeds, rules, constraints, and tolerances
- [`src/smart_auth_engine/`](src/smart_auth_engine/) — Library and FastAPI service implementation
- [`data/fixtures/`](data/fixtures/) — Valid, invalid, boundary, and hand-calculated reference oracle JSON fixtures
- [`tests/`](tests/) — Unit, integration, boundary, operational, and reproducibility test suites
- [`scripts/`](scripts/) — `run_demo.py`, `validate_all.py`, and `serve_api.py`
- [`docs/architecture.md`](docs/architecture.md) — Component diagram, data flow, invariants, and trade-offs
- [`docs/adr/0001-separated-calibrated-policy-and-deterministic-fallback.md`](docs/adr/0001-separated-calibrated-policy-and-deterministic-fallback.md) — First Architecture Decision Record
- [`docs/contracts.md`](docs/contracts.md) — Input/output schemas, units, timestamps, and error contracts
- [`docs/integration.md`](docs/integration.md) — Adapter boundaries, optional Redis notes, and dependency licenses
- [`docs/runbook.md`](docs/runbook.md) — Setup, demo, diagnostics, failure modes, and recovery procedures
- [`reports/acceptance.md`](reports/acceptance.md) — Acceptance criteria mapped to measured test and benchmark evidence
- [`reports/`](reports/) — Versioned machine-readable JSON outputs (`baseline_vs_calibrated_metrics.json`, `policy_tradeoff_frontier.json`, `chronological_evaluation.json`, `step_up_simulation_summary.json`, `run_metadata.json`, `decision_log.jsonl`)
- [`../../.github/workflows/smart-payment-authentication-engine.yml`](../../.github/workflows/smart-payment-authentication-engine.yml) — CI validation workflow

---

## Assumptions and Known Limitations

1. **Synthetic Data Only:** Fraud probabilities, label maturity delays (3–18 days for fraud, 14 days for non-fraud), and step-up pass/abandon rates are generated by a fixed-seed synthetic model and do not reflect any real issuer or merchant portfolio.
2. **Holdout Variance:** Because synthetic amounts are heavy-tailed, holdout test fraud-loss rate (`3.61%`) can exceed the validation-set expected target (`<= 1.5%`) even when challenge rate (`19.68%`) remains inside budget (`<= 22%`).
3. **Optional Redis Backend Not Live-Tested:** `OptionalRedisDecisionStore` is verified against a fake client stub; live Redis persistence is marked **not run** in [`reports/acceptance.md`](reports/acceptance.md).
4. **No Production Compliance or Security Hardening:** Real 3DS 2.x integration, HSM/key management, authentication middleware, and regulatory review are explicitly outside scope.
