# Account Takeover Detection Engine

An independently developed research prototype for detecting anomalous login and session behavior with strictly causal features, chronological validation, calibrated risk scoring, adaptive step-up recommendations, analyst evidence, drift alerts, and deterministic fallback policies.

> **Scope notice:** Synthetic simulation and offline/local execution only. Uses RFC 5737 documentation IP addresses (`192.0.2.0/24`, `198.51.100.0/24`, `203.0.113.0/24`) and synthetic identifiers. Real credentials, customer records, live transactions, and proprietary code are strictly excluded.

## Capabilities

- **Strictly causal feature engineering** (`src/ato_engine/features.py`): 22 versioned rolling user, device, network, and session features enforcing `history_ts < event_ts`, 30-day TTL expiry (`profile_expired`), and explicit missing-telemetry tracking.
- **Chronological validation & label maturity** (`src/ato_engine/models.py`): Enforces `label_available_at <= train_cutoff` to eliminate delayed-label leakage; trains a hand-calculated reference rule oracle, `IsolationForest` anomaly baseline, `LogisticRegression`, and `HistGradientBoostingClassifier`, calibrated on a dedicated chronological holdout split.
- **Adaptive thresholds, analyst evidence & fallback** (`src/ato_engine/policy.py`): Recommends `ALLOW`, `CHALLENGE_OTP`, `CHALLENGE_WEBAUTHN`, or `BLOCK_AND_RESET` with account-tier tightening, shared-network dampening, repeated-attempt guardrails, ranked feature attributions, and automatic conservative rule fallback on missing telemetry, store outages, stale models, or critical drift.
- **Drift alerts, mandatory audit logging & FastAPI service** (`src/ato_engine/drift.py`, `audit.py`, `api.py`): Score PSI and feature/missingness/expiry drift alerts, append-only JSONL decision audit logging (every decision logged with deterministic `decision_id`), replay verification, and typed REST endpoints.

## Project Layout

```text
projects/account-takeover-detection-engine/
├── config/default.yaml                  # Pinned default engine configuration
├── data/fixtures/                       # Reference oracle, invalid, unsupported & boundary fixtures
├── docs/
│   ├── adr/0001-architecture-and-boundaries.md
│   ├── architecture.md                  # Mermaid diagram, data flow, invariants, trade-offs
│   ├── contracts.md                     # Input/output schemas, units, versions, errors
│   ├── integration.md                   # Adapter boundaries & dependency licenses
│   └── runbook.md                       # Setup, diagnostics, failure modes & recovery
├── reports/
│   ├── acceptance.md                    # Acceptance criteria mapped to measured evidence
│   ├── evaluation_results.json          # Raw versioned benchmark & scenario outputs
│   ├── decision_audit.jsonl             # Structured JSONL audit log from evaluation run
│   └── model_bundle.joblib(.manifest.json)
├── scripts/
│   ├── run_evaluation.py                # Reproducible train/calibrate/evaluate/scenario pipeline
│   └── run_demo.py                      # Concise end-to-end walkthrough
├── src/ato_engine/                      # Core Python package
├── tests/                               # Unit, integration, boundary, and FastAPI tests
├── pyproject.toml
├── requirements.txt
└── plan.md
```

## Quickstart

```bash
cd projects/account-takeover-detection-engine

# 1. Create virtual environment and install pinned dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Run unit, integration, boundary, and FastAPI tests
.venv/bin/pytest -v

# 3. Run reproducible chronological evaluation and generate reports/
.venv/bin/python scripts/run_evaluation.py --config config/default.yaml --output-dir reports

# 4. Run end-to-end demo walkthrough
.venv/bin/python scripts/run_demo.py

# 5. (Optional) Start local FastAPI service
PYTHONPATH=src .venv/bin/uvicorn ato_engine.api:create_app --factory --host 127.0.0.1 --port 8000
```

## Example Demo Output

Running `.venv/bin/python scripts/run_demo.py` evaluates a 4-step sequence (`returning trusted login` -> `benign new device` -> `compromised session` -> `missing telemetry fallback`) and logs every decision:

```json
{
  "decision_id": "dec_6df62fbcbb885df0",
  "action": "BLOCK_AND_RESET",
  "calibrated_risk_score": 1.0,
  "anomaly_score": 1.0,
  "fallback_used": false,
  "fallback_reason": "none",
  "triggered_rules": [
    "IMPOSSIBLE_TRAVEL_DETECTED",
    "NEW_DEVICE_SENSITIVE_ACTION_BURST",
    "AUTH_FACTOR_DOWNGRADE_ON_NEW_DEVICE"
  ],
  "summary": "Recommended BLOCK_AND_RESET (risk=1.000) via calibrated ML + adaptive threshold policy; 3 rule signal(s) triggered.",
  "top_factor": {
    "factor": "sensitive_action_count",
    "direction": "risk_increasing",
    "contribution": 8.0347,
    "observed_value": 3.0,
    "explanation": "Number of high-risk account changes attempted in the session."
  }
}
```

## Measured Results (Synthetic Simulation Only)

The table below reports **actual measured results** from `scripts/run_evaluation.py` on the deterministic synthetic dataset (`seed=42`, `2,985` events; `1,657` mature train rows with `134` immature delayed-label rows excluded, `597` calibration rows, `597` chronological test rows). These metrics reflect synthetic simulation separability and are **not** estimates of real-world production accuracy.

| Model / Baseline | ROC-AUC | PR-AUC | Brier Score | ECE | Recall @ 1% FPR | Recall @ 2% FPR | Recall @ 5% FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rule_oracle_baseline` | 0.9994 | 0.9981 | 0.047378 | 0.160949 | 0.9790 | 1.0000 | 1.0000 |
| `anomaly_isolation_forest` | 0.9230 | 0.7481 | 0.121605 | 0.144861 | 0.2098 | 0.2517 | 0.4685 |
| `logistic_regression` | 1.0000 | 1.0000 | 0.000165 | 0.001903 | 1.0000 | 1.0000 | 1.0000 |
| `supervised_uncalibrated_blend` | 1.0000 | 1.0000 | 0.005093 | 0.061413 | 1.0000 | 1.0000 | 1.0000 |
| `calibrated_ensemble` | 1.0000 | 1.0000 | 0.000003 | 0.000491 | 1.0000 | 1.0000 | 1.0000 |

- **Tests:** `14 passed` (`pytest -v`), all 6 required domain/integration/boundary/drift scenarios passing.
- **Audit completeness:** `597/597` test decisions logged (`100%`) in `reports/decision_audit.jsonl`.
- **Replay reproducibility:** `597/597` decisions matched identically on independent replay (`0` mismatches).
- **Full acceptance mapping:** See [reports/acceptance.md](reports/acceptance.md) and [reports/evaluation_results.json](reports/evaluation_results.json).

## Documentation & Evidence

- [Architecture & Component Diagram](docs/architecture.md)
- [ADR-0001: Boundaries, Causality, Calibration & Fallback](docs/adr/0001-architecture-and-boundaries.md)
- [Input/Output Data Contracts & Error Taxonomy](docs/contracts.md)
- [Integration Boundaries & Dependency Licenses](docs/integration.md)
- [Operator & Developer Runbook](docs/runbook.md)
- [Acceptance Evidence Report](reports/acceptance.md)
- [CI Validation Workflow](../../.github/workflows/account-takeover-detection-engine.yml)

## Assumptions & Known Limitations

1. **Synthetic distribution separability:** Even with benign travel/VPN noise and stealthy ATO patterns, synthetic event streams are cleaner than adversarial production traffic; real-world recall at 1% FPR and calibration drift will require ongoing retraining on production-labeled fraud investigations.
2. **Optional Redis backend not live-tested:** `RedisStateStore` (`src/ato_engine/store.py`) is implemented as an isolated adapter, but live Redis server integration was **not run** in this environment because no local Redis daemon is installed; all automated tests execute against `InMemoryStateStore` (including simulated store-outage fault injection).
3. **Single-process prototype:** File-backed JSONL audit logging and in-memory state are intended for local evaluation and CI; production deployment requires distributed state, durable audit streaming, authentication/authorization on API endpoints, and a full security/privacy/legal review.
