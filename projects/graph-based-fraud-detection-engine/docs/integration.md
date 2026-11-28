# Integration Boundaries — Graph-Based Fraud Detection Engine

## 1. Repository context (`Risk-ML`)

This project lives independently under `projects/graph-based-fraud-detection-engine/` within the `Risk-ML` workspace. Neighboring Risk-ML projects (such as `account-takeover-detection-engine`, `real-time-transaction-risk-platform`, and `chargeback-prediction-engine`) are siblings with independent virtual environments and lifecycles.

To prevent fragile cross-project imports and hidden coupling:
- This project has **zero runtime imports** from sibling projects and zero private/employer/SAAT dependencies.
- All integration surfaces use versioned JSON contracts (`schema_version = "1.0.0"`) or typed Python interfaces.

## 2. Reusable Adapter Interfaces

### Upstream Transaction / Entity Adapters
Any sibling Risk-ML service (e.g., `real-time-transaction-risk-platform` or `account-takeover-detection-engine`) can call this engine by:
1. Pseudonymizing local synthetic IDs via `graph_fraud_engine.privacy.pseudonymize_id(raw_id, prefix, salt)` before submitting requests.
2. Sending a `RiskScoreRequest` JSON payload to `POST /v1/score` (or invoking `FraudDetectionService.score_transaction(...)` directly in Python).

### Downstream Case Management / Chargeback Adapters
1. Analyst queues pull open cases from `GET /v1/reviews?status=open`.
2. Investigation outcomes (`confirmed_fraud`, `false_positive`, `escalated`, `needs_info`) are posted back via `POST /v1/reviews/{case_id}/disposition`, which logs an auditable `AnalystDispositionRecord` and registers matured `analyst_review` labels with `label_available_at = recorded_at`.
3. Delayed chargeback systems (such as `chargeback-prediction-engine`) can supply `FraudLabelRecord` batches where `label_available_at` reflects actual dispute arrival time, preserving strict no-leakage guarantees.

## 3. Dependency & License Inventory

All runtime and development dependencies are open-source and pinned in `requirements.txt` and `pyproject.toml`:

| Package | Version | License | Purpose |
|---|---|---|---|
| `numpy` | `1.26.4` | BSD-3-Clause | Vector math and native 2-layer TemporalGraphSAGE GNN |
| `scipy` | `1.13.1` | BSD-3-Clause | Numerical support for scikit-learn |
| `scikit-learn` | `1.5.2` | BSD-3-Clause | Logistic baseline, HistGradientBoosting, calibration, metrics |
| `networkx` | `3.2.1` | BSD-3-Clause | Time-aware k-hop subgraph projection & path verification |
| `fastapi` | `0.115.6` | MIT | HTTP serving API |
| `uvicorn` | `0.32.1` | BSD-3-Clause | Local ASGI server |
| `pydantic` | `2.10.4` | MIT | Strict typed data & config contracts |
| `PyYAML` | `6.0.2` | MIT | YAML configuration parsing |
| `joblib` | `1.4.2` | BSD-3-Clause | Versioned model artifact persistence |
| `pytest` | `8.3.4` | MIT | Automated test suite (dev) |
| `httpx` | `0.28.1` | BSD-3-Clause | FastAPI TestClient transport (dev) |

Optional (not installed in default environment): `torch` / `torch-geometric` (`[project.optional-dependencies].pyg`).
