# Integration Guide & Dependency Inventory

## 1. Repository Context (`Risk-ML`)

Verified workspace inspection of `Risk-ML/projects/` shows sibling planned projects (`account-takeover-detection-engine`, `chargeback-prediction-engine`, `credit-risk-scoring-platform`, `graph-based-fraud-detection-engine`, `payment-authorization-optimization-engine`, `real-time-transaction-risk-platform`, `smart-payment-authentication-engine`), each currently containing only its `plan.md`. Because no shared library modules exist yet in `Risk-ML`, this project is structured as a self-contained runnable package (`src/aml_engine/`) exposing stable adapter boundaries so future sibling projects can integrate cleanly without copying code.

## 2. Reusable Adapter Boundaries

| Boundary | Interface / Module | Integration Usage |
|---|---|---|
| Transaction Graph Export | `aml_engine.features.graph.TransactionGraph.to_networkx()` | Exports directed temporal multi-graph to `networkx.MultiDiGraph` for interoperability with `graph-based-fraud-detection-engine` |
| Monitoring Batch Facade | `aml_engine.api.service.AMLMonitoringService` | Framework-agnostic Python service facade reusable by `real-time-transaction-risk-platform` or external orchestrators |
| HTTP / ASGI API | `aml_engine.api.http_server.dispatch_http_request`, `create_http_server`, `aml_engine.api.fastapi_app.create_fastapi_app` | Zero-dependency HTTP router + drop-in FastAPI ASGI app exposing `/health`, `/monitor`, `/alerts`, `/cases`, `/cases/{id}/review`, `/runs` |
| Relational Persistence | `aml_engine.storage.repository.CaseRepository` (`SQLiteCaseRepository`, `PostgresCaseRepository`) | Pluggable storage protocol with PostgreSQL DDL (`src/aml_engine/storage/schema_postgres.sql`) |
| Model Artifact Contract | `MLPrioritizer.save_artifact()` / `load_artifact()` | Versioned JSON artifact (`feature_names`, scaler stats, weights, Platt parameters, `config_hash`, training metadata) |

## 3. Dependency & License Inventory

### Core Runtime Dependencies
- **None** (Python Standard Library `>= 3.9`, PSF License).

### Optional Pinned Extras (`requirements-optional.txt`)
| Package | Pinned Version | License | Purpose |
|---|---|---|---|
| `fastapi` | `0.115.6` | MIT | Optional ASGI REST framework (`create_fastapi_app`) |
| `uvicorn` | `0.34.0` | BSD-3-Clause | Optional ASGI server for FastAPI |
| `pydantic` | `2.10.4` | MIT | FastAPI request/response modeling |
| `networkx` | `3.2.1` | BSD-3-Clause | Optional graph interoperability (`TransactionGraph.to_networkx`) |
| `psycopg[binary]` | `3.2.3` | LGPL-3.0 | Optional PostgreSQL driver (`PostgresCaseRepository`) |
| `pytest` | `8.3.4` | MIT | Optional test runner (all tests also run natively via `unittest`) |
| `httpx` | `0.28.1` | BSD-3-Clause | Optional FastAPI test client |

## 4. Non-Production / Exclusion Boundary
- No proprietary employer or SAAT code, schemas, credentials, or private registries are referenced.
- Live PostgreSQL, external KYC providers, and production compliance filing endpoints remain explicitly disabled and outside scope.
