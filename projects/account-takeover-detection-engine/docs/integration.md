# Integration Guide — Account Takeover Detection Engine

## 1. Repository Context & Sibling Projects

This project resides in `Risk-ML/projects/account-takeover-detection-engine/`. At implementation time, the sibling directories under `Risk-ML/projects/` (`aml-transaction-monitoring-engine`, `chargeback-prediction-engine`, `credit-risk-scoring-platform`, `graph-based-fraud-detection-engine`, `payment-authorization-optimization-engine`, `real-time-transaction-risk-platform`, `smart-payment-authentication-engine`) contain planning documents (`plan.md`) only — no shared Python library exists yet in `Risk-ML`.

Accordingly, this project is structured as an independent, cleanly packaged library (`ato_engine`) with narrow adapter seams so future `Risk-ML` components can integrate without copying code:

- **Downstream Authentication Orchestrator (`smart-payment-authentication-engine`)**: Consume `PolicyDecision.action` (`ALLOW`, `CHALLENGE_OTP`, `CHALLENGE_WEBAUTHN`, `BLOCK_AND_RESET`) and `calibrated_risk_score` via either direct Python import (`ATODetectionEngine.evaluate`) or `POST /v1/evaluate`.
- **Real-time Transaction Risk (`real-time-transaction-risk-platform`)**: Attach `decision_id`, `calibrated_risk_score`, and `evidence.triggered_rules` as upstream session-risk context before payment authorization.
- **Graph Fraud Detection (`graph-based-fraud-detection-engine`)**: Export `reports/decision_audit.jsonl` or synthetic `LoginSessionEvent` streams containing `(user_id, device_id, ip_address)` edges for downstream link analysis.

## 2. Adapter Boundaries

| Adapter | Interface | Implementation | Notes |
|---|---|---|---|
| Library API | `ATODetectionEngine` | `src/ato_engine/engine.py` | Pure Python interface isolated from HTTP/CLI I/O |
| HTTP Service | FastAPI app factory `create_app()` | `src/ato_engine/api.py` | Stateless HTTP adapter over `ATODetectionEngine` |
| State Store | `StateStore` Protocol | `InMemoryStateStore` (default), `RedisStateStore` (optional) | Swap via `store.backend` in YAML config; live Redis server is optional and marked **not run** in local acceptance checks |
| Audit Sink | `DecisionAuditLogger` | Append-only JSONL file + in-memory buffer | Easily wrapped for Kafka/SIEM shipping in future production work |

## 3. Dependency Licenses and Provenance

All dependencies are open-source and pinned in `requirements.txt` / `pyproject.toml`:

- `numpy` (BSD-3-Clause)
- `scipy` (BSD-3-Clause)
- `scikit-learn` (BSD-3-Clause)
- `joblib` (BSD-3-Clause)
- `fastapi` (MIT)
- `uvicorn` (BSD-3-Clause)
- `pydantic` (MIT)
- `PyYAML` (MIT)
- `pytest` (MIT, dev)
- `httpx` (BSD-3-Clause, dev)
- `redis` (MIT, optional extra)

**Provenance & Compliance Statement:** All source code, synthetic generators, and fixtures are original to this prototype. Zero proprietary employer/SAAT code, real customer records, real credentials, or live transaction logs are included.
