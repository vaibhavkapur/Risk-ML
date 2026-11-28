# Integration, Adapter Boundaries, and Dependencies

## 1. Repository Context (`Risk-ML`)

This prototype lives under `projects/smart-payment-authentication-engine/` inside the `Risk-ML` repository. Sibling projects in `projects/` (such as `account-takeover-detection-engine`, `chargeback-prediction-engine`, `payment-authorization-optimization-engine`, and `real-time-transaction-risk-platform`) are independent standalone prototypes.

To maintain strict isolation and zero fragile cross-directory imports, this project does **not** import code directly from sibling project directories. Instead, it exposes clean, versioned adapter boundaries so any sibling service can integrate via HTTP/JSON or Python protocol adapters:

| Boundary | Interface | Purpose |
|---|---|---|
| **Risk Scorer Adapter** | `RiskScorerProtocol` (`src/smart_auth_engine/inference.py`) | Swap in an external risk scoring model (e.g., from `real-time-transaction-risk-platform` or `account-takeover-detection-engine`) as long as it implements `predict_proba_event(event)` and declares `model_version` / `feature_version`. |
| **Decision Audit Store** | `DecisionStore` ABC (`src/smart_auth_engine/store.py`) | Pluggable persistence: `InMemoryDecisionStore`, `JsonlDecisionStore`, or `OptionalRedisDecisionStore`. |
| **HTTP Service Contract** | FastAPI REST API (`src/smart_auth_engine/api.py`) | Language-agnostic JSON integration (`/v1/authenticate`, `/v1/step-up/simulate`, `/v1/decisions/{decision_id}`, `/v1/replay/{decision_id}`, `/v1/metrics`, `/health`). |

## 2. Optional Redis Integration

- `OptionalRedisDecisionStore` supports persisting decisions and replay state to Redis when `redis` (`pip install .[redis]`) is installed and a Redis instance is running.
- If `storage.backend: "redis"` is configured without the `redis` package installed, initialization fails loudly with an actionable `RuntimeError` rather than silently dropping records.
- Unit tests verify the adapter logic against an in-memory Redis-compatible stub (`_FakeRedisClient`); live Redis server integration is explicitly marked **not run** in `reports/acceptance.md` because no Redis daemon is provisioned in the default offline test environment.

## 3. Dependency Licenses and Provenance

All dependencies are pinned in `requirements.txt` and `pyproject.toml`:

| Package | Version | License | Role |
|---|---|---|---|
| `numpy` | `1.26.4` | BSD-3-Clause | Numerical arrays & deterministic RNG |
| `scipy` | `1.13.1` | BSD-3-Clause | Scientific routines (scikit-learn dependency) |
| `scikit-learn` | `1.5.2` | BSD-3-Clause | Gradient boosting risk model & calibration |
| `fastapi` | `0.115.6` | MIT | HTTP API framework |
| `uvicorn` | `0.32.1` | BSD-3-Clause | ASGI server |
| `pydantic` | `2.10.4` | MIT | Strict schema & configuration validation |
| `PyYAML` | `6.0.2` | MIT | YAML configuration parsing |
| `joblib` | `1.4.2` | BSD-3-Clause | Model artifact serialization |
| `pytest` | `8.3.4` | MIT | Test runner (dev) |
| `httpx` | `0.28.1` | BSD-3-Clause | FastAPI TestClient transport (dev) |
| `redis` (optional) | `5.2.1` | MIT | Optional Redis store backend |

**Data & Code Provenance:** 100% original code and 100% deterministic synthetic fixtures generated with fixed seeds. Zero real transactions, customer records, credentials, or proprietary/employer artifacts.

## 4. Production Out-of-Scope Boundaries

The following production integrations are intentionally disabled / excluded:
- Live 3DS 2.x EMVCo directory server / ACS / challenge SDK integration
- Real payment gateways, card networks, or live capital authorization
- Real customer PII, PANs, or external identity/KYC providers
