# Integration Guide — `Risk-ML` Ecosystem Boundaries

## 1. Standalone Isolation Policy

This project (`projects/payment-authorization-optimization-engine/`) is an **independently runnable** Python package and FastAPI prototype. It has **zero hard runtime imports** from sibling projects in `Risk-ML` and **zero dependencies** on private employer, SAAT, or external payment gateway SDKs.

## 2. Sibling Projects in `Risk-ML` & Adapter Interfaces

Verified existing sibling directories under `Risk-ML/projects/`:

| Sibling Project | Verified Path | Conceptual Upstream / Downstream Role | Recommended Adapter Pattern |
|---|---|---|---|
| Real-Time Transaction Risk Platform | `projects/real-time-transaction-risk-platform/` | Upstream real-time fraud/risk scorer supplying `risk_score`, `velocity_1h`, `velocity_24h` | Map upstream score payload to `AuthorizationAttempt` fields; if upstream times out, pass `None` so missing-feature conservatism engages cleanly |
| Smart Payment Authentication Engine | `projects/smart-payment-authentication-engine/` | Downstream 3DS / SCA orchestration when this engine selects `STEP_UP_3DS` | Consume `DecisionRecord` (`action == "STEP_UP_3DS"`) over HTTP/JSON and pass `attempt_id`, `transaction_id`, `amount_cents`, `currency` |
| Chargeback Prediction Engine | `projects/chargeback-prediction-engine/` | Downstream post-authorization dispute risk & matured loss feedback | Feed matured dispute outcomes back into `HistoricalOutcome` (`fraud_confirmed`, `loss_cents`, `label_maturity_days`) for periodic retraining |
| Account Takeover Detection Engine | `projects/account-takeover-detection-engine/` | Upstream session/account risk signal | Combine ATO probability into upstream `risk_score` before calling `/v1/authorize` |

## 3. Integration Contract Rules

1. **Adapter over copy-paste:** Keep integrations as thin JSON/HTTP or typed-schema translation adapters rather than duplicating domain modules across `Risk-ML` projects.
2. **Idempotency propagation:** Callers MUST generate a deterministic `idempotency_key` per logical submission attempt and preserve `transaction_id` across retries so `IdempotencyLedger` prevents double-authorization.
3. **Health-state updates:** External gateway monitors can update processor state via `POST /v1/processors/{processor_id}/health` (`HEALTHY`, `DEGRADED`, `OUTAGE`) without restarting the inference service.
4. **Offline prototype boundary:** Production payment processors, PCI-DSS cardholder vaults, live acquirer credentials, and real customer PII are explicitly excluded and disabled.
