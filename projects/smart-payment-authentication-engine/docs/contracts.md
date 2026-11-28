# Data and Action Contracts (`schema_version = "1.0.0"`)

All schemas are implemented with strict Pydantic v2 models (`extra="forbid"`) in [`src/smart_auth_engine/contracts.py`](../src/smart_auth_engine/contracts.py).

## 1. Units, Timestamps, and Identifiers

- **Monetary amounts (`amount_minor`, `tx_amount_24h_minor`, `fraud_loss_minor`):** Positive/non-negative integers in **minor currency units** (e.g., USD/EUR/GBP/CAD/AUD cents; JPY yen for zero-decimal currencies). `amount_minor` must satisfy `1 <= amount_minor <= 100_000_000`.
- **Currencies:** Supported ISO 4217 subset: `USD`, `EUR`, `GBP`, `CAD`, `AUD`, `JPY`. Unsupported currencies fail validation explicitly.
- **Countries (`billing_country`, `card_bin_country`, `device.ip_country`):** 2-letter uppercase ISO-3166-1 alpha-2 codes.
- **Timestamps (`timestamp`, `event_timestamp`, `label_available_at`, `completed_at`, `audit_timestamp`):** Must be **timezone-aware ISO-8601 timestamps**; normalized internally to UTC. Naive timestamps without offsets are rejected.
- **Identifiers:**
  - `payment_id`, `account_id`, `merchant_id`: non-empty synthetic strings (1–128 chars). No real PANs, PII, or customer IDs permitted.
  - `decision_id`: deterministic SHA-256 prefix (`dec_<20 hex chars>`) derived from `(payment_id, timestamp_utc, model_version, feature_version, policy_version)` so replays produce identical IDs.

## 2. Input Contracts

### `PaymentEvent`
| Field | Type | Constraints / Semantics |
|---|---|---|
| `schema_version` | `str` | Must equal `"1.0.0"` |
| `payment_id` | `str` | Required, 1–128 chars |
| `account_id` | `str` | Required, 1–128 chars |
| `merchant_id` | `str` | Required, 1–128 chars |
| `merchant_category` | `str` | One of `retail`, `digital_goods`, `travel`, `crypto_exchange`, `gambling`, `groceries`, `utilities`, `luxury` |
| `amount_minor` | `int` | `1 <= amount_minor <= 100_000_000` |
| `currency` | `str` | One of `USD`, `EUR`, `GBP`, `CAD`, `AUD`, `JPY` |
| `timestamp` | `datetime` | Timezone-aware ISO-8601 (UTC) |
| `billing_country` | `str` | 2-letter uppercase ISO |
| `card_bin_country` | `str` | 2-letter uppercase ISO |
| `is_recurring` | `bool` | Default `false` |
| `is_cross_border` | `Optional[bool]` | Derived automatically if omitted (`ip_country != billing_country` or `card_bin_country != billing_country`) |
| `account_age_days` | `Optional[float]` | `>= 0` |
| `prior_chargebacks_90d` | `Optional[int]` | `0 <= x <= 1000` |
| `tx_count_24h` | `Optional[int]` | `>= 0` |
| `tx_amount_24h_minor` | `Optional[int]` | `>= 0` |
| `device` | `DeviceSignals` | Optional telemetry fields; `ip_velocity_24h >= ip_velocity_1h` enforced when both present |

### `RiskLabel` (Delayed Ground Truth)
| Field | Type | Constraints / Semantics |
|---|---|---|
| `payment_id` | `str` | Joins to `PaymentEvent.payment_id` |
| `event_timestamp` | `datetime` | Must match `PaymentEvent.timestamp` (UTC) |
| `is_fraud` | `bool` | Fraud ground truth |
| `label_available_at` | `datetime` | UTC timestamp when label matured (`>= event_timestamp`) |
| `label_source` | `str` | `chargeback`, `issuer_confirmed_fraud`, or `matured_non_fraud` |
| `fraud_loss_minor` | `int` | `>= 0`; must be `0` when `is_fraud == false` |

## 3. Risk and Action Output Contracts

### `AuthAction`
- `approve`: Frictionless authentication.
- `challenge`: Step-up authentication required (`otp_sms`, `biometric_app`, or `hardware_token`).
- `decline`: Hard rejection (override, model decline threshold, or fallback amount cap).
- `abstain`: Insufficient telemetry (`missing_feature_ratio >= abstain_missing_feature_ratio`) with no mandatory rule/override; routed to `held_for_review`.

### `PolicyDecision`
Includes `decision_id`, `payment_id`, `timestamp`, `action`, `calibrated_fraud_prob` (`null` on fallback/outage), `raw_fraud_prob`, `challenge_method`, `mandatory_rule_triggered`, `override_triggered`, `fallback_used`, `reason_codes`, `explanations` (top-5 local sensitivity attributions), `missing_features`, `missing_feature_ratio`, `model_version`, `feature_version`, and `policy_version`.

### `StepUpExecutionResult`
Wraps `decision`, optional `challenge_outcome` (`passed`, `failed`, `abandoned`, `timed_out`, `latency_ms`, `completed_at`), `final_outcome` (`authorized`, `declined`, `abandoned`, `held_for_review`), `expected_fraud_loss_minor`, `friction_cost_minor`, and `audit_timestamp`.

## 4. Validation Errors and Fixtures

- HTTP validation failures return HTTP `422` with `ErrorResponse(error_code="VALIDATION_ERROR", message=..., details=...)`.
- Missing decisions return HTTP `404` with `ErrorResponse(error_code="DECISION_NOT_FOUND", ...)`.
- Canonical fixtures live under `data/fixtures/`:
  - [`valid_payments.json`](../data/fixtures/valid_payments.json): Valid USD, cross-border EUR, and zero-decimal JPY payments.
  - [`invalid_payments.json`](../data/fixtures/invalid_payments.json): Negative amount, naive timestamp, unsupported currency/category/schema version, inconsistent velocities, and forbidden extra fields (e.g., raw PAN).
  - [`boundary_payments.json`](../data/fixtures/boundary_payments.json): Exact 1-minor minimum, exact mandatory challenge boundary (`50000` vs `49999`), and total telemetry loss (`abstain`).
  - [`reference_oracle.json`](../data/fixtures/reference_oracle.json): Hand-calculated analytical reference oracle cases verifying probabilities to `1e-6` and exact actions.
