# Data Contracts — Payment Authorization Optimization Engine

**Contract Schema Version:** `1.0.0`  
**Feature Schema Version:** `1.0.0`

All inputs and outputs are strictly typed (`extra="forbid"`) via Pydantic models in [schemas.py](../src/payment_auth_engine/schemas.py).

## 1. Canonical Input: `AuthorizationAttempt`

| Field | Type | Units / Format | Constraints & Validation |
|---|---|---|---|
| `schema_version` | `string` | SemVer | Must equal `"1.0.0"` (`UNSUPPORTED_SCHEMA_VERSION`) |
| `attempt_id` | `string` | Unique attempt ID | Non-empty, 1–128 chars |
| `transaction_id` | `string` | Logical payment ID | Non-empty, 1–128 chars; links retries |
| `idempotency_key` | `string` | Deduplication key | Non-empty, 1–128 chars; prevents double-charge |
| `timestamp_utc` | `datetime` | ISO-8601 UTC | Must include timezone (`MISSING_TIMEZONE`); normalized to UTC |
| `amount_cents` | `integer` | Minor currency units | `1 <= amount_cents <= 100_000_000` (no floating-point currency) |
| `currency` | `string` | ISO-4217 uppercase | Supported set: `USD`, `EUR`, `GBP`, `CAD`, `AUD`, `JPY` (`UNSUPPORTED_CURRENCY`) |
| `merchant_id` | `string` | Synthetic merchant ID | Non-empty, 1–64 chars |
| `merchant_category_code` | `string` | 4-digit MCC | Regex `^[0-9]{4}$` (`INVALID_MCC`) |
| `card_network` | `enum` | Lowercase string | `visa`, `mastercard`, `amex`, `discover` |
| `issuer_country` | `string` | ISO-3166 alpha-2 | Regex `^[A-Z]{2}$` (`INVALID_COUNTRY_CODE`) |
| `merchant_country` | `string` | ISO-3166 alpha-2 | Regex `^[A-Z]{2}$` (`INVALID_COUNTRY_CODE`) |
| `entry_mode` | `enum` | Lowercase string | `chip`, `contactless`, `ecommerce`, `recurring`, `moto` |
| `is_recurring` | `boolean` | Flag | Default `false` |
| `has_3ds` | `boolean` | Flag | Default `false` |
| `has_network_token` | `boolean` | Flag | Default `false` |
| `attempt_number` | `integer` | 1-indexed counter | `1 <= attempt_number <= 5` |
| `prior_response_code` | `string \| null` | ISO-8583 / gateway code | Required when `attempt_number > 1` & `prior_outcome_known=true` (`MISSING_PRIOR_RESPONSE_CODE`); forbidden when `attempt_number == 1` (`UNEXPECTED_PRIOR_RESPONSE_CODE`) |
| `prior_outcome_known` | `boolean` | Certainty flag | Default `true`; `false` triggers `HOLD_RECONCILE` |
| `risk_score` | `float \| null` | Probability `[0, 1]` | Optional; finite in `[0.0, 1.0]` (`INVALID_RISK_SCORE`) |
| `velocity_1h` | `integer \| null` | Count | Optional; `0 <= v <= 10_000` |
| `velocity_24h` | `integer \| null` | Count | Optional; `0 <= v <= 100_000` |
| `account_age_days` | `integer \| null` | Days | Optional; `0 <= d <= 36_500` |
| `bin_approval_rate_30d` | `float \| null` | Ratio `[0, 1]` | Optional; finite in `[0.0, 1.0]` (`INVALID_BIN_APPROVAL_RATE`) |
| `processor_id` | `string` | Primary route ID | Default `"proc_a"` |

## 2. Processor Response Taxonomy (`taxonomy.py`)

| Response Code | Category | Retry Eligible? | Engine Action / Rule |
|---|---|---|---|
| `00` | `APPROVED` | No | `DEDUPLICATE_RETURN_PRIOR` (terminal) |
| `14`, `54`, `57`, `59`, `63` | `HARD_DECLINE` | **Never** | `BLOCK_HARD_DECLINE` |
| `91`, `96` | `TECHNICAL_RETRIABLE` | Conditional | `ROUTE_SECONDARY` failover (if within `max_retries` and enabled) |
| `65` | `SOFT_DECLINE` | Conditional | `STEP_UP_3DS` (if `has_3ds=false` and within `max_retries`) |
| `51`, `61` | `SOFT_DECLINE` | Recurring only | Allowed for scheduled `is_recurring=true`; blocked for immediate online retry |
| `05` | `SOFT_DECLINE` | Single retry | Max 1 smart secondary failover (`attempt_number == 2`) |
| `TIMEOUT`, `NETWORK_ERROR`, `UNKNOWN`, any unmapped code | `UNCERTAIN` | **Never blindly** | `HOLD_RECONCILE` (fail-closed pending reconciliation) |

## 3. Canonical Output: `DecisionRecord`

| Field | Type | Semantics |
|---|---|---|
| `schema_version` | `string` | `"1.0.0"` |
| `decision_id` | `string` | Deterministic SHA-256 prefix (`dec_<20hex>`) over `(attempt_id, transaction_id, idempotency_key, policy_name, model_version, config_hash)` |
| `attempt_id`, `transaction_id`, `idempotency_key` | `string` | Echoed identifiers |
| `policy_name`, `model_version`, `feature_schema_version`, `config_hash` | `string` | Full reproducibility provenance |
| `action` | `PolicyAction` | Selected action (`APPROVE_PRIMARY`, `ROUTE_SECONDARY`, `STEP_UP_3DS`, `DECLINE_POLICY`, `BLOCK_HARD_DECLINE`, `HOLD_RECONCILE`, `DEDUPLICATE_RETURN_PRIOR`, `BLOCK_MAX_RETRIES`, `BLOCK_PROCESSOR_OUTAGE`) |
| `processor_selected` | `string \| null` | Active processor route or `null` when blocked/declined |
| `predicted_acceptance_prob` | `float` | Calibrated $P(\text{approved}\mid x,a)$ in `[0, 1]` |
| `predicted_loss_prob` | `float` | Calibrated $P(\text{fraud}\mid x,\text{approved},a)$ in `[0, 1]` |
| `expected_loss_cents` | `float` | Expected fraud + dispute loss in minor units (`>= 0`) |
| `expected_net_value_cents` | `float` | Expected margin minus expected loss minus action cost |
| `estimated_latency_ms` | `float` | Estimated authorization latency in milliseconds |
| `constraint_checks` | `list[ConstraintCheckResult]` | Per-constraint limit, observed value, pass/fail boolean, message |
| `any_constraint_breach` | `boolean` | Explicit flag (`false` when all guards pass) |
| `explanation` | `DecisionExplanation` | `reason_codes`, `missing_features`, `top_contributing_factors`, `policy_notes` |

## 4. Validation & Failure Error Codes

- **Contract Errors (`ContractValidationError`, HTTP 422):** `CONTRACT_VALIDATION_ERROR`, `UNSUPPORTED_SCHEMA_VERSION`, `MISSING_TIMEZONE`, `UNSUPPORTED_CURRENCY`, `INVALID_MCC`, `INVALID_COUNTRY_CODE`, `INVALID_RISK_SCORE`, `INVALID_BIN_APPROVAL_RATE`, `MISSING_PRIOR_RESPONSE_CODE`, `UNEXPECTED_PRIOR_RESPONSE_CODE`.
- **Configuration Errors (`ConfigurationError`, CLI exit code 2):** `CONFIG_NOT_FOUND`, `CONFIG_NOT_FILE`, `CONFIG_PARSE_ERROR`, `CONFIG_EMPTY_OR_INVALID`, `CONFIG_VALIDATION_ERROR`.
- **Counterfactual Assumption Errors (`CounterfactualAssumptionError`):** `EMPTY_EVALUATION_SET`, `DIMENSION_MISMATCH`, `INVALID_PROPENSITY`, `POSITIVITY_VIOLATION`, `NON_FINITE_REWARDS`, `INSUFFICIENT_SUPPORT`, `ZERO_OVERLAP`, `ESS_COLLAPSE`.

## 5. Fixtures

- [valid_attempts.json](../data/fixtures/valid_attempts.json): 5 valid attempts (initial, technical failover, SCA step-up, recurring NSF, missing optional features).
- [invalid_attempts.json](../data/fixtures/invalid_attempts.json): 8 invalid payloads mapped to expected error codes.
- [boundary_attempts.json](../data/fixtures/boundary_attempts.json): 6 boundary cases (1-cent min, 1M-dollar max, hard decline `54`, uncertain `TIMEOUT`, unmapped `ZZ`, retry cap `attempt_number=5`).
- [reference_oracle.json](../data/fixtures/reference_oracle.json): Hand-calculated exact arithmetic oracles for DM/IPS/SNIPS/DR estimators, expected net value/loss bps economics, and response taxonomy classification.
