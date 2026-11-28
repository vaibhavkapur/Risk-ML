# Data Contracts (`tx.v1`, `feat.v1`, `rtrp.model.v1`, `audit.v1`)

This document defines the canonical input schemas, units, timestamp semantics, identifiers, validation errors, and output contracts for the **Real-Time Transaction Risk Platform** prototype.

---

## 1. Transaction Event Input (`tx.v1`)

All monetary quantities use integer minor currency units (`amount_cents`) to avoid floating-point rounding discrepancies. All timestamps MUST be timezone-aware ISO-8601 UTC strings (canonicalized to epoch milliseconds `event_ts_ms`).

| Field | Type | Constraints / Units | Notes |
|---|---|---|---|
| `schema_version` | `string` | Must be in `schema.supported_versions` (`"tx.v1"`) | Unsupported values raise `UnsupportedSchemaVersionError` (HTTP 422). |
| `event_id` | `string` | Non-empty unique idempotency key | Duplicate with identical payload hash is ignored (`DUPLICATE_IGNORED`); different payload hash raises `DuplicateConflictError` (HTTP 409). |
| `account_id` | `string` | Non-empty synthetic token (e.g. `acct_syn_0001`) | Raw 13–19 digit PAN patterns rejected (`SensitiveDataViolationError`). |
| `card_token` | `string` | Non-empty synthetic token (e.g. `card_tok_0001`) | Raw 13–19 digit PAN patterns rejected (`SensitiveDataViolationError`). |
| `merchant_id` | `string` | Non-empty synthetic token (e.g. `merch_syn_001`) | Used for 24h distinct-merchant velocity. |
| `merchant_category` | `string` | Lowercase MCC label | Known set: `grocery`, `electronics`, `travel`, `gaming`, `digital_goods`, `utilities`, `restaurant`, `crypto_exchange`, `unknown`. Unknown values trigger conservative policy escalation. |
| `amount_cents` | `int` | `1 <= amount_cents <= 100_000_000` (minor units) | Booleans and non-integers rejected (`ContractValidationError`). |
| `currency` | `string` | ISO-4217 uppercase in `{"USD", "EUR", "GBP"}` | Converted to USD via versioned config `schema.fx_to_usd`. Unsupported currencies raise `UnsupportedCurrencyError` (HTTP 422). |
| `country_code` | `string` | ISO-3166 alpha-2 (`^[A-Z]{2}$`) | Countries outside known table trigger conservative `UNKNOWN_COUNTRY` flag. |
| `channel` | `string` | Enum: `card_present`, `ecommerce`, `in_app`, `recurring` | Invalid values raise `ContractValidationError`. |
| `device_id` | `string` | Non-empty synthetic device token | Used for 24h distinct-device velocity. |
| `event_ts` | `string` | Timezone-aware ISO-8601 UTC (e.g. `2026-01-15T10:00:00Z`) | Naive timestamps without offset are rejected (`ContractValidationError`). |
| `ingest_ts` | `string \| null` | Optional timezone-aware ISO-8601 UTC | When omitted, freshness defaults to `event_ts`. |
| `ip_risk_score` | `float` | `[0.0, 1.0]` | Synthetic upstream risk signal. Out-of-range values raise `ContractValidationError`. |

### Fixtures

- Valid hand-calculated oracle: [reference_transactions.json](../data/fixtures/reference_transactions.json)
- Invalid validation cases: [invalid_transactions.json](../data/fixtures/invalid_transactions.json)
- Boundary cases (1-cent minimum, exact 5m half-open edge, timestamp tie-break, unknown MCC/country): [boundary_transactions.json](../data/fixtures/boundary_transactions.json)

---

## 2. Feature Contract (`feat.v1`)

### Point-in-Time Window Semantics

For target event `e` of account `A` at timestamp `t = e.event_ts_ms`, a rolling window of duration `W` milliseconds includes all historical events `h` for account `A` satisfying:

1. `h.event_id != e.event_id`
2. `(h.event_ts_ms, h.event_id) < (e.event_ts_ms, e.event_id)` (strict causal ordering with lexicographic tie-breaking)
3. `t - W < h.event_ts_ms <= t` (half-open interval `(t - W, t]`)

### Feature Definitions (`feat.v1`)

| # | Feature | Unit / Range | Definition |
|---|---|---|---|
| 1 | `amount_usd_log1p` | `ln(USD + 1)` | $\ln(1 + \text{amount\_usd})$ |
| 2 | `tx_count_5m` | count | Prior account events in `(t - 300s, t]` |
| 3 | `tx_count_1h` | count | Prior account events in `(t - 3600s, t]` |
| 4 | `tx_count_24h` | count | Prior account events in `(t - 86400s, t]` |
| 5 | `tx_amount_sum_1h_usd` | USD | Sum of `amount_usd` of prior events in `(t - 3600s, t]` |
| 6 | `tx_amount_sum_24h_usd` | USD | Sum of `amount_usd` of prior events in `(t - 86400s, t]` |
| 7 | `amount_ratio_to_24h_avg` | ratio | `amount_usd / (avg_24h_usd + 1.0)` if `tx_count_24h > 0` else `1.0` |
| 8 | `distinct_merchants_24h` | count | Distinct `merchant_id` in prior 24h window |
| 9 | `distinct_devices_24h` | count | Distinct `device_id` in prior 24h window |
| 10 | `country_changed_from_prev` | `{0.0, 1.0}` | `1.0` if most recent prior 24h event had different `country_code`, else `0.0` |
| 11 | `seconds_since_prev_tx_log1p` | `ln(sec + 1)` | `ln(1 + min(86400, (t - prev_ts) / 1000))` or `ln(86401)` if no prior 24h event |
| 12 | `high_risk_context_score` | `[0.0, 1.5]` | `min(1.5, channel_risk + mcc_risk + 0.5 * ip_risk_score)` |

---

## 3. Outcome Labels (`OutcomeLabel`)

| Field | Type | Constraints |
|---|---|---|
| `event_id` | `string` | Joins to `TransactionEvent.event_id` |
| `is_fraud` | `int` | `0` (legit) or `1` (fraud) |
| `label_ts` | `string` | ISO-8601 UTC timestamp when label matured; temporal training enforces `label_ts <= split_snapshot_ts` |
| `label_source` | `string` | `"synthetic_chargeback"` or `"synthetic_confirmed_legit"` |

---

## 4. Model Artifact Contract (`rtrp.model.v1`)

Serialized as deterministic sorted JSON containing:
- `artifact_format`: `"rtrp.model.v1"`
- `model_version`, `feature_version`, `feature_names`
- `scaler_mean`, `scaler_scale`, `weights`, `intercept`
- `calibrator`: `{method: "platt_sigmoid", platt_a, platt_b}`
- `training_metadata`: seed, maturity window, temporal split cutoffs, evaluation metrics (ROC-AUC, PR-AUC, Brier, ECE), and per-feature reference histograms/moments for skew detection
- `artifact_sha256`: SHA-256 over canonical JSON body excluding `artifact_sha256`; verified strictly on load (`ArtifactIntegrityError` on tamper or schema mismatch).

---

## 5. Scoring Response & Error Contracts

### `POST /v1/score` Output (`RiskDecision`)

- `decision_id`: deterministic `dec_<sha256[:20]>` derived from `(event_id, model_version, feature_version, policy_version, as_of_ts_ms)`
- `action`: `"APPROVE" | "REVIEW" | "DECLINE"`
- `risk_score`: calibrated fraud probability `[0.0, 1.0]` (or conservative fallback score)
- `raw_score`: pre-calibration probability (`null` on rule fallback)
- `model_version`, `feature_version`, `policy_version`
- `feature_freshness_ms`: age in milliseconds relative to `as_of_ts` (`null` on `COLD_START`)
- `feature_status`: `"FRESH" | "COLD_START" | "STALE"`
- `fallback_used`: `bool`; `fallback_reason`: `null | "MODEL_TIMEOUT" | "STALE_FEATURES" | "UNKNOWN_INPUT_CONSERVATIVE" | "MODEL_ERROR"`
- `reason_codes`: deterministic human-auditable trigger list
- `top_attributions`: top-5 standardized linear feature contributions
- Traceability HTTP response headers: `X-Decision-Id`, `X-Model-Version`, `X-Feature-Version`, `X-Policy-Version`

### Error Mapping

| Exception | HTTP Status | `error_code` |
|---|---|---|
| `SensitiveDataViolationError` | `422` | `SENSITIVE_DATA_VIOLATION` |
| `UnsupportedSchemaVersionError` | `422` | `UNSUPPORTED_SCHEMA_VERSION` |
| `UnsupportedCurrencyError` | `422` | `UNSUPPORTED_CURRENCY` |
| `ContractValidationError` | `422` | `CONTRACT_VALIDATION_ERROR` |
| `DuplicateConflictError` | `409` | `DUPLICATE_CONFLICT` |
| `ConfigurationError` | Startup fail-fast | N/A |
| `ArtifactIntegrityError` | Load fail-fast | N/A |
