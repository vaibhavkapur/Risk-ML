# Data Contracts — Chargeback Prediction Engine

**Schema version:** `1.0.0`  
**Feature set version:** `1.0.0`

## 1. Timestamp and currency semantics

- **Timestamps:** All timestamps MUST be ISO-8601 strings with explicit UTC offset (`Z` or `+00:00`). Naive timestamps without timezone info raise `ContractValidationError`.
- **Units:**
  - Monetary amounts (`amount`, `dispute_amount`) are positive finite floats in the record's native `currency`.
  - Normalized amounts (`amount_usd`, `dispute_amount_usd`) are converted deterministically using static offline synthetic FX rates (`USD: 1.00`, `EUR: 1.08`, `GBP: 1.27`, `CAD: 0.74`). Unsupported currencies raise `UnsupportedInputError`.
  - Durations (`customer_tenure_days`, `avg_fulfillment_days`, `days_since_settlement`, `dispute_delay_days`) are in calendar days.

## 2. Input schemas

### 2.1 Merchants (`MerchantRecord`)

| Field | Type | Constraints | Notes |
|---|---|---|---|
| `merchant_id` | string | Required, unique, non-empty | Canonical merchant identifier |
| `mcc_group` | string | Required, one of `digital_goods`, `travel`, `retail`, `subscription`, `hospitality`, `services` | Unsupported values raise `UnsupportedInputError` |
| `onboarding_ts` | ISO-8601 UTC | Required | Merchant onboarding timestamp |
| `country` | string | Required, 2-letter alphabetic ISO code | Normalized uppercase |
| `avg_fulfillment_days` | float | Required, finite, `>= 0` | Average order fulfillment latency (days) |

### 2.2 Transactions (`TransactionRecord`)

| Field | Type | Constraints | Notes |
|---|---|---|---|
| `transaction_id` | string | Required, unique, non-empty | Canonical transaction identifier |
| `merchant_id` | string | Required, must exist in merchants (when `require_known_merchant=True`) | Referential integrity enforced |
| `transaction_ts` | ISO-8601 UTC | Required | Authorization/capture event time |
| `settlement_ts` | ISO-8601 UTC | Optional (defaults to `transaction_ts`), `>= transaction_ts` | Start of label maturity window |
| `amount` | float | Required, finite, `> 0` | Native currency amount |
| `currency` | string | Required, one of `USD`, `EUR`, `GBP`, `CAD` | Normalized to `amount_usd` |
| `channel` | string | Required, one of `card_not_present`, `card_present`, `recurring`, `digital_wallet` | Unsupported values raise `UnsupportedInputError` |
| `is_cross_border` | bool | Optional (default `false`) | Cross-border indicator |
| `has_3ds` | bool | Optional (default `false`) | 3DS authentication indicator |
| `has_avs_match` | bool | Optional (default `true`) | AVS match indicator |
| `customer_tenure_days` | int | Optional (default `0`), `>= 0` | Customer account age in days |
| `maturity_date` | ISO-8601 UTC | Optional, if present must be `> settlement_ts` | Overrides `settlement_ts + default_maturity_days` |

### 2.3 Disputes (`DisputeRecord`)

| Field | Type | Constraints | Notes |
|---|---|---|---|
| `dispute_id` | string | Required, unique, non-empty | Canonical dispute identifier |
| `transaction_id` | string | Required, must exist in transactions | Duplicate references controlled by `maturity.duplicate_dispute_policy` |
| `dispute_ts` | ISO-8601 UTC | Required, `>= transaction.transaction_ts` | Date dispute was reported/filed |
| `reason_code` | string or null | Optional | `null`/blank/`"UNKNOWN"` mapped deterministically to `UNKNOWN` (`reason_category="unknown"`, `is_missing_reason_code=true`) |
| `dispute_amount` | float | Required, finite, `> 0`, `<= transaction.amount_usd * 1.05` (in USD) | Disputed amount |
| `currency` | string | Optional (default `USD`), supported currency | Normalized to `dispute_amount_usd` |

### 2.4 Reason code taxonomy

| Synthetic Reason Code | Canonical `reason_category` |
|---|---|
| `10.4`, `10.5`, `4837` | `fraud` |
| `13.1`, `13.3`, `4853` | `consumer_dispute` |
| `11.1`, `4808` | `authorization` |
| `12.5`, `4834` | `processing_error` |
| `null`, `""`, `"UNKNOWN"` | `unknown` (`is_missing_reason_code=true`) |
| Any other non-empty string | `unknown` (`is_unrecognized_reason_code=true`, code preserved) |

## 3. Output contracts

### 3.1 Label maturity table

Adds: `maturity_date`, `days_since_settlement`, `is_mature`, `has_observed_dispute`, `dispute_ts`, `dispute_delay_days`, `reason_code`, `reason_category`, `is_missing_reason_code`, `is_unrecognized_reason_code`, `label_status` (`mature_positive`, `mature_negative`, `immature_positive`, `immature_negative`), `label` (`1`, `0`, or `null`), `is_trainable` (`bool`).

### 3.2 Pipeline summary JSON (`reports/demo_summary.json`)

Top-level keys: `run_id`, `schema_version`, `config_hash`, `feature_metadata`, `maturity_diagnostics`, `split_summary`, `models`, `test_metrics`, `cold_start_slice_metrics`, `policy_comparison`, `delayed_outcome_monitoring`, `sample_explanations`, `structured_events`.

## 4. Error taxonomy

| Exception | Trigger | CLI Exit Code |
|---|---|---|
| `ConfigError` | Missing/malformed YAML, invalid parameter ranges, `allow_automated_merchant_punishment=true` | `2` |
| `ContractValidationError` | Negative/NaN amounts, naive timestamps, `settlement_ts < transaction_ts`, `dispute_ts < transaction_ts`, unknown FK | `2` |
| `UnsupportedInputError` | Unsupported `currency`, `channel`, or `mcc_group` | `2` |
| `DuplicateDisputeError` | Multiple disputes for the same `transaction_id` under `duplicate_dispute_policy="raise"` | `2` |
| `TemporalLeakageError` | Split timestamp overlap, empty split after embargo, or null/immature label in split | `2` |

## 5. Fixtures

- Valid & boundary reference fixtures: [`data/fixtures/reference_merchants.json`](../data/fixtures/reference_merchants.json), [`reference_transactions.json`](../data/fixtures/reference_transactions.json), [`reference_disputes.json`](../data/fixtures/reference_disputes.json), [`reference_expected.json`](../data/fixtures/reference_expected.json).
- Invalid contract fixtures: [`data/fixtures/invalid_cases.json`](../data/fixtures/invalid_cases.json).
