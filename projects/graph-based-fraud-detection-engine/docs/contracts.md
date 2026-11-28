# Data Contracts — Graph-Based Fraud Detection Engine

**Schema Version:** `1.0.0`  
**Feature Version:** `1.0.0`  
**Timestamp Standard:** Timezone-aware ISO-8601 UTC (naive timestamps are rejected).  
**Monetary Units:** USD (`amount_usd`, strictly `0 < amount_usd <= 1,000,000.0`).

## 1. Privacy-Safe Identifiers

All entity and transaction IDs must conform to deterministic HMAC-SHA256 pseudonyms with typed prefixes:

| Entity | Format Regex | Example |
|---|---|---|
| Account | `^acc_[0-9a-f]{8,32}$` | `acc_8a1b2c3d4e5f` |
| Device | `^dev_[0-9a-f]{8,32}$` | `dev_1f2e3d4c5b6a` |
| Merchant | `^mer_[0-9a-f]{8,32}$` | `mer_998877665544` |
| Transaction | `^tx_[0-9a-f]{8,32}$` | `tx_a1b2c3d4e5f6` |

Raw strings containing email addresses, SSNs, or card PAN-like digit sequences are explicitly rejected with `PrivacyViolationError`.

## 2. Canonical Input Schemas

### `AccountRecord`
- `account_id` (`str`, required): privacy-safe `acc_*` ID
- `created_at` (`datetime`, required, UTC)
- `account_age_days_at_start` (`float >= 0.0`)
- `kyc_tier` (`"basic" | "standard" | "enhanced"`)
- `home_region` (`str`, 2–16 chars)
- `is_synthetic_ring_member` (`bool`, synthetic metadata only — never used as a model feature)
- `is_shared_household_member` (`bool`, synthetic metadata only — never used as a model feature)

### `DeviceRecord`
- `device_id` (`str`, required): privacy-safe `dev_*` ID
- `first_seen_at` (`datetime`, required, UTC)
- `device_type`: one of `mobile_ios`, `mobile_android`, `desktop_web`, `kiosk`, `emulator`
- `is_shared_legitimate` (`bool`, synthetic metadata only — never used as a model feature)
- `device_risk_hint` (`float` in `[0.0, 1.0]`, synthetic metadata only)

### `MerchantRecord`
- `merchant_id` (`str`, required): privacy-safe `mer_*` ID
- `onboarded_at` (`datetime`, required, UTC)
- `category`: one of `groceries`, `utilities`, `retail`, `travel`, `digital_goods`, `crypto_onramp`, `gift_cards`, `electronics`, `restaurant`
- `country` (`str`, 2–8 chars)
- `base_chargeback_rate` (`float` in `[0.0, 1.0]`)

### `TransactionRecord`
- `tx_id` (`str`, `tx_*`)
- `account_id` (`str`, `acc_*`)
- `device_id` (`str`, `dev_*`)
- `merchant_id` (`str`, `mer_*`)
- `tx_timestamp` (`datetime`, UTC)
- `amount_usd` (`float`, `(0.0, 1_000_000.0]`)
- `currency`: one of `USD`, `EUR`, `GBP`, `CAD`
- `channel`: one of `web`, `mobile`, `pos`, `api`
- `ip_prefix` (`str`, 3–32 chars)

### `FraudLabelRecord`
- `tx_id` (`str`, `tx_*`)
- `is_fraud` (`0 | 1`)
- `tx_timestamp` (`datetime`, UTC — must equal the transaction's `tx_timestamp`)
- `label_available_at` (`datetime`, UTC — must satisfy `label_available_at >= tx_timestamp`)
- `label_source`: `"chargeback" | "analyst_review" | "rule_confirmed" | "matured_clean"`
- `fraud_pattern` (`Optional[str]`): synthetic diagnostic tag for segment analysis

## 3. Cross-Table Validation & Error Contract

`validate_dataset_bundle(bundle)` enforces:
1. `schema_version == "1.0.0"`
2. Unique `account_id`, `device_id`, `merchant_id`, `tx_id`, and one label per `tx_id`
3. Referential integrity: every transaction references known accounts, devices, and merchants; every label references a known transaction with matching `tx_timestamp`
4. Label causality: `label_available_at >= tx_timestamp`

Violations raise `SchemaValidationError` containing all detected error strings in `.errors`.

## 4. Output Contracts

### `RiskScoreResponse` (`POST /v1/score`)
- `decision_id` (`str`): deterministic `dec_<hash>` identifier
- `schema_version`, `feature_version`, `model_name`, `model_version`
- `tx_id` (`str`)
- `raw_score` (`float` in `[0, 1]`)
- `calibrated_probability` (`float` in `[0, 1]`)
- `decision`: `"approve" | "review" | "block"`
- `cold_start_flags`: `{"account_unseen": bool, "device_unseen": bool, "merchant_unseen": bool}`
- `explanation`: `GraphExplanation` with `reason_codes`, `top_feature_contributions`, `linked_entities`, `evidence_paths`, `ring_cluster_size`, `shared_device_accounts_prior`, `legitimate_shared_device_mitigant`, `graph_audit_max_edge_timestamp`
- `review_case_id` (`Optional[str]`): populated for `review` / `block` decisions when `record_for_review=true`
- `diagnostics`: `run_id`, `max_edge_timestamp_used`, `max_label_available_at_used`

### HTTP Error Codes
- `400 Bad Request`: unsupported model name or invalid review status filter
- `404 Not Found`: unknown `case_id`
- `422 Unprocessable Entity`: Pydantic/contract validation failure (PII ID, naive timestamp, unsupported currency, negative amount)
- `503 Service Unavailable`: scoring or review called before service initialization

## 5. Fixtures

Generated deterministically by `scripts/generate_fixtures.py` into `data/fixtures/`:
- `valid_reference_bundle.json` + `reference_expected_oracle.json`: hand-calculated 6-account / 3-device / 2-merchant / 8-transaction fixture
- `invalid_contract_cases.json`: PII ID, naive timestamp, negative amount, unsupported currency, and pre-transaction label maturity
- `boundary_cases.json`: unseen cold-start entities (`$0.01`) and upper-bound amount (`$999,999.99`)
