# Data Contracts & Schemas (`schema_version: 1.0.0`)

## 1. Units, Identifiers, and Timestamp Semantics

- **Monetary Amounts**: Floating-point positive amounts (`> 0.0`, finite). Original currency is preserved alongside normalized USD (`amount_usd = round(original_amount * fx_rate_to_usd, 6)`), using the versioned FX table in `config/default_config.json`.
- **Timestamps**: Strictly ISO-8601 strings with explicit timezone (`Z` or explicit UTC offset, e.g., `2026-01-10T09:00:00Z`). Naive timestamps without timezone info are rejected with `ValidationError`. All internal computations normalize to UTC (`timezone.utc`).
- **Identifiers**:
  - Input IDs: non-empty strings (`entity_id`, `account_id`, `tx_id`).
  - Output IDs (deterministic SHA-256 derived): `ALT-<16HEX>`, `EVD-<16HEX>`, `CASE-<16HEX>`, `AUD-<16HEX>`, `RUN-<16HEX>`.

---

## 2. Input Contracts

### 2.1 Entity (`entities[]`)
| Field | Type | Required | Constraints |
|---|---|---|---|
| `entity_id` | string | Yes | Non-empty unique identifier |
| `name` | string | Yes | Non-empty synthetic display name |
| `risk_tier` | enum string | No (default `LOW`) | `LOW`, `MEDIUM`, `HIGH` |
| `jurisdiction` | string | No (default `US`) | ISO-style uppercase jurisdiction code |

### 2.2 Account (`accounts[]`)
| Field | Type | Required | Constraints |
|---|---|---|---|
| `account_id` | string | Yes | Non-empty unique identifier |
| `entity_id` | string | Yes | Links to `Entity.entity_id` (missing entity is flagged explicitly during normalization) |
| `account_type` | string | No (default `RETAIL`) | `RETAIL`, `CORPORATE`, `MERCHANT`, `PAYROLL_DISBURSEMENT`, `TREASURY` |
| `opened_at` | string | No | ISO-8601 UTC timestamp |
| `baseline_monthly_usd` | float | No (default `5000.0`) | Finite `>= 0.0` |

### 2.3 Raw Transaction (`transactions[]`)
| Field | Type | Required | Constraints |
|---|---|---|---|
| `tx_id` | string | Yes | Non-empty; duplicates are deduplicated & counted |
| `source_account_id` | string | Yes | Non-empty; must differ from `target_account_id` |
| `target_account_id` | string | Yes | Non-empty; self-transfers rejected (`ValidationError`) |
| `amount` | float | Yes | Finite, strictly `> 0.0` |
| `currency` | string | Yes | Must exist in `config.fx_rates_to_usd` (`UnsupportedInputError` otherwise) |
| `timestamp` | string | Yes | ISO-8601 with timezone (`ValidationError` otherwise) |
| `channel` | string | Yes | Must exist in `config.supported_channels` (`UnsupportedInputError` otherwise) |
| `scenario_label` | string / null | No | Synthetic evaluation label only (never read by detection rules or ML inference) |

---

## 3. Output Contracts

### 3.1 Advisory Alert (`Alert`)
- `alert_id`: Stable SHA-256 ID (`ALT-*`).
- `subject_account_id`, `subject_entity_id`: Account/entity under review.
- `scenarios[]`: Sorted list of triggered scenario names.
- `involved_tx_ids[]`: Sorted list of triggering transactions.
- `prioritization`: `{rule_score, ml_raw_score, ml_calibrated_probability, combined_score, priority, policy_version, model_version, feature_contributions[]}`.
- `reasons[]`: Structured human-readable `ReasonContribution` list (`RULE`, `GRAPH`, `ML_FEATURE`, `DATA_QUALITY`).
- `evidence`: `EvidenceRecord` with `config_hash`, `feature_version`, `policy_version`, `model_version`, `source_tx_hashes[]`, `graph_path[]`, `entity_mapping_warnings[]`, and cryptographic `provenance_hash`.
- `advisory_notice`: Mandatory non-accusatory disclaimer.

### 3.2 Case & Audit Trail (`Case`, `AuditEvent`)
- Starts in `status="OPEN"`, `disposition="PENDING_ANALYST_REVIEW"`.
- Allowed transitions: `OPEN -> {IN_REVIEW, ESCALATED_FOR_REVIEW, CLOSED_BENIGN, CLOSED_DUPLICATE}`; `CLOSED_* -> {REOPENED}`.
- Reviewer override conditions (priority change away from `system_priority`, closing `P1_HIGH` as benign, or escalating `P3_LOW`) enforce:
  - Non-empty human `reviewer_id` (`!= SYSTEM_MONITOR`)
  - Valid `override_reason_code` in `{VERIFIED_BUSINESS_ACTIVITY, CORROBORATED_SOURCE_OF_FUNDS, KNOWN_PAYROLL_OR_TREASURY_SWEEP, MANUAL_INTELLIGENCE_ESCALATION, RELATED_CASE_CONSOLIDATION, FALSE_POSITIVE_RULE_ARTIFACT}`
  - `rationale` length `>= 10` characters.
- Each transition appends a hash-chained `AuditEvent` (`prev_hash` -> `event_hash`).

---

## 4. Explicit Error Taxonomy

| Exception Class | HTTP Status | Trigger Conditions |
|---|---:|---|
| `ConfigurationError` | N/A (startup) | Missing config file, malformed JSON, missing required config keys, invalid FX rate |
| `ValidationError` | 400 | Missing required fields, non-positive amount, self-transfer, invalid/naive timestamp, invalid status/priority enum |
| `UnsupportedInputError` | 400 | Unsupported currency (e.g., `BTC`, `XYZ`) or unsupported channel (e.g., `CRYPTO_MIXER`) |
| `CaseWorkflowError` | 400 | Unknown case ID, illegal status transition, `SYSTEM_MONITOR` acting as reviewer, missing/invalid reviewer override code or short rationale |

---

## 5. Fixture Coverage Summary

- **Valid Reference Oracle**: `data/fixtures/reference_hand_calc.json`
- **Valid Legitimate High-Volume**: `data/fixtures/valid_legitimate_high_volume.json`
- **Valid Suspicious Scenarios**: `data/fixtures/valid_suspicious_chains.json`
- **Invalid Schema & Unsupported Inputs**: `data/fixtures/invalid_inputs.json`
- **Boundary & Resilience Cases**: `data/fixtures/boundary_cases.json`
