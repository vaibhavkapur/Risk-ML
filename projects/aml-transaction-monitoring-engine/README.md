# AML Transaction Monitoring Engine

**Project 36 of 67 (`Risk-ML`)** — Educational / research prototype for generating **reproducible, explainable synthetic transaction-monitoring alerts** with rolling temporal features, directed transaction-graph motifs, temporally validated ML alert prioritization, and an auditable case-review & provenance workflow.

> **Important Scope & Non-Accusation Notice:** This project is an independently developed educational prototype using 100% synthetic data. It is **not** a legal or regulatory compliance certification, and it produces **advisory investigative indicators only** — never automated accusations, legal determinations, or SAR/STR filings. Human analyst judgment is explicitly preserved and audited.

---

## Core Capabilities

1. **Transaction Normalization & Rolling Aggregations** (`src/aml_engine/normalization.py`, `src/aml_engine/features/rolling.py`)
   - Deterministic schema validation, multi-currency FX normalization to USD, UTC timestamp enforcement, canonical SHA-256 event hashing, duplicate-event deduplication, and explicit diagnostics for missing entity mappings.
   - Strict point-in-time rolling windows (`1h`, `24h`, `7d`, `30d`) with zero future leakage: structuring-band counters, unique counterparties, pass-through ratios, baseline velocity ratios, and dormancy gaps.
2. **Configurable Suspicious-Pattern Detection & Graph Features** (`src/aml_engine/features/graph.py`, `src/aml_engine/detection/scenarios.py`)
   - Directed temporal multi-graph engine (pure Python with optional `NetworkX` adapter via `to_networkx()`).
   - Five configurable synthetic scenarios with explicit exemptions for legitimate high-volume profiles (merchants, payroll, treasury):
     - `STRUCTURING_SMURFING`
     - `RAPID_LAYERING_CHAIN`
     - `ROUND_TRIP_CYCLE`
     - `MULE_FAN_IN_FAN_OUT`
     - `DORMANT_ACCOUNT_SPIKE`
3. **Calibrated ML Prioritization & Workload Management** (`src/aml_engine/ml/prioritizer.py`, `src/aml_engine/evaluation.py`)
   - Strict separation of feature generation (`v1.0.0`), temporal train/calibration/test splitting with label maturity buffers (`7` days), L2-regularized Logistic Regression training, Platt probability calibration, policy tiering (`P1_HIGH`, `P2_MEDIUM`, `P3_LOW`), and exact per-feature log-odds explainability.
4. **Case Review, Cryptographic Evidence Provenance & Reviewer Override Audit** (`src/aml_engine/case_management/workflow.py`, `src/aml_engine/storage/repository.py`)
   - Deterministic content-derived IDs (`ALT-*`, `EVD-*`, `CASE-*`) for idempotent replay and restart recovery.
   - SHA-256 evidence provenance verification (`verify_evidence_provenance`) and append-only hash-chained audit trail (`verify_audit_chain`) enforcing mandatory `reviewer_id`, `override_reason_code`, and substantive `rationale` on reviewer overrides.
5. **Typed Service, HTTP API & CLI** (`src/aml_engine/api/`, `src/aml_engine/cli.py`)
   - Zero-dependency local HTTP/JSON server (`create_http_server`), optional FastAPI ASGI app (`create_fastapi_app`), SQLite persistence default, and optional PostgreSQL adapter (`PostgresCaseRepository` + `schema_postgres.sql`).

---

## Quickstart & Setup

### Prerequisites
- Python **3.9+** (zero mandatory third-party packages required for core execution, tests, CLI, and evaluation).
- Optional extras (FastAPI, NetworkX, PostgreSQL `psycopg`, pytest) are pinned in [`requirements-optional.txt`](requirements-optional.txt).

```bash
cd projects/aml-transaction-monitoring-engine

# 1. Run the full unit, integration, and boundary test suite (15 tests)
PYTHONPATH=src python3 -m unittest discover -s tests -p "test_*.py" -v

# 2. Run temporal rule-vs-ML evaluation and save versioned reports + model artifact
python3 scripts/run_evaluation.py

# 3. Run the end-to-end reproducible demo (ingestion, replay idempotency, override audit)
python3 scripts/run_demo.py

# 4. Monitor a specific fixture via CLI
PYTHONPATH=src python3 -m aml_engine.cli monitor --input data/fixtures/reference_hand_calc.json --strict

# 5. (Optional) Start local HTTP API server
PYTHONPATH=src python3 -m aml_engine.cli serve --host 127.0.0.1 --port 8000
```

---

## Example Measured Output (Synthetic Dataset, Seed 42)

All numbers below are **actual measured observations** recorded on the deterministic 90-day synthetic dataset (`synthetic_aml_seed42_d90`, 2,280 normalized transactions, Python 3.9.6, `config_hash=a738d18172e4413b`). See raw machine-readable outputs in [`reports/baseline_vs_ml_evaluation.json`](reports/baseline_vs_ml_evaluation.json) and [`reports/demo_run_output.json`](reports/demo_run_output.json).

| Strategy (Out-of-Time Test Window, n=12 candidates, 9 true synthetic positives) | Precision | Recall | F1 | PR-AUC | Review Volume | Workload Reduction vs Rules |
|---|---:|---:|---:|---:|---:|---:|
| Rules-Only Baseline (`threshold=0.50`) | 0.7500 | 1.0000 | 0.8571 | 0.8623 | 12 | 0.0% (baseline) |
| Calibrated ML Only (`threshold=0.50`, ECE=0.0151, Brier=0.0012) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 9 | 25.0% |
| Hybrid Rule + ML (`P1_HIGH` Queue, `threshold=0.75`) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 9 | **25.0%** |

> **Distinguishing Synthetic Observations from Production Targets:** The 100% precision/recall on the out-of-time test slice reflects clean separation in our synthetic generator (49 true synthetic suspicious episodes vs 13 borderline benign decoys). Real-world AML distributions exhibit much heavier class imbalance, concept drift, and noisy labels; these figures are synthetic verification results, not production guarantees.

---

## Documentation & Evidence Index

| Artifact | Purpose |
|---|---|
| [`plan.md`](plan.md) | Original M1–M4 implementation specification and acceptance criteria |
| [`docs/architecture.md`](docs/architecture.md) | Component diagram, pipeline data flow, invariants, trade-offs, and ADR-001 |
| [`docs/contracts.md`](docs/contracts.md) | Canonical input/output schemas, units, UTC semantics, error classes, fixtures |
| [`docs/integration.md`](docs/integration.md) | Adapter boundaries with `Risk-ML`, optional stack adapters, dependency/license inventory |
| [`docs/runbook.md`](docs/runbook.md) | Demo, validation, diagnostics, replay/recovery procedures, troubleshooting |
| [`reports/acceptance.md`](reports/acceptance.md) | Acceptance criteria & test matrix mapped to measured evidence and explicit not-run items |
| [`reports/baseline_vs_ml_evaluation.json`](reports/baseline_vs_ml_evaluation.json) | Raw versioned evaluation report and runtime metadata |
| [`reports/demo_run_output.json`](reports/demo_run_output.json) | Raw end-to-end demo output, replay proof, sample explainable alerts, audit chain |
| [`data/fixtures/PROVENANCE.md`](data/fixtures/PROVENANCE.md) | Synthetic fixture inventory and non-confidentiality provenance statement |
| [`../../.github/workflows/aml-transaction-monitoring-engine.yml`](../../.github/workflows/aml-transaction-monitoring-engine.yml) | Automated CI validation workflow |

---

## Assumptions & Known Limitations

- **Synthetic Data Only:** All entities, accounts, FX rates, and scenario labels are synthetic; no real transactions, customer PII, or production credentials exist in this repository.
- **No Automated Filing / No Accusation:** Outputs are advisory triage records only (`CaseDisposition.PENDING_ANALYST_REVIEW` by default).
- **Optional External Infrastructure:** Live PostgreSQL (`PostgresCaseRepository`), FastAPI/Uvicorn (`create_fastapi_app`), and `networkx` (`TransactionGraph.to_networkx`) adapters are implemented and documented, but were **not executed** in this offline stdlib-only validation environment (explicitly marked `Not Run` in [`reports/acceptance.md`](reports/acceptance.md)).
- **Batch/Micro-Batch Graph Scope:** Multi-hop layering and cycle detection operate over the ingested temporal batch window rather than an unbounded distributed streaming graph database.
