# Real-Time Transaction Risk Platform

An independently developed offline research prototype for **low-latency, versioned transaction risk scoring** with streaming time-window features, exact offline/online parity, temporal training with label maturity and Platt calibration, bounded-latency timeout fallbacks, training-serving skew monitoring, and a hash-chained replayable audit log.

> **Scope & Safety Notice:** Offline educational/research prototype using 100% synthetic tokenized data (`acct_syn_*`, `card_tok_*`, `merch_syn_*`). Raw 13–19 digit card numbers (PANs) are actively rejected at contract validation. Live capital, real customer records, and production credentials are excluded.

---

## Key Capabilities

- **Idempotent Event Ingestion & Sliding-Window Features (`src/risk_platform/store.py`, `features.py`)**: Deduplicates events by `event_id` + SHA-256 payload hash (rejecting conflicting duplicates), enforces watermark lateness windows (`ACCEPTED`, `LATE_ACCEPTED`, `LATE_DROPPED`), tracks feature freshness (`FRESH`, `COLD_START`, `STALE`), and supports atomic JSON checkpointing for restart recovery.
- **Guaranteed Offline/Online Feature Parity**: Both `OfflineFeatureExtractor` and `InMemoryFeatureStore` execute the exact same pure domain function `compute_point_in_time_features` (`feat.v1`, 12 features across 5m/1h/24h half-open windows).
- **Temporal Training & Calibration (`src/risk_platform/training.py`, `model.py`)**: Strict chronological train/calibration/test split with 24h warm-up exclusion and 24h label maturity filtering; exports a pure-math SHA-256 verified JSON model artifact (`rtrp.model.v1`) with Platt scaling and per-feature linear attributions.
- **Conservative Policy & Bounded Timeout Fallback (`src/risk_platform/policy.py`, `service.py`)**: Enforces `inference_timeout_ms` (default `25 ms`), falling back deterministically to `fallback.rules.v1` on model timeout or error, and conservatively escalating `STALE` caches or unknown MCC/country inputs.
- **FastAPI Scoring Service, Monitoring & Replayable Audit Log (`api.py`, `audit.py`, `monitoring.py`, `metrics.py`)**: Exposes `/health`, `/metrics`, `/v1/models/active`, `/v1/events/ingest`, `/v1/score`, and `/v1/audit/replay`, emitting traceability headers (`X-Decision-Id`, `X-Model-Version`, `X-Feature-Version`, `X-Policy-Version`) and hash-chained JSONL audit records.

---

## Project Layout

```text
projects/real-time-transaction-risk-platform/
├── config/default.yaml                     # Versioned schema, ingestion, serving, training & skew config
├── data/fixtures/                          # Hand-calculated oracle, invalid & boundary fixtures
├── docs/
│   ├── adr/0001-streaming-feature-parity-and-serving-architecture.md
│   ├── architecture.md                     # Component diagram, invariants & trade-offs
│   ├── contracts.md                        # Input/output schemas, units, versions & errors
│   ├── integration.md                      # Adapter boundaries, licenses & provenance
│   └── runbook.md                          # Setup, demo, API smoke tests & recovery procedures
├── reports/
│   ├── acceptance.md                       # Acceptance criteria mapped to measured evidence
│   └── *.json / audit_log.jsonl            # Raw machine-readable demo & benchmark outputs
├── scripts/
│   ├── run_demo.py                         # Reproducible end-to-end demo & benchmark runner
│   └── serve_api.py                        # Local FastAPI/Uvicorn server launcher
├── src/risk_platform/                      # Pure domain + service + API package
├── tests/                                  # Unit, parity, skew, API, load & recovery tests
├── plan.md                                 # Implementation plan
├── pyproject.toml
└── requirements.txt
```

---

## Quickstart

### 1. Setup

```bash
cd projects/real-time-transaction-risk-platform
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2. Run Automated Tests

```bash
.venv/bin/pytest -v
```

### 3. Run End-to-End Reproducible Demo & Benchmark

```bash
.venv/bin/python scripts/run_demo.py --benchmark-requests 2000
```

### 4. Start the Local Scoring API

```bash
.venv/bin/python scripts/serve_api.py --host 127.0.0.1 --port 8000
```

See [docs/runbook.md](docs/runbook.md) for `curl` examples, diagnostics, and recovery steps.

---

## Measured Results (Local Prototype Run)

All figures below are **actual local measurements** on Python `3.9.6` (`macOS-26.5.1-arm64`), `config/default.yaml`, `seed = 42`. Raw outputs are stored in `reports/`.

- **Automated Test Suite:** `13 passed in 1.40s` (`pytest -v`)
- **Offline/Online Feature Parity:**
  - Hand-calculated 4-event oracle fixture: **PASS** (`max_abs_diff = 0.0`)
  - 800-event held-out streaming slice: **PASS** (`max_abs_diff = 0.0`)
  - Checkpoint restart recovery: **PASS** (`max_abs_diff = 0.0`)
- **Local In-Process Scoring Latency & Throughput (`2,000` requests, active timeout executor):**
  - Throughput: **34,364.38 req/s**
  - Latency: **p50 = 0.0280 ms** | **p95 = 0.0363 ms** | **p99 = 0.0453 ms** | **max = 0.1262 ms**
- **Temporal Model Quality (Synthetic Held-Out Test Split, `n = 704`, 7.53% positive rate):**
  - ROC-AUC: **0.8876** | PR-AUC: **0.7452** | Calibrated Brier: **0.0299** | Calibrated ECE: **0.0187**
- **Training-Serving Skew Detection:**
  - In-distribution held-out slice: `OK` (`max_psi = 0.0270`)
  - Simulated drifted serving slice: `CRITICAL_SKEW` (`max_psi = 1.9038`, alert triggered)
- **Timeout Fallback & Audit Replay:**
  - Injected 60ms slow model vs 25ms budget -> `MODEL_TIMEOUT` fallback to `fallback.rules.v1` in `28.19 ms`
  - Audit hash chain: **valid** (`800` records); deterministic replay: **800/800 exact matches**

### Example Scoring Decision Output

```json
{
  "decision_id": "dec_8593002eddbc9194a9be",
  "event_id": "evt_syn_003205",
  "account_id": "acct_syn_0006",
  "action": "REVIEW",
  "risk_score": 0.568214,
  "raw_score": 0.669946,
  "model_version": "model.logreg_cal.v1",
  "feature_version": "feat.v1",
  "policy_version": "policy.v1",
  "feature_freshness_ms": 86080,
  "feature_status": "FRESH",
  "fallback_used": false,
  "fallback_reason": null,
  "reason_codes": [
    "HIGH_RISK_CONTEXT",
    "SCORE_ABOVE_REVIEW_THRESHOLD"
  ],
  "latency_ms": 0.0392,
  "decided_at": "2026-01-08T00:48:00.696Z"
}
```

---

## Documentation & Acceptance Evidence

- [reports/acceptance.md](reports/acceptance.md) — Criterion-by-criterion acceptance verification
- [docs/architecture.md](docs/architecture.md) — Component diagram, boundaries, invariants, and trade-offs
- [docs/contracts.md](docs/contracts.md) — Canonical schemas, units, window semantics, and error codes
- [docs/integration.md](docs/integration.md) — Adapter boundaries, pinned dependency licenses, and fixture provenance
- [docs/runbook.md](docs/runbook.md) — Setup, demo, smoke testing, diagnostics, and recovery
- [docs/adr/0001-streaming-feature-parity-and-serving-architecture.md](docs/adr/0001-streaming-feature-parity-and-serving-architecture.md) — First Architecture Decision Record
- [`../../.github/workflows/real-time-transaction-risk-platform.yml`](../../.github/workflows/real-time-transaction-risk-platform.yml) — Automated CI validation workflow

---

## Assumptions & Known Limitations

1. **Synthetic Data Only:** Model metrics (`ROC-AUC = 0.8876`) reflect a synthetic transaction-and-attack generator and do not represent production fraud detection accuracy on real payment traffic.
2. **Single-Process Feature Store:** The default `InMemoryFeatureStore` provides thread-safe sliding windows and atomic JSON checkpoints for local/CI reproducibility. Live Kafka/Redpanda and distributed Redis backends are explicitly **not run** in this offline prototype (`storage.kafka_enabled: false`, `storage.redis_enabled: false`).
3. **In-Process Benchmark:** The reported `34,364 req/s` / `0.045 ms p99` benchmark measures the in-process scoring path (feature lookup + timeout-bounded model inference + policy evaluation), excluding network/TLS/HTTP serialization and synchronous disk audit flushes.
4. **Baseline Linear Model:** `CalibratedLinearRiskModel` prioritizes deterministic explainability, sub-millisecond inference, and tamper-evident JSON serialization over higher-capacity tree ensembles.
