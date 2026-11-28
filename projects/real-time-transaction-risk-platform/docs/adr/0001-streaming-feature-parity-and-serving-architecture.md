# ADR 0001: Shared Pure Feature Engine and JSON Model Artifact for Offline/Online Parity

- **Status:** Accepted
- **Date:** 2026-11-28
- **Project:** Real-Time Transaction Risk Platform

## Context

Real-time fraud and transaction risk platforms frequently suffer from training-serving skew caused by:
1. Separate offline SQL/Pandas batch feature pipelines and online imperative serving code diverging on window boundaries (inclusive vs. exclusive edges, tie-breaking, FX rounding, cold-start defaults).
2. Binary model serialization (`pickle`/`joblib`) that couples serving to exact library internals, prevents straightforward checksum verification, and introduces deserialization latency and security risk.

## Decision

1. **Single Pure-Python Domain Feature Function (`compute_point_in_time_features`)**:
   - Both `OfflineFeatureExtractor` and `InMemoryFeatureStore` call the exact same pure function over strictly prior account events ordered by `(event_ts_ms, event_id)` with half-open `(t - W, t]` windows.
   - Monetary amounts are stored as integer minor units (`amount_cents`) and converted using versioned FX rates in `config/default.yaml`.
2. **Self-Contained JSON Model Artifact (`rtrp.model.v1`)**:
   - Training uses `scikit-learn` (`StandardScaler` + `LogisticRegression` + Platt calibration), then exports parameters, calibration constants, evaluation metrics, and reference feature histograms into a canonical JSON artifact protected by `artifact_sha256`.
   - Online inference runs pure math directly over `FeatureVector`, achieving sub-millisecond latency with zero scikit-learn runtime dependency in the hot path.
3. **Conservative Policy Fallbacks**:
   - Stale feature caches, unknown inputs (`unknown` MCC / unseen country), and model timeouts exceeding `inference_timeout_ms` deterministically escalate or fall back to `fallback.rules.v1`.

## Consequences

- **Positive:** Exact `0.0` / `< 1e-9` offline-vs-online parity on fixtures and streaming replays; deterministic audit replay; tamper-evident model loading; fast local execution without external broker daemons.
- **Negative / Limitations:** Linear calibrated baseline has lower representational capacity than gradient-boosted trees; single-process in-memory store does not horizontally shard across nodes (Redis/Kafka adapters remain optional integration boundaries outside offline prototype scope).
