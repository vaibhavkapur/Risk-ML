# ADR-0001: Architectural Boundaries, Causal Features, Calibration, and Fallback Policy

- **Status:** Accepted
- **Date:** 2026-11-28
- **Project:** Account Takeover Detection Engine (`projects/account-takeover-detection-engine`)

## Context

Account takeover (ATO) models frequently fail in practice due to four recurring engineering hazards:
1. **Temporal leakage** — computing rolling user/device aggregates using future or same-timestamp events, or training on fraud labels that were not yet confirmed at the training cutoff.
2. **Uncalibrated scores coupled directly to static thresholds** — causing step-up rates to swing wildly when class priors or attack patterns shift.
3. **Shared-network false positives** — penalizing legitimate users behind corporate or campus NAT IPs as credential stuffing.
4. **Silent degradation** — failing open (returning `ALLOW`) when telemetry fields are missing, the feature store is unavailable, or the model artifact is stale/drifting.

## Decision

1. **Strict module boundaries**: Isolate pure domain logic (`features.py`, `models.py`, `policy.py`, `drift.py`) from state storage (`store.py`), audit persistence (`audit.py`), and HTTP transport (`api.py`).
2. **Causal feature invariant**: Enforce `history_ts < event_ts` both in `StateStore` queries and via explicit assertion inside `compute_features()`. Track profile expiry past the 30-day TTL (`profile_expired`) explicitly.
3. **Chronological split with label maturity**: Partition synthetic streams strictly by timestamp into 60% train / 20% calibration / 20% test, dropping training rows whose `label_available_at > train_cutoff`.
4. **Hybrid scoring + post-hoc calibration**: Establish a hand-calculated `rule_oracle_score` baseline first, then train an `IsolationForest` anomaly baseline alongside `LogisticRegression` and `HistGradientBoostingClassifier`, calibrated via Platt scaling on the calibration split.
5. **Adaptive thresholds + explicit fallback policy**: Adjust thresholds for account tier, verified shared networks, and active drift. Automatically engage a deterministic rule-based fallback (`FallbackReason`) with conservative step-up guardrails on model unavailability/staleness, store outages, excessive missing features, or critical drift circuit-breaker activation.
6. **Mandatory structured audit logging**: Assign every decision a deterministic `decision_id` and record it to an append-only JSONL audit trail.

## Consequences

- **Positive**: Fully deterministic, reproducible offline evaluation and replay; zero label/feature leakage; auditable analyst evidence for every decision; safe degraded-mode behavior.
- **Negative**: Excluding immature labels reduces effective training set size near `train_cutoff`; conservative fallback increases OTP challenges during telemetry outages.
- **Non-goals**: Live Redis clustering, real customer PII/credential collection, and production deployment security hardening remain out of scope for this research prototype.
