# ADR-0001: Risk-Aware Payment Authorization Optimization Architecture

- **Status:** Accepted
- **Date:** 2026-11-28
- **Project:** Payment Authorization Optimization Engine (`projects/payment-authorization-optimization-engine/`)

## Context

Payment authorization involves joint optimization over multiple competing objectives: maximizing legitimate approval rate (net acceptance), containing expected fraud/chargeback losses, controlling end-to-end authorization latency, and enforcing strict payment-network retry safety rules. Naive ML-only systems risk:

1. Retrying hard-declined cards (violating card-network rules and incurring scheme penalties).
2. Blindly retrying timed-out or uncertain authorization attempts (causing duplicate captures and customer double-charges).
3. Leaking future fraud labels during training due to delayed chargeback maturity (typically 30–90 days).
4. Over-optimistic offline policy comparisons when evaluating counterfactual actions without checking positivity or effective sample size (ESS).

## Decision

We adopt a strictly layered, modular architecture with hard separation of concerns:

1. **Deterministic Safety & Taxonomy Gate First:** Every request passes contract validation (`schemas.py`), idempotency/deduplication checks (`IdempotencyLedger`), and the deterministic response taxonomy state machine (`taxonomy.py`) *before* ML inference or policy optimization executes. Hard declines (`14`, `54`, `57`, `59`, `63`) and uncertain outcomes (`TIMEOUT`, `NETWORK_ERROR`, `UNKNOWN`, `prior_outcome_known=False`) are hard-stopped and never retried blindly.
2. **Versioned Features & Strict Temporal Calibration:** Feature imputation statistics (`FeatureExtractor`) are fit exclusively on the chronological training fold (`FEATURE_SCHEMA_VERSION = "1.0.0"`). Missing optional features produce explicit missingness flags rather than silent defaults. Acceptance and fraud models (`PlattCalibratedClassifier`) are trained on the training fold and Platt-calibrated on a strictly later chronological calibration fold. Fraud models train exclusively on matured approved labels (`label_maturity_days >= min_label_maturity_days`).
3. **Dual-Constrained Risk-Aware Policy:** Rather than thresholding a raw risk score (`baseline_threshold`) or maximizing unconstrained expected value (`unconstrained_ev`), `ConstrainedRiskAwarePolicy` tunes Lagrangian dual multipliers $(\lambda_{\text{loss}}, \lambda_{\text{latency}})$ on the calibration fold to maximize expected net value and acceptance subject to explicit expected-loss-bps and P95-latency budgets, plus conservative uncertainty inflation when features are missing and processor-outage failover/fail-closed guards.
4. **Audited Off-Policy Counterfactual Evaluation:** Candidate policies are compared on logged synthetic data using Direct Method (DM), Inverse Propensity Scoring (IPS), Self-Normalized IPS (SNIPS), and Doubly Robust (DR) estimators with seeded bootstrap confidence intervals and explicit fail-fast checks for positivity violations, unsupported actions, and ESS collapse.

## Consequences

- **Positive:** Safety invariants cannot be overridden by ML score drift; zero hidden constraint breaches; reproducible runs with deterministic decision IDs and replay verification; exact numerical verification against hand-calculated reference oracles.
- **Trade-offs:** Logistic/Platt base models prioritize calibration stability, interpretability, and reproducibility over non-linear tree complexity; synthetic data enables ground-truth oracle validation but cannot substitute for randomized production experiments (A/B or contextual bandits) when claiming real-world causal uplift.
