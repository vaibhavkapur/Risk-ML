# ADR-0001: Observation Maturity Windows and Point-in-Time Causal Aggregates

- **Status:** Accepted
- **Date:** 2026-11-28
- **Project:** Chargeback Prediction Engine (`projects/chargeback-prediction-engine`)

## Context

Chargebacks and payment disputes arrive weeks or months after a transaction settles (typically 15–90+ calendar days). Treating every recently settled transaction without an observed dispute as a negative label (`label = 0`) introduces severe **right-censoring bias**: recent transactions appear artificially safe, depressing predicted probabilities and distorting expected-loss rankings.

Conversely, excluding only immature negatives while keeping early-arriving disputes from immature cohorts introduces **selection/prevalence bias** in the opposite direction: immature cohorts only contribute positives, inflating the apparent dispute rate and degrading probability calibration.

A second source of silent leakage occurs during feature engineering: joining final `as_of_ts` dispute flags onto historical merchant transactions allows a transaction at time $t$ to "see" disputes on prior transactions that were only reported at $t_{\text{dispute}} > t$.

## Decision

1. **Four-state label maturity taxonomy:** Every transaction observed at snapshot `as_of_ts` is assigned a `maturity_date` (explicit per-transaction override or `settlement_ts + default_maturity_days`) and classified into one of four states:
   - `mature_positive` (`as_of_ts >= maturity_date`, observed dispute `dispute_ts <= as_of_ts`): `label = 1`, trainable.
   - `mature_negative` (`as_of_ts >= maturity_date`, no observed dispute): `label = 0`, trainable.
   - `immature_positive` (`as_of_ts < maturity_date`, observed dispute `dispute_ts <= as_of_ts`): `label = 1`, preserved for delayed-outcome monitoring; excluded from training/calibration under default `cohort_mature` mode to keep base rates unbiased.
   - `immature_negative` (`as_of_ts < maturity_date`, no observed dispute): `label = None`, **always excluded from training and negative evaluation**.
2. **Strict event-time causality in merchant features:**
   - Transaction velocity/volume features at time $t$ aggregate only prior transactions of the same merchant with `transaction_ts < t`.
   - Rolling reported-dispute features at time $t$ count only prior disputes whose `dispute_ts < t`.
   - Mature merchant dispute rates at time $t$ count only prior transactions with `maturity_date <= t` (and whose dispute, if any, was reported at `dispute_ts <= t`), smoothed toward the training-set global prior via Empirical Bayes shrinkage to handle merchant cold start gracefully.
3. **Explicit missing reason code representation:** Missing or null dispute reason codes are mapped deterministically to `reason_code = "UNKNOWN"` and `reason_category = "unknown"` with explicit boolean diagnostics rather than being dropped or crashing feature generation.

## Consequences

- **Positive:** Calibrated probabilities align with true matured dispute prevalence; zero future-dispute leakage in historical merchant aggregates; cold-start merchants receive smooth, well-behaved priors rather than extreme 0%/100% rates.
- **Trade-off:** Default `cohort_mature` training waits for the full maturity window (e.g., 90 days) before incorporating the newest cohort into model weights. To mitigate concept-drift lag, the engine exposes delayed-outcome cohort monitoring (`policy.py`) that tracks early observed dispute curves against expected maturation curves without contaminating training labels.
