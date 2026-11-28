# ADR-0001: Separated Calibrated Risk Policy and Deterministic Fallback

- **Status:** Accepted
- **Date:** 2026-11-28
- **Project:** Smart Payment Authentication Engine (Project 34 of 67)

## Context

Payment authentication systems must balance three competing objectives—fraud loss, customer step-up friction, and genuine acceptance—while strictly respecting non-negotiable business/regulatory rules (mandatory step-up challenges and high-risk overrides) and remaining safe during partial telemetry loss or model outages.

Coupling rule overrides inside ML training targets or letting threshold optimization override mandatory rules creates two critical failure modes:
1. Policy optimizers can silently bypass mandatory challenge rules to satisfy a tight `max_challenge_rate` budget.
2. Missing device telemetry or model timeouts can either crash authorization flows or silently approve high-risk payments using imputed neutral values.

## Decision

We enforce a strict separation between **calibrated risk estimation**, **mandatory/override rules**, **constrained threshold policy**, and **deterministic fallback**, with the following fixed decision precedence:

1. **High-risk overrides (`DECLINE`)**: Always evaluated first; cannot be downgraded.
2. **Feature/model health gate -> Deterministic fallback**: If the risk model is unavailable/corrupted, any critical feature is missing, or `missing_feature_ratio` exceeds the configured threshold, route to pure deterministic fallback rules (`evaluate_deterministic_fallback`), which preserve mandatory challenge rules and return `abstain` when telemetry is severely absent (`missing_feature_ratio >= 0.65`) and no mandatory rule triggers.
3. **Mandatory challenge guard (`CHALLENGE`)**: Evaluated independently of ML thresholds; can be escalated to `DECLINE` by high risk, but can **never** be downgraded to `APPROVE` or `ABSTAIN`.
4. **Calibrated risk thresholds**: Applied only when model and features are healthy, using post-hoc calibrated probabilities trained strictly on maturity-filtered delayed labels (`label_available_at <= as_of`).

## Consequences

- **Positive:** Mandatory challenge non-bypass and fallback determinism are structurally guaranteed and directly unit-testable in isolation from I/O.
- **Positive:** Threshold optimization can be re-run whenever business constraints (`max_challenge_rate`, `max_fraud_loss_rate`) change without retraining the underlying risk model.
- **Trade-off:** When mandatory challenge rules cover a large fraction of volume (e.g., high-amount or high-risk categories), the achievable minimum `challenge_rate` is bounded below by the mandatory rule floor; the optimizer reports `constraints_satisfied=False` explicitly if constraints are stricter than the mandatory floor rather than silently violating rules.
