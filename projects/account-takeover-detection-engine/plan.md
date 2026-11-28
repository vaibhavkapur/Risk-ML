# Account Takeover Detection Engine — Implementation Plan

**Project:** 37 of 67  
**Target repository:** `Risk-ML` (existing)  
**Suggested directory:** `projects/account-takeover-detection-engine/`  
**Status:** Proposed / not implemented by this plan  
**Suggested stack:** Python, scikit-learn, FastAPI, Redis optional  
**Scope:** Independently developed educational or research prototype; offline, mock, paper or testnet execution only as applicable.

## 1. Objective and scope

Detect anomalous login and session behavior with calibrated step-up recommendations.

**Primary inputs:** Synthetic login, device, network and session events.

**Core deliverables**

- Build session and rolling user/device features.
- Train anomaly and supervised baselines with chronological validation.
- Evaluate adaptive thresholds and step-up recommendations.
- Provide analyst evidence, drift alerts and fallback policy.

**Out of scope / limitations:** Avoid collecting real credentials or device fingerprints; synthetic simulation only. Live capital, real customer records and production credentials are excluded. Any later production use requires a separate security, operational, legal and risk review as appropriate.

## 2. Architecture and implementation decisions

- **Boundaries:** Separate feature generation, training, calibration, policy and inference. Use temporal validation, label maturity and versioned features; record model artifacts and decision explanations. Use synthetic or appropriately licensed data only.
- **Data contracts:** Define input schema, units, timestamp semantics where applicable, identifiers, validation errors and output contract in `docs/contracts.md`. Include valid, invalid and boundary fixtures.
- **Reproducibility:** Pin dependencies and configuration; version inputs and outputs; use fixed seeds for stochastic scenarios and record runtime environment. Establish a simple reference, hand-calculated fixture, or existing library oracle before adding complexity; record deviations and tolerances.
- **Integration:** Document any reusable interfaces from `Risk-ML` or neighboring repositories in `docs/integration.md`. Prefer independent runnable adapters over copied implementations; verify existing paths before importing. No dependency on private employer or SAAT code.
- **Delivery layout:** `src/`, `tests/`, `config/`, `data/fixtures/`, `scripts/`, `docs/`, `reports/`, dependency/build manifest. Include this `plan.md`, a project `README.md`, and documented setup, demo and validation commands. Adapt commands to the selected stack.

## 3. Implementation milestones

### M1 — Foundation and reference fixtures

- [ ] Create scaffold, dependency lock/build configuration, project README and local setup instructions.
- [ ] Define the canonical inputs (synthetic login, device, network and session events) and expected outputs; include schema, validation and error cases.
- [ ] Build session and rolling user/device features.
- [ ] Build a small deterministic fixture and an independent expected-result example; write architecture notes and the first ADR.

**Exit gate:** Clean setup and a minimal reference scenario run successfully; input and output contracts are documented and tested.

### M2 — Core domain implementation

- [ ] Train anomaly and supervised baselines with chronological validation.
- [ ] Isolate the domain algorithm/state transitions from I/O; expose a typed library or service interface.
- [ ] Implement deterministic unit tests for: New device vs compromised session.
- [ ] Add input validation, error propagation and explicit handling for unsupported cases; capture baseline outputs for comparison.

**Exit gate:** Core behavior passes reference fixtures, including edge cases, with no undocumented assumptions.

### M3 — Integration and evaluation

- [ ] Evaluate adaptive thresholds and step-up recommendations.
- [ ] Run integration scenarios covering: Shared network, missing data and delayed label.
- [ ] Add structured logs or diagnostic counters, replay or rerun support as appropriate, and stable output identifiers.
- [ ] Compare results against the reference baseline with a documented tolerance or evaluation protocol; retain raw outputs in `reports/`.

**Exit gate:** End-to-end demo runs reproducibly, integration fixtures pass, and discrepancies are explained.

### M4 — Hardening, benchmarking and delivery

- [ ] Provide analyst evidence, drift alerts and fallback policy.
- [ ] Add failure or boundary scenarios for: Repeated attempts and feature expiry.
- [ ] Run the acceptance checks below, record measured results and configuration, and mark untested capabilities explicitly.
- [ ] Add CI validation, operator/developer runbook, limitation notes, architecture diagram and a short demo walkthrough.

**Exit gate:** All applicable acceptance checks are evidenced; unsupported or untested features are identified; another developer can reproduce the demo from a clean checkout.

## 4. Test and validation matrix

| Test area | Required scenario | Expected evidence |
|---|---|---|
| Domain correctness | New device vs compromised session | Reference comparison and automated assertions |
| Integration and resilience | Shared network, missing data and delayed label | Integration/scenario test report |
| Boundary and failure behavior | Repeated attempts and feature expiry | Explicit error handling and regression test |
| Reproducibility | Same versioned inputs/configuration and fixed seed where applicable | Matching logical outputs or documented stochastic tolerance |
| Operational review | Missing configuration, unsupported inputs, restart/rerun where relevant | Clear diagnostics, documented recovery steps and no silent success |

## 5. Project-specific acceptance criteria

- [ ] Report recall at fixed false-positive rate and calibration.
- [ ] No future events in features.
- [ ] Every decision logged.
- [ ] All required test scenarios pass in the supported environment, or any unavailable hardware/data dependency is explicitly marked not run.
- [ ] README documents setup, demo, validation, assumptions, and known limitations; measured results distinguish actual observations from synthetic fixtures, estimates and targets.
- [ ] No real transactions, customer data, credentials or proprietary employer/SAAT artifacts in source, fixtures, logs or reports.

## 6. Required final evidence

| Artifact | Purpose |
|---|---|
| `README.md` | Overview, quickstart, example output and limitations |
| `docs/architecture.md` | Component diagram, data flow, invariants and trade-offs |
| `docs/contracts.md` | Input/output schemas, units, versions and errors |
| `docs/integration.md` | Adapter boundaries, dependencies and compatibility |
| `docs/runbook.md` | Demo, validation, diagnostics and recovery instructions |
| `tests/` | Unit, integration and required failure scenarios |
| `reports/acceptance.md` | Each acceptance criterion mapped to test or benchmark evidence |
| `reports/` raw results | Versioned machine-readable outputs and runtime/configuration metadata |

## 7. Completion checklist

- [ ] M1–M4 completed and reviewed.
- [ ] Project-specific acceptance evidence linked from README.
- [ ] Dependency licenses and fixture provenance documented; original code and no confidential inputs.
- [ ] Reproducible example and CI workflow verified on the stated environment.
- [ ] Any production-only integrations remain disabled or explicitly outside scope.

> Numerical thresholds and technology choices in this plan are proposed engineering targets, not measured results or guarantees. Adjust them only with documented rationale and retain baseline evidence.
