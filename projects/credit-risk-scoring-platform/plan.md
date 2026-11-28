# Credit Risk Scoring Platform — Implementation Plan

**Project:** 38 of 67  
**Target repository:** `Risk-ML` (existing)  
**Suggested directory:** `projects/credit-risk-scoring-platform/`  
**Status:** Implemented (verified locally on Python 3.9.6; optional `xgboost`/`shap` extras documented as not run)  
**Suggested stack:** Python, scikit-learn, XGBoost optional, SHAP optional  
**Scope:** Independently developed educational or research prototype; offline, mock, paper or testnet execution only as applicable.

## 1. Objective and scope

Estimate probability of default with transparent validation, calibration and fairness diagnostics.

**Primary inputs:** Public appropriately licensed or synthetic loan features, default labels, observation windows.

**Core deliverables**

- Define point-in-time applicant feature and label contracts.
- Train logistic regression and boosted baselines with temporal splits.
- Calibrate PD and evaluate discrimination, stability and subgroup metrics where appropriate.
- Build model card, decision explanations and manual review interface.

**Out of scope / limitations:** Educational risk model only; not a deployed credit decision or proof of fairness/compliance. Live capital, real customer records and production credentials are excluded. Any later production use requires a separate security, operational, legal and risk review as appropriate.

## 2. Architecture and implementation decisions

- **Boundaries:** Separate feature generation, training, calibration, policy and inference. Use temporal validation, label maturity and versioned features; record model artifacts and decision explanations. Use synthetic or appropriately licensed data only.
- **Data contracts:** Define input schema, units, timestamp semantics where applicable, identifiers, validation errors and output contract in `docs/contracts.md`. Include valid, invalid and boundary fixtures.
- **Reproducibility:** Pin dependencies and configuration; version inputs and outputs; use fixed seeds for stochastic scenarios and record runtime environment. Establish a simple reference, hand-calculated fixture, or existing library oracle before adding complexity; record deviations and tolerances.
- **Integration:** Document any reusable interfaces from `Risk-ML` or neighboring repositories in `docs/integration.md`. Prefer independent runnable adapters over copied implementations; verify existing paths before importing. No dependency on private employer or SAAT code.
- **Delivery layout:** `src/`, `tests/`, `config/`, `data/fixtures/`, `scripts/`, `docs/`, `reports/`, dependency/build manifest. Include this `plan.md`, a project `README.md`, and documented setup, demo and validation commands. Adapt commands to the selected stack.

## 3. Implementation milestones

### M1 — Foundation and reference fixtures

- [x] Create scaffold, dependency lock/build configuration, project README and local setup instructions.
- [x] Define the canonical inputs (public appropriately licensed or synthetic loan features, default labels, observation windows) and expected outputs; include schema, validation and error cases.
- [x] Define point-in-time applicant feature and label contracts.
- [x] Build a small deterministic fixture and an independent expected-result example; write architecture notes and the first ADR.

**Exit gate:** Clean setup and a minimal reference scenario run successfully; input and output contracts are documented and tested.

### M2 — Core domain implementation

- [x] Train logistic regression and boosted baselines with temporal splits.
- [x] Isolate the domain algorithm/state transitions from I/O; expose a typed library or service interface.
- [x] Implement deterministic unit tests for: Missing values, distribution shift and leakage from post-origination data.
- [x] Add input validation, error propagation and explicit handling for unsupported cases; capture baseline outputs for comparison.

**Exit gate:** Core behavior passes reference fixtures, including edge cases, with no undocumented assumptions.

### M3 — Integration and evaluation

- [x] Calibrate PD and evaluate discrimination, stability and subgroup metrics where appropriate.
- [x] Run integration scenarios covering: Calibration and threshold monotonicity.
- [x] Add structured logs or diagnostic counters, replay or rerun support as appropriate, and stable output identifiers.
- [x] Compare results against the reference baseline with a documented tolerance or evaluation protocol; retain raw outputs in `reports/`.

**Exit gate:** End-to-end demo runs reproducibly, integration fixtures pass, and discrepancies are explained.

### M4 — Hardening, benchmarking and delivery

- [x] Build model card, decision explanations and manual review interface.
- [x] Add failure or boundary scenarios for: Insufficient subgroup sample.
- [x] Run the acceptance checks below, record measured results and configuration, and mark untested capabilities explicitly.
- [x] Add CI validation, operator/developer runbook, limitation notes, architecture diagram and a short demo walkthrough.

**Exit gate:** All applicable acceptance checks are evidenced; unsupported or untested features are identified; another developer can reproduce the demo from a clean checkout.

## 4. Test and validation matrix

| Test area | Required scenario | Expected evidence |
|---|---|---|
| Domain correctness | Missing values, distribution shift and leakage from post-origination data | Reference comparison and automated assertions |
| Integration and resilience | Calibration and threshold monotonicity | Integration/scenario test report |
| Boundary and failure behavior | Insufficient subgroup sample | Explicit error handling and regression test |
| Reproducibility | Same versioned inputs/configuration and fixed seed where applicable | Matching logical outputs or documented stochastic tolerance |
| Operational review | Missing configuration, unsupported inputs, restart/rerun where relevant | Clear diagnostics, documented recovery steps and no silent success |

## 5. Project-specific acceptance criteria

- [x] Report AUROC, PR-AUC, Brier score and calibration plots.
- [x] Document limitations and uncertainty.
- [x] No automated lending decision.
- [x] All required test scenarios pass in the supported environment, or any unavailable hardware/data dependency is explicitly marked not run.
- [x] README documents setup, demo, validation, assumptions, and known limitations; measured results distinguish actual observations from synthetic fixtures, estimates and targets.
- [x] No real transactions, customer data, credentials or proprietary employer/SAAT artifacts in source, fixtures, logs or reports.

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

- [x] M1–M4 completed and reviewed.
- [x] Project-specific acceptance evidence linked from README.
- [x] Dependency licenses and fixture provenance documented; original code and no confidential inputs.
- [x] Reproducible example and CI workflow verified on the stated environment.
- [x] Any production-only integrations remain disabled or explicitly outside scope.

> Numerical thresholds and technology choices in this plan are proposed engineering targets, not measured results or guarantees. Adjust them only with documented rationale and retain baseline evidence.
