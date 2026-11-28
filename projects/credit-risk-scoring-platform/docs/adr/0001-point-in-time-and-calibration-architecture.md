# ADR-0001: Point-in-Time Feature Contracts, Temporal Validation, and Calibration Boundaries

- **Status:** Accepted
- **Date:** 2026-11-28
- **Project:** `credit-risk-scoring-platform` (Project 38 of 67)

## Context

Credit default modeling is unusually vulnerable to silent methodological failures:
1. **Post-origination leakage:** Including post-funding delinquency, payment history, collections, or charge-off signals inflates offline AUROC while failing completely at origination.
2. **Right-censored label bias:** Treating immature recent loans (whose 365-day observation window has not yet elapsed) as non-defaults artificially lowers default rates and biases probability of default (PD) calibration downward.
3. **Random-split optimism:** Randomly shuffling loans across time masks macroeconomic distribution shift and overstates calibration stability.
4. **Conflating PD estimation with lending decisions:** An educational/research prototype must not embed automated approval or denial logic.

## Decision

1. **Strict Point-in-Time Contracts (`contracts.py`):**
   - Separate `ApplicantRecord` (origination snapshot) from `LabelRecord` (outcome over an explicit `observation_window_days`).
   - Enforce `feature_as_of_timestamp <= application_timestamp` and block a denylist of post-origination columns (`FORBIDDEN_POST_ORIGINATION_FIELDS`) with `PointInTimeLeakageError`.
   - Enforce `as_of_cutoff_date >= label_maturity_date` for any `MATURED` label; automatically route immature labels to `CENSORED_IMMATURE` and exclude them from temporal splits when `require_label_maturity=true`.
2. **Three-Way Temporal Splitting (`data.py`):**
   - Strictly partition matured observations chronologically into `train` (`<= train_end_date`), `calibration` (`<= calibration_end_date`), and `test` (`<= test_end_date`).
   - Fit imputation medians, scaling statistics, and base models exclusively on `train`; fit post-hoc calibrators exclusively on `calibration`; report all headline discrimination, calibration, stability (PSI/CSI), and subgroup metrics on `test`.
3. **Protected Subgroup Isolation (`features.py` & `evaluation.py`):**
   - Retain `age_band` and `region_tier` strictly for post-hoc subgroup diagnostics and assert they never enter `FeaturePipeline.feature_names_`.
   - Enforce `min_subgroup_sample_size` with explicit error/withholding behavior (`InsufficientSubgroupSampleError` in strict mode; `INSUFFICIENT_SAMPLE` status + warning in reporting mode).
4. **Decision-Support Manual Review Only (`policy.py` & `review.py`):**
   - Reject `policy.automated_decision_allowed: true` with `UnsupportedInputError`.
   - Map calibrated PD monotonically into human underwriting review queues and SLAs with per-applicant log-odds explanations and an audit-logged manual review workbench.

## Consequences

- **Positive:** Prevents silent leakage, enforces monotone calibrated probabilities and risk bands, preserves exact replay reproducibility via deterministic SHA-256 decision IDs, and keeps boundaries cleanly testable.
- **Trade-offs:**
  - Holding out a dedicated temporal calibration window reduces training sample size compared to cross-fitted calibration, but mirrors real chronological deployment.
  - Defaulting to scikit-learn `HistGradientBoostingClassifier` and marginal baseline log-odds attribution removes heavy native dependency friction (`xgboost`, `shap` remain optional extras) while keeping deterministic behavior on Python 3.9+.
