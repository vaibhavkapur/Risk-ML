# Model Card — Credit Risk Scoring Platform (Educational Prototype)

> **Important Notice:** This model card documents an educational/research prototype trained exclusively on deterministic synthetic loan data. **Automated lending decisions are strictly disabled (`automated_decision_allowed = false`).** Outputs are decision-support estimates for manual review prioritization only and are NOT proof of legal fairness, regulatory compliance, or creditworthiness in production.

## 1. Model Details

- **Project / Version:** `credit-risk-scoring-platform` v`0.1.0`
- **Schema / Feature / Policy Versions:** `schema=1.0.0`, `features=1.0.0`, `policy=1.0.0`
- **Primary Model:** `boosted_trees` (backend: `sklearn_hist_gb`)
- **Reference Baseline:** `logistic_regression` (`sklearn.linear_model.LogisticRegression`)
- **Selected Calibration Method:** `sigmoid` (compared against `sigmoid, isotonic`)
- **Random Seed / Config SHA-256:** `42` / `6eca6ff09edfee4f...`
- **Python / scikit-learn:** `3.9.6` / `1.5.2`
- **Optional Backends:** `xgboost=not_installed`, `shap=not_installed`

## 2. Intended Use and Out-of-Scope Uses

- **Intended Use:** Educational demonstration of point-in-time feature contracts, temporal validation with label maturity windows, post-hoc PD calibration, stability monitoring (PSI/CSI), subgroup diagnostics with minimum-sample guards, local explanations, and human manual-review triage.
- **Out-of-Scope Uses:** Automated credit approval/denial, adverse-action notice generation without legal/compliance review, pricing/interest-rate setting, or scoring real customer records.

## 3. Data Provenance and Temporal Splits

- **Data Source:** Deterministic synthetic generator (`synthetic_generator_v1`) with controlled missingness and post-`2024-01-01` temporal distribution shift.
- **Real Customer Records:** None (`contains_real_customer_data = false`).
- **Observation Window:** `365` days; label cutoff `2025-06-15`.
- **Train Split (`<= 2023-06-30`):** `n=2730`, default rate=`0.1718`
- **Calibration Split (`<= 2023-12-31`):** `n=890`, default rate=`0.1551`
- **Test Split (`<= 2024-06-30`):** `n=797`, default rate=`0.2020`
- **Right-Censored Immature Excluded:** `n=83`
- **Protected Subgroup Attributes:** `age_band` and `region_tier` are strictly excluded from model feature matrices and retained solely for post-hoc subgroup diagnostics.

## 4. Measured Performance (Held-Out Temporal Test Split)

| Model Variant | AUROC | PR-AUC | KS | Brier Score | ECE | Mean Pred PD | Observed Default Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression (Uncalibrated) | 0.7640 | 0.4602 | 0.4170 | 0.1363 | 0.0273 | 0.1945 | 0.2020 |
| Logistic Regression (Calibrated: sigmoid) | 0.7640 | 0.4602 | 0.4170 | 0.1364 | 0.0367 | 0.1868 | 0.2020 |
| Boosted Trees (Uncalibrated) | 0.7433 | 0.4625 | 0.3713 | 0.1383 | 0.0207 | 0.1863 | 0.2020 |
| Boosted Trees (Calibrated: sigmoid) | 0.7433 | 0.4625 | 0.3713 | 0.1388 | 0.0299 | 0.1745 | 0.2020 |

- **Baseline Comparison (`boosted_trees` vs `logistic_regression`):** AUROC delta=`-0.0207`, Brier delta=`+0.0024`, Candidate ECE=`0.0299`, All tolerance checks passed=`False`.
- **Calibration Plots:** `reports/calibration_plot.png` and `reports/calibration_plot.svg`.

## 5. Stability and Distribution Shift Diagnostics (Train vs Temporal Test)

- **Calibrated Score PSI:** `0.020567` (status: `STABLE`; warning >= `0.1`, alert >= `0.25`)
- **Max Feature CSI:** `0.084519`
- **Features with Warning CSI:** `None`
- **Features with Alert CSI:** `None`

## 6. Subgroup Diagnostics (Test Split, Monitoring Only)

Minimum subgroup sample threshold: `n >= 30` (groups below this threshold have metrics withheld and raise explicit warnings/errors).

### Subgroup Dimension: `age_band`

| Group | Status | N | Observed DR | Mean Pred PD | Cal Gap | Elevated Review Rate | TPR | FPR | AUROC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `18_29` | `EVALUATED` | 178 | 0.2022 | 0.1795 | -0.0228 | 0.4607 | 0.7500 | 0.3873 | 0.7745 |
| `30_44` | `EVALUATED` | 290 | 0.2172 | 0.1713 | -0.0460 | 0.4103 | 0.6190 | 0.3524 | 0.7292 |
| `45_59` | `EVALUATED` | 211 | 0.1848 | 0.1712 | -0.0136 | 0.4313 | 0.7436 | 0.3605 | 0.7566 |
| `60_PLUS` | `EVALUATED` | 118 | 0.1949 | 0.1811 | -0.0139 | 0.4492 | 0.6957 | 0.3895 | 0.7190 |

- Disparity summary: max elevated-review-rate gap=`0.050329`, max TPR gap=`0.130952`, max |calibration gap|=`0.045957`.

### Subgroup Dimension: `region_tier`

| Group | Status | N | Observed DR | Mean Pred PD | Cal Gap | Elevated Review Rate | TPR | FPR | AUROC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `METRO` | `EVALUATED` | 372 | 0.2043 | 0.1787 | -0.0256 | 0.4651 | 0.7368 | 0.3953 | 0.7387 |
| `RURAL` | `EVALUATED` | 133 | 0.2105 | 0.1815 | -0.0290 | 0.3910 | 0.6786 | 0.3143 | 0.8160 |
| `SUBURBAN` | `EVALUATED` | 292 | 0.1952 | 0.1661 | -0.0291 | 0.4110 | 0.6316 | 0.3574 | 0.7106 |

- Disparity summary: max elevated-review-rate gap=`0.074076`, max TPR gap=`0.105263`, max |calibration gap|=`0.029112`.

## 7. Limitations, Uncertainty, and Governance Notes

- **Synthetic Data Only:** Performance numbers reflect synthetic feature-to-default relationships and do not generalize to real borrowers or macroeconomic regimes.
- **Calibration Under Shift:** Post-hoc calibration is fitted on the pre-shift calibration window; under severe macroeconomic shift (`PSI >= 0.25`), recalibration and underwriting review are required.
- **Missingness Uncertainty:** Median imputation with missingness indicators stabilizes inference, but applicants with multiple missing bureau fields carry wider epistemic uncertainty and trigger mandatory manual verification items.
- **Subgroup Uncertainty:** Small subgroups (`n < min_subgroup_sample_size`) produce wide confidence intervals; metrics are deliberately withheld rather than silently reported.
- **No Automated Lending:** Risk bands only govern human manual-review queue priority and SLA.
