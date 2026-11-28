# Integration Boundaries — Chargeback Prediction Engine

## 1. Repository context (`Risk-ML`)

This project lives under `projects/chargeback-prediction-engine/` inside the `Risk-ML` repository alongside sibling prototypes (`real-time-transaction-risk-platform`, `payment-authorization-optimization-engine`, `smart-payment-authentication-engine`, `account-takeover-detection-engine`, etc.).

**Zero cross-project code duplication:** At the time of implementation, sibling `Risk-ML` projects are independent directories without a shared published library package. This engine exposes a clean, self-contained typed Python library (`chargeback_engine`) with no runtime dependency on sibling paths, private repositories, or external services.

## 2. Reusable library adapter boundaries

Downstream `Risk-ML` components can integrate with `chargeback_engine` via explicit, stateless functional boundaries:

| Interface | Module | Inputs | Outputs | Potential `Risk-ML` Consumer |
|---|---|---|---|---|
| Contract validation | `chargeback_engine.contracts.validate_dataset_contracts` | Raw merchant/tx/dispute dicts | Typed validated record lists | Shared ingestion adapters |
| Label maturity engine | `chargeback_engine.maturity.assign_maturity_labels` | Validated records + `as_of_ts` + `MaturityConfig` | Polars DataFrame + `MaturityDiagnostics` | Delayed-label risk pipelines (credit, disputes, fraud) |
| Causal feature builder | `chargeback_engine.features.build_causal_features` | Labeled Polars DataFrame + `FeatureConfig` | Feature Polars DataFrame + `FeatureSchemaMetadata` | Offline feature store / training pipelines |
| Expected-loss scoring & policy | `chargeback_engine.policy.prioritize_reviews` | Scored DataFrame + calibrated probabilities + `EconomicsConfig` + `PolicyConfig` | Typed `ReviewDecision` list + policy summary | Manual review queue / case management simulator |
| Delayed-outcome monitor | `chargeback_engine.policy.simulate_delayed_outcome_monitoring` | Monitored cohort + delay CDF + checkpoints | Cohort maturation & O/E drift snapshots | Model risk monitoring dashboards |

## 3. Dependency licenses and provenance

All runtime and development dependencies are open-source packages pinned in [`requirements.lock`](../requirements.lock):

| Package | License | Purpose |
|---|---|---|
| `polars` | MIT | Typed tabular dataframe transformations |
| `scikit-learn` | BSD-3-Clause | Calibrated classification, scaling, metrics, permutation importance |
| `numpy` | BSD-3-Clause | Vectorized numerical & expected-loss calculations |
| `scipy` | BSD-3-Clause | Scientific math backend for `scikit-learn` |
| `PyYAML` | MIT | Configuration loading |
| `pytest` | MIT | Automated test suite (dev only) |

All data fixtures in `data/fixtures/` and generated datasets in `synthetic.py` are 100% synthetic and independently created for this prototype. No real customer records, card numbers, live merchant data, credentials, or proprietary artifacts are used.

## 4. Production exclusions

Any live or production integration (streaming Kafka/Flink ingestion, live payment gateway webhooks, card-network representment APIs, automated merchant underwriting/holds) is **explicitly disabled and out of scope**. Production adoption requires separate security, PCI/privacy, legal, and model-risk governance reviews.
