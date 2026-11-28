# Risk ML

Nine independent risk, fraud, and payment machine learning projects. Eight offline Python prototypes use synthetic data to cover real-time transaction risk scoring, temporal graph fraud detection, payment authentication policy optimization, delayed chargeback prediction, AML transaction monitoring, account takeover detection, credit risk scoring, and payment authorization optimization. The collection also includes a Ruby on Rails application for real-time fraud scoring with XGBoost on AWS SageMaker.

## Getting started

Start with the AML Transaction Monitoring Engine. Its offline demo and test suite require Python 3.9 or later and use only the Python standard library to normalize synthetic payment flows, compute rolling and graph alerts, run temporal rule-versus-ML prioritization, and verify hash-chained case audit trails.

```bash
git clone https://github.com/vaibhavkapur/Risk-ML.git
cd Risk-ML/projects/aml-transaction-monitoring-engine
PYTHONPATH=src python3 -m unittest discover -s tests -p "test_*.py" -v
python3 scripts/run_evaluation.py
python3 scripts/run_demo.py
```

The evaluation and demo write versioned machine-readable outputs under [`projects/aml-transaction-monitoring-engine/reports/`](projects/aml-transaction-monitoring-engine/reports/). Each project runs independently under `projects/`. Run its setup, demo, and validation commands from that project's directory. For the Rails and AWS SageMaker project, see the [local development instructions](projects/Sagemaker-Fraud-Detector/README.md#local-development) and [SageMaker guide](projects/Sagemaker-Fraud-Detector/sagemaker/README.md).

## Projects

- [Real-Time Transaction Risk Platform](projects/real-time-transaction-risk-platform/) — Scores synthetic transaction streams with identical offline and online sliding-window features, Platt calibration, bounded-latency timeout fallbacks, training-serving skew monitoring, and hash-chained audit replay.
- [Graph-Based Fraud Detection Engine](projects/graph-based-fraud-detection-engine/) — Builds time-aware account, device, and merchant entity graphs with privacy-safe identifiers, leak-free graph features, causal GraphSAGE and boosted-tree models, graph-traced explanations, and analyst review workflows.
- [Smart Payment Authentication Engine](projects/smart-payment-authentication-engine/) — Selects simulated approve, challenge, decline, and abstain authentication actions using delayed-label calibrated models, constrained policy optimization, non-bypassable mandatory rules, and deterministic outage fallbacks.
- [Chargeback Prediction Engine](projects/chargeback-prediction-engine/) — Models delayed payment disputes with label maturity windows, causal point-in-time Polars features, calibrated expected dispute loss, budgeted manual review prioritization, and cohort maturation monitoring.
- [AML Transaction Monitoring Engine](projects/aml-transaction-monitoring-engine/) — Normalizes synthetic payment flows, computes rolling and directed-graph suspicious-pattern alerts, prioritizes cases with temporally validated calibrated ML, and audits analyst dispositions with cryptographic evidence provenance.
- [Account Takeover Detection Engine](projects/account-takeover-detection-engine/) — Detects anomalous synthetic login and session behavior using strictly causal rolling user and device features, chronological anomaly and supervised baselines, adaptive step-up policies, drift alerts, and fallback guardrails.
- [Credit Risk Scoring Platform](projects/credit-risk-scoring-platform/) — Estimates calibrated probability of default on synthetic loans with point-in-time leakage checks, Platt and isotonic monotonicity verification, stability and subgroup diagnostics, local reason codes, and a manual underwriting review workbench.
- [Payment Authorization Optimization Engine](projects/payment-authorization-optimization-engine/) — Evaluates constrained risk-aware authorization and retry policies under ISO-8583-style decline safety rules, processor failover guards, and audited doubly robust counterfactual estimators.
- [Sagemaker-Fraud-Detector](projects/Sagemaker-Fraud-Detector/) — Scores transactions in a Ruby on Rails application using XGBoost on AWS SageMaker with Redis-backed historical decline-rate caching.

Each Python project README links to its architecture, data contracts, integration boundaries, operator runbook, and validation evidence.
