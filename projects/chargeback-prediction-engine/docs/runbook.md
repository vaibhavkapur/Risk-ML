# Developer & Operator Runbook — Chargeback Prediction Engine

## 1. Local setup (clean checkout)

```bash
cd projects/chargeback-prediction-engine
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
```

> Note: On Python 3.10+ with pip >= 23, you may also run `.venv/bin/python -m pip install --no-deps -e .` to register the `chargeback-engine` console script. On Python 3.9 with pip 21, run the module directly via `PYTHONPATH=src .venv/bin/python -m chargeback_engine.cli ...` (all scripts in `scripts/` configure `sys.path` automatically).

## 2. Standard commands

### Verify hand-calculated reference fixture

```bash
PYTHONPATH=src .venv/bin/python -m chargeback_engine.cli verify-reference --fixtures-dir data/fixtures
```

### Run full validation test suite

```bash
.venv/bin/pytest
```

### Run reproducible end-to-end demo

```bash
.venv/bin/python scripts/run_demo.py --config config/default.yaml --output-dir reports
```

Outputs written:
- `reports/demo_summary.json` — full metrics, maturity diagnostics, model metadata, cold-start slice metrics, structured events.
- `reports/review_simulation.json` — review budget policy comparison, delayed-outcome monitoring trajectory, sample decision explanations.
- `reports/runtime_metadata.json` — exact Python/package versions, `config_hash`, seed, and full config snapshot.

### Run reproducibility & imbalance/cold-start benchmarks

```bash
.venv/bin/python scripts/run_benchmarks.py --config config/default.yaml --cold-start-config config/cold_start.yaml --output-dir reports
```

Outputs written:
- `reports/reproducibility_check.json` — verifies bit-for-bit identical logical outputs across two independent runs with the same seed and config.
- `reports/baseline_vs_calibrated.json` — baseline logistic vs uncalibrated GBDT vs calibrated GBDT metrics under both default and severe imbalance/cold-start scenarios.

## 3. Diagnostics and troubleshooting

| Symptom / Error | Root Cause | Recovery Action |
|---|---|---|
| `ConfigError: Configuration file not found` (exit code 2) | Missing or mistyped `--config` path | Verify path (e.g., `config/default.yaml`) and rerun. |
| `ConfigError: Punitive automated merchant decisions are prohibited` | `policy.allow_automated_merchant_punishment` set to `true` | Set to `false`; disputes are not equivalent to fraud. |
| `DuplicateDisputeError: Duplicate dispute records detected` | Multiple disputes reference the same `transaction_id` under `duplicate_dispute_policy: "raise"` | Inspect upstream dispute feed for duplicates, or explicitly set `maturity.duplicate_dispute_policy: "deduplicate_earliest"` and monitor `maturity_diagnostics.duplicate_disputes_deduplicated`. |
| `UnsupportedInputError: Unsupported currency ...` | Input contains a currency outside `{USD, EUR, GBP, CAD}` | Normalize upstream or add verified FX mapping to `contracts.SUPPORTED_CURRENCIES` with a schema version bump. |
| `TemporalLeakageError` | Overlapping split timestamps, overly wide `embargo_days`, or immature/null labels passed into split | Increase observation span, reduce `split.embargo_days`, or verify `assign_maturity_labels` ran before splitting. |
| `monitoring_status: "DRIFT_ALERT"` in delayed-outcome monitoring | Observed dispute arrival significantly diverges from empirical delay-adjusted expected disputes (`O/E < 0.70` or `> 1.35` with `expected >= 5`) | Audit recent merchant mix / reason-code shifts and schedule model recalibration once the cohort matures. |

## 4. Replay and rerun protocol

Every run emits a deterministic `run_id`, `config_hash`, `feature_schema_hash`, and `model_id` along with `reports/runtime_metadata.json`. To replay a previous run exactly:
1. Restore the recorded `config` snapshot from `runtime_metadata.json`.
2. Ensure the same `requirements.lock` environment.
3. Re-run `scripts/run_demo.py` and confirm `run_id` and `config_hash` match identically.
