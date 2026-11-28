# Operator & Developer Runbook

## 1. Clean Environment Setup

```bash
cd projects/graph-based-fraud-detection-engine
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 2. Generate Fixtures & Run Full Evaluation Pipeline

```bash
.venv/bin/python scripts/generate_fixtures.py
.venv/bin/python scripts/run_pipeline.py --config config/default.yaml --save-artifacts
```

Outputs written to:
- `data/fixtures/`: valid, invalid, boundary, and hand-calculated reference fixtures
- `reports/evaluation_report.json`: full raw & calibrated metrics, temporal split counts, leakage audit
- `reports/runtime_metadata.json`: Python/library versions, seed, GNN backend, PyG status
- `reports/reproducibility_check.json`: dual-run determinism verification
- `reports/sample_explanations.json`: example fraud-ring vs legitimate shared-device graph explanations & analyst disposition
- `artifacts/model_bundle_<run_id>.joblib`: serialized model & temporal graph state (when `--save-artifacts` is passed)

## 3. Run End-to-End Demo

```bash
.venv/bin/python scripts/run_demo.py
```

## 4. Run Local API Server (Synthetic/Offline Only)

```bash
.venv/bin/python scripts/serve_api.py --host 127.0.0.1 --port 8000
```

Verify health & diagnostics:
```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metrics
```

## 5. Run Automated Validation Suite

```bash
.venv/bin/pytest -v
```

## 6. Diagnostics & Failure Recovery

| Failure Mode | Symptom / Error | Recovery Action |
|---|---|---|
| Missing or invalid config | `ConfigurationError` at startup (no silent fallback) | Verify `--config` path and YAML schema against `config/default.yaml` (`train_end_day < val_end_day < test_end_day`, `review_threshold < block_threshold`) |
| PII or malformed ID in payload | HTTP `422` or `PrivacyViolationError` | Pseudonymize synthetic IDs via `pseudonymize_id()` using typed prefixes (`acc_`, `dev_`, `mer_`, `tx_`) |
| Naive timestamp | HTTP `422` / `ValidationError` | Append explicit UTC offset (`Z` or `+00:00`) to all timestamps |
| Future edge or immature label detected | `TemporalLeakageError` | Audit upstream event timestamps and `label_available_at >= tx_timestamp`; never bypass `enforce_label_maturity` |
| Uninitialized API service | HTTP `503` on `/v1/score` or `/metrics`, `/health` returns `degraded` | Start via `scripts/serve_api.py` or pass a trained `FraudDetectionService` to `create_app()` |
| Cold-start unseen nodes | `cold_start_flags` set to `true`, `COLD_START_UNSEEN_*` reason codes | Expected behavior; transactions use `cold_start_review_threshold` and avoid hard auto-blocks below `0.90` unless corroborated |
| Restart / rerun | Deterministic `run_id` and `decision_id` | Re-run `scripts/run_pipeline.py` with the same config and seed, or reload `artifacts/model_bundle_<run_id>.joblib` via `load_artifacts()` |
