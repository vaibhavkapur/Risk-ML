# Operator & Developer Runbook

## 1. Local Setup (Clean Checkout)

```bash
cd projects/real-time-transaction-risk-platform
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 2. Running Automated Validation

Run the full unit, integration, parity, skew, load, and restart-recovery test suite:

```bash
.venv/bin/pytest -v
```

## 3. Running the End-to-End Demo & Benchmark

Runs reference oracle verification, synthetic data generation (seed `42`), temporal training + Platt calibration, offline/online parity across held-out streaming events, training-serving skew checks, checkpoint restart recovery, timeout fallback verification, audit log replay, and a 2,000-request local latency benchmark:

```bash
.venv/bin/python scripts/run_demo.py --benchmark-requests 2000
```

Outputs generated under `reports/`:
- `reports/acceptance_summary.json`
- `reports/benchmark.json`
- `reports/parity_report.json`
- `reports/skew_report.json`
- `reports/runtime_metadata.json`
- `reports/audit_log.jsonl`

## 4. Running the Local Scoring HTTP API

```bash
.venv/bin/python scripts/serve_api.py --host 127.0.0.1 --port 8000
```

Quick smoke checks (in a second terminal):

```bash
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
curl -s http://127.0.0.1:8000/v1/models/active | python3 -m json.tool
curl -s -X POST http://127.0.0.1:8000/v1/events/ingest \
  -H "Content-Type: application/json" \
  -d '{"schema_version":"tx.v1","event_id":"evt_smoke_1","account_id":"acct_syn_smoke","card_token":"card_tok_smoke","merchant_id":"merch_syn_1","merchant_category":"grocery","amount_cents":4500,"currency":"USD","country_code":"US","channel":"card_present","device_id":"dev_smoke_1","event_ts":"2026-01-15T10:00:00Z","ip_risk_score":0.05}'
curl -s -X POST http://127.0.0.1:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{"event":{"schema_version":"tx.v1","event_id":"evt_smoke_2","account_id":"acct_syn_smoke","card_token":"card_tok_smoke","merchant_id":"merch_syn_2","merchant_category":"electronics","amount_cents":12000,"currency":"USD","country_code":"GB","channel":"ecommerce","device_id":"dev_smoke_2","event_ts":"2026-01-15T10:02:00Z","ip_risk_score":0.65},"ingest_after_score":true}'
curl -s http://127.0.0.1:8000/metrics | python3 -m json.tool
curl -s -X POST http://127.0.0.1:8000/v1/audit/replay | python3 -m json.tool
```

## 5. Diagnostics & Recovery Procedures

| Symptom | Diagnostic Check | Recovery Action |
|---|---|---|
| Startup fails with `ConfigurationError` | Check YAML path (`--config`) and verify `0 < review_threshold < decline_threshold < 1`, split fractions sum to `1.0`, and `retention_window_ms >= 86400000`. | Fix `config/default.yaml`; service fails fast by design rather than silently substituting defaults. |
| Model load fails with `ArtifactIntegrityError` | Artifact was modified or `feature_version` changed without retraining. | Re-run `scripts/run_demo.py` to regenerate a valid signed `rtrp.model.v1` artifact or restore from known-good artifact backup. |
| Elevated `model_timeout_fallbacks` in `/metrics` | Inspect `scoring_latency.p99_ms` vs `serving.inference_timeout_ms`. | Service automatically serves `fallback.rules.v1` conservative decisions; investigate host CPU contention or model backend latency before restoring primary model. |
| Elevated `feature_status_stale` in `/metrics` | Stream ingestion lag exceeds `serving.max_feature_staleness_ms`. | Check upstream ingestion pipeline lag; policy conservatively escalates stale high-value transactions to `REVIEW` until watermark catches up. |
| Process crash / restart | Inspect `storage.checkpoint_path` (`data/artifacts/feature_store_checkpoint.json`) and `reports/audit_log.jsonl`. | Restart `serve_api.py` (automatically loads latest atomic checkpoint); call `POST /v1/audit/replay` to verify hash-chain integrity and replay determinism. |
| `CRITICAL_SKEW` in `reports/skew_report.json` | Inspect per-feature `psi` and `mean_shift_sigma`. | Investigate upstream data/feature pipeline changes or genuine fraud regime shift; collect matured labels and retrain/recalibrate a new versioned artifact. |
