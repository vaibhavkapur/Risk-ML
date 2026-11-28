# Operator & Developer Runbook — Account Takeover Detection Engine

## 1. Local Setup

```bash
cd projects/account-takeover-detection-engine
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 2. Validation, Evaluation, and Demo Commands

```bash
# Run full unit, integration, boundary, and API test suite
.venv/bin/pytest -v

# Run reproducible chronological training, calibration, evaluation & scenario suite
.venv/bin/python scripts/run_evaluation.py --config config/default.yaml --output-dir reports

# Run concise end-to-end demo walkthrough
.venv/bin/python scripts/run_demo.py

# Optional: start local FastAPI server (offline / synthetic requests only)
PYTHONPATH=src .venv/bin/uvicorn ato_engine.api:create_app --factory --host 127.0.0.1 --port 8000
```

## 3. Diagnostics & Health Inspection

- **Service health**: `GET /health` reports `store_healthy`, `model_loaded`, `model_version`, `schema_version`, `feature_version`, `policy_version`, `drift_alert_active`, `critical_drift_breaker`, and `audit_decisions_logged`.
- **Diagnostic counters**: `GET /v1/metrics` reports `total_decisions`, `decisions_by_action`, `fallbacks_by_reason`, `validation_errors`, `unsupported_input_errors`, `drift_checks_run`, `critical_drift_events`, and `avg_latency_ms`.
- **Decision audit log**: Inspect `reports/decision_audit.jsonl` (one JSON object per decision with deterministic `decision_id`, scores, thresholds, rules, and fallback status).

## 4. Failure Modes & Recovery Steps

| Failure / Alert | Symptom | Automatic Behavior | Recovery Action |
|---|---|---|---|
| **Missing or invalid config** | `ConfigurationError` on startup | Fails fast; no silent defaults | Verify `config/default.yaml` exists and passes threshold/window/fraction invariants. |
| **Missing or tampered model artifact** | `ModelArtifactError` on load | Fails fast on checksum mismatch; if engine started with `model_bundle=None`, uses `MODEL_UNAVAILABLE` fallback | Re-run `scripts/run_evaluation.py` to regenerate `reports/model_bundle.joblib` and its `.manifest.json`, then verify SHA-256. |
| **State store outage (e.g., Redis down)** | `store_healthy: false`, `fallback_reason: store_unavailable` | Switches to stateless single-event features + conservative fallback (minimum `CHALLENGE_OTP`); logs every decision | Restore store connectivity; verify `GET /health` returns `status: ok`. |
| **Excessive missing telemetry** | `fallback_reason: excessive_missing_features` | Adds uncertainty cushion and blocks silent `ALLOW` | Inspect upstream login/device telemetry collectors for dropped fields (`evidence.missing_fields`). |
| **Stale model (> 90 days)** | `fallback_reason: model_stale` | Engages conservative rule fallback while continuing audit logging | Retrain on newly matured chronological labels and deploy new versioned artifact. |
| **Critical drift detected** | `POST /v1/drift/check` reports `critical_drift_detected: true` | Tightens thresholds and activates `critical_drift_circuit_breaker` fallback | Inspect `DriftReport` (`score_psi`, `feature_psi`, `missingness_rate`, `expiry_rate`), investigate attack wave or schema shift, retrain and recalibrate before resetting breaker. |
| **Deterministic replay verification** | Need to verify restart/rerun consistency | `compare_decision_replays()` checks `decision_id`, `action`, `fallback_reason`, and calibrated score within `1e-6` | Re-run `scripts/run_evaluation.py` with identical `config/default.yaml` and compare `replay_reproducibility.matched`. |
