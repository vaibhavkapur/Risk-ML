# Operator & Developer Runbook

## 1. Local Setup (Clean Checkout)

```bash
cd projects/payment-authorization-optimization-engine
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 2. Running the Reproducible Demo

```bash
.venv/bin/python scripts/run_demo.py
```

Outputs written to `reports/`:
- `baseline_outputs.json` — baseline threshold policy metrics, audit, and sample decisions.
- `policy_comparison.json` — calibration metrics and DM/IPS/SNIPS/DR counterfactual comparison across `baseline_threshold`, `unconstrained_ev`, and `constrained_risk_aware` with 95% confidence bounds and synthetic oracle errors.
- `audit_report.json` — portfolio & per-decision constraint audit plus structured diagnostics.
- `model_manifest.json` — versioned model artifact metadata and training imputation statistics.
- `runtime_metadata.json` — Python/library versions, seed, config hash, and provenance.

## 3. Running Full Validation & Replay Verification

```bash
.venv/bin/python scripts/validate_all.py
```

This executes:
1. Full `pytest` suite (`tests/`).
2. End-to-end reproducible demo (`scripts/run_demo.py`).
3. Deterministic replay check (`payment_auth_engine.cli replay`).

## 4. Running the Local FastAPI Prototype (Offline Synthetic Mode)

```bash
PYTHONPATH=src .venv/bin/python -c "
import uvicorn
from payment_auth_engine.api import create_app
from payment_auth_engine.config import load_config
from payment_auth_engine.engine import AuthorizationEngine
from payment_auth_engine.models import AuthorizationModelBundle
from payment_auth_engine.policy import ConstrainedRiskAwarePolicy
from payment_auth_engine.synthetic import generate_synthetic_dataset

cfg = load_config('config/default.yaml')
ds = generate_synthetic_dataset(cfg)
bundle, _, cal_idx, _ = AuthorizationModelBundle.train(ds.attempts, ds.outcomes, cfg, ds.as_of_utc)
pol = ConstrainedRiskAwarePolicy(cfg)
pol.fit_dual_multipliers([ds.attempts[int(i)] for i in cal_idx], [bundle.predict_all_actions(ds.attempts[int(i)]) for i in cal_idx])
uvicorn.run(create_app(AuthorizationEngine(config=cfg, model_bundle=bundle, policy=pol)), host='127.0.0.1', port=8000)
"
```

## 5. Diagnostics & Operational Recovery

| Failure Mode | Symptom / Error Code | Recovery Action |
|---|---|---|
| Missing or invalid config | CLI exits with code `2`; `ConfigurationError` (`CONFIG_NOT_FOUND`, `CONFIG_PARSE_ERROR`, `CONFIG_VALIDATION_ERROR`) | Verify `--config` path and YAML schema against `config/default.yaml`; verify split fractions sum to `1.0`. |
| Invalid request payload | HTTP `422` with `error_code` (e.g., `UNSUPPORTED_CURRENCY`, `MISSING_TIMEZONE`, `MISSING_PRIOR_RESPONSE_CODE`) | Fix caller payload per `docs/contracts.md`; do not bypass validation. |
| Uncertain prior authorization outcome | Decision returns `HOLD_RECONCILE` (`uncertain_outcome_holds` counter increments) | Query processor settlement/reconciliation report out-of-band before initiating any new transaction; never blindly retry. |
| Primary processor outage | `outage_failovers` increments and traffic routes to secondary processor | Monitor `/v1/diagnostics`; restore health via `POST /v1/processors/proc_a/health` with `{"status": "HEALTHY"}` once recovered. |
| Total processor outage | Decisions fail closed with `BLOCK_PROCESSOR_OUTAGE` (`outage_fail_closed_blocks` increments) | Restore at least one processor to `HEALTHY` or `DEGRADED`; no silent approvals occur during outage. |
| Counterfactual evaluation error | `CounterfactualAssumptionError` (`POSITIVITY_VIOLATION`, `INSUFFICIENT_SUPPORT`, `ESS_COLLAPSE`) | Inspect logging policy exploration floor and propensity distribution; do not force off-policy point estimates when overlap fails. |
| Replay / restart rerun | Re-running with identical `config/default.yaml` and `seed` | Run `PYTHONPATH=src .venv/bin/python -m payment_auth_engine.cli replay --config config/default.yaml` to verify 100% matching `decision_id`s and actions. |
