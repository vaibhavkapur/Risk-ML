# Graph-Based Fraud Detection Engine

An independently developed educational and research prototype for scoring coordinated fraud rings using **time-aware entity graphs**, **privacy-safe identifiers**, **strictly leak-free graph features**, **calibrated tabular & graph neural network models**, **graph-traced explanations**, and an **analyst review workflow**.

> **Scope & Limitations:** Offline/synthetic prototype only. Uses exclusively synthetic accounts, devices, merchants, transactions, and delayed fraud labels. Synthetic accuracy and PR-AUC metrics do not represent real-world performance. Live capital, real customer records, and production credentials are excluded.

---

## Highlights

- **Time-Aware Entity Graph (`TemporalEntityGraph`)**: Models `account`, `device`, and `merchant` entities with privacy-safe HMAC-SHA256 IDs (`acc_*`, `dev_*`, `mer_*`, `tx_*`) and timestamped `USES_DEVICE`, `TRANSACTS_AT`, and `OBSERVED_AT` edges.
- **Zero Future-Edge & Delayed-Label Leakage**:
  - Historical feature queries strictly enforce `edge.timestamp < tx_timestamp` and `label_available_at <= tx_timestamp`.
  - Temporal splits exclude immature labels (`label_available_at > split_cutoff`).
  - The 2-layer causal `TemporalGraphSAGEModel` uses a strictly lower-triangular temporal adjacency matrix `A` (`triu(A) == 0` and `triu(A @ A) == 0`).
- **Fraud Rings vs Legitimate Shared Devices**: Distinguishes coordinated mule rings from households/kiosks using multi-account 24h bursts, 30d device high-risk merchant ratios, 2-hop matured fraud exposure, merchant triangulation, and an explicit `LEGITIMATE_SHARED_DEVICE_MITIGANT`.
- **Calibrated Scoring & Analyst Review API**: FastAPI service (`/v1/score`, `/v1/reviews`, `/v1/reviews/{case_id}/disposition`, `/health`, `/metrics`) returning calibrated probabilities, cold-start flags, graph-traced evidence paths, and auditable analyst dispositions.

---

## Quickstart

```bash
cd projects/graph-based-fraud-detection-engine
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 1. Generate deterministic fixtures
```bash
.venv/bin/python scripts/generate_fixtures.py
```

### 2. Run the reproducible training, calibration & evaluation pipeline
```bash
.venv/bin/python scripts/run_pipeline.py --config config/default.yaml --save-artifacts
```

### 3. Run the end-to-end demo (fraud ring vs legitimate shared device + analyst disposition)
```bash
.venv/bin/python scripts/run_demo.py
```

### 4. Run the full test suite
```bash
.venv/bin/pytest -v
```

### 5. Optional: Start the local HTTP API
```bash
.venv/bin/python scripts/serve_api.py --host 127.0.0.1 --port 8000
```

---

## Measured Synthetic Benchmark Results (`run_b737379321a3`, seed `42`)

Evaluated on the held-out temporal test split (`708` synthetic transactions, `29` positives / `4.10%` fraud rate) after Platt calibration on the temporal validation split (`516` matured transactions; `381` immature train labels and `428` immature validation labels excluded):

| Model | Test PR-AUC | Recall @ 1% FPR | Recall @ 5% FPR | Calibrated ECE | Calibrated Brier | Fraud Ring Det. | Legit Shared Device FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `logistic_tabular_baseline` (10 tabular) | `0.7274` | `0.5172` | `0.9310` | `0.0179` | `0.0149` | `1.0000` | `0.0000` |
| `logistic` (20 graph) | `0.8558` | `0.8276` | `0.9310` | `0.0184` | `0.0133` | `0.9259` | `0.0000` |
| `boosted` (HistGradientBoosting, 20 graph) | **`0.8955`** | `0.8621` | `0.9310` | **`0.0079`** | **`0.0095`** | `0.9259` | `0.0000` |
| `gnn` (2-layer causal TemporalGraphSAGE) | `0.8428` | **`0.8966`** | `0.9310` | `0.0182` | `0.0118` | `0.9630` | `0.0086` |

- **Leakage Audit:** `PASS` (`0` future-edge or immature-label violations across `3,589` audited transactions).
- **Exact Reproducibility:** `true` across independent runs with seed `42`.
- **Automated Tests:** `16 passed` (`pytest -v`).

---

## Example Output (`scripts/run_demo.py`)

```json
{
  "tx_id": "tx_eb94386458d1",
  "decision_id": "dec_ac378d7682533d",
  "calibrated_probability": 0.934765,
  "decision": "block",
  "review_case_id": "case_e7faa1be0598",
  "reason_codes": [
    "DEVICE_PRIOR_MATURED_FRAUD",
    "TWO_HOP_RING_MATURED_FRAUD",
    "COORDINATED_MERCHANT_TRIANGULATION",
    "SHARED_DEVICE_HIGH_RISK_BURST"
  ],
  "evidence_paths": [
    "Shared device dev_bc7fa8d7e5ac previously used by co-account acc_28eeb634b475 with matured prior fraud",
    "Co-account acc_93e7e5917a9e previously used shared device dev_bc7fa8d7e5ac and transacted at same merchant mer_57198144a521 (matured fraud linked)"
  ]
}
```

By contrast, a legitimate household shared-device transaction (`tx_c540f0dd0804`) receives `calibrated_probability: 0.001644`, `decision: "approve"`, and `reason_codes: ["LEGITIMATE_SHARED_DEVICE_MITIGANT"]`.

---

## Project Layout & Documentation

- [`plan.md`](plan.md) — Original implementation plan and acceptance criteria
- [`docs/architecture.md`](docs/architecture.md) — Component diagram, data flow, causal invariants, and trade-offs
- [`docs/contracts.md`](docs/contracts.md) — Canonical schemas, units, UTC timestamps, privacy-safe IDs, and error codes
- [`docs/integration.md`](docs/integration.md) — Adapter boundaries and open-source dependency/license inventory
- [`docs/runbook.md`](docs/runbook.md) — Setup, pipeline, demo, validation, diagnostics, and recovery procedures
- [`docs/adr/0001-temporal-graph-and-gnn-architecture.md`](docs/adr/0001-temporal-graph-and-gnn-architecture.md) — Architecture Decision Record
- [`reports/acceptance.md`](reports/acceptance.md) — Full acceptance evidence mapped to tests and raw JSON reports
- [`reports/`](reports/) — Machine-readable evaluation, runtime metadata, reproducibility, and sample explanation artifacts
- [`../../.github/workflows/graph-based-fraud-detection-engine.yml`](../../.github/workflows/graph-based-fraud-detection-engine.yml) — Automated CI validation workflow

---

## Assumptions & Known Limitations

1. **Synthetic distributions**: Ring bursts, merchant risk categories, and household sharing patterns are generated synthetically and do not mirror any institution's real traffic.
2. **Optional PyTorch Geometric backend not run**: `torch-geometric` is declared under `[project.optional-dependencies].pyg` and is not installed in the default CPU environment; the engine uses its deterministic native NumPy 2-layer `TemporalGraphSAGEModel`.
3. **In-memory storage & dense training adjacency**: Designed for research-scale datasets (thousands of transactions). Production scale requires durable graph/case storage and mini-batch temporal neighbor sampling.
4. **No live production integration**: Authentication, rate limiting, encryption key management, and live payment blocking are outside scope.
