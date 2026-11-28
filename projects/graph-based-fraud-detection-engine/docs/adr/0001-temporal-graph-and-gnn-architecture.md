# ADR-0001: Time-Aware Entity Graph, Delayed Label Maturity, and Native Causal GraphSAGE

**Date:** 2026-11-28  
**Status:** Accepted

## Context

Coordinated fraud rings (e.g., mule accounts sharing emulator devices to cash out at gift-card or crypto-onramp merchants) exhibit multi-hop relational patterns that tabular transaction-level features miss. However, naive graph construction introduces two classic production failure modes:

1. **Future-edge leakage**: Building a static graph over all transactions before extracting features allows future device-sharing links and future fraud labels to leak backward into training and evaluation splits, artificially inflating PR-AUC.
2. **Legitimate shared-device false positives**: Naive rules that penalize any device shared by multiple accounts heavily misclassify households and corporate kiosks.
3. **Environment fragility**: Requiring PyTorch Geometric (`torch-geometric`) native C++/CUDA wheels makes clean CPU reproduction slow and brittle across Python environments.

## Decision

1. **Strict Causal Temporal Graph (`TemporalEntityGraph`)**:
   - Store all `USES_DEVICE`, `TRANSACTS_AT`, and `OBSERVED_AT` edges with UTC timestamps and transaction IDs.
   - Enforce `edge.timestamp < as_of` on all historical neighborhood queries and `label_available_at <= as_of` (and `tx_timestamp < as_of`) on all fraud label lookups.
   - Enforce `label_available_at <= split_cutoff` on temporal training and validation splits so immature delayed labels are excluded rather than leaked or mislabeled as clean.
2. **Behavioral Context on Shared Devices**:
   - Pair shared-device degree with 24-hour multi-account burst counts, 30-day high-risk MCC ratios, 2-hop co-account matured fraud exposure, and coordinated merchant triangulation.
   - Emit an explicit `LEGITIMATE_SHARED_DEVICE_MITIGANT` signal when a shared device has zero matured fraud and routine merchant mix.
3. **Deterministic Native 2-Layer Causal GraphSAGE (`TemporalGraphSAGEModel`)**:
   - Implement a 2-layer GraphSAGE-style message-passing network in NumPy over a strictly lower-triangular causal transaction adjacency matrix `A` (where `A[i, j] > 0` only if `t_j < t_i` within `neighbor_window_days`).
   - Keep PyTorch Geometric as an explicit optional extra (`[project.optional-dependencies].pyg`) and report its installation status transparently in `runtime_metadata.json` and `reports/acceptance.md`.

## Consequences

- **Positive**: 100% deterministic CPU training and inference, mathematical guarantee against 1-hop and 2-hop future leakage (`triu(A) == 0` and `triu(A @ A) == 0`), fast test execution, and verifiable graph-traced explanations.
- **Trade-off**: Dense matrix propagation (`O(N^2)` in the training prefix) is fast for synthetic research datasets (thousands of transactions) but would be replaced by sparse/mini-batch temporal neighbor sampling (e.g., PyG `NeighborLoader` or TGN) at million-node production scale.
