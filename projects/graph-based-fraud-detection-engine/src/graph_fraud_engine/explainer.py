"""Graph-traced explanation engine.

Every explanation maps directly back to nodes, edges, and matured labels in the
temporal entity graph strictly prior to the scored transaction's timestamp.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from graph_fraud_engine.config import FeatureConfig
from graph_fraud_engine.features import FeatureExtractionResult
from graph_fraud_engine.graph import TemporalEntityGraph
from graph_fraud_engine.schemas import (
    EvidenceEdge,
    EvidencePath,
    GraphExplanation,
    LinkedEntitySummary,
    TemporalLeakageError,
    TransactionRecord,
)


def _stable_path_id(tx_id: str, nodes: List[str], tag: str) -> str:
    raw = f"{tx_id}|{tag}|{'->'.join(nodes)}".encode("utf-8")
    return f"path_{hashlib.sha256(raw).hexdigest()[:10]}"


def build_graph_explanation(
    graph: TemporalEntityGraph,
    tx: TransactionRecord,
    feature_result: FeatureExtractionResult,
    feature_attributions: Dict[str, float],
    feature_config: Optional[FeatureConfig] = None,
    max_paths: int = 5,
) -> GraphExplanation:
    """Construct a verifiable, graph-traced explanation for a scored transaction."""
    cfg = feature_config or FeatureConfig()
    as_of = tx.tx_timestamp
    long_w = timedelta(days=cfg.long_window_days)
    feats = feature_result.features

    # Build strictly historical NetworkX subgraph for structural audit
    hist_nx = graph.build_historical_nx_subgraph(
        seed_entities=[tx.account_id, tx.device_id, tx.merchant_id],
        as_of=as_of,
        max_hops=cfg.max_hops,
        window=long_w,
        exclude_tx_id=tx.tx_id,
    )
    max_hist_edge_ts: Optional[datetime] = graph.max_edge_timestamp_in_nx(hist_nx)
    if max_hist_edge_ts is not None and max_hist_edge_ts >= as_of:
        raise TemporalLeakageError(
            f"Explanation subgraph contains future edge {max_hist_edge_ts} >= {as_of}"
        )

    # 1. Linked entity summaries for the transaction's direct entities
    linked_entities: List[LinkedEntitySummary] = []
    for eid, etype in (
        (tx.account_id, "account"),
        (tx.device_id, "device"),
        (tx.merchant_id, "merchant"),
    ):
        prior_rel = "USES_DEVICE" if etype in ("account", "device") else "TRANSACTS_AT"
        prior_count = len(
            graph.prior_edges(eid, as_of=as_of, relation=prior_rel, exclude_tx_id=tx.tx_id)
        )
        matured_fraud = graph.matured_fraud_count(eid, as_of=as_of, exclude_tx_id=tx.tx_id)
        linked_entities.append(
            LinkedEntitySummary(
                entity_id=eid,
                entity_type=etype,  # type: ignore[arg-type]
                prior_tx_count=prior_count,
                prior_known_fraud_count=matured_fraud,
                unseen_cold_start=(prior_count == 0),
            )
        )

    # 2. Graph-traced evidence paths (all historical edges strictly < as_of)
    evidence_paths: List[EvidencePath] = []
    co_accounts, _ = graph.compute_two_hop_account_ring(
        account_id=tx.account_id,
        device_id=tx.device_id,
        as_of=as_of,
        window=long_w,
        exclude_tx_id=tx.tx_id,
    )

    # Sort co-accounts deterministically: highest matured fraud first, then ID
    scored_co_accounts = sorted(
        co_accounts,
        key=lambda aid: (-graph.matured_fraud_count(aid, as_of=as_of, exclude_tx_id=tx.tx_id), aid),
    )

    # Add up to 3 co-accounts to linked_entities for analyst visibility
    for co_acc in scored_co_accounts[:3]:
        co_prior = len(
            graph.prior_account_tx_edges(co_acc, as_of=as_of, exclude_tx_id=tx.tx_id)
        )
        co_fraud = graph.matured_fraud_count(co_acc, as_of=as_of, exclude_tx_id=tx.tx_id)
        linked_entities.append(
            LinkedEntitySummary(
                entity_id=co_acc,
                entity_type="account",
                prior_tx_count=co_prior,
                prior_known_fraud_count=co_fraud,
                unseen_cold_start=False,
            )
        )

    # 2a. Coordinated merchant triangulation paths:
    #     tx.account_id --(CURRENT_TX)--> tx.device_id <--(USES_DEVICE, <as_of)-- co_acc --(TRANSACTS_AT, <as_of)--> tx.merchant_id
    for co_acc in scored_co_accounts:
        if len(evidence_paths) >= max_paths:
            break
        dev_edges = [
            e
            for e in graph.prior_edges(
                co_acc, as_of=as_of, relation="USES_DEVICE", window=long_w, exclude_tx_id=tx.tx_id
            )
            if e.dst == tx.device_id
        ]
        mer_edges = [
            e
            for e in graph.prior_edges(
                co_acc, as_of=as_of, relation="TRANSACTS_AT", window=long_w, exclude_tx_id=tx.tx_id
            )
            if e.dst == tx.merchant_id
        ]
        co_has_fraud = graph.matured_fraud_count(co_acc, as_of=as_of, exclude_tx_id=tx.tx_id) > 0

        if dev_edges and mer_edges:
            latest_dev_edge = dev_edges[-1]
            latest_mer_edge = mer_edges[-1]
            nodes = [tx.account_id, tx.device_id, co_acc, tx.merchant_id]
            edges = [
                EvidenceEdge(
                    source=tx.account_id,
                    target=tx.device_id,
                    relation="CURRENT_TX_USES_DEVICE",
                    timestamp=as_of,
                    tx_id=tx.tx_id,
                ),
                EvidenceEdge(
                    source=co_acc,
                    target=tx.device_id,
                    relation="PRIOR_USES_DEVICE",
                    timestamp=latest_dev_edge.timestamp,
                    tx_id=latest_dev_edge.tx_id,
                ),
                EvidenceEdge(
                    source=co_acc,
                    target=tx.merchant_id,
                    relation="PRIOR_TRANSACTS_AT",
                    timestamp=latest_mer_edge.timestamp,
                    tx_id=latest_mer_edge.tx_id,
                ),
            ]
            evidence_paths.append(
                EvidencePath(
                    path_id=_stable_path_id(tx.tx_id, nodes, "triangulation"),
                    summary=(
                        f"Co-account {co_acc} previously used shared device {tx.device_id} and "
                        f"transacted at same merchant {tx.merchant_id}"
                        + (" (matured fraud linked)" if co_has_fraud else "")
                    ),
                    nodes=nodes,
                    edges=edges,
                    contains_known_fraud=co_has_fraud,
                )
            )
        elif dev_edges:
            latest_dev_edge = dev_edges[-1]
            nodes = [tx.account_id, tx.device_id, co_acc]
            edges = [
                EvidenceEdge(
                    source=tx.account_id,
                    target=tx.device_id,
                    relation="CURRENT_TX_USES_DEVICE",
                    timestamp=as_of,
                    tx_id=tx.tx_id,
                ),
                EvidenceEdge(
                    source=co_acc,
                    target=tx.device_id,
                    relation="PRIOR_USES_DEVICE",
                    timestamp=latest_dev_edge.timestamp,
                    tx_id=latest_dev_edge.tx_id,
                ),
            ]
            evidence_paths.append(
                EvidencePath(
                    path_id=_stable_path_id(tx.tx_id, nodes, "shared_device"),
                    summary=(
                        f"Shared device {tx.device_id} previously used by co-account {co_acc}"
                        + (" with matured prior fraud" if co_has_fraud else "")
                    ),
                    nodes=nodes,
                    edges=edges,
                    contains_known_fraud=co_has_fraud,
                )
            )

    # 2b. If no co-account paths, still trace the direct transaction-to-device/merchant historical context
    if not evidence_paths:
        prior_mer_edges = graph.prior_edges(
            tx.merchant_id, as_of=as_of, relation="TRANSACTS_AT", window=long_w, exclude_tx_id=tx.tx_id
        )
        if prior_mer_edges:
            latest_e = prior_mer_edges[-1]
            mer_fraud = graph.matured_fraud_count(tx.merchant_id, as_of=as_of, exclude_tx_id=tx.tx_id) > 0
            nodes = [tx.account_id, tx.merchant_id, latest_e.src]
            evidence_paths.append(
                EvidencePath(
                    path_id=_stable_path_id(tx.tx_id, nodes, "merchant_context"),
                    summary=f"Direct transaction at merchant {tx.merchant_id} (prior activity trace)",
                    nodes=nodes,
                    edges=[
                        EvidenceEdge(
                            source=tx.account_id,
                            target=tx.merchant_id,
                            relation="CURRENT_TX_TRANSACTS_AT",
                            timestamp=as_of,
                            tx_id=tx.tx_id,
                        ),
                        EvidenceEdge(
                            source=latest_e.src,
                            target=tx.merchant_id,
                            relation="PRIOR_TRANSACTS_AT",
                            timestamp=latest_e.timestamp,
                            tx_id=latest_e.tx_id,
                        ),
                    ],
                    contains_known_fraud=mer_fraud,
                )
            )
        else:
            nodes = [tx.account_id, tx.device_id, tx.merchant_id]
            evidence_paths.append(
                EvidencePath(
                    path_id=_stable_path_id(tx.tx_id, nodes, "cold_start_tx"),
                    summary="Cold-start transaction with no prior 30-day historical neighborhood edges",
                    nodes=nodes,
                    edges=[
                        EvidenceEdge(
                            source=tx.account_id,
                            target=tx.device_id,
                            relation="CURRENT_TX_USES_DEVICE",
                            timestamp=as_of,
                            tx_id=tx.tx_id,
                        ),
                        EvidenceEdge(
                            source=tx.account_id,
                            target=tx.merchant_id,
                            relation="CURRENT_TX_TRANSACTS_AT",
                            timestamp=as_of,
                            tx_id=tx.tx_id,
                        ),
                    ],
                    contains_known_fraud=False,
                )
            )

    # 3. Domain-interpretable reason codes and legitimate shared-device mitigant
    reason_codes: List[str] = []
    shared_prior = int(feats["device_prior_distinct_accounts_30d"])
    legit_shared_mitigant = (
        shared_prior >= 1
        and feats["device_matured_fraud_count"] == 0.0
        and feats["device_high_risk_mcc_ratio_30d"] <= 0.2
        and feats["device_distinct_accounts_24h"] <= 1.0
    )

    if feats["device_matured_fraud_count"] >= 1.0:
        reason_codes.append("DEVICE_PRIOR_MATURED_FRAUD")
    if feats["two_hop_ring_matured_fraud_count"] >= 1.0:
        reason_codes.append("TWO_HOP_RING_MATURED_FRAUD")
    if feats["shared_merchant_with_co_accounts_30d"] >= 1.0 and feats["is_high_risk_mcc"] >= 1.0:
        reason_codes.append("COORDINATED_MERCHANT_TRIANGULATION")
    if shared_prior >= 2 and feats["device_high_risk_mcc_ratio_30d"] >= 0.5:
        reason_codes.append("SHARED_DEVICE_HIGH_RISK_BURST")
    if feats["is_emulator_or_kiosk"] >= 0.9:
        reason_codes.append("EMULATOR_DEVICE_RISK")
    if feats["account_unseen_cold_start"] >= 1.0:
        reason_codes.append("COLD_START_UNSEEN_ACCOUNT")
    if feats["device_unseen_cold_start"] >= 1.0:
        reason_codes.append("COLD_START_UNSEEN_DEVICE")
    if legit_shared_mitigant:
        reason_codes.append("LEGITIMATE_SHARED_DEVICE_MITIGANT")
    if not reason_codes:
        reason_codes.append("LOW_GRAPH_RISK_PROFILE")

    # Top 6 feature attributions by absolute magnitude
    sorted_attrs = dict(
        sorted(feature_attributions.items(), key=lambda kv: (-abs(kv[1]), kv[0]))[:6]
    )

    return GraphExplanation(
        as_of_timestamp=as_of,
        reason_codes=reason_codes,
        top_feature_contributions=sorted_attrs,
        linked_entities=linked_entities,
        evidence_paths=evidence_paths,
        ring_cluster_size=len(co_accounts) + 1,
        shared_device_accounts_prior=shared_prior,
        legitimate_shared_device_mitigant=legit_shared_mitigant,
        graph_audit_max_edge_timestamp=max_hist_edge_ts,
    )
