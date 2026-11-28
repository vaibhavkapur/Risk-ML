"""Configurable suspicious-pattern scenario detectors (pure domain logic).

Detects five configurable synthetic AML typologies using rolling temporal
aggregations and directed graph features while preserving explicit exemptions
for legitimate high-volume profiles (e.g., merchant settlements, payroll).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from typing import Dict, List, Sequence, Tuple

from aml_engine.config import EngineConfig
from aml_engine.features.graph import TemporalChain, TransactionGraph
from aml_engine.models import (
    Account,
    GraphFeatureVector,
    NormalizedTransaction,
    ReasonContribution,
    RollingFeatureVector,
    ScenarioHit,
)


def detect_scenarios(
    transactions: Sequence[NormalizedTransaction],
    accounts_by_id: Dict[str, Account],
    rolling_features: Dict[Tuple[str, str], RollingFeatureVector],
    graph: TransactionGraph,
    config: EngineConfig,
) -> Tuple[List[ScenarioHit], Dict[str, GraphFeatureVector]]:
    """Run all enabled scenario detectors and return hits + account graph features."""
    sc_cfg = config.scenarios
    hits: List[ScenarioHit] = []

    # Graph motif searches
    lay_cfg = sc_cfg.get("rapid_layering", {})
    layering_chains: List[TemporalChain] = []
    if lay_cfg.get("enabled", True):
        layering_chains = graph.find_rapid_layering_chains(
            min_hops=int(lay_cfg.get("min_hops", 3)),
            max_hops=int(lay_cfg.get("max_hops", 5)),
            max_window_hours=float(lay_cfg.get("max_chain_window_hours", 12.0)),
            min_initial_amount_usd=float(lay_cfg.get("min_initial_amount_usd", 10000.0)),
            min_preservation_ratio=float(lay_cfg.get("min_value_preservation_ratio", 0.80)),
            max_preservation_ratio=float(lay_cfg.get("max_value_preservation_ratio", 1.02)),
        )

    cyc_cfg = sc_cfg.get("round_trip_cycle", {})
    round_trip_cycles: List[TemporalChain] = []
    if cyc_cfg.get("enabled", True):
        round_trip_cycles = graph.find_round_trip_cycles(
            min_hops=int(cyc_cfg.get("min_hops", 3)),
            max_hops=int(cyc_cfg.get("max_hops", 5)),
            max_window_hours=float(cyc_cfg.get("max_cycle_window_hours", 48.0)),
            min_amount_usd=float(cyc_cfg.get("min_amount_usd", 10000.0)),
            min_return_ratio=float(cyc_cfg.get("min_return_ratio", 0.75)),
        )

    graph_features = graph.compute_account_graph_features(layering_chains, round_trip_cycles)

    # 1. STRUCTURING_SMURFING
    str_cfg = sc_cfg.get("structuring", {})
    if str_cfg.get("enabled", True):
        hits.extend(
            _detect_structuring(transactions, accounts_by_id, rolling_features, str_cfg)
        )

    # 2. RAPID_LAYERING_CHAIN
    if lay_cfg.get("enabled", True):
        hits.extend(_detect_rapid_layering(layering_chains, lay_cfg))

    # 3. ROUND_TRIP_CYCLE
    if cyc_cfg.get("enabled", True):
        hits.extend(_detect_round_trip_cycles(round_trip_cycles, cyc_cfg))

    # 4. MULE_FAN_IN_FAN_OUT
    mule_cfg = sc_cfg.get("mule_fan_in_fan_out", {})
    if mule_cfg.get("enabled", True):
        hits.extend(
            _detect_mule_fan_in_fan_out(transactions, accounts_by_id, rolling_features, mule_cfg)
        )

    # 5. DORMANT_ACCOUNT_SPIKE
    dorm_cfg = sc_cfg.get("dormant_spike", {})
    if dorm_cfg.get("enabled", True):
        hits.extend(
            _detect_dormant_spike(transactions, accounts_by_id, rolling_features, dorm_cfg)
        )

    hits.sort(
        key=lambda h: (
            h.trigger_timestamp_iso,
            h.subject_account_id,
            h.scenario_name,
            h.involved_tx_ids,
        )
    )
    return hits, graph_features


def _detect_structuring(
    transactions: Sequence[NormalizedTransaction],
    accounts_by_id: Dict[str, Account],
    rolling_features: Dict[Tuple[str, str], RollingFeatureVector],
    cfg: Dict[str, object],
) -> List[ScenarioHit]:
    band_min = float(cfg.get("band_min_usd", 7500.0))
    band_max = float(cfg.get("band_max_usd", 9999.99))
    min_count = int(cfg.get("min_count_24h", 3))
    min_total = float(cfg.get("min_total_usd_24h", 24000.0))
    channels = {str(c).upper() for c in cfg.get("channels", [])}
    base_weight = float(cfg.get("weight", 0.85))

    inbound_by_target: Dict[str, List[NormalizedTransaction]] = defaultdict(list)
    for tx in transactions:
        inbound_by_target[tx.target_account_id].append(tx)

    hits: List[ScenarioHit] = []
    # Deduplicate overlapping hits per account within the same 24h cluster
    best_by_account_day: Dict[Tuple[str, str], ScenarioHit] = {}

    for tx in transactions:
        fv = rolling_features.get((tx.tx_id, "TARGET"))
        if fv is None:
            continue
        if fv.structuring_band_count_24h < min_count or fv.structuring_band_sum_24h_usd < min_total:
            continue

        acc = accounts_by_id.get(tx.target_account_id)
        if acc and acc.account_type in {"PAYROLL_DISBURSEMENT", "TREASURY"}:
            continue

        cutoff = tx.timestamp_utc - timedelta(hours=24)
        contributing = [
            item
            for item in inbound_by_target[tx.target_account_id]
            if cutoff <= item.timestamp_utc <= tx.timestamp_utc
            and band_min <= item.amount_usd <= band_max
            and (not channels or item.channel in channels)
        ]
        if len(contributing) < min_count:
            continue

        total_sum = sum(c.amount_usd for c in contributing)
        score = min(1.0, base_weight + 0.03 * max(0, len(contributing) - min_count))
        involved_txs = tuple(c.tx_id for c in contributing)
        involved_accs = tuple(sorted({tx.target_account_id} | {c.source_account_id for c in contributing}))
        reason = ReasonContribution(
            code="RULE_STRUCTURING_24H_BAND",
            category="RULE",
            weight=score,
            description=(
                f"Account {tx.target_account_id} received {len(contributing)} transfers "
                f"in the sub-threshold band (${band_min:,.0f}-${band_max:,.2f}) within 24h "
                f"totaling ${total_sum:,.2f} USD."
            ),
            evidence_metrics={
                "band_count_24h": len(contributing),
                "band_sum_24h_usd": round(total_sum, 2),
                "unique_senders_24h": fv.unique_inbound_senders_24h,
            },
        )
        hit = ScenarioHit(
            scenario_name="STRUCTURING_SMURFING",
            subject_account_id=tx.target_account_id,
            subject_entity_id=tx.target_entity_id,
            trigger_timestamp_iso=tx.timestamp_iso,
            rule_score=round(score, 6),
            involved_tx_ids=involved_txs,
            involved_accounts=involved_accs,
            graph_path=(tx.target_account_id,),
            reasons=(reason,),
            metrics={
                "structuring_count_24h": len(contributing),
                "structuring_sum_24h_usd": round(total_sum, 2),
            },
        )
        cluster_key = (tx.target_account_id, tx.timestamp_iso[:10])
        prev = best_by_account_day.get(cluster_key)
        if prev is None or len(hit.involved_tx_ids) >= len(prev.involved_tx_ids):
            best_by_account_day[cluster_key] = hit

    hits.extend(best_by_account_day.values())
    return hits


def _detect_rapid_layering(
    chains: Sequence[TemporalChain],
    cfg: Dict[str, object],
) -> List[ScenarioHit]:
    base_weight = float(cfg.get("weight", 0.90))
    hits: List[ScenarioHit] = []
    for chain in chains:
        first_tx = chain.transactions[0]
        last_tx = chain.transactions[-1]
        hops = len(chain.transactions)
        score = min(1.0, base_weight + 0.02 * max(0, hops - 3))
        reason = ReasonContribution(
            code="GRAPH_RAPID_LAYERING_CHAIN",
            category="GRAPH",
            weight=score,
            description=(
                f"Detected {hops}-hop transfer chain {' -> '.join(chain.account_path)} "
                f"moving ${first_tx.amount_usd:,.2f} to ${last_tx.amount_usd:,.2f} USD "
                f"({chain.value_preservation_ratio:.1%} value preserved) in {chain.elapsed_hours:.2f}h."
            ),
            evidence_metrics={
                "hops": hops,
                "elapsed_hours": chain.elapsed_hours,
                "value_preservation_ratio": chain.value_preservation_ratio,
                "initial_amount_usd": round(first_tx.amount_usd, 2),
                "final_amount_usd": round(last_tx.amount_usd, 2),
            },
        )
        hits.append(
            ScenarioHit(
                scenario_name="RAPID_LAYERING_CHAIN",
                subject_account_id=first_tx.source_account_id,
                subject_entity_id=first_tx.source_entity_id,
                trigger_timestamp_iso=last_tx.timestamp_iso,
                rule_score=round(score, 6),
                involved_tx_ids=tuple(t.tx_id for t in chain.transactions),
                involved_accounts=chain.account_path,
                graph_path=chain.account_path,
                reasons=(reason,),
                metrics={
                    "hops": hops,
                    "elapsed_hours": chain.elapsed_hours,
                    "value_preservation_ratio": chain.value_preservation_ratio,
                },
            )
        )
    return hits


def _detect_round_trip_cycles(
    cycles: Sequence[TemporalChain],
    cfg: Dict[str, object],
) -> List[ScenarioHit]:
    base_weight = float(cfg.get("weight", 0.92))
    hits: List[ScenarioHit] = []
    for cyc in cycles:
        first_tx = cyc.transactions[0]
        last_tx = cyc.transactions[-1]
        hops = len(cyc.transactions)
        score = min(1.0, base_weight + 0.02 * max(0, hops - 3))
        reason = ReasonContribution(
            code="GRAPH_ROUND_TRIP_CYCLE",
            category="GRAPH",
            weight=score,
            description=(
                f"Detected {hops}-hop circular flow {' -> '.join(cyc.account_path)} "
                f"returning ${last_tx.amount_usd:,.2f} of ${first_tx.amount_usd:,.2f} USD "
                f"({cyc.value_preservation_ratio:.1%}) within {cyc.elapsed_hours:.2f}h."
            ),
            evidence_metrics={
                "hops": hops,
                "elapsed_hours": cyc.elapsed_hours,
                "return_ratio": cyc.value_preservation_ratio,
                "origin_amount_usd": round(first_tx.amount_usd, 2),
                "returned_amount_usd": round(last_tx.amount_usd, 2),
            },
        )
        hits.append(
            ScenarioHit(
                scenario_name="ROUND_TRIP_CYCLE",
                subject_account_id=first_tx.source_account_id,
                subject_entity_id=first_tx.source_entity_id,
                trigger_timestamp_iso=last_tx.timestamp_iso,
                rule_score=round(score, 6),
                involved_tx_ids=tuple(t.tx_id for t in cyc.transactions),
                involved_accounts=cyc.account_path,
                graph_path=cyc.account_path,
                reasons=(reason,),
                metrics={
                    "hops": hops,
                    "elapsed_hours": cyc.elapsed_hours,
                    "return_ratio": cyc.value_preservation_ratio,
                },
            )
        )
    return hits


def _detect_mule_fan_in_fan_out(
    transactions: Sequence[NormalizedTransaction],
    accounts_by_id: Dict[str, Account],
    rolling_features: Dict[Tuple[str, str], RollingFeatureVector],
    cfg: Dict[str, object],
) -> List[ScenarioHit]:
    window_h = float(cfg.get("window_hours", 24.0))
    min_senders = int(cfg.get("min_distinct_senders", 4))
    min_inbound = float(cfg.get("min_inbound_total_usd", 15000.0))
    min_pt = float(cfg.get("min_passthrough_ratio", 0.80))
    max_pt = float(cfg.get("max_passthrough_ratio", 1.15))
    max_out_recips = int(cfg.get("max_outbound_recipients", 2))
    exempt_types = {str(t).upper() for t in cfg.get("exempt_account_types", [])}
    base_weight = float(cfg.get("weight", 0.82))

    txs_by_account: Dict[str, List[NormalizedTransaction]] = defaultdict(list)
    for tx in transactions:
        txs_by_account[tx.source_account_id].append(tx)
        txs_by_account[tx.target_account_id].append(tx)

    best_by_account_day: Dict[Tuple[str, str], ScenarioHit] = {}
    for tx in transactions:
        # Check on outbound consolidation event from potential mule hub
        hub_id = tx.source_account_id
        acc = accounts_by_id.get(hub_id)
        if acc and acc.account_type in exempt_types:
            continue

        fv = rolling_features.get((tx.tx_id, "SOURCE"))
        if fv is None:
            continue
        if (
            fv.unique_inbound_senders_24h < min_senders
            or fv.inbound_sum_24h_usd < min_inbound
            or not (min_pt <= fv.passthrough_ratio_24h <= max_pt)
            or fv.unique_outbound_recipients_24h == 0
            or fv.unique_outbound_recipients_24h > max_out_recips
        ):
            continue

        cutoff = tx.timestamp_utc - timedelta(hours=window_h)
        window_txs = [
            item
            for item in txs_by_account[hub_id]
            if cutoff <= item.timestamp_utc <= tx.timestamp_utc
        ]
        involved_txs = tuple(sorted({item.tx_id for item in window_txs}))
        senders = sorted({item.source_account_id for item in window_txs if item.target_account_id == hub_id})
        recipients = sorted({item.target_account_id for item in window_txs if item.source_account_id == hub_id})
        involved_accs = tuple(sorted({hub_id} | set(senders) | set(recipients)))

        score = min(1.0, base_weight + 0.02 * max(0, fv.unique_inbound_senders_24h - min_senders))
        reason = ReasonContribution(
            code="GRAPH_MULE_CONSOLIDATION_PASSTHROUGH",
            category="GRAPH",
            weight=score,
            description=(
                f"Account {hub_id} collected ${fv.inbound_sum_24h_usd:,.2f} USD from "
                f"{fv.unique_inbound_senders_24h} distinct senders and passed through "
                f"{fv.passthrough_ratio_24h:.1%} to {fv.unique_outbound_recipients_24h} "
                f"recipient(s) within {window_h:.0f}h."
            ),
            evidence_metrics={
                "unique_inbound_senders_24h": fv.unique_inbound_senders_24h,
                "inbound_sum_24h_usd": round(fv.inbound_sum_24h_usd, 2),
                "outbound_sum_24h_usd": round(fv.outbound_sum_24h_usd, 2),
                "passthrough_ratio_24h": fv.passthrough_ratio_24h,
                "unique_outbound_recipients_24h": fv.unique_outbound_recipients_24h,
            },
        )
        hit = ScenarioHit(
            scenario_name="MULE_FAN_IN_FAN_OUT",
            subject_account_id=hub_id,
            subject_entity_id=tx.source_entity_id,
            trigger_timestamp_iso=tx.timestamp_iso,
            rule_score=round(score, 6),
            involved_tx_ids=involved_txs,
            involved_accounts=involved_accs,
            graph_path=tuple([senders[0], hub_id, recipients[0]]) if senders and recipients else (hub_id,),
            reasons=(reason,),
            metrics={
                "unique_inbound_senders_24h": fv.unique_inbound_senders_24h,
                "passthrough_ratio_24h": fv.passthrough_ratio_24h,
            },
        )
        key = (hub_id, tx.timestamp_iso[:10])
        prev = best_by_account_day.get(key)
        if prev is None or len(hit.involved_tx_ids) >= len(prev.involved_tx_ids):
            best_by_account_day[key] = hit

    return list(best_by_account_day.values())


def _detect_dormant_spike(
    transactions: Sequence[NormalizedTransaction],
    accounts_by_id: Dict[str, Account],
    rolling_features: Dict[Tuple[str, str], RollingFeatureVector],
    cfg: Dict[str, object],
) -> List[ScenarioHit]:
    min_dormancy = float(cfg.get("min_dormancy_days", 30.0))
    min_burst = float(cfg.get("min_burst_amount_usd", 20000.0))
    min_ratio = float(cfg.get("min_burst_ratio_vs_baseline", 5.0))
    exempt_types = {str(t).upper() for t in cfg.get("exempt_account_types", [])}
    base_weight = float(cfg.get("weight", 0.65))

    hits: List[ScenarioHit] = []
    seen: set[Tuple[str, str]] = set()
    for tx in transactions:
        for role, acc_id, ent_id in (
            ("TARGET", tx.target_account_id, tx.target_entity_id),
            ("SOURCE", tx.source_account_id, tx.source_entity_id),
        ):
            acc = accounts_by_id.get(acc_id)
            if acc and acc.account_type in exempt_types:
                continue
            fv = rolling_features.get((tx.tx_id, role))
            if fv is None:
                continue
            if (
                fv.dormancy_days_prior >= min_dormancy
                and tx.amount_usd >= min_burst
                and fv.velocity_ratio_vs_baseline >= min_ratio
            ):
                key = (acc_id, tx.tx_id)
                if key in seen:
                    continue
                seen.add(key)
                score = min(1.0, base_weight + min(0.20, 0.01 * (fv.velocity_ratio_vs_baseline - min_ratio)))
                reason = ReasonContribution(
                    code="RULE_DORMANT_ACCOUNT_SPIKE",
                    category="RULE",
                    weight=score,
                    description=(
                        f"Account {acc_id} was inactive for {fv.dormancy_days_prior:.1f} days "
                        f"before a ${tx.amount_usd:,.2f} USD transfer ({fv.velocity_ratio_vs_baseline:.1f}x baseline)."
                    ),
                    evidence_metrics={
                        "dormancy_days_prior": round(fv.dormancy_days_prior, 2),
                        "amount_usd": round(tx.amount_usd, 2),
                        "velocity_ratio_vs_baseline": round(fv.velocity_ratio_vs_baseline, 2),
                    },
                )
                hits.append(
                    ScenarioHit(
                        scenario_name="DORMANT_ACCOUNT_SPIKE",
                        subject_account_id=acc_id,
                        subject_entity_id=ent_id,
                        trigger_timestamp_iso=tx.timestamp_iso,
                        rule_score=round(score, 6),
                        involved_tx_ids=(tx.tx_id,),
                        involved_accounts=(tx.source_account_id, tx.target_account_id),
                        graph_path=(tx.source_account_id, tx.target_account_id),
                        reasons=(reason,),
                        metrics={
                            "dormancy_days_prior": round(fv.dormancy_days_prior, 2),
                            "velocity_ratio_vs_baseline": round(fv.velocity_ratio_vs_baseline, 2),
                        },
                    )
                )
    return hits
