"""Directed temporal transaction graph and NetworkX-compatible graph feature extractor.

Pure domain module with zero mandatory third-party dependencies. Provides an
optional `to_networkx()` adapter when `networkx` is installed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Sequence, Set, Tuple

from aml_engine.models import GraphFeatureVector, NormalizedTransaction


@dataclass(frozen=True)
class TemporalChain:
    transactions: Tuple[NormalizedTransaction, ...]
    account_path: Tuple[str, ...]
    elapsed_hours: float
    value_preservation_ratio: float


class TransactionGraph:
    """Directed temporal multi-graph of normalized financial transfers."""

    def __init__(self, transactions: Sequence[NormalizedTransaction]) -> None:
        self.transactions: List[NormalizedTransaction] = sorted(
            transactions, key=lambda t: (t.timestamp_utc, t.tx_id)
        )
        self.nodes: Set[str] = set()
        self.out_edges: Dict[str, List[NormalizedTransaction]] = defaultdict(list)
        self.in_edges: Dict[str, List[NormalizedTransaction]] = defaultdict(list)
        self.account_entities: Dict[str, str] = {}

        for tx in self.transactions:
            self.nodes.add(tx.source_account_id)
            self.nodes.add(tx.target_account_id)
            self.out_edges[tx.source_account_id].append(tx)
            self.in_edges[tx.target_account_id].append(tx)
            if tx.source_entity_id:
                self.account_entities[tx.source_account_id] = tx.source_entity_id
            if tx.target_entity_id:
                self.account_entities[tx.target_account_id] = tx.target_entity_id

    def to_networkx(self) -> Any:
        """Convert this graph to a networkx.MultiDiGraph if networkx is installed."""
        try:
            import networkx as nx  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Optional dependency 'networkx' is not installed. "
                "Install with: pip install -r requirements-optional.txt"
            ) from exc

        g = nx.MultiDiGraph()
        for acc in sorted(self.nodes):
            g.add_node(acc, entity_id=self.account_entities.get(acc))
        for tx in self.transactions:
            g.add_edge(
                tx.source_account_id,
                tx.target_account_id,
                key=tx.tx_id,
                amount_usd=tx.amount_usd,
                timestamp_iso=tx.timestamp_iso,
                channel=tx.channel,
            )
        return g

    def find_rapid_layering_chains(
        self,
        min_hops: int = 3,
        max_hops: int = 5,
        max_window_hours: float = 12.0,
        min_initial_amount_usd: float = 10000.0,
        min_preservation_ratio: float = 0.80,
        max_preservation_ratio: float = 1.02,
    ) -> List[TemporalChain]:
        """Find simple directed temporal transfer chains preserving value rapidly."""
        max_delta = timedelta(hours=max_window_hours)
        found: List[TemporalChain] = []

        def dfs(
            start_tx: NormalizedTransaction,
            path_txs: List[NormalizedTransaction],
            visited: Set[str],
        ) -> None:
            last_tx = path_txs[-1]
            hops = len(path_txs)
            extended = False

            if hops < max_hops:
                for nxt in self.out_edges.get(last_tx.target_account_id, []):
                    if nxt.timestamp_utc < last_tx.timestamp_utc:
                        continue
                    if nxt.timestamp_utc - start_tx.timestamp_utc > max_delta:
                        break
                    if nxt.target_account_id in visited:
                        continue
                    hop_ratio = nxt.amount_usd / last_tx.amount_usd
                    if not (min_preservation_ratio <= hop_ratio <= max_preservation_ratio):
                        continue
                    extended = True
                    visited.add(nxt.target_account_id)
                    path_txs.append(nxt)
                    dfs(start_tx, path_txs, visited)
                    path_txs.pop()
                    visited.remove(nxt.target_account_id)

            if hops >= min_hops and not extended:
                total_ratio = last_tx.amount_usd / start_tx.amount_usd
                if total_ratio >= min_preservation_ratio:
                    elapsed_h = (
                        last_tx.timestamp_utc - start_tx.timestamp_utc
                    ).total_seconds() / 3600.0
                    acc_path = tuple(
                        [path_txs[0].source_account_id] + [t.target_account_id for t in path_txs]
                    )
                    found.append(
                        TemporalChain(
                            transactions=tuple(path_txs),
                            account_path=acc_path,
                            elapsed_hours=round(elapsed_h, 6),
                            value_preservation_ratio=round(total_ratio, 6),
                        )
                    )

        for tx in self.transactions:
            if tx.amount_usd < min_initial_amount_usd:
                continue
            dfs(tx, [tx], {tx.source_account_id, tx.target_account_id})

        # Keep only maximal chains (exclude sub-chains strictly contained in a longer chain)
        found.sort(key=lambda c: (-len(c.transactions), c.transactions[0].timestamp_utc, c.transactions[0].tx_id))
        maximal: List[TemporalChain] = []
        seen_tx_sets: List[Set[str]] = []
        for chain in found:
            tx_set = {t.tx_id for t in chain.transactions}
            if any(tx_set.issubset(existing) for existing in seen_tx_sets):
                continue
            seen_tx_sets.append(tx_set)
            maximal.append(chain)

        maximal.sort(key=lambda c: (c.transactions[-1].timestamp_utc, c.transactions[0].tx_id))
        return maximal

    def find_round_trip_cycles(
        self,
        min_hops: int = 3,
        max_hops: int = 5,
        max_window_hours: float = 48.0,
        min_amount_usd: float = 10000.0,
        min_return_ratio: float = 0.75,
    ) -> List[TemporalChain]:
        """Find directed temporal cycles returning funds to origin account or origin entity."""
        max_delta = timedelta(hours=max_window_hours)
        raw_cycles: Dict[Tuple[str, ...], TemporalChain] = {}

        def dfs(
            start_tx: NormalizedTransaction,
            path_txs: List[NormalizedTransaction],
            intermediate_visited: Set[str],
        ) -> None:
            last_tx = path_txs[-1]
            hops = len(path_txs)
            if hops >= max_hops:
                return

            origin_acc = start_tx.source_account_id
            origin_ent = start_tx.source_entity_id

            for nxt in self.out_edges.get(last_tx.target_account_id, []):
                if nxt.timestamp_utc < last_tx.timestamp_utc:
                    continue
                if nxt.timestamp_utc - start_tx.timestamp_utc > max_delta:
                    break
                if nxt.amount_usd < min_amount_usd * min_return_ratio:
                    continue

                hop_ratio = nxt.amount_usd / last_tx.amount_usd
                if hop_ratio < min_return_ratio or hop_ratio > 1.10:
                    continue

                is_return_to_origin = nxt.target_account_id == origin_acc or (
                    origin_ent is not None
                    and nxt.target_entity_id == origin_ent
                    and nxt.target_account_id != last_tx.target_account_id
                )

                if is_return_to_origin and (hops + 1) >= min_hops:
                    full_txs = tuple(path_txs + [nxt])
                    return_ratio = nxt.amount_usd / start_tx.amount_usd
                    if return_ratio >= min_return_ratio:
                        elapsed_h = (
                            nxt.timestamp_utc - start_tx.timestamp_utc
                        ).total_seconds() / 3600.0
                        acc_path = tuple(
                            [full_txs[0].source_account_id] + [t.target_account_id for t in full_txs]
                        )
                        key = tuple(sorted(t.tx_id for t in full_txs))
                        if key not in raw_cycles:
                            raw_cycles[key] = TemporalChain(
                                transactions=full_txs,
                                account_path=acc_path,
                                elapsed_hours=round(elapsed_h, 6),
                                value_preservation_ratio=round(return_ratio, 6),
                            )
                elif nxt.target_account_id not in intermediate_visited and nxt.target_account_id != origin_acc:
                    intermediate_visited.add(nxt.target_account_id)
                    path_txs.append(nxt)
                    dfs(start_tx, path_txs, intermediate_visited)
                    path_txs.pop()
                    intermediate_visited.remove(nxt.target_account_id)

        for tx in self.transactions:
            if tx.amount_usd < min_amount_usd:
                continue
            dfs(tx, [tx], {tx.target_account_id})

        cycles = list(raw_cycles.values())
        cycles.sort(key=lambda c: (c.transactions[-1].timestamp_utc, c.transactions[0].tx_id))
        return cycles

    def compute_account_graph_features(
        self,
        layering_chains: Sequence[TemporalChain],
        round_trip_cycles: Sequence[TemporalChain],
    ) -> Dict[str, GraphFeatureVector]:
        """Compute structural and flow-topology graph features per account."""
        layering_accounts: Set[str] = set()
        for chain in layering_chains:
            layering_accounts.update(chain.account_path)

        cycle_accounts: Set[str] = set()
        for cyc in round_trip_cycles:
            cycle_accounts.update(cyc.account_path)

        # Precompute undirected neighbor adjacency for ego density
        neighbors_map: Dict[str, Set[str]] = defaultdict(set)
        directed_pairs: Set[Tuple[str, str]] = set()
        for tx in self.transactions:
            neighbors_map[tx.source_account_id].add(tx.target_account_id)
            neighbors_map[tx.target_account_id].add(tx.source_account_id)
            directed_pairs.add((tx.source_account_id, tx.target_account_id))

        features: Dict[str, GraphFeatureVector] = {}
        for acc in sorted(self.nodes):
            in_txs = self.in_edges.get(acc, [])
            out_txs = self.out_edges.get(acc, [])
            in_neighbors = {t.source_account_id for t in in_txs}
            out_neighbors = {t.target_account_id for t in out_txs}

            total_in = sum(t.amount_usd for t in in_txs)
            total_out = sum(t.amount_usd for t in out_txs)
            denom_flow = total_in + total_out
            net_flow_ratio = abs(total_in - total_out) / denom_flow if denom_flow > 0 else 1.0

            fan_ratio = (
                len(in_neighbors) / max(1, len(out_neighbors))
                if len(in_neighbors) > 0
                else 0.0
            )

            ego_nodes = neighbors_map[acc] | {acc}
            k = len(ego_nodes)
            if k > 1:
                internal_edges = sum(
                    1
                    for u in ego_nodes
                    for v in ego_nodes
                    if u != v and (u, v) in directed_pairs
                )
                ego_density = internal_edges / (k * (k - 1))
            else:
                ego_density = 0.0

            acc_ent = self.account_entities.get(acc)
            shared_ent = 0
            if acc_ent is not None:
                for nb in neighbors_map[acc]:
                    if self.account_entities.get(nb) == acc_ent:
                        shared_ent += 1

            features[acc] = GraphFeatureVector(
                account_id=acc,
                in_degree=len(in_txs),
                out_degree=len(out_txs),
                unique_in_neighbors=len(in_neighbors),
                unique_out_neighbors=len(out_neighbors),
                fan_in_fan_out_ratio=round(fan_ratio, 6),
                total_inbound_usd=round(total_in, 6),
                total_outbound_usd=round(total_out, 6),
                net_flow_ratio=round(net_flow_ratio, 6),
                ego_density=round(ego_density, 6),
                shared_entity_counterparties=shared_ent,
                in_layering_chain=acc in layering_accounts,
                in_round_trip_cycle=acc in cycle_accounts,
            )

        return features
