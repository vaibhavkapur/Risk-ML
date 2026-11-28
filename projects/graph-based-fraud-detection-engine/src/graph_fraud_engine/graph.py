"""Time-aware heterogeneous entity graph with strict no-future-edge guarantees.

Entities (nodes):
- Accounts (acc_*)
- Devices (dev_*)
- Merchants (mer_*)

Timestamped relations (edges):
- USES_DEVICE: account <-> device (tagged with tx_id, tx_timestamp, amount_usd)
- TRANSACTS_AT: account <-> merchant (tagged with tx_id, tx_timestamp, amount_usd)
- OBSERVED_AT: device <-> merchant (tagged with tx_id, tx_timestamp, amount_usd)

Causality invariant:
Every historical query at time `as_of` strictly enforces:
- Prior edges: `edge.timestamp < as_of` (and `edge.tx_id != exclude_tx_id`)
- Known fraud labels: `label.label_available_at <= as_of` AND `label.tx_timestamp < as_of`
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx

from graph_fraud_engine.schemas import (
    AccountRecord,
    DatasetBundle,
    DeviceRecord,
    FraudLabelRecord,
    MerchantRecord,
    TemporalLeakageError,
    TransactionRecord,
)


@dataclass(frozen=True)
class TemporalEdge:
    src: str
    dst: str
    relation: str
    timestamp: datetime
    tx_id: str
    amount_usd: float


@dataclass(frozen=True)
class MaturedLabelIndexEntry:
    tx_id: str
    account_id: str
    device_id: str
    merchant_id: str
    is_fraud: int
    tx_timestamp: datetime
    label_available_at: datetime
    fraud_pattern: Optional[str]


class TemporalEntityGraph:
    """In-memory time-aware heterogeneous transaction graph with leak-free queries."""

    def __init__(self) -> None:
        self.accounts: Dict[str, AccountRecord] = {}
        self.devices: Dict[str, DeviceRecord] = {}
        self.merchants: Dict[str, MerchantRecord] = {}
        self.transactions: Dict[str, TransactionRecord] = {}

        # Adjacency lists indexed by node_id -> sorted list of TemporalEdge
        self._edges_by_node: Dict[str, List[TemporalEdge]] = defaultdict(list)
        # Directed unique edges (one per relation per transaction) for NetworkX snapshots
        self._all_edges: List[TemporalEdge] = []

        # Labels indexed by entity for fast matured-label lookup
        self._labels_by_tx: Dict[str, MaturedLabelIndexEntry] = {}
        self._labels_by_entity: Dict[str, List[MaturedLabelIndexEntry]] = defaultdict(list)

        # First observed transaction timestamp per entity
        self._first_tx_ts: Dict[str, datetime] = {}

    @classmethod
    def from_dataset(cls, bundle: DatasetBundle) -> "TemporalEntityGraph":
        graph = cls()
        for acc in bundle.accounts:
            graph.add_account(acc)
        for dev in bundle.devices:
            graph.add_device(dev)
        for mer in bundle.merchants:
            graph.add_merchant(mer)
        for tx in sorted(bundle.transactions, key=lambda t: (t.tx_timestamp, t.tx_id)):
            graph.add_transaction(tx)
        for lbl in bundle.labels:
            graph.add_label(lbl)
        graph.finalize_indices()
        return graph

    def add_account(self, account: AccountRecord) -> None:
        self.accounts[account.account_id] = account

    def add_device(self, device: DeviceRecord) -> None:
        self.devices[device.device_id] = device

    def add_merchant(self, merchant: MerchantRecord) -> None:
        self.merchants[merchant.merchant_id] = merchant

    def add_transaction(self, tx: TransactionRecord) -> None:
        if tx.tx_id in self.transactions:
            return
        self.transactions[tx.tx_id] = tx

        triples = [
            (tx.account_id, tx.device_id, "USES_DEVICE"),
            (tx.account_id, tx.merchant_id, "TRANSACTS_AT"),
            (tx.device_id, tx.merchant_id, "OBSERVED_AT"),
        ]
        for u, v, rel in triples:
            edge = TemporalEdge(
                src=u,
                dst=v,
                relation=rel,
                timestamp=tx.tx_timestamp,
                tx_id=tx.tx_id,
                amount_usd=tx.amount_usd,
            )
            self._all_edges.append(edge)
            self._edges_by_node[u].append(edge)
            self._edges_by_node[v].append(edge)

        for eid in (tx.account_id, tx.device_id, tx.merchant_id):
            prev = self._first_tx_ts.get(eid)
            if prev is None or tx.tx_timestamp < prev:
                self._first_tx_ts[eid] = tx.tx_timestamp

    def add_label(self, label: FraudLabelRecord) -> None:
        tx = self.transactions.get(label.tx_id)
        if tx is None:
            return
        entry = MaturedLabelIndexEntry(
            tx_id=label.tx_id,
            account_id=tx.account_id,
            device_id=tx.device_id,
            merchant_id=tx.merchant_id,
            is_fraud=label.is_fraud,
            tx_timestamp=label.tx_timestamp,
            label_available_at=label.label_available_at,
            fraud_pattern=label.fraud_pattern,
        )
        self._labels_by_tx[label.tx_id] = entry
        for eid in (tx.account_id, tx.device_id, tx.merchant_id):
            self._labels_by_entity[eid].append(entry)

    def finalize_indices(self) -> None:
        """Sort edge and label lists chronologically for deterministic queries."""
        self._all_edges.sort(key=lambda e: (e.timestamp, e.tx_id, e.relation))
        for eid in self._edges_by_node:
            self._edges_by_node[eid].sort(key=lambda e: (e.timestamp, e.tx_id, e.relation))
        for eid in self._labels_by_entity:
            self._labels_by_entity[eid].sort(key=lambda l: (l.label_available_at, l.tx_timestamp, l.tx_id))

    @staticmethod
    def entity_type(entity_id: str) -> str:
        if entity_id.startswith("acc_"):
            return "account"
        if entity_id.startswith("dev_"):
            return "device"
        if entity_id.startswith("mer_"):
            return "merchant"
        return "unknown"

    def is_unseen_prior_to(self, entity_id: str, as_of: datetime) -> bool:
        """Return True if entity has zero transaction edges strictly before `as_of`."""
        first_ts = self._first_tx_ts.get(entity_id)
        return first_ts is None or first_ts >= as_of

    def prior_edges(
        self,
        entity_id: str,
        as_of: datetime,
        relation: Optional[str] = None,
        window: Optional[timedelta] = None,
        exclude_tx_id: Optional[str] = None,
    ) -> List[TemporalEdge]:
        """Return all incident edges with timestamp strictly < as_of (and within optional window)."""
        min_ts = (as_of - window) if window is not None else None
        out: List[TemporalEdge] = []
        for edge in self._edges_by_node.get(entity_id, ()):
            if edge.timestamp >= as_of:
                # Edges are sorted chronologically, but check safely even before finalize_indices.
                continue
            if min_ts is not None and edge.timestamp < min_ts:
                continue
            if relation is not None and edge.relation != relation:
                continue
            if exclude_tx_id is not None and edge.tx_id == exclude_tx_id:
                continue
            out.append(edge)
        return out

    def prior_neighbors(
        self,
        entity_id: str,
        as_of: datetime,
        relation: Optional[str] = None,
        window: Optional[timedelta] = None,
        exclude_tx_id: Optional[str] = None,
    ) -> Set[str]:
        """Return distinct neighbor entity IDs connected strictly before `as_of`."""
        neighbors: Set[str] = set()
        for edge in self.prior_edges(
            entity_id=entity_id,
            as_of=as_of,
            relation=relation,
            window=window,
            exclude_tx_id=exclude_tx_id,
        ):
            other = edge.dst if edge.src == entity_id else edge.src
            neighbors.add(other)
        return neighbors

    def matured_labels_for_entity(
        self,
        entity_id: str,
        as_of: datetime,
        exclude_tx_id: Optional[str] = None,
    ) -> List[MaturedLabelIndexEntry]:
        """Return labels whose `label_available_at <= as_of` and `tx_timestamp < as_of`."""
        out: List[MaturedLabelIndexEntry] = []
        for entry in self._labels_by_entity.get(entity_id, ()):
            if entry.tx_timestamp >= as_of:
                continue
            if entry.label_available_at > as_of:
                continue
            if exclude_tx_id is not None and entry.tx_id == exclude_tx_id:
                continue
            out.append(entry)
        return out

    def matured_fraud_count(
        self,
        entity_id: str,
        as_of: datetime,
        exclude_tx_id: Optional[str] = None,
    ) -> int:
        return sum(
            1
            for entry in self.matured_labels_for_entity(
                entity_id=entity_id, as_of=as_of, exclude_tx_id=exclude_tx_id
            )
            if entry.is_fraud == 1
        )

    def label_for_tx(self, tx_id: str) -> Optional[MaturedLabelIndexEntry]:
        return self._labels_by_tx.get(tx_id)

    def is_label_matured_at(self, tx_id: str, cutoff: datetime) -> bool:
        entry = self._labels_by_tx.get(tx_id)
        if entry is None:
            return False
        return entry.label_available_at <= cutoff

    def prior_account_tx_edges(
        self,
        account_id: str,
        as_of: datetime,
        window: Optional[timedelta] = None,
        exclude_tx_id: Optional[str] = None,
    ) -> List[TemporalEdge]:
        """Each transaction generates exactly one USES_DEVICE edge for the account."""
        return self.prior_edges(
            entity_id=account_id,
            as_of=as_of,
            relation="USES_DEVICE",
            window=window,
            exclude_tx_id=exclude_tx_id,
        )

    def build_historical_nx_subgraph(
        self,
        seed_entities: Iterable[str],
        as_of: datetime,
        max_hops: int = 2,
        window: Optional[timedelta] = None,
        exclude_tx_id: Optional[str] = None,
    ) -> nx.Graph:
        """Build a NetworkX undirected projection of the k-hop neighborhood strictly before `as_of`.

        Every edge retains its latest historical timestamp (`< as_of`), relation, and `tx_id`.
        """
        G = nx.Graph()
        frontier: Set[str] = set(seed_entities)
        visited: Set[str] = set(seed_entities)
        for node in seed_entities:
            G.add_node(node, entity_type=self.entity_type(node))

        for _ in range(max_hops):
            next_frontier: Set[str] = set()
            for node in sorted(frontier):
                for edge in self.prior_edges(
                    entity_id=node,
                    as_of=as_of,
                    window=window,
                    exclude_tx_id=exclude_tx_id,
                ):
                    if edge.timestamp >= as_of:
                        raise TemporalLeakageError(
                            f"Future edge detected in historical subgraph: {edge.tx_id} at {edge.timestamp} >= {as_of}"
                        )
                    other = edge.dst if edge.src == node else edge.src
                    if other not in G:
                        G.add_node(other, entity_type=self.entity_type(other))
                    if G.has_edge(node, other):
                        existing = G[node][other]
                        existing["count"] += 1
                        if edge.timestamp > existing["timestamp"]:
                            existing["timestamp"] = edge.timestamp
                            existing["tx_id"] = edge.tx_id
                            existing["relation"] = edge.relation
                    else:
                        G.add_edge(
                            node,
                            other,
                            relation=edge.relation,
                            timestamp=edge.timestamp,
                            tx_id=edge.tx_id,
                            count=1,
                        )
                    if other not in visited:
                        visited.add(other)
                        next_frontier.add(other)
            frontier = next_frontier
            if not frontier:
                break

        return G

    def max_edge_timestamp_in_nx(self, G: nx.Graph) -> Optional[datetime]:
        """Audit helper returning the maximum edge timestamp present in a NetworkX subgraph."""
        timestamps = [data["timestamp"] for _, _, data in G.edges(data=True) if "timestamp" in data]
        return max(timestamps) if timestamps else None

    def shared_device_accounts_prior(
        self,
        device_id: str,
        account_id: str,
        as_of: datetime,
        window: Optional[timedelta] = None,
        exclude_tx_id: Optional[str] = None,
    ) -> Set[str]:
        """Other accounts that used `device_id` strictly before `as_of`."""
        users = self.prior_neighbors(
            entity_id=device_id,
            as_of=as_of,
            relation="USES_DEVICE",
            window=window,
            exclude_tx_id=exclude_tx_id,
        )
        users.discard(account_id)
        return users

    def compute_two_hop_account_ring(
        self,
        account_id: str,
        device_id: str,
        as_of: datetime,
        window: Optional[timedelta] = None,
        exclude_tx_id: Optional[str] = None,
    ) -> Tuple[Set[str], Set[str]]:
        """Return (co_accounts, linked_devices) connected via 1-2 hops of USES_DEVICE strictly before `as_of`."""
        # Start from both the account's prior devices AND the transaction's device
        prior_devices = self.prior_neighbors(
            entity_id=account_id,
            as_of=as_of,
            relation="USES_DEVICE",
            window=window,
            exclude_tx_id=exclude_tx_id,
        )
        candidate_devices = set(prior_devices) | {device_id}

        co_accounts: Set[str] = set()
        for dev in candidate_devices:
            dev_accounts = self.prior_neighbors(
                entity_id=dev,
                as_of=as_of,
                relation="USES_DEVICE",
                window=window,
                exclude_tx_id=exclude_tx_id,
            )
            co_accounts.update(dev_accounts)
        co_accounts.discard(account_id)

        second_hop_devices: Set[str] = set(candidate_devices)
        for co_acc in co_accounts:
            second_hop_devices.update(
                self.prior_neighbors(
                    entity_id=co_acc,
                    as_of=as_of,
                    relation="USES_DEVICE",
                    window=window,
                    exclude_tx_id=exclude_tx_id,
                )
            )
        return co_accounts, second_hop_devices
