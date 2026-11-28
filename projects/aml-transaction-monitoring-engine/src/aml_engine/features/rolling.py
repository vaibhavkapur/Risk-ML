"""Point-in-time rolling temporal aggregation features (pure, I/O-free).

All rolling windows are strictly inclusive on [t - W, t] (no future leakage)
where transactions are processed in chronological order (timestamp_utc, tx_id).
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Deque, Dict, List, Sequence, Tuple

from aml_engine.config import EngineConfig
from aml_engine.models import Account, NormalizedTransaction, RollingFeatureVector


class _AccountRollingState:
    """Maintains chronological inbound/outbound deques up to 30 days."""

    def __init__(self) -> None:
        self.inbound_30d: Deque[NormalizedTransaction] = deque()
        self.outbound_30d: Deque[NormalizedTransaction] = deque()
        self.last_activity_ts: datetime | None = None

    def evict_older_than(self, cutoff: datetime) -> None:
        while self.inbound_30d and self.inbound_30d[0].timestamp_utc < cutoff:
            self.inbound_30d.popleft()
        while self.outbound_30d and self.outbound_30d[0].timestamp_utc < cutoff:
            self.outbound_30d.popleft()


def compute_rolling_features(
    transactions: Sequence[NormalizedTransaction],
    accounts_by_id: Dict[str, Account],
    config: EngineConfig,
) -> Dict[Tuple[str, str], RollingFeatureVector]:
    """Compute point-in-time rolling feature vectors for (tx_id, role).

    Returns a dictionary keyed by (tx_id, "SOURCE") and (tx_id, "TARGET").
    """
    windows = config.rolling_windows_hours
    w_1h = timedelta(hours=windows.get("short", 1.0))
    w_24h = timedelta(hours=windows.get("day", 24.0))
    w_7d = timedelta(hours=windows.get("week", 168.0))
    w_30d = timedelta(hours=windows.get("month", 720.0))

    str_cfg = config.scenarios.get("structuring", {})
    band_min = float(str_cfg.get("band_min_usd", 7500.0))
    band_max = float(str_cfg.get("band_max_usd", 9999.99))
    str_channels = {str(c).upper() for c in str_cfg.get("channels", [])}

    ordered = sorted(transactions, key=lambda t: (t.timestamp_utc, t.tx_id))
    states: Dict[str, _AccountRollingState] = defaultdict(_AccountRollingState)
    results: Dict[Tuple[str, str], RollingFeatureVector] = {}

    def _snapshot(
        tx: NormalizedTransaction,
        account_id: str,
        role: str,
        prior_dormancy_days: float,
    ) -> RollingFeatureVector:
        st = states[account_id]
        now = tx.timestamp_utc
        cutoff_1h = now - w_1h
        cutoff_24h = now - w_24h
        cutoff_7d = now - w_7d
        cutoff_30d = now - w_30d
        st.evict_older_than(cutoff_30d)

        in_1h_c = in_24h_c = in_7d_c = 0
        in_1h_s = in_24h_s = in_7d_s = in_30d_s = 0.0
        str_c = 0
        str_s = 0.0
        senders_24h = set()

        for item in st.inbound_30d:
            amt = item.amount_usd
            in_30d_s += amt
            if item.timestamp_utc >= cutoff_7d:
                in_7d_c += 1
                in_7d_s += amt
            if item.timestamp_utc >= cutoff_24h:
                in_24h_c += 1
                in_24h_s += amt
                senders_24h.add(item.source_account_id)
                if (
                    band_min <= amt <= band_max
                    and (not str_channels or item.channel in str_channels)
                ):
                    str_c += 1
                    str_s += amt
            if item.timestamp_utc >= cutoff_1h:
                in_1h_c += 1
                in_1h_s += amt

        out_1h_c = out_24h_c = out_7d_c = 0
        out_1h_s = out_24h_s = out_7d_s = out_30d_s = 0.0
        recipients_24h = set()

        for item in st.outbound_30d:
            amt = item.amount_usd
            out_30d_s += amt
            if item.timestamp_utc >= cutoff_7d:
                out_7d_c += 1
                out_7d_s += amt
            if item.timestamp_utc >= cutoff_24h:
                out_24h_c += 1
                out_24h_s += amt
                recipients_24h.add(item.target_account_id)
            if item.timestamp_utc >= cutoff_1h:
                out_1h_c += 1
                out_1h_s += amt

        passthrough = (out_24h_s / in_24h_s) if in_24h_s > 0.0 else 0.0

        acc = accounts_by_id.get(account_id)
        baseline_monthly = acc.baseline_monthly_usd if acc and acc.baseline_monthly_usd > 0 else 5000.0
        total_24h_vol = max(in_24h_s, out_24h_s)
        velocity_ratio = total_24h_vol / baseline_monthly

        return RollingFeatureVector(
            tx_id=tx.tx_id,
            account_id=account_id,
            role=role,
            as_of_iso=tx.timestamp_iso,
            feature_version=config.feature_version,
            inbound_count_1h=in_1h_c,
            inbound_sum_1h_usd=round(in_1h_s, 6),
            inbound_count_24h=in_24h_c,
            inbound_sum_24h_usd=round(in_24h_s, 6),
            inbound_count_7d=in_7d_c,
            inbound_sum_7d_usd=round(in_7d_s, 6),
            inbound_sum_30d_usd=round(in_30d_s, 6),
            outbound_count_1h=out_1h_c,
            outbound_sum_1h_usd=round(out_1h_s, 6),
            outbound_count_24h=out_24h_c,
            outbound_sum_24h_usd=round(out_24h_s, 6),
            outbound_count_7d=out_7d_c,
            outbound_sum_7d_usd=round(out_7d_s, 6),
            outbound_sum_30d_usd=round(out_30d_s, 6),
            structuring_band_count_24h=str_c,
            structuring_band_sum_24h_usd=round(str_s, 6),
            unique_inbound_senders_24h=len(senders_24h),
            unique_outbound_recipients_24h=len(recipients_24h),
            passthrough_ratio_24h=round(passthrough, 6),
            velocity_ratio_vs_baseline=round(velocity_ratio, 6),
            dormancy_days_prior=round(prior_dormancy_days, 6),
        )

    for tx in ordered:
        src_state = states[tx.source_account_id]
        tgt_state = states[tx.target_account_id]

        src_dormancy = (
            (tx.timestamp_utc - src_state.last_activity_ts).total_seconds() / 86400.0
            if src_state.last_activity_ts is not None
            else 0.0
        )
        tgt_dormancy = (
            (tx.timestamp_utc - tgt_state.last_activity_ts).total_seconds() / 86400.0
            if tgt_state.last_activity_ts is not None
            else 0.0
        )

        src_state.outbound_30d.append(tx)
        tgt_state.inbound_30d.append(tx)
        src_state.last_activity_ts = tx.timestamp_utc
        tgt_state.last_activity_ts = tx.timestamp_utc

        results[(tx.tx_id, "SOURCE")] = _snapshot(tx, tx.source_account_id, "SOURCE", src_dormancy)
        results[(tx.tx_id, "TARGET")] = _snapshot(tx, tx.target_account_id, "TARGET", tgt_dormancy)

    return results
