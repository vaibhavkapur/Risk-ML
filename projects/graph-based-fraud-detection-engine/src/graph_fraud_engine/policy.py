"""Decision policy mapping calibrated probabilities and graph context to actions."""

from __future__ import annotations

import hashlib
from typing import Dict

from graph_fraud_engine.config import PolicyConfig
from graph_fraud_engine.schemas import DECISION_ACTIONS, GraphExplanation


def make_stable_decision_id(
    tx_id: str,
    model_name: str,
    model_version: str,
    feature_version: str,
    calibrated_probability: float,
) -> str:
    """Create a deterministic decision identifier for reproducible audit trails."""
    payload = f"{tx_id}|{model_name}|{model_version}|{feature_version}|{calibrated_probability:.6f}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:14]
    return f"dec_{digest}"


def evaluate_policy(
    calibrated_probability: float,
    cold_start_flags: Dict[str, bool],
    explanation: GraphExplanation,
    policy_config: PolicyConfig,
) -> DECISION_ACTIONS:
    """Map calibrated probability and cold-start/graph flags to approve | review | block."""
    any_cold_start = any(cold_start_flags.values())

    # Block requires crossing block_threshold.
    # For pure cold-start (both account and device completely unseen), route to analyst review
    # rather than automatic hard block unless probability is extreme (>= 0.90).
    both_unseen = cold_start_flags.get("account_unseen", False) and cold_start_flags.get(
        "device_unseen", False
    )
    if calibrated_probability >= policy_config.block_threshold:
        if both_unseen and calibrated_probability < 0.90:
            return "review"
        return "block"

    review_thresh = (
        policy_config.cold_start_review_threshold
        if any_cold_start and not explanation.legitimate_shared_device_mitigant
        else policy_config.review_threshold
    )
    if calibrated_probability >= review_thresh:
        return "review"

    return "approve"
