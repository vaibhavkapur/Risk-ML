"""Manual-review routing policy with monotone PD risk bands.

Strictly decision-support ONLY: automated lending decisions (auto-approve or
auto-decline) are structurally disallowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from credit_risk.config import PolicyConfig, RiskBandConfig
from credit_risk.errors import UnsupportedInputError


@dataclass(frozen=True)
class PolicyRoutingDecision:
    risk_band: str
    band_index: int
    queue_priority: str
    review_sla_hours: int
    recommended_action: str
    automated_decision_allowed: bool
    manual_review_required: bool
    review_checklist: List[str]


class ManualReviewPolicy:
    """Monotone PD-to-review-queue routing policy (no automated lending decisions)."""

    def __init__(self, cfg: PolicyConfig, policy_version: str = "1.0.0") -> None:
        if cfg.automated_decision_allowed:
            raise UnsupportedInputError(
                "Automated lending decisions are prohibited; policy must require manual review."
            )
        self.cfg = cfg
        self.policy_version = policy_version
        self.bands: List[RiskBandConfig] = list(cfg.risk_bands)

    def route(
        self,
        calibrated_pd: float,
        missing_fields: List[str] | None = None,
    ) -> PolicyRoutingDecision:
        if not (0.0 <= calibrated_pd <= 1.0):
            raise ValueError(f"calibrated_pd must be in [0, 1]; got {calibrated_pd}")

        missing = missing_fields or []
        chosen_idx = len(self.bands) - 1
        chosen = self.bands[-1]
        for idx, band in enumerate(self.bands):
            if calibrated_pd <= band.max_pd:
                chosen_idx = idx
                chosen = band
                break

        checklist = [
            "Verify applicant identity and income documentation independently.",
            "Review top positive and negative risk-factor explanations.",
            "Confirm no out-of-band underwriting overrides or policy exceptions apply.",
        ]
        if missing:
            checklist.append(
                f"Request verification for missing origination fields: {', '.join(missing)}."
            )
        if chosen_idx >= 2:
            checklist.append(
                "Perform senior analyst cash-flow and debt-service stress review."
            )

        return PolicyRoutingDecision(
            risk_band=chosen.name,
            band_index=chosen_idx,
            queue_priority=chosen.queue_priority,
            review_sla_hours=chosen.review_sla_hours,
            recommended_action=chosen.recommended_action,
            automated_decision_allowed=False,
            manual_review_required=True,
            review_checklist=checklist,
        )

    def verify_threshold_monotonicity(self, grid_points: int = 500) -> Dict[str, Any]:
        """Verify that risk band severity is non-decreasing and review SLA is non-increasing with PD."""
        grid = np.linspace(0.0, 1.0, grid_points)
        band_indices: List[int] = []
        slas: List[int] = []
        for pd_val in grid:
            decision = self.route(float(pd_val))
            band_indices.append(decision.band_index)
            slas.append(decision.review_sla_hours)

        band_monotone = all(band_indices[i] <= band_indices[i + 1] for i in range(len(band_indices) - 1))
        sla_monotone = all(slas[i] >= slas[i + 1] for i in range(len(slas) - 1))
        if not band_monotone:
            raise ValueError("Policy risk band assignment violated monotonicity.")

        return {
            "policy_version": self.policy_version,
            "automated_decision_allowed": False,
            "band_severity_monotone": band_monotone,
            "sla_urgency_monotone": sla_monotone,
            "grid_points": grid_points,
        }
