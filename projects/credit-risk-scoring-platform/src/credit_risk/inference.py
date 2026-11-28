"""Typed inference service with deterministic decision IDs, replay support, and diagnostics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import joblib

from credit_risk.calibration import PDCalibrator
from credit_risk.contracts import (
    ApplicantRecord,
    SCHEMA_VERSION,
    ScoringDecisionOutput,
    validate_applicant_record,
)
from credit_risk.errors import ContractValidationError, PointInTimeLeakageError
from credit_risk.explainability import DecisionExplainer
from credit_risk.features import FeaturePipeline
from credit_risk.models import BasePDModel, LogisticRegressionPDModel
from credit_risk.policy import ManualReviewPolicy

logger = logging.getLogger("credit_risk.inference")


def compute_stable_decision_id(
    record: ApplicantRecord,
    model_id: str,
    feature_version: str,
    calibration_method: str,
    policy_version: str,
) -> str:
    """Compute a deterministic SHA-256 decision identifier for replay & auditability."""
    canonical = {
        "application_id": record.application_id,
        "applicant_id": record.applicant_id,
        "application_timestamp": record.application_timestamp,
        "feature_as_of_timestamp": record.feature_as_of_timestamp,
        "model_id": model_id,
        "feature_version": feature_version,
        "calibration_method": calibration_method,
        "policy_version": policy_version,
        "payload": record.to_dict(),
    }
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"DEC-{digest.upper()}"


@dataclass
class InferenceDiagnostics:
    """Structured diagnostic counters tracked across inference calls."""

    total_requests: int = 0
    total_scored: int = 0
    validation_errors: int = 0
    leakage_errors: int = 0
    records_with_missing_features: int = 0
    by_risk_band: Dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.by_risk_band is None:
            self.by_risk_band = {}

    def record_band(self, band: str) -> None:
        self.by_risk_band[band] = self.by_risk_band.get(band, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "total_scored": self.total_scored,
            "validation_errors": self.validation_errors,
            "leakage_errors": self.leakage_errors,
            "records_with_missing_features": self.records_with_missing_features,
            "by_risk_band": dict(sorted(self.by_risk_band.items())),
        }


class CreditRiskScorer:
    """Typed inference service exposing calibrated PD, explanations, and manual-review routing."""

    def __init__(
        self,
        model: BasePDModel,
        feature_pipeline: FeaturePipeline,
        calibrator: PDCalibrator,
        policy: ManualReviewPolicy,
        top_k_factors: int = 5,
        use_shap_if_available: bool = False,
    ) -> None:
        self.model = model
        self.feature_pipeline = feature_pipeline
        self.calibrator = calibrator
        self.policy = policy
        self.explainer = DecisionExplainer(
            model=model,
            feature_pipeline=feature_pipeline,
            top_k=top_k_factors,
            use_shap_if_available=use_shap_if_available,
        )
        self.diagnostics = InferenceDiagnostics()

    def _transform_for_model(self, records: Sequence[ApplicantRecord]):
        standardized = isinstance(self.model, LogisticRegressionPDModel)
        return self.feature_pipeline.transform(records, standardized=standardized)

    def score_applicant(
        self,
        payload: Union[ApplicantRecord, Dict[str, Any]],
        scored_at_utc: Optional[str] = None,
    ) -> ScoringDecisionOutput:
        """Validate, score, calibrate, explain, and route a single applicant for manual review."""
        self.diagnostics.total_requests += 1
        try:
            record = (
                payload
                if isinstance(payload, ApplicantRecord)
                else validate_applicant_record(payload)
            )
        except PointInTimeLeakageError:
            self.diagnostics.leakage_errors += 1
            logger.warning(
                json.dumps(
                    {
                        "event": "point_in_time_leakage_blocked",
                        "application_id": (
                            payload.get("application_id") if isinstance(payload, dict) else None
                        ),
                    }
                )
            )
            raise
        except ContractValidationError:
            self.diagnostics.validation_errors += 1
            logger.warning(
                json.dumps(
                    {
                        "event": "contract_validation_failed",
                        "application_id": (
                            payload.get("application_id") if isinstance(payload, dict) else None
                        ),
                    }
                )
            )
            raise

        X = self._transform_for_model([record])
        raw_score = float(self.model.predict_raw_proba(X)[0])
        calibrated_pd = float(self.calibrator.transform([raw_score])[0])

        missing_fields = self.feature_pipeline.missing_fields_for_record(record)
        warnings: List[str] = []
        if missing_fields:
            self.diagnostics.records_with_missing_features += 1
            warnings.append(
                f"Imputed missing origination features using training medians: {', '.join(missing_fields)}"
            )

        routing = self.policy.route(calibrated_pd=calibrated_pd, missing_fields=missing_fields)
        explanations = self.explainer.explain(record)
        decision_id = compute_stable_decision_id(
            record=record,
            model_id=self.model.model_id,
            feature_version=self.feature_pipeline.feature_version,
            calibration_method=self.calibrator.method,
            policy_version=self.policy.policy_version,
        )

        timestamp = scored_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        output = ScoringDecisionOutput(
            decision_id=decision_id,
            application_id=record.application_id,
            schema_version=SCHEMA_VERSION,
            feature_version=self.feature_pipeline.feature_version,
            model_id=self.model.model_id,
            calibration_method=self.calibrator.method,
            policy_version=self.policy.policy_version,
            raw_score=round(raw_score, 6),
            calibrated_pd=round(calibrated_pd, 6),
            risk_band=routing.risk_band,
            queue_priority=routing.queue_priority,
            review_sla_hours=routing.review_sla_hours,
            recommended_action=routing.recommended_action,
            automated_decision_allowed=False,
            manual_review_required=True,
            explanations=explanations,
            warnings=warnings,
            scored_at_utc=timestamp,
        )

        self.diagnostics.total_scored += 1
        self.diagnostics.record_band(routing.risk_band)
        logger.info(
            json.dumps(
                {
                    "event": "applicant_scored",
                    "decision_id": decision_id,
                    "application_id": record.application_id,
                    "model_id": self.model.model_id,
                    "calibrated_pd": output.calibrated_pd,
                    "risk_band": output.risk_band,
                    "manual_review_required": True,
                }
            )
        )
        return output

    def score_batch(
        self,
        payloads: Sequence[Union[ApplicantRecord, Dict[str, Any]]],
        scored_at_utc: Optional[str] = None,
    ) -> List[ScoringDecisionOutput]:
        return [self.score_applicant(p, scored_at_utc=scored_at_utc) for p in payloads]

    def replay_decision(
        self,
        payload: Union[ApplicantRecord, Dict[str, Any]],
        expected_decision_id: str,
        scored_at_utc: str = "2025-07-01T00:00:00Z",
    ) -> Dict[str, Any]:
        """Replay a prior scoring call and confirm deterministic decision ID and PD match."""
        replayed = self.score_applicant(payload, scored_at_utc=scored_at_utc)
        return {
            "expected_decision_id": expected_decision_id,
            "replayed_decision_id": replayed.decision_id,
            "decision_id_match": replayed.decision_id == expected_decision_id,
            "replayed_decision": replayed.to_dict(),
        }

    def save_bundle(self, artifact_dir: Union[str, Path], filename: str = "scorer_bundle.joblib") -> Path:
        out_dir = Path(artifact_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = out_dir / filename
        bundle = {
            "model": self.model,
            "feature_pipeline_state": self.feature_pipeline.to_state_dict(),
            "calibrator": self.calibrator,
            "policy_config": self.policy.cfg,
            "policy_version": self.policy.policy_version,
            "top_k_factors": self.explainer.top_k,
            "use_shap_if_available": self.explainer.use_shap_if_available,
        }
        joblib.dump(bundle, bundle_path)
        return bundle_path

    @classmethod
    def load_bundle(cls, bundle_path: Union[str, Path]) -> "CreditRiskScorer":
        bundle = joblib.load(Path(bundle_path))
        feature_pipeline = FeaturePipeline.from_state_dict(bundle["feature_pipeline_state"])
        policy = ManualReviewPolicy(
            cfg=bundle["policy_config"],
            policy_version=bundle["policy_version"],
        )
        return cls(
            model=bundle["model"],
            feature_pipeline=feature_pipeline,
            calibrator=bundle["calibrator"],
            policy=policy,
            top_k_factors=bundle["top_k_factors"],
            use_shap_if_available=bundle["use_shap_if_available"],
        )
