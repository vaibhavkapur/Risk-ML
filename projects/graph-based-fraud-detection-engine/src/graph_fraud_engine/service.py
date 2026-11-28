"""Inference service, analyst review queue, disposition audit log, and diagnostic counters."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from graph_fraud_engine import FEATURE_VERSION, SCHEMA_VERSION
from graph_fraud_engine.config import EngineConfig
from graph_fraud_engine.explainer import build_graph_explanation
from graph_fraud_engine.features import FEATURE_NAMES, extract_transaction_features
from graph_fraud_engine.graph import TemporalEntityGraph
from graph_fraud_engine.models import TemporalGraphSAGEModel
from graph_fraud_engine.pipeline import TrainedArtifacts
from graph_fraud_engine.policy import evaluate_policy, make_stable_decision_id
from graph_fraud_engine.schemas import (
    AnalystDispositionRecord,
    AnalystDispositionRequest,
    FraudLabelRecord,
    ModelNotReadyError,
    ReviewCase,
    RiskScoreRequest,
    RiskScoreResponse,
    UnsupportedOperationError,
)

logger = logging.getLogger("graph_fraud_engine.service")


class ReviewCaseNotFoundError(KeyError):
    """Raised when an analyst requests or updates a non-existent review case."""


class FraudDetectionService:
    """Serves calibrated risk scores, graph-traced evidence, and analyst review workflows."""

    def __init__(
        self,
        config: EngineConfig,
        graph: TemporalEntityGraph,
        models: Dict[str, Any],
        run_id: str = "in-memory",
    ) -> None:
        self.config = config
        self.graph = graph
        self.models = models
        self.run_id = run_id

        self._review_cases: Dict[str, ReviewCase] = {}
        self._counters: Dict[str, int] = {
            "total_scored": 0,
            "approved": 0,
            "reviewed": 0,
            "blocked": 0,
            "cold_start_transactions": 0,
            "review_cases_created": 0,
            "dispositions_recorded": 0,
        }

    @classmethod
    def from_artifacts(cls, artifacts: TrainedArtifacts) -> "FraudDetectionService":
        return cls(
            config=artifacts.config,
            graph=artifacts.graph,
            models=artifacts.models,
            run_id=artifacts.run_id,
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "schema_version": SCHEMA_VERSION,
            "feature_version": FEATURE_VERSION,
            "model_version": self.config.model_version,
            "available_models": sorted(self.models.keys()),
            "default_model": self.config.serving.default_model,
            "counters": dict(self._counters),
            "open_review_cases": sum(1 for c in self._review_cases.values() if c.status == "open"),
            "resolved_review_cases": sum(1 for c in self._review_cases.values() if c.status == "resolved"),
        }

    def score_transaction(self, request: RiskScoreRequest) -> RiskScoreResponse:
        """Score a transaction with leak-free graph features, calibrated probability, and graph evidence."""
        if not self.models:
            raise ModelNotReadyError("No trained models loaded in FraudDetectionService")

        model_name = request.model_name or self.config.serving.default_model
        if model_name not in self.models:
            raise UnsupportedOperationError(
                f"Model '{model_name}' is not available. Loaded models: {sorted(self.models.keys())}"
            )

        tx = request.transaction
        feat_res = extract_transaction_features(self.graph, tx, self.config.features)
        x_vec = feat_res.to_vector(FEATURE_NAMES)

        mdl = self.models[model_name]
        if isinstance(mdl, TemporalGraphSAGEModel):
            raw_prob, cal_prob, attributions = mdl.predict_single_inductive(tx, x_vec)
        else:
            raw_prob = float(mdl.predict_raw(x_vec.reshape(1, -1))[0])
            cal_prob = float(mdl.predict_calibrated(x_vec.reshape(1, -1))[0])
            attributions = mdl.feature_attributions(x_vec)

        explanation = build_graph_explanation(
            graph=self.graph,
            tx=tx,
            feature_result=feat_res,
            feature_attributions=attributions,
            feature_config=self.config.features,
            max_paths=self.config.policy.max_evidence_paths,
        )

        decision = evaluate_policy(
            calibrated_probability=cal_prob,
            cold_start_flags=feat_res.cold_start_flags,
            explanation=explanation,
            policy_config=self.config.policy,
        )

        decision_id = make_stable_decision_id(
            tx_id=tx.tx_id,
            model_name=model_name,
            model_version=self.config.model_version,
            feature_version=FEATURE_VERSION,
            calibrated_probability=cal_prob,
        )

        # Update counters
        self._counters["total_scored"] += 1
        if decision == "approve":
            self._counters["approved"] += 1
        elif decision == "review":
            self._counters["reviewed"] += 1
        elif decision == "block":
            self._counters["blocked"] += 1
        if any(feat_res.cold_start_flags.values()):
            self._counters["cold_start_transactions"] += 1

        # Create or reuse deterministic analyst review case for review/block decisions
        review_case_id: Optional[str] = None
        if request.record_for_review and decision in ("review", "block"):
            case_digest = hashlib.sha256(f"{tx.tx_id}|{decision_id}".encode("utf-8")).hexdigest()[:12]
            review_case_id = f"case_{case_digest}"
            if review_case_id not in self._review_cases:
                self._review_cases[review_case_id] = ReviewCase(
                    case_id=review_case_id,
                    decision_id=decision_id,
                    tx_id=tx.tx_id,
                    account_id=tx.account_id,
                    device_id=tx.device_id,
                    merchant_id=tx.merchant_id,
                    tx_timestamp=tx.tx_timestamp,
                    calibrated_probability=round(cal_prob, 6),
                    decision=decision,
                    status="open",
                    explanation=explanation,
                )
                self._counters["review_cases_created"] += 1

        logger.info(
            "scored_transaction tx_id=%s model=%s decision=%s prob=%.4f cold_start=%s case_id=%s",
            tx.tx_id,
            model_name,
            decision,
            cal_prob,
            feat_res.cold_start_flags,
            review_case_id,
        )

        return RiskScoreResponse(
            decision_id=decision_id,
            schema_version=SCHEMA_VERSION,
            feature_version=FEATURE_VERSION,
            model_name=model_name,
            model_version=self.config.model_version,
            tx_id=tx.tx_id,
            raw_score=round(float(raw_prob), 6),
            calibrated_probability=round(float(cal_prob), 6),
            decision=decision,
            cold_start_flags=feat_res.cold_start_flags,
            explanation=explanation,
            review_case_id=review_case_id,
            diagnostics={
                "run_id": self.run_id,
                "max_edge_timestamp_used": (
                    feat_res.max_edge_timestamp_used.isoformat()
                    if feat_res.max_edge_timestamp_used is not None
                    else None
                ),
                "max_label_available_at_used": (
                    feat_res.max_label_available_at_used.isoformat()
                    if feat_res.max_label_available_at_used is not None
                    else None
                ),
            },
        )

    def list_review_cases(self, status: Optional[str] = None) -> List[ReviewCase]:
        cases = list(self._review_cases.values())
        if status is not None:
            if status not in {"open", "resolved"}:
                raise UnsupportedOperationError(f"Unsupported review status filter: '{status}'")
            cases = [c for c in cases if c.status == status]
        cases.sort(key=lambda c: (-c.calibrated_probability, c.case_id))
        return cases

    def get_review_case(self, case_id: str) -> ReviewCase:
        case = self._review_cases.get(case_id)
        if case is None:
            raise ReviewCaseNotFoundError(f"Review case not found: {case_id}")
        return case

    def submit_disposition(
        self,
        case_id: str,
        request: AnalystDispositionRequest,
        recorded_at: Optional[datetime] = None,
    ) -> AnalystDispositionRecord:
        """Record an analyst disposition on a review case and optionally register the matured label."""
        case = self.get_review_case(case_id)
        ts = (recorded_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        # Ensure label maturity timestamp is never earlier than transaction timestamp
        if ts < case.tx_timestamp:
            ts = case.tx_timestamp

        disp_hash = hashlib.sha256(
            f"{case_id}|{request.analyst_id}|{request.disposition}|{len(case.dispositions)}".encode("utf-8")
        ).hexdigest()[:12]
        record = AnalystDispositionRecord(
            disposition_id=f"disp_{disp_hash}",
            case_id=case_id,
            tx_id=case.tx_id,
            analyst_id=request.analyst_id,
            disposition=request.disposition,
            notes=request.notes,
            recorded_at=ts,
        )
        case.dispositions.append(record)
        if request.disposition in ("confirmed_fraud", "false_positive"):
            case.status = "resolved"
            # Register analyst label in the temporal graph with label_available_at = ts
            if case.tx_id in self.graph.transactions:
                self.graph.add_label(
                    FraudLabelRecord(
                        tx_id=case.tx_id,
                        is_fraud=1 if request.disposition == "confirmed_fraud" else 0,
                        tx_timestamp=case.tx_timestamp,
                        label_available_at=ts,
                        label_source="analyst_review",
                        fraud_pattern="analyst_confirmed" if request.disposition == "confirmed_fraud" else "analyst_cleared",
                    )
                )
                self.graph.finalize_indices()

        self._counters["dispositions_recorded"] += 1
        logger.info(
            "recorded_disposition case_id=%s tx_id=%s analyst=%s disposition=%s",
            case_id,
            case.tx_id,
            request.analyst_id,
            request.disposition,
        )
        return record
