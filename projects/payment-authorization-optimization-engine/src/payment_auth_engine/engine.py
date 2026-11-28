"""Authorization engine orchestration, replay support, and full evaluation runner."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import sklearn

from payment_auth_engine import FEATURE_SCHEMA_VERSION, SCHEMA_VERSION, __version__
from payment_auth_engine.config import EngineConfig
from payment_auth_engine.counterfactual import PolicyCounterfactualReport, evaluate_policy_counterfactual
from payment_auth_engine.diagnostics import (
    DiagnosticsCollector,
    IdempotencyLedger,
    ProcessorHealthRegistry,
)
from payment_auth_engine.models import AuthorizationModelBundle
from payment_auth_engine.policy import (
    AuthorizationPolicy,
    BaselineThresholdPolicy,
    ConstrainedRiskAwarePolicy,
    UnconstrainedEVPolicy,
    audit_batch_constraints,
)
from payment_auth_engine.schemas import (
    AuthorizationAttempt,
    DecisionExplanation,
    DecisionRecord,
    PolicyAction,
)
from payment_auth_engine.synthetic import generate_synthetic_dataset
from payment_auth_engine.taxonomy import evaluate_retry_eligibility


def make_decision_id(
    *,
    attempt_id: str,
    transaction_id: str,
    idempotency_key: str,
    policy_name: str,
    model_version: str,
    config_hash: str,
) -> str:
    """Create a stable deterministic decision identifier for audit and replay verification."""
    raw = f"{attempt_id}|{transaction_id}|{idempotency_key}|{policy_name}|{model_version}|{config_hash}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    return f"dec_{digest}"


class AuthorizationEngine:
    """Core authorization engine isolating domain decisions from I/O."""

    def __init__(
        self,
        *,
        config: EngineConfig,
        model_bundle: AuthorizationModelBundle,
        policy: AuthorizationPolicy,
        health_registry: Optional[ProcessorHealthRegistry] = None,
        idempotency_ledger: Optional[IdempotencyLedger] = None,
        diagnostics: Optional[DiagnosticsCollector] = None,
    ) -> None:
        self.config = config
        self.model_bundle = model_bundle
        self.policy = policy
        self.health_registry = health_registry or ProcessorHealthRegistry(config)
        self.idempotency_ledger = idempotency_ledger or IdempotencyLedger()
        self.diagnostics = diagnostics or DiagnosticsCollector()
        self._config_hash = config.config_hash()

    def decide(
        self,
        attempt: AuthorizationAttempt,
        *,
        record_in_ledger: bool = True,
        _precomputed_predictions: Optional[Dict[PolicyAction, Any]] = None,
    ) -> DecisionRecord:
        """Evaluate retry safety, model predictions, policy constraints, and return DecisionRecord."""
        # 1. Safety & Retry Eligibility Layer (runs BEFORE ML policy)
        idem_seen = self.idempotency_ledger.has_idempotency_key(attempt.idempotency_key)
        prior_approved = self.idempotency_ledger.is_transaction_approved(attempt.transaction_id)

        eligibility = evaluate_retry_eligibility(
            attempt,
            idempotency_seen=idem_seen,
            prior_approved=prior_approved,
            max_retries=self.config.retry.max_retries,
            allow_soft_decline_retry=self.config.retry.allow_soft_decline_retry,
            allow_technical_failover=self.config.retry.allow_technical_failover,
        )

        decision_id = make_decision_id(
            attempt_id=attempt.attempt_id,
            transaction_id=attempt.transaction_id,
            idempotency_key=attempt.idempotency_key,
            policy_name=self.policy.name,
            model_version=self.model_bundle.model_version,
            config_hash=self._config_hash,
        )

        if not eligibility.eligible:
            prior = self.idempotency_ledger.get_by_idempotency_key(attempt.idempotency_key)
            extra_note = (
                f" Prior decision_id={prior.decision_id}, action={prior.action.value}."
                if prior is not None
                else ""
            )
            record = DecisionRecord(
                schema_version=SCHEMA_VERSION,
                decision_id=decision_id,
                attempt_id=attempt.attempt_id,
                transaction_id=attempt.transaction_id,
                idempotency_key=attempt.idempotency_key,
                policy_name=self.policy.name,
                model_version=self.model_bundle.model_version,
                feature_schema_version=FEATURE_SCHEMA_VERSION,
                config_hash=self._config_hash,
                action=eligibility.recommended_action,
                processor_selected=None,
                predicted_acceptance_prob=0.0,
                predicted_loss_prob=0.0,
                expected_loss_cents=0.0,
                expected_net_value_cents=0.0,
                estimated_latency_ms=10.0,
                constraint_checks=[],
                any_constraint_breach=False,
                explanation=DecisionExplanation(
                    reason_codes=[eligibility.reason_code],
                    missing_features=[],
                    top_contributing_factors={},
                    policy_notes=eligibility.explanation + extra_note,
                ),
            )
            self.diagnostics.record_decision(record)
            return record

        # 2. Restrict allowed actions if retry eligibility specifies targeted recovery
        allowed_actions: Optional[List[PolicyAction]] = None
        if attempt.attempt_number > 1:
            if eligibility.recommended_action == PolicyAction.STEP_UP_3DS:
                allowed_actions = [PolicyAction.STEP_UP_3DS, PolicyAction.DECLINE_POLICY]
            elif eligibility.recommended_action == PolicyAction.ROUTE_SECONDARY:
                allowed_actions = [PolicyAction.ROUTE_SECONDARY, PolicyAction.DECLINE_POLICY]

        # 3. Calibrated Model Inference Layer
        if _precomputed_predictions is not None:
            predictions = _precomputed_predictions
        else:
            health_states = self.health_registry.get_all()
            predictions = self.model_bundle.predict_all_actions(attempt, health_states=health_states)

        # 4. Decision Policy Layer
        choice = self.policy.choose(attempt, predictions, allowed_actions=allowed_actions)
        pred = choice.prediction

        record = DecisionRecord(
            schema_version=SCHEMA_VERSION,
            decision_id=decision_id,
            attempt_id=attempt.attempt_id,
            transaction_id=attempt.transaction_id,
            idempotency_key=attempt.idempotency_key,
            policy_name=self.policy.name,
            model_version=self.model_bundle.model_version,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            config_hash=self._config_hash,
            action=choice.action,
            processor_selected=pred.processor_selected if choice.action != PolicyAction.BLOCK_PROCESSOR_OUTAGE else None,
            predicted_acceptance_prob=round(pred.acceptance_prob, 6),
            predicted_loss_prob=round(pred.fraud_prob_given_approved, 6),
            expected_loss_cents=round(pred.expected_loss_cents, 4),
            expected_net_value_cents=round(pred.expected_net_value_cents, 4),
            estimated_latency_ms=round(pred.estimated_latency_ms, 2),
            constraint_checks=choice.constraint_checks,
            any_constraint_breach=choice.any_constraint_breach,
            explanation=DecisionExplanation(
                reason_codes=choice.reason_codes,
                missing_features=pred.missing_features,
                top_contributing_factors=pred.top_factors,
                policy_notes=choice.policy_notes,
            ),
        )

        if record_in_ledger:
            self.idempotency_ledger.record(record)
        self.diagnostics.record_decision(record)
        return record

    def decide_batch(
        self, attempts: Sequence[AuthorizationAttempt], *, record_in_ledger: bool = True
    ) -> List[DecisionRecord]:
        health_states = self.health_registry.get_all()
        batch_preds = self.model_bundle.predict_batch_all_actions(attempts, health_states=health_states)
        records: List[DecisionRecord] = []
        for attempt, preds in zip(attempts, batch_preds):
            records.append(self.decide(attempt, record_in_ledger=record_in_ledger, _precomputed_predictions=preds))
        return records

    def replay(self, attempts: Sequence[AuthorizationAttempt]) -> List[DecisionRecord]:
        """Deterministically replay a sequence of attempts on a fresh isolated ledger."""
        fresh_engine = AuthorizationEngine(
            config=self.config,
            model_bundle=self.model_bundle,
            policy=self.policy,
            health_registry=self.health_registry,
            idempotency_ledger=IdempotencyLedger(),
            diagnostics=DiagnosticsCollector(),
        )
        return fresh_engine.decide_batch(attempts, record_in_ledger=True)


def run_full_experiment(
    config: EngineConfig,
    *,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run reproducible end-to-end training, calibration, counterfactual evaluation, and audit."""
    dataset = generate_synthetic_dataset(config)
    bundle, train_idx, cal_idx, test_idx = AuthorizationModelBundle.train(
        dataset.attempts,
        dataset.outcomes,
        config,
        dataset.as_of_utc,
    )

    cal_attempts = [dataset.attempts[int(i)] for i in cal_idx]
    cal_preds = bundle.predict_batch_all_actions(cal_attempts)

    test_attempts = [dataset.attempts[int(i)] for i in test_idx]
    test_outcomes = [dataset.outcomes[int(i)] for i in test_idx]
    test_oracles = [dataset.oracles[int(i)] for i in test_idx]
    test_preds = bundle.predict_batch_all_actions(test_attempts)

    baseline_policy = BaselineThresholdPolicy(config)
    unconstrained_policy = UnconstrainedEVPolicy(config)
    constrained_policy = ConstrainedRiskAwarePolicy(config)
    cal_tuning = constrained_policy.fit_dual_multipliers(cal_attempts, cal_preds)

    policies: List[AuthorizationPolicy] = [baseline_policy, unconstrained_policy, constrained_policy]
    policy_reports: Dict[str, Dict[str, Any]] = {}
    audit_reports: Dict[str, Dict[str, Any]] = {}
    diagnostics_by_policy: Dict[str, Dict[str, Any]] = {}
    sample_decisions_by_policy: Dict[str, List[Dict[str, Any]]] = {}

    for pol in policies:
        engine = AuthorizationEngine(config=config, model_bundle=bundle, policy=pol)
        decisions = engine.decide_batch(test_attempts)
        chosen_actions = [d.action for d in decisions]

        cf_report: PolicyCounterfactualReport = evaluate_policy_counterfactual(
            policy_name=pol.name,
            attempts=test_attempts,
            outcomes=test_outcomes,
            predictions_per_attempt=test_preds,
            candidate_actions=chosen_actions,
            config=config,
            oracles=test_oracles,
        )
        audit = audit_batch_constraints(
            test_attempts,
            decisions,
            max_expected_loss_bps=config.policy.max_expected_loss_bps,
            max_p95_latency_ms=config.policy.max_p95_latency_ms,
        )
        policy_reports[pol.name] = cf_report.to_dict()
        audit_reports[pol.name] = audit
        diagnostics_by_policy[pol.name] = engine.diagnostics.snapshot()
        sample_decisions_by_policy[pol.name] = [d.model_dump(mode="json") for d in decisions[:10]]

    runtime_metadata = {
        "package_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "config_version": config.config_version,
        "config_hash": config.config_hash(),
        "seed": config.seed,
        "model_version": bundle.model_version,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_provenance": "100% synthetic generated via payment_auth_engine.synthetic (no real transactions or PII)",
    }

    result = {
        "runtime_metadata": runtime_metadata,
        "model_summary": asdict(bundle.summary),
        "constrained_policy_calibration": cal_tuning,
        "policy_comparison": policy_reports,
        "policy_audit": audit_reports,
        "diagnostics": diagnostics_by_policy,
        "baseline_sample_decisions": sample_decisions_by_policy["baseline_threshold"],
        "constrained_sample_decisions": sample_decisions_by_policy["constrained_risk_aware"],
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        bundle.save_manifest(output_dir / "model_manifest.json")
        (output_dir / "runtime_metadata.json").write_text(
            json.dumps(runtime_metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        (output_dir / "baseline_outputs.json").write_text(
            json.dumps(
                {
                    "model_version": bundle.model_version,
                    "config_hash": config.config_hash(),
                    "baseline_policy_metrics": policy_reports["baseline_threshold"],
                    "baseline_audit": audit_reports["baseline_threshold"],
                    "sample_decisions": sample_decisions_by_policy["baseline_threshold"],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (output_dir / "policy_comparison.json").write_text(
            json.dumps(
                {
                    "model_summary": asdict(bundle.summary),
                    "constrained_policy_calibration": cal_tuning,
                    "policy_comparison": policy_reports,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (output_dir / "audit_report.json").write_text(
            json.dumps(
                {
                    "config_hash": config.config_hash(),
                    "model_version": bundle.model_version,
                    "constraints": {
                        "max_expected_loss_bps": config.policy.max_expected_loss_bps,
                        "max_p95_latency_ms": config.policy.max_p95_latency_ms,
                    },
                    "policy_audit": audit_reports,
                    "diagnostics": diagnostics_by_policy,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    return result
