"""End-to-end AML transaction monitoring orchestration pipeline.

Orchestrates validation, normalization/deduplication, rolling & graph features,
configurable scenario detection, calibrated ML prioritization, deterministic
alert/case provenance creation, structured logging, and replay idempotency.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from aml_engine.case_management.workflow import build_alert_and_evidence, create_case_for_alert
from aml_engine.config import EngineConfig
from aml_engine.detection.scenarios import detect_scenarios
from aml_engine.features.graph import TransactionGraph
from aml_engine.features.rolling import compute_rolling_features
from aml_engine.ml.prioritizer import (
    MLPrioritizer,
    apply_prioritization_policy,
    extract_candidate_features,
)
from aml_engine.models import (
    Account,
    Alert,
    Case,
    Entity,
    GraphFeatureVector,
    NormalizedTransaction,
    ReviewPriority,
    RollingFeatureVector,
    RunDiagnostics,
    ScenarioHit,
)
from aml_engine.normalization import normalize_transactions
from aml_engine.storage.repository import CaseRepository
from aml_engine.validation import validate_account_dict, validate_entity_dict


logger = logging.getLogger("aml_engine.pipeline")


@dataclass
class PipelineRunOutput:
    diagnostics: RunDiagnostics
    normalized_transactions: List[NormalizedTransaction]
    rolling_features: Dict[Tuple[str, str], RollingFeatureVector]
    graph_features: Dict[str, GraphFeatureVector]
    scenario_hits: List[ScenarioHit]
    alerts: List[Alert]
    cases: List[Case]
    candidate_features_by_alert_id: Dict[str, Dict[str, float]]


def _group_hits_into_alert_clusters(
    hits: Sequence[ScenarioHit],
) -> List[Tuple[str, List[ScenarioHit]]]:
    """Group scenario hits by (subject_account_id, trigger_date) deterministically."""
    grouped: Dict[Tuple[str, str], List[ScenarioHit]] = defaultdict(list)
    for hit in hits:
        day_bucket = hit.trigger_timestamp_iso[:10]
        grouped[(hit.subject_account_id, day_bucket)].append(hit)
    clusters: List[Tuple[str, List[ScenarioHit]]] = []
    for (acc_id, _day), group_hits in sorted(grouped.items()):
        clusters.append((acc_id, group_hits))
    return clusters


class MonitoringPipeline:
    """Deterministic monitoring pipeline separating pure domain logic from storage."""

    def __init__(
        self,
        config: EngineConfig,
        repository: Optional[CaseRepository] = None,
        prioritizer: Optional[MLPrioritizer] = None,
    ) -> None:
        self.config = config
        self.repository = repository
        self.prioritizer = prioritizer

    def run_batch(
        self,
        payload: Dict[str, Any],
        strict_validation: bool = False,
        run_id: Optional[str] = None,
    ) -> PipelineRunOutput:
        raw_entities = payload.get("entities", [])
        raw_accounts = payload.get("accounts", [])
        raw_transactions = payload.get("transactions", [])

        entities_by_id: Dict[str, Entity] = {}
        for item in raw_entities:
            ent = validate_entity_dict(item)
            entities_by_id[ent.entity_id] = ent

        accounts_by_id: Dict[str, Account] = {}
        for item in raw_accounts:
            acc = validate_account_dict(item)
            accounts_by_id[acc.account_id] = acc

        norm_result = normalize_transactions(
            raw_transactions=raw_transactions,
            accounts_by_id=accounts_by_id,
            entities_by_id=entities_by_id,
            config=self.config,
            strict=strict_validation,
        )

        normalized_txs = norm_result.normalized_transactions
        tx_by_id = {t.tx_id: t for t in normalized_txs}

        rolling_features = compute_rolling_features(
            transactions=normalized_txs,
            accounts_by_id=accounts_by_id,
            config=self.config,
        )
        graph = TransactionGraph(normalized_txs)
        scenario_hits, graph_features = detect_scenarios(
            transactions=normalized_txs,
            accounts_by_id=accounts_by_id,
            rolling_features=rolling_features,
            graph=graph,
            config=self.config,
        )

        clusters = _group_hits_into_alert_clusters(scenario_hits)
        alerts: List[Alert] = []
        cases: List[Case] = []
        candidate_features: Dict[str, Dict[str, float]] = {}
        cases_created = 0
        cases_reused = 0

        for subject_acc_id, cluster_hits in clusters:
            rule_score = max(h.rule_score for h in cluster_hits)
            feats = extract_candidate_features(
                hits=cluster_hits,
                tx_by_id=tx_by_id,
                accounts_by_id=accounts_by_id,
                entities_by_id=entities_by_id,
                rolling_features=rolling_features,
                graph_features=graph_features,
            )
            prioritization = apply_prioritization_policy(
                rule_score=rule_score,
                features=feats,
                config=self.config,
                prioritizer=self.prioritizer,
            )
            if prioritization.priority == ReviewPriority.SUPPRESSED_BELOW_THRESHOLD:
                continue

            alert = build_alert_and_evidence(
                subject_account_id=subject_acc_id,
                hits=cluster_hits,
                tx_by_id=tx_by_id,
                prioritization=prioritization,
                config=self.config,
            )
            case = create_case_for_alert(alert)
            if self.repository is not None:
                created, persisted_case = self.repository.save_alert_and_case(alert, case)
                if created:
                    cases_created += 1
                else:
                    cases_reused += 1
                case = persisted_case
            else:
                cases_created += 1

            alerts.append(alert)
            cases.append(case)
            candidate_features[alert.alert_id] = feats

        alerts.sort(key=lambda a: (-a.prioritization.combined_score, a.alert_id))
        cases.sort(key=lambda c: (c.created_at_iso, c.case_id))

        computed_run_id = run_id or self._deterministic_run_id(normalized_txs)
        diagnostics = RunDiagnostics(
            run_id=computed_run_id,
            config_version=self.config.config_version,
            config_hash=self.config.config_hash,
            feature_version=self.config.feature_version,
            policy_version=self.config.policy_version,
            model_version=self.prioritizer.model_version if self.prioritizer else None,
            total_raw_events=len(raw_transactions),
            valid_normalized_events=len(normalized_txs),
            invalid_events_rejected=len(norm_result.rejected_events),
            duplicate_events_skipped=len(norm_result.duplicate_tx_ids),
            missing_entity_mappings=len(norm_result.missing_entity_tx_ids),
            scenario_hits_count=len(scenario_hits),
            alerts_generated=len(alerts),
            cases_created=cases_created,
            cases_reused_on_replay=cases_reused,
            warnings=norm_result.warnings,
            errors=norm_result.rejected_events,
        )

        if self.repository is not None:
            self.repository.save_run_diagnostics(diagnostics)

        logger.info(
            json.dumps(
                {
                    "event": "monitoring_batch_completed",
                    "run_id": diagnostics.run_id,
                    "valid_events": diagnostics.valid_normalized_events,
                    "duplicates_skipped": diagnostics.duplicate_events_skipped,
                    "missing_entity_mappings": diagnostics.missing_entity_mappings,
                    "alerts_generated": diagnostics.alerts_generated,
                    "cases_created": diagnostics.cases_created,
                    "cases_reused_on_replay": diagnostics.cases_reused_on_replay,
                },
                sort_keys=True,
            )
        )

        return PipelineRunOutput(
            diagnostics=diagnostics,
            normalized_transactions=normalized_txs,
            rolling_features=rolling_features,
            graph_features=graph_features,
            scenario_hits=scenario_hits,
            alerts=alerts,
            cases=cases,
            candidate_features_by_alert_id=candidate_features,
        )

    def _deterministic_run_id(self, normalized_txs: Sequence[NormalizedTransaction]) -> str:
        payload = {
            "config_hash": self.config.config_hash,
            "model_version": self.prioritizer.model_version if self.prioritizer else "RULES_ONLY",
            "tx_hashes": [t.canonical_event_hash for t in normalized_txs],
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16].upper()
        return f"RUN-{digest}"
