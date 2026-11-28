"""Evaluation harness comparing rule-only baseline vs ML vs hybrid prioritization.

Measures precision, recall, F1, PR-AUC, calibration (Brier/ECE), false-positive
suppression, and analyst review workload under strict temporal train/cal/test
splitting with label maturity enforcement.
"""

from __future__ import annotations

import platform
from typing import Any, Dict, List, Sequence, Tuple

from aml_engine.config import EngineConfig
from aml_engine.ml.prioritizer import (
    CandidateRecord,
    MLPrioritizer,
    compute_brier_score,
    compute_ece,
    temporal_maturity_split,
)
from aml_engine.pipeline import MonitoringPipeline
from aml_engine.validation import parse_utc_timestamp


def _binary_classification_metrics(
    scores: Sequence[float],
    labels: Sequence[int],
    threshold: float,
) -> Dict[str, Any]:
    tp = fp = tn = fn = 0
    for s, y in zip(scores, labels):
        pred = 1 if s >= threshold else 0
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 0:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    review_volume = tp + fp
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "review_volume": review_volume,
        "total_candidates": len(labels),
        "total_positives": sum(labels),
    }


def _compute_pr_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    total_pos = sum(labels)
    if total_pos == 0 or not scores:
        return 0.0
    ranked = sorted(zip(scores, labels), key=lambda x: -x[0])
    tp = 0
    fp = 0
    prev_recall = 0.0
    auc = 0.0
    for _score, y in ranked:
        if y == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / total_pos
        precision = tp / (tp + fp)
        auc += (recall - prev_recall) * precision
        prev_recall = recall
    return round(auc, 4)


def _precision_recall_at_k(
    scores: Sequence[float],
    labels: Sequence[int],
    k: int,
) -> Dict[str, Any]:
    if not scores:
        return {"k": k, "precision_at_k": 0.0, "recall_at_k": 0.0, "reviewed": 0}
    k_eff = min(k, len(scores))
    ranked = sorted(zip(scores, labels), key=lambda x: -x[0])[:k_eff]
    tp = sum(y for _s, y in ranked)
    total_pos = sum(labels)
    return {
        "k": k_eff,
        "precision_at_k": round(tp / k_eff, 4) if k_eff > 0 else 0.0,
        "recall_at_k": round(tp / total_pos, 4) if total_pos > 0 else 0.0,
        "reviewed": k_eff,
        "positives_captured": tp,
        "total_positives": total_pos,
    }


def evaluate_rule_vs_ml_prioritization(
    dataset: Dict[str, Any],
    config: EngineConfig,
) -> Tuple[MLPrioritizer, Dict[str, Any]]:
    """Run baseline rules, build temporal candidate dataset, train/calibrate ML, and evaluate."""
    # Step 1: Run rules-only baseline to obtain all rule-triggered candidate alerts
    baseline_pipeline = MonitoringPipeline(config=config, repository=None, prioritizer=None)
    baseline_out = baseline_pipeline.run_batch(dataset, strict_validation=False)
    tx_by_id = {t.tx_id: t for t in baseline_out.normalized_transactions}

    candidates: List[CandidateRecord] = []
    for alert in baseline_out.alerts:
        involved_labels = [
            tx_by_id[tid].scenario_label
            for tid in alert.involved_tx_ids
            if tid in tx_by_id and tx_by_id[tid].scenario_label is not None
        ]
        is_suspicious = 1 if any(lbl != "BENIGN" for lbl in involved_labels) else 0
        candidates.append(
            CandidateRecord(
                candidate_id=alert.alert_id,
                subject_account_id=alert.subject_account_id,
                trigger_timestamp_utc=parse_utc_timestamp(alert.trigger_timestamp_iso),
                trigger_timestamp_iso=alert.trigger_timestamp_iso,
                features=baseline_out.candidate_features_by_alert_id[alert.alert_id],
                label=is_suspicious,
                rule_score=alert.prioritization.rule_score,
            )
        )

    ml_cfg = config.ml
    split = temporal_maturity_split(
        records=candidates,
        train_fraction=float(ml_cfg.get("train_fraction", 0.60)),
        calibration_fraction=float(ml_cfg.get("calibration_fraction", 0.20)),
        maturity_days=int(ml_cfg.get("label_maturity_days", 7)),
    )

    prioritizer = MLPrioritizer.fit(
        train_records=split.train,
        calibration_records=split.calibration,
        config=config,
    )

    # Evaluate on strictly out-of-time test window
    test_records = split.test
    test_labels = [r.label for r in test_records]
    rule_scores = [r.rule_score for r in test_records]

    ml_cal_scores: List[float] = []
    hybrid_scores: List[float] = []
    w_rule = float(config.policy.get("hybrid_rule_weight", 0.45))
    w_ml = float(config.policy.get("hybrid_ml_weight", 0.55))

    for rec in test_records:
        _raw_p, cal_p, _contribs = prioritizer.predict_and_explain(rec.features)
        ml_cal_scores.append(cal_p)
        hybrid_scores.append(round(w_rule * rec.rule_score + w_ml * cal_p, 6))

    p1_threshold = float(config.policy["priority_thresholds"]["P1_HIGH"])
    p2_threshold = float(config.policy["priority_thresholds"]["P2_MEDIUM"])
    rule_threshold = float(config.policy.get("rule_only_alert_threshold", 0.50))

    baseline_test = _binary_classification_metrics(rule_scores, test_labels, threshold=rule_threshold)
    ml_test_p2 = _binary_classification_metrics(ml_cal_scores, test_labels, threshold=0.50)
    hybrid_test_p2 = _binary_classification_metrics(hybrid_scores, test_labels, threshold=p2_threshold)
    hybrid_test_p1 = _binary_classification_metrics(hybrid_scores, test_labels, threshold=p1_threshold)

    capacity_k = max(1, sum(test_labels))
    baseline_vol = baseline_test["review_volume"]
    hybrid_p1_vol = hybrid_test_p1["review_volume"]
    workload_reduction_p1 = (
        round((baseline_vol - hybrid_p1_vol) / baseline_vol, 4) if baseline_vol > 0 else 0.0
    )

    # Also compute dataset-level scenario recall (synthetic episodes vs detected)
    true_susp_tx_ids = {
        t.tx_id
        for t in baseline_out.normalized_transactions
        if t.scenario_label and t.scenario_label != "BENIGN"
    }
    covered_susp_tx_ids = {
        tid
        for alert in baseline_out.alerts
        for tid in alert.involved_tx_ids
        if tid in true_susp_tx_ids
    }
    scenario_tx_recall = (
        round(len(covered_susp_tx_ids) / len(true_susp_tx_ids), 4)
        if true_susp_tx_ids
        else 0.0
    )

    report = {
        "evaluation_type": "synthetic_temporal_rule_vs_ml_prioritization",
        "dataset_id": dataset.get("dataset_id", "unknown"),
        "seed": config.seed,
        "runtime_environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "versions": {
            "config_version": config.config_version,
            "config_hash": config.config_hash,
            "feature_version": config.feature_version,
            "policy_version": config.policy_version,
            "model_version": prioritizer.model_version,
        },
        "dataset_summary": {
            "total_normalized_transactions": len(baseline_out.normalized_transactions),
            "total_rule_candidates": len(candidates),
            "true_suspicious_candidates": sum(r.label for r in candidates),
            "benign_rule_false_positives": sum(1 - r.label for r in candidates),
            "synthetic_suspicious_tx_coverage_recall": scenario_tx_recall,
        },
        "temporal_split": {
            "train_count": len(split.train),
            "train_positives": sum(r.label for r in split.train),
            "excluded_immature_train": split.excluded_immature_train,
            "calibration_count": len(split.calibration),
            "calibration_positives": sum(r.label for r in split.calibration),
            "excluded_immature_cal": split.excluded_immature_cal,
            "test_count": len(split.test),
            "test_positives": sum(test_labels),
            "train_end_iso": split.train_end_iso,
            "cal_end_iso": split.cal_end_iso,
            "label_maturity_days": split.maturity_days,
        },
        "out_of_time_test_metrics": {
            "rules_only_baseline": {
                **baseline_test,
                "pr_auc": _compute_pr_auc(rule_scores, test_labels),
                "at_positive_capacity": _precision_recall_at_k(rule_scores, test_labels, capacity_k),
            },
            "ml_calibrated_only": {
                **ml_test_p2,
                "pr_auc": _compute_pr_auc(ml_cal_scores, test_labels),
                "brier_score": round(compute_brier_score(ml_cal_scores, test_labels), 6),
                "ece": round(compute_ece(ml_cal_scores, test_labels), 6),
                "at_positive_capacity": _precision_recall_at_k(ml_cal_scores, test_labels, capacity_k),
            },
            "hybrid_rule_plus_ml_p2_and_above": {
                **hybrid_test_p2,
                "pr_auc": _compute_pr_auc(hybrid_scores, test_labels),
                "at_positive_capacity": _precision_recall_at_k(hybrid_scores, test_labels, capacity_k),
            },
            "hybrid_rule_plus_ml_p1_high_queue": {
                **hybrid_test_p1,
                "workload_reduction_vs_rules_only": workload_reduction_p1,
            },
        },
        "calibration_metadata": prioritizer.training_metadata.get("calibration", {}),
        "note": (
            "Measured on synthetic data generated with fixed seed. "
            "Not a regulatory compliance benchmark or production performance guarantee."
        ),
    }
    return prioritizer, report
