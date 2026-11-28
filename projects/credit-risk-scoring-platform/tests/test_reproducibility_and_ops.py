"""Reproducibility, operational review, explainability, and manual review interface tests."""

from pathlib import Path
import pytest

from credit_risk.config import load_config, parse_config_dict
from credit_risk.contracts import validate_applicant_record
from credit_risk.data import load_json_fixture
from credit_risk.errors import ConfigurationError, UnsupportedInputError
from credit_risk.inference import CreditRiskScorer
from credit_risk.pipeline import run_end_to_end_pipeline
from credit_risk.review import ManualReviewStore, render_review_queue_html

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = PROJECT_ROOT / "data" / "fixtures"


def test_end_to_end_reproducibility_same_seed_same_outputs(tmp_path):
    cfg = load_config(PROJECT_ROOT / "config" / "reference.yaml")
    run1_dir = tmp_path / "run1"
    run2_dir = tmp_path / "run2"

    res1 = run_end_to_end_pipeline(cfg, base_dir=run1_dir, n_review_queue_samples=4)
    res2 = run_end_to_end_pipeline(cfg, base_dir=run2_dir, n_review_queue_samples=4)

    er1 = res1["evaluation_report"]
    er2 = res2["evaluation_report"]

    assert er1["models"]["logistic_regression"]["calibrated_test"] == er2["models"]["logistic_regression"]["calibrated_test"]
    assert er1["models"]["boosted_trees"]["calibrated_test"] == er2["models"]["boosted_trees"]["calibrated_test"]
    assert er1["stability"]["score_psi_train_vs_test"] == er2["stability"]["score_psi_train_vs_test"]
    assert [d["decision_id"] for d in res1["sample_decisions"]] == [d["decision_id"] for d in res2["sample_decisions"]]
    assert [d["calibrated_pd"] for d in res1["sample_decisions"]] == [d["calibrated_pd"] for d in res2["sample_decisions"]]
    assert er1["replay_verification"]["decision_id_match"] is True

    # Verify calibration plots & model card were generated and non-empty.
    for key in ("calibration_plot_png", "calibration_plot_svg", "model_card", "evaluation_report"):
        p = Path(res1["outputs"][key])
        assert p.exists() and p.stat().st_size > 100


def test_bundle_reload_replay_and_explanations(tmp_path):
    cfg = load_config(PROJECT_ROOT / "config" / "reference.yaml")
    res = run_end_to_end_pipeline(cfg, base_dir=tmp_path, n_review_queue_samples=3)

    scorer = CreditRiskScorer.load_bundle(res["outputs"]["scorer_bundle"])
    valid_apps = [validate_applicant_record(r) for r in load_json_fixture(FIXTURES_DIR / "valid_applicants.json")]

    first = scorer.score_applicant(valid_apps[0], scored_at_utc="2025-07-01T12:00:00Z")
    replay = scorer.replay_decision(valid_apps[0], expected_decision_id=first.decision_id)

    assert replay["decision_id_match"] is True
    assert first.automated_decision_allowed is False
    assert first.manual_review_required is True
    assert len(first.explanations) == cfg.explainability.top_k_factors
    assert all(e.reason_code.startswith("RC") for e in first.explanations)

    # Missing-field record triggers explicit missing-feature warning and diagnostic counter.
    missing_dec = scorer.score_applicant(valid_apps[1], scored_at_utc="2025-07-01T12:00:00Z")
    assert len(missing_dec.warnings) >= 1
    assert scorer.diagnostics.records_with_missing_features >= 1


def test_operational_error_handling_missing_and_invalid_configs(tmp_path):
    with pytest.raises(ConfigurationError):
        load_config(tmp_path / "does_not_exist.yaml")

    bad_yaml = tmp_path / "malformed.yaml"
    bad_yaml.write_text("project: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config(bad_yaml)

    # Unsupported backend rejected explicitly.
    raw = {
        "project": {"name": "x", "version": "0.1", "schema_version": "1.0.0", "feature_version": "1.0.0", "policy_version": "1.0.0", "seed": 1},
        "data": {"n_applicants": 100, "start_date": "2022-01-01", "end_date": "2024-06-30", "observation_window_days": 365, "as_of_cutoff_date": "2025-07-01", "missing_rate": 0.05, "shift_start_date": "2024-01-01", "shift_magnitude": 0.1},
        "temporal_split": {"train_end_date": "2023-06-30", "calibration_end_date": "2023-12-31", "test_end_date": "2024-06-30"},
        "models": {"logistic_regression": {}, "boosted_trees": {"backend": "catboost_unsupported"}},
        "calibration": {"method": "isotonic"},
        "evaluation": {"min_subgroup_sample_size": 20, "subgroup_columns": ["age_band"], "psi_bins": 5, "psi_warning_threshold": 0.1, "psi_alert_threshold": 0.25, "decision_threshold": 0.15, "baseline_comparison": {}},
        "policy": {"automated_decision_allowed": False, "risk_bands": [{"name": "A", "max_pd": 0.5, "queue_priority": "low", "review_sla_hours": 24, "recommended_action": "R1"}, {"name": "B", "max_pd": 1.0, "queue_priority": "high", "review_sla_hours": 4, "recommended_action": "R2"}]},
        "explainability": {"top_k_factors": 3},
        "paths": {"artifacts_dir": "a", "reports_dir": "r", "review_store_path": "r/q.json"},
    }
    with pytest.raises(UnsupportedInputError):
        parse_config_dict(raw)


def test_manual_review_store_and_html_workbench(tmp_path):
    cfg = load_config(PROJECT_ROOT / "config" / "reference.yaml")
    res = run_end_to_end_pipeline(cfg, base_dir=tmp_path, n_review_queue_samples=3)

    store = ManualReviewStore(res["outputs"]["manual_review_queue"])
    cases = store.list_cases()
    assert len(cases) == 3

    target_id = cases[0]["decision_id"]
    updated = store.record_human_review(
        decision_id=target_id,
        reviewer_id="analyst-qa-02",
        disposition="HUMAN_ESCALATED_COMMITTEE",
        rationale="Elevated DTI and utilization require manual cash-flow verification.",
    )
    assert updated["review_status"] == "HUMAN_ESCALATED_COMMITTEE"
    assert updated["human_reviews"][-1]["automated_decision"] is False

    # Unsupported disposition fails explicitly.
    with pytest.raises(UnsupportedInputError):
        store.record_human_review(
            decision_id=target_id,
            reviewer_id="analyst-qa-02",
            disposition="AUTO_APPROVE",
            rationale="Should fail",
        )

    # HTML workbench renders human-review banner and cases.
    html_page = render_review_queue_html(store.list_cases())
    assert "Manual Review Workbench" in html_page
    assert "Automated lending decisions are disabled" in html_page
    assert target_id in html_page
