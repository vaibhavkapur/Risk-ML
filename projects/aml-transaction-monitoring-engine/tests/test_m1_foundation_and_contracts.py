"""M1 tests: configuration, data contracts, normalization, and hand-calculated reference oracle."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from aml_engine.config import load_config
from aml_engine.models import EntityMappingStatus
from aml_engine.pipeline import MonitoringPipeline
from aml_engine.validation import (
    ConfigurationError,
    UnsupportedInputError,
    ValidationError,
    validate_transaction_dict,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = PROJECT_ROOT / "data" / "fixtures"
CONFIG_PATH = PROJECT_ROOT / "config" / "default_config.json"


class TestM1FoundationAndContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(CONFIG_PATH)

    def test_missing_config_raises_explicit_error(self) -> None:
        with self.assertRaises(ConfigurationError):
            load_config(PROJECT_ROOT / "config" / "nonexistent_config.json")

    def test_invalid_inputs_fixture_rejected_explicitly(self) -> None:
        invalid_fixture = json.loads((FIXTURES_DIR / "invalid_inputs.json").read_text(encoding="utf-8"))
        error_map = {
            "ValidationError": ValidationError,
            "UnsupportedInputError": UnsupportedInputError,
        }
        for case in invalid_fixture["cases"]:
            expected_cls = error_map[case["expected_error"]]
            with self.subTest(case=case["case_name"]):
                with self.assertRaises(expected_cls):
                    validate_transaction_dict(
                        case["transaction"],
                        supported_currencies=self.config.supported_currencies,
                        supported_channels=self.config.supported_channels,
                    )

    def test_reference_hand_calculated_oracle(self) -> None:
        fixture = json.loads((FIXTURES_DIR / "reference_hand_calc.json").read_text(encoding="utf-8"))
        oracle = fixture["expected_oracle"]
        tol = float(oracle["numeric_tolerance"])

        pipeline = MonitoringPipeline(config=self.config)
        out = pipeline.run_batch(fixture, strict_validation=True)

        self.assertEqual(out.diagnostics.invalid_events_rejected, 0)
        self.assertEqual(out.diagnostics.duplicate_events_skipped, 0)
        self.assertEqual(out.diagnostics.missing_entity_mappings, 0)

        # 1. Verify exact USD currency normalization against hand calculations
        norm_by_id = {t.tx_id: t for t in out.normalized_transactions}
        for tx_id, expected_usd in oracle["normalized_amounts_usd"].items():
            self.assertIn(tx_id, norm_by_id)
            self.assertAlmostEqual(norm_by_id[tx_id].amount_usd, expected_usd, delta=tol)
            self.assertEqual(norm_by_id[tx_id].entity_mapping_status, EntityMappingStatus.MAPPED)

        # 2. Verify point-in-time rolling aggregations at TX-REF-003 target
        expected_roll = oracle["rolling_at_tx_ref_003_target"]
        fv = out.rolling_features[("TX-REF-003", "TARGET")]
        self.assertEqual(fv.account_id, expected_roll["account_id"])
        self.assertEqual(fv.inbound_count_24h, expected_roll["inbound_count_24h"])
        self.assertAlmostEqual(fv.inbound_sum_24h_usd, expected_roll["inbound_sum_24h_usd"], delta=tol)
        self.assertEqual(fv.structuring_band_count_24h, expected_roll["structuring_band_count_24h"])
        self.assertAlmostEqual(
            fv.structuring_band_sum_24h_usd, expected_roll["structuring_band_sum_24h_usd"], delta=tol
        )
        self.assertEqual(fv.unique_inbound_senders_24h, expected_roll["unique_inbound_senders_24h"])

        # 3. Verify expected scenarios and layering metrics
        hit_pairs = {(h.scenario_name, h.subject_account_id) for h in out.scenario_hits}
        for exp in oracle["expected_scenarios"]:
            self.assertIn((exp["scenario"], exp["subject_account_id"]), hit_pairs)

        layer_hit = next(h for h in out.scenario_hits if h.scenario_name == "RAPID_LAYERING_CHAIN")
        self.assertAlmostEqual(
            layer_hit.metrics["value_preservation_ratio"],
            oracle["layering_chain_preservation_ratio"],
            delta=tol,
        )
        self.assertAlmostEqual(
            layer_hit.metrics["elapsed_hours"],
            oracle["layering_chain_elapsed_hours"],
            delta=tol,
        )
        self.assertEqual(
            layer_hit.graph_path,
            ("ACC-LAY-0", "ACC-LAY-1", "ACC-LAY-2", "ACC-LAY-3"),
        )

    def test_exact_24h_rolling_window_boundary(self) -> None:
        fixture = json.loads((FIXTURES_DIR / "boundary_cases.json").read_text(encoding="utf-8"))
        pipeline = MonitoringPipeline(config=self.config)
        out = pipeline.run_batch(fixture, strict_validation=True)

        # TX-BND-WIN-1 is at 2026-01-20T00:00:00Z and TX-BND-WIN-3 is at 2026-01-21T00:00:01Z
        # (24h + 1s later), so only 2 transfers fall within 24h at TX-BND-WIN-3.
        fv_win3 = out.rolling_features[("TX-BND-WIN-3", "TARGET")]
        self.assertEqual(fv_win3.structuring_band_count_24h, 2)
        self.assertAlmostEqual(fv_win3.structuring_band_sum_24h_usd, 16000.0, delta=1e-6)
        # Structuring requires >= 3 in 24h, so ACC-BND-1 must NOT trigger
        str_hits = [h for h in out.scenario_hits if h.scenario_name == "STRUCTURING_SMURFING"]
        self.assertEqual(str_hits, [])


if __name__ == "__main__":
    unittest.main()
