"""M2 tests: legitimate high-volume traffic, synthetic suspicious chains, and error propagation."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from aml_engine.config import load_config
from aml_engine.pipeline import MonitoringPipeline
from aml_engine.validation import UnsupportedInputError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = PROJECT_ROOT / "data" / "fixtures"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default_config.json"
STRICT_CONFIG = PROJECT_ROOT / "config" / "strict_config.json"


class TestM2DomainDetection(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(DEFAULT_CONFIG)

    def test_legitimate_high_volume_activity_produces_zero_false_positives(self) -> None:
        fixture = json.loads(
            (FIXTURES_DIR / "valid_legitimate_high_volume.json").read_text(encoding="utf-8")
        )
        pipeline = MonitoringPipeline(config=self.config)
        out = pipeline.run_batch(fixture, strict_validation=True)

        self.assertEqual(out.diagnostics.valid_normalized_events, 10)
        self.assertEqual(out.scenario_hits, [])
        self.assertEqual(out.alerts, [])

    def test_synthetic_suspicious_chains_all_five_scenarios_detected(self) -> None:
        fixture = json.loads(
            (FIXTURES_DIR / "valid_suspicious_chains.json").read_text(encoding="utf-8")
        )
        pipeline = MonitoringPipeline(config=self.config)
        out = pipeline.run_batch(fixture, strict_validation=True)

        detected_scenarios = {h.scenario_name for h in out.scenario_hits}
        expected_scenarios = {
            "STRUCTURING_SMURFING",
            "RAPID_LAYERING_CHAIN",
            "ROUND_TRIP_CYCLE",
            "MULE_FAN_IN_FAN_OUT",
            "DORMANT_ACCOUNT_SPIKE",
        }
        self.assertEqual(detected_scenarios, expected_scenarios)

        # Verify graph feature flags match detected motifs
        self.assertTrue(out.graph_features["ACC-L-0"].in_layering_chain)
        self.assertTrue(out.graph_features["ACC-L-3"].in_layering_chain)
        self.assertTrue(out.graph_features["ACC-C-1"].in_round_trip_cycle)
        self.assertFalse(out.graph_features["ACC-S-TARGET"].in_round_trip_cycle)

        # Every alert must carry non-empty human-readable reason contributions
        self.assertGreaterEqual(len(out.alerts), 5)
        for alert in out.alerts:
            self.assertTrue(len(alert.reasons) >= 1)
            for reason in alert.reasons:
                self.assertTrue(bool(reason.description.strip()))

    def test_unsupported_input_propagation_strict_vs_non_strict(self) -> None:
        payload = {
            "entities": [{"entity_id": "E1", "name": "Test Entity", "risk_tier": "LOW"}],
            "accounts": [
                {"account_id": "A1", "entity_id": "E1", "account_type": "RETAIL"},
                {"account_id": "A2", "entity_id": "E1", "account_type": "RETAIL"},
            ],
            "transactions": [
                {
                    "tx_id": "TX-UNSUP-1",
                    "source_account_id": "A1",
                    "target_account_id": "A2",
                    "amount": 1000.0,
                    "currency": "XYZ",
                    "timestamp": "2026-01-10T10:00:00Z",
                    "channel": "WIRE",
                }
            ],
        }
        pipeline = MonitoringPipeline(config=self.config)
        with self.assertRaises(UnsupportedInputError):
            pipeline.run_batch(payload, strict_validation=True)

        non_strict = pipeline.run_batch(payload, strict_validation=False)
        self.assertEqual(non_strict.diagnostics.invalid_events_rejected, 1)
        self.assertEqual(non_strict.diagnostics.valid_normalized_events, 0)
        self.assertEqual(non_strict.diagnostics.errors[0]["error_type"], "UnsupportedInputError")

    def test_strict_config_profile_loads_with_distinct_config_hash(self) -> None:
        strict_cfg = load_config(STRICT_CONFIG)
        self.assertNotEqual(self.config.config_hash, strict_cfg.config_hash)
        fixture = json.loads(
            (FIXTURES_DIR / "valid_suspicious_chains.json").read_text(encoding="utf-8")
        )
        out = MonitoringPipeline(config=strict_cfg).run_batch(fixture, strict_validation=True)
        self.assertEqual(out.diagnostics.config_version, "1.0.0-strict")
        self.assertGreaterEqual(len(out.alerts), 5)


if __name__ == "__main__":
    unittest.main()
