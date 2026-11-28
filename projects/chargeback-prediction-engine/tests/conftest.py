"""Shared pytest fixtures for Chargeback Prediction Engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from chargeback_engine.config import EngineConfig, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = PROJECT_ROOT / "data" / "fixtures"
CONFIG_DIR = PROJECT_ROOT / "config"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def default_config() -> EngineConfig:
    return load_config(CONFIG_DIR / "default.yaml")


@pytest.fixture(scope="session")
def cold_start_config() -> EngineConfig:
    return load_config(CONFIG_DIR / "cold_start.yaml")


@pytest.fixture(scope="session")
def reference_raw_data() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    merchants = json.loads((FIXTURES_DIR / "reference_merchants.json").read_text("utf-8"))
    transactions = json.loads((FIXTURES_DIR / "reference_transactions.json").read_text("utf-8"))
    disputes = json.loads((FIXTURES_DIR / "reference_disputes.json").read_text("utf-8"))
    expected = json.loads((FIXTURES_DIR / "reference_expected.json").read_text("utf-8"))
    return merchants, transactions, disputes, expected


@pytest.fixture(scope="session")
def invalid_cases() -> Dict[str, Any]:
    return json.loads((FIXTURES_DIR / "invalid_cases.json").read_text("utf-8"))
