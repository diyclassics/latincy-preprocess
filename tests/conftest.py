"""Shared fixtures for latincy-preprocess tests."""

import json
from pathlib import Path

import pytest

from latincy_preprocess.uv import UVNormalizerRules
from latincy_preprocess.long_s import LongSNormalizer

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def uv_normalizer() -> UVNormalizerRules:
    """Return a fresh UV normalizer instance."""
    return UVNormalizerRules()


@pytest.fixture
def long_s_normalizer() -> LongSNormalizer:
    """Return a fresh LongS normalizer instance."""
    return LongSNormalizer()


@pytest.fixture
def curated_uv_tests() -> list[dict]:
    """Load curated U/V test cases from JSON file."""
    test_file = DATA_DIR / "curated_test.json"
    with open(test_file) as f:
        data = json.load(f)
    return data["test_cases"]


@pytest.fixture
def approved_corrections() -> dict:
    """Load the 54 manually approved long-s corrections."""
    path = DATA_DIR / "approved_corrections.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def long_s_allowlist() -> list[str]:
    """Load the allowlist of legitimate f- words."""
    path = DATA_DIR / "allowlist.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
