"""Test configuration and shared fixtures."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pytest

from src.config_loader import load_config


@pytest.fixture(scope="session")
def config():
    """Load project config once per test session."""
    return load_config()


@pytest.fixture
def sample_md_path() -> Path:
    return _root / "data" / "documents" / "sample_intro.md"


@pytest.fixture
def sample_txt_path() -> Path:
    return _root / "data" / "documents" / "sample_concepts.txt"
