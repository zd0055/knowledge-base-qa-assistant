"""Ingestion pipeline tests."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

import pytest


@pytest.fixture
def config():
    from src.config_loader import load_config
    return load_config()


def test_pipeline_creation(config):
    """Verify IngestionPipeline can be instantiated."""
    from src.ingestion import IngestionPipeline
    pipeline = IngestionPipeline(config)
    assert pipeline is not None
    assert pipeline.loader is not None
    assert pipeline.splitter is not None


def test_doc_id_uniqueness():
    """Verify _doc_id produces unique IDs for different content."""
    from src.ingestion.pipeline import _doc_id
    id1 = _doc_id("hello", {"source": "a.md"})
    id2 = _doc_id("world", {"source": "a.md"})
    id3 = _doc_id("hello", {"source": "b.md"})
    assert id1 != id2, "Different content should have different IDs"
    assert id1 != id3, "Different metadata should have different IDs"
    assert isinstance(id1, str)
    assert len(id1) == 32, "MD5 hash should be 32 hex chars"
