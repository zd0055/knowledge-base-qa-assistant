"""Retriever tests."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

import pytest


@pytest.fixture
def config():
    from src.config_loader import load_config
    return load_config()


def test_retriever_creation(config):
    """Verify Retriever can be instantiated."""
    from src.embedding import EmbeddingManager
    from src.retrieval import Retriever
    em = EmbeddingManager(config)
    retriever = Retriever(config, em)
    assert retriever is not None
    assert retriever.top_k > 0
    assert repr(retriever) is not None


def test_retriever_get_collection_stats_empty(config):
    """Verify collection stats returns safe defaults when empty."""
    from src.embedding import EmbeddingManager
    from src.retrieval import Retriever
    em = EmbeddingManager(config)
    retriever = Retriever(config, em)
    stats = retriever.get_collection_stats()
    assert "total_chunks" in stats
    assert "sources" in stats
    assert "num_documents" in stats
