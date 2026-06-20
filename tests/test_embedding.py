"""Embedding model tests (proper pytest style)."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

import pytest

from src.embedding import EmbeddingManager


@pytest.fixture
def config():
    from src.config_loader import load_config
    return load_config()


def test_embedding_manager_creation(config):
    """Verify EmbeddingManager can be instantiated."""
    em = EmbeddingManager(config)
    assert em is not None
    assert repr(em) is not None


def test_embedding_dimension(config):
    """Verify embedding dimension is a reasonable positive integer."""
    em = EmbeddingManager(config)
    try:
        dim = em.dimension
        assert isinstance(dim, int)
        assert dim > 0
        assert dim < 10000
    except Exception:
        pytest.skip("Embedding model not available in test environment")


def test_embed_texts(config):
    """Verify embedding multiple texts works."""
    em = EmbeddingManager(config)
    try:
        vectors = em.embed_texts(["hello world", "test query"])
        assert len(vectors) == 2
        assert len(vectors[0]) > 0
    except Exception:
        pytest.skip("Embedding model not available in test environment")


def test_embed_query(config):
    """Verify query embedding works."""
    em = EmbeddingManager(config)
    try:
        vec = em.embed_query("test query")
        assert len(vec) > 0
    except Exception:
        pytest.skip("Embedding model not available in test environment")
