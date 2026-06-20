"""RAG chain tests."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

import pytest


@pytest.fixture
def config():
    from src.config_loader import load_config
    return load_config()


def test_rag_chain_creation(config):
    """Verify RAGChain can be instantiated."""
    from src.embedding import EmbeddingManager
    from src.retrieval import Retriever
    from src.rag import RAGChain
    em = EmbeddingManager(config)
    retriever = Retriever(config, em)
    chain = RAGChain(config, retriever)
    assert chain is not None
    assert repr(chain) is not None


def test_rag_chain_no_docs(config):
    """Verify query returns helpful message when no documents exist."""
    from src.embedding import EmbeddingManager
    from src.retrieval import Retriever
    from src.rag import RAGChain
    em = EmbeddingManager(config)
    retriever = Retriever(config, em)
    chain = RAGChain(config, retriever)
    try:
        result = chain.query("test question")
        assert "sources" in result
        assert "answer" in result
        assert result["source_count"] == 0
    except (RuntimeError, OSError) as e:
        err_str = str(e)
        if 'download' in err_str.lower() or 'cannot send' in err_str.lower():
            pytest.skip("Embedding model not available (no network)")
        raise
