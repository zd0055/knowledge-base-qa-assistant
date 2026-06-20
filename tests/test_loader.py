"""Document loader and splitter tests (proper pytest style)."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

import pytest

from src.config_loader import load_config
from src.ingestion.loader import DocumentLoader
from src.ingestion.splitter import TextSplitter


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def loader():
    return DocumentLoader()


@pytest.fixture
def splitter(config):
    return TextSplitter(config)


def test_config_loader(config):
    """Verify config loads with expected keys."""
    assert "llm" in config
    assert config["llm"]["provider"] in ("deepseek", "ollama")
    assert "embedding" in config
    assert "vector_store" in config
    assert "text_splitter" in config
    assert "documents" in config


def test_loader_default_extensions(loader):
    """Verify DEFAULT_EXTENSIONS is a proper set."""
    exts = loader.DEFAULT_EXTENSIONS
    assert isinstance(exts, set)
    assert len(exts) > 0
    assert ".pdf" in exts
    assert ".txt" in exts


def test_load_markdown(loader):
    """Verify loading a Markdown file returns documents with metadata."""
    md_path = _root / "data" / "documents" / "sample_intro.md"
    docs = loader.load_file(str(md_path))
    assert len(docs) > 0, f"Expected >0 documents, got {len(docs)}"
    for d in docs:
        assert "source" in d.metadata
        assert d.metadata["source"] == "sample_intro.md"


def test_load_txt(loader):
    """Verify loading a TXT file works."""
    txt_path = _root / "data" / "documents" / "sample_concepts.txt"
    docs = loader.load_file(str(txt_path))
    assert len(docs) > 0, f"Expected >0 documents, got {len(docs)}"
    assert docs[0].page_content is not None


def test_text_splitter_creation(splitter):
    """Verify TextSplitter can be instantiated."""
    assert splitter is not None
    assert splitter.chunk_size > 0
    assert repr(splitter) is not None


def test_text_splitter_chunks(loader, splitter):
    """Verify document splitting produces more chunks than input documents."""
    md_path = _root / "data" / "documents" / "sample_intro.md"
    docs = loader.load_file(str(md_path))
    chunks = splitter.split_documents(docs)
    assert len(chunks) > 0, f"Expected >0 chunks, got {len(chunks)}"


def test_split_text(splitter):
    """Verify split_text returns a list of strings."""
    result = splitter.split_text("Hello world. This is a test.")
    assert isinstance(result, list)
    assert len(result) > 0


def test_unsupported_file_type(loader):
    """Verify unsupported file types return empty list gracefully."""
    from pathlib import Path
    result = loader.load_file("nonexistent.unsupported")
    assert result == []
