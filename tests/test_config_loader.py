"""Config loader tests."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

import pytest

from src.config_loader import load_config, resolve_path, get_project_root


def test_load_config():
    """Verify config loads all required sections."""
    config = load_config()
    assert "llm" in config
    assert "embedding" in config
    assert "vector_store" in config
    assert "text_splitter" in config
    assert "documents" in config
    assert "ui" in config
    assert "logging" in config


def test_llm_config():
    """Verify LLM config contains expected keys."""
    config = load_config()
    llm = config["llm"]
    assert llm["provider"] in ("deepseek", "ollama")
    assert "deepseek" in llm
    assert "ollama" in llm
    assert "max_daily_calls" in llm
    assert "max_daily_tokens" in llm


def test_resolve_path_relative():
    """Verify resolve_path produces absolute path from relative."""
    config = load_config()
    result = resolve_path(config, "documents.directory", "data/documents")
    path = Path(result)
    assert path.is_absolute()
    assert path.name == "documents"


def test_resolve_path_default():
    """Verify resolve_path returns default when key is missing."""
    config = load_config()
    result = resolve_path(config, "nonexistent.key", "fallback/path")
    assert result == "fallback/path"


def test_get_project_root():
    """Verify get_project_root returns an existing directory."""
    root = get_project_root()
    assert root.exists()
    assert (root / "src").exists()
    assert (root / "config").exists()
