"""LLM generator tests."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

import pytest

from src.generation.generator import (
    build_prompt, LLMFactory, _check_quota,
    QuotaExceededError, reset_daily_quota, _track_usage,
)


@pytest.fixture
def config():
    from src.config_loader import load_config
    return load_config()


def test_build_prompt():
    """Verify prompt construction includes context and question."""
    from langchain_core.documents import Document
    docs = [Document(page_content="Test content", metadata={"source": "test.md"})]
    prompt = build_prompt("What is this?", docs)
    assert "Test content" in prompt
    assert "What is this?" in prompt
    assert "test.md" in prompt


def test_build_prompt_empty_docs():
    """Verify empty docs list produces basic prompt."""
    prompt = build_prompt("Hello", [])
    assert "Hello" in prompt


def test_quota_check_below_limit(config):
    """Verify quota check passes when under limits."""
    reset_daily_quota()
    # Should not raise
    _check_quota(config, prompt_tokens=10)


def test_quota_exceeded_calls(config):
    """Verify QuotaExceededError is raised when call limit is hit."""
    reset_daily_quota()
    low_config = {"llm": {"max_daily_calls": 1, "max_daily_tokens": 100000}}
    _check_quota(low_config)  # First call passes
    _track_usage(0, 0)  # Track the call
    with pytest.raises(QuotaExceededError):
        _check_quota(low_config)  # Second call exceeds limit


def test_quota_exceeded_tokens(config):
    """Verify QuotaExceededError is raised when token limit is hit."""
    reset_daily_quota()
    low_config = {"llm": {"max_daily_calls": 100, "max_daily_tokens": 50}}
    _check_quota(low_config, prompt_tokens=40)  # Passes (40 < 50)
    _track_usage(40, 10)  # Track 50 tokens
    with pytest.raises(QuotaExceededError):
        _check_quota(low_config, prompt_tokens=20)  # Fails (50+20 > 50)


def test_llm_factory_unknown_provider():
    """Verify unknown provider raises ValueError."""
    with pytest.raises(ValueError):
        LLMFactory.create({"llm": {"provider": "unknown"}})
