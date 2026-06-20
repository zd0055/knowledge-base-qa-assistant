"""LLM generation module - supports DeepSeek API and Ollama local models"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

# In-memory daily cost tracker
_daily_call_count = 0
_daily_token_count = 0


class QuotaExceededError(RuntimeError):
    """Raised when daily LLM quota is exceeded."""
    pass


def _check_quota(config: Dict[str, Any], prompt_tokens: int = 0):
    """Check daily call/token quota before making an API call."""
    global _daily_call_count, _daily_token_count
    max_calls = config.get("llm", {}).get("max_daily_calls", 0)
    max_tokens = config.get("llm", {}).get("max_daily_tokens", 0)
    if max_calls > 0 and _daily_call_count >= max_calls:
        raise QuotaExceededError("Daily call limit reached. Try again tomorrow.")
    if max_tokens > 0 and _daily_token_count + prompt_tokens > max_tokens:
        raise QuotaExceededError("Daily token limit reached. Try again tomorrow.")


def _track_usage(prompt_tokens: int, completion_tokens: int):
    """Track daily API usage."""
    global _daily_call_count, _daily_token_count
    _daily_call_count += 1
    _daily_token_count += prompt_tokens + completion_tokens


def reset_daily_quota():
    """Reset daily usage counters (call at midnight or from monitoring)."""
    global _daily_call_count, _daily_token_count
    _daily_call_count = 0
    _daily_token_count = 0


class BaseGenerator(ABC):
    """Base generator abstract class."""

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Synchronous generation."""
        ...

    @abstractmethod
    def stream(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        """Streaming generation."""
        ...


class DeepSeekGenerator(BaseGenerator):
    """DeepSeek API generator (OpenAI-compatible protocol)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        dsc = config.get("llm", {}).get("deepseek", {})
        api_key = dsc.get("api_key", "")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is not configured.")
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model = dsc.get("model", "deepseek-chat")
        self.temperature = dsc.get("temperature", 0.1)
        self.max_tokens = dsc.get("max_tokens", 4096)
        self.top_p = dsc.get("top_p", 0.9)

    def _make_kwargs(self, override: dict) -> dict:
        return {
            "model": override.get("model", self.model),
            "temperature": override.get("temperature", self.temperature),
            "max_tokens": override.get("max_tokens", self.max_tokens),
            "top_p": override.get("top_p", self.top_p),
        }

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        params = self._make_kwargs(kwargs)
        _check_quota(self.config)
        response = self.client.chat.completions.create(messages=messages, **params)
        content = response.choices[0].message.content or ""
        usage = response.usage
        if usage:
            _track_usage(usage.prompt_tokens, usage.completion_tokens)
            logger.info("DeepSeek: input=%d output=%d tokens, model=%s", usage.prompt_tokens, usage.completion_tokens, params["model"])
        return content

    def stream(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        params = self._make_kwargs(kwargs)
        _check_quota(self.config)
        response = self.client.chat.completions.create(messages=messages, stream=True, **params)
        for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content


class OllamaGenerator(BaseGenerator):
    """Ollama local model generator."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        oc = config.get("llm", {}).get("ollama", {})
        self.base_url = oc.get("base_url", "http://localhost:11434")
        self.model = oc.get("model", "qwen2:7b")
        self.temperature = oc.get("temperature", 0.1)
        self.max_tokens = oc.get("max_tokens", 4096)
        api_key = oc.get("api_key", "ollama")
        from openai import OpenAI as OllamaClient
        self.client = OllamaClient(base_url=f"{self.base_url}/v1", api_key=api_key)

    def _make_kwargs(self, override: dict) -> dict:
        return {
            "model": override.get("model", self.model),
            "temperature": override.get("temperature", self.temperature),
            "max_tokens": override.get("max_tokens", self.max_tokens),
        }

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        params = self._make_kwargs(kwargs)
        _check_quota(self.config)
        response = self.client.chat.completions.create(messages=messages, **params)
        return response.choices[0].message.content or ""

    def stream(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        params = self._make_kwargs(kwargs)
        _check_quota(self.config)
        response = self.client.chat.completions.create(messages=messages, stream=True, **params)
        for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content


class LLMFactory:
    """Factory to create the appropriate generator based on config."""

    @staticmethod
    def create(config: Dict[str, Any]) -> BaseGenerator:
        provider = config.get("llm", {}).get("provider", "deepseek")
        if provider == "deepseek":
            logger.info("Using LLM backend: DeepSeek API")
            return DeepSeekGenerator(config)
        elif provider == "ollama":
            logger.info("Using LLM backend: Ollama (%s)", config.get("llm", {}).get("ollama", {}).get("model", "qwen2:7b"))
            return OllamaGenerator(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider} (options: deepseek, ollama)")


_prompt_template = """You are a knowledge base Q&A assistant. Answer concisely based on the reference documents below.
## Guidelines
- If the documents do not contain enough information, state that directly - do not fabricate answers
- When quoting document content, indicate the source filename
- If the question is unrelated to the documents, politely guide the user
## Reference Documents
{context}

## Question
{question}

## Answer"""


def build_prompt(question: str, documents: List[Any]) -> str:
    """Build RAG prompt with context and question."""
    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "Unknown")
        context_parts.append(f"[{i}] Source: {source}")
        context_parts.append(doc.page_content)
        context_parts.append("---")
    context = "\n".join(context_parts)
    return _prompt_template.format(context=context, question=question)
