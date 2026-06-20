"""RAG chain: connects retrieval, prompt building, and generation."""

import json
import logging
from typing import Any, Dict, Generator, List

from langchain_core.documents import Document

from src.generation.generator import BaseGenerator, LLMFactory, build_prompt
from src.retrieval import Retriever

logger = logging.getLogger(__name__)


class RAGChain:
    """RAG chain: retrieve -> build prompt -> generate answer."""

    def __init__(
        self,
        config: Dict[str, Any],
        retriever: Retriever,
        generator: BaseGenerator | None = None,
    ):
        self.config = config
        self.retriever = retriever
        self.generator = generator or LLMFactory.create(config)

    def query(self, question: str) -> Dict[str, Any]:
        """Execute a full RAG query.

        Returns:
            {"answer": str, "sources": List[Document], "source_count": int}
        """
        docs = self.retriever.retrieve(question)
        logger.info("Retrieved %d relevant document chunks", len(docs))

        if not docs:
            return {
                "answer": "No relevant documents found in the knowledge base. Please try rephrasing or upload related documents.",
                "sources": [],
                "source_count": 0,
            }

        prompt = build_prompt(question, docs)
        messages = [{"role": "user", "content": prompt}]
        answer = self.generator.generate(messages)

        return {"answer": answer, "sources": docs, "source_count": len(docs)}

    def query_stream(self, question: str) -> Generator[str, None, None]:
        """Streaming RAG query.

        Yields JSON strings for structured parsing:
          {"type": "sources", "count": N}
          {"type": "answer", "text": "..."}
        """
        docs = self.retriever.retrieve(question)
        logger.info("Streaming query: retrieved %d relevant document chunks", len(docs))

        if not docs:
            yield json.dumps({
                "type": "no_docs",
                "message": "No relevant documents found. Please try rephrasing or upload documents first.",
            })
            return

        prompt = build_prompt(question, docs)
        messages = [{"role": "user", "content": prompt}]

        yield json.dumps({"type": "sources", "count": len(docs)})

        for chunk in self.generator.stream(messages):
            yield json.dumps({"type": "answer", "text": chunk})

    def query_with_sources(
        self, question: str
    ) -> tuple[str, List[Document]]:
        """Execute RAG and return (answer, sources) tuple."""
        result = self.query(question)
        return result["answer"], result["sources"]

    def get_generator(self) -> BaseGenerator:
        return self.generator

    def __repr__(self) -> str:
        provider = self.config.get("llm", {}).get("provider", "deepseek")
        return f"RAGChain(provider={provider}, retriever={self.retriever})"
