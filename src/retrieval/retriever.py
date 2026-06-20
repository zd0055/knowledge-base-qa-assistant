"""向量检索模块"""

import logging
from typing import Any, Dict, List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.embedding import EmbeddingManager

logger = logging.getLogger(__name__)


class Retriever:
    """向量检索器，支持相似度搜索和 MMR 多样性搜索"""

    def __init__(self, config: Dict[str, Any], embedding_manager: EmbeddingManager):
        self.config = config
        self.embedding_manager = embedding_manager
        vs_cfg = config.get("vector_store", {})
        self.top_k = vs_cfg.get("top_k", 5)
        self.use_mmr = vs_cfg.get("use_mmr", True)
        self.mmr_fetch_multiplier = vs_cfg.get("mmr_fetch_multiplier", 2)
        self.mmr_diversity = vs_cfg.get("mmr_diversity", 0.3)
        self._vector_store: Chroma | None = None

    def get_vector_store(self) -> Chroma:
        """获取向量存储实例（惰性初始化）"""
        if self._vector_store is not None:
            return self._vector_store

        persist_dir = self.config.get("vector_store", {}).get(
            "persist_directory", "data/chroma_db"
        )

        self._vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embedding_manager.get_embeddings(),
        )
        return self._vector_store

    def retrieve(self, query: str, k: int | None = None) -> List[Document]:
        """检索与查询最相关的文档块"""
        vs = self.get_vector_store()
        top_k = k or self.top_k

        if self.use_mmr:
            docs = vs.max_marginal_relevance_search(
                query,
                k=top_k,
                fetch_k=top_k * self.mmr_fetch_multiplier,
                lambda_mult=self.mmr_diversity,
            )
            logger.debug(
                "MMR 检索: query='%s...' k=%d fetch_k=%d diversity=%.2f",
                query[:50], top_k, top_k * self.mmr_fetch_multiplier, self.mmr_diversity,
            )
        else:
            docs = vs.similarity_search(query, k=top_k)
            logger.debug(
                "相似度检索: query='%s...' k=%d", query[:50], top_k
            )

        return docs

    def retrieve_with_scores(self, query: str, k: int | None = None) -> List[tuple]:
        """检索并返回带距离分数的结果"""
        vs = self.get_vector_store()
        top_k = k or self.top_k
        return vs.similarity_search_with_relevance_scores(query, k=top_k)

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取向量集合统计信息"""
        try:
            vs = self.get_vector_store()
            collection = vs.get()
            sources = {}
            for meta in collection.get("metadatas", []):
                if meta and "source" in meta:
                    src = meta["source"]
                    sources[src] = sources.get(src, 0) + 1
            return {
                "total_chunks": len(collection.get("ids", [])),
                "sources": sources,
                "num_documents": len(sources),
            }
        except Exception as e:
            logger.warning("获取向量库统计失败: %s", e)
            return {"total_chunks": 0, "sources": {}, "num_documents": 0}

    def clear(self):
        """清空向量库"""
        if self._vector_store is not None:
            try:
                collection_data = self._vector_store.get()
                ids = collection_data.get("ids", [])
                if ids:
                    self._vector_store.delete(ids=ids)
                logger.info("向量库已清空 (%d 条)", len(ids))
            except Exception as e:
                logger.error("清空向量库失败: %s", e)
            self._vector_store = None

    def __repr__(self) -> str:
        return (
            f"Retriever(top_k={self.top_k}, use_mmr={self.use_mmr}, "
            f"mmr_diversity={self.mmr_diversity})"
        )
