"""文本分块模块（中文优化）"""

import logging
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class TextSplitter:
    """中文优化的递归文本分割器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        splitter_cfg = config.get("text_splitter", {})

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=splitter_cfg.get("chunk_size", 512),
            chunk_overlap=splitter_cfg.get("chunk_overlap", 128),
            separators=splitter_cfg.get(
                "separators",
                ["\n\n", "\n", "。", "！", "？", ".", " ", ""],
            ),
            length_function=len,
            add_start_index=True,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """将文档拆分为块"""
        if not documents:
            return []

        chunks = self._splitter.split_documents(documents)
        logger.info(
            "分块完成: %d 个文档 -> %d 个块 (chunk_size=%d, overlap=%d)",
            len(documents),
            len(chunks),
            self.config.get("text_splitter", {}).get("chunk_size", 512),
            self.config.get("text_splitter", {}).get("chunk_overlap", 128),
        )
        return chunks

    def split_text(self, text: str) -> List[str]:
        """直接分割文本"""
        return self._splitter.split_text(text)

    def create_documents(
        self, texts: List[str], metadatas: List[Dict] | None = None
    ) -> List[Document]:
        """从文本列表创建文档块"""
        return self._splitter.create_documents(texts, metadatas)

    @property
    def chunk_size(self) -> int:
        return self.config.get("text_splitter", {}).get("chunk_size", 512)

    def __repr__(self) -> str:
        sc = self.config.get("text_splitter", {})
        return (
            f"TextSplitter(chunk_size={sc.get('chunk_size')}, "
            f"overlap={sc.get('chunk_overlap')})"
        )
