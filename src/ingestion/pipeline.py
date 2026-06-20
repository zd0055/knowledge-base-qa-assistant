"""文档入库管道：加载 → 分块 → 嵌入 → 存入向量库"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Set

from langchain_core.documents import Document

from src.config_loader import resolve_path
from src.ingestion.loader import DocumentLoader
from src.ingestion.splitter import TextSplitter

logger = logging.getLogger(__name__)


def _doc_id(content: str, metadata: dict) -> str:
    """根据内容和元数据生成文档唯一 ID"""
    raw = content + str(sorted(metadata.items()))
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


class IngestionPipeline:
    """完整的文档入库管道"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.loader = DocumentLoader(
            supported_extensions=set(
                config.get("documents", {}).get("supported_extensions", [])
            )
        )
        self.splitter = TextSplitter(config)

    def get_existing_ids(self, vector_store) -> Set[str]:
        """获取向量库中已有的文档 ID"""
        try:
            collection = vector_store.get()
            return set(collection.get("ids", []))
        except Exception:
            return set()

    def run(
        self,
        vector_store,
        file_paths: List[str] | None = None,
    ) -> Dict[str, int]:
        """执行入库流程

        Args:
            vector_store: Chroma 向量库实例
            file_paths: 指定文件列表；None 则扫描配置目录

        Returns:
            {"loaded": N, "chunks": N, "new": N, "skipped": N}
        """
        # 1. 加载文档
        if file_paths:
            all_docs: List[Document] = []
            for fp in file_paths:
                all_docs.extend(self.loader.load_file(fp))
        else:
            doc_dir = resolve_path(
                self.config, "documents.directory", "data/documents"
            )
            all_docs = self.loader.load_directory(doc_dir)

        if not all_docs:
            logger.info("没有新文档需要处理")
            return {"loaded": 0, "chunks": 0, "new": 0, "skipped": 0}

        # 2. 分块
        chunks = self.splitter.split_documents(all_docs)

        # 3. 去重：检查向量库中已有的 ID
        existing_ids = self.get_existing_ids(vector_store)
        new_chunks: List[Document] = []
        new_ids: List[str] = []

        for chunk in chunks:
            cid = _doc_id(chunk.page_content, chunk.metadata)
            if cid not in existing_ids:
                new_chunks.append(chunk)
                new_ids.append(cid)

        if not new_chunks:
            logger.info("所有文档已在向量库中，跳过")
            return {"loaded": len(all_docs), "chunks": len(chunks), "new": 0, "skipped": len(chunks)}

        # 4. 入库
        texts = [doc.page_content for doc in new_chunks]
        metadatas = [doc.metadata for doc in new_chunks]

        vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=new_ids,
        )

        logger.info(
            "入库完成: 新增 %d / 跳过 %d (共 %d 块)",
            len(new_chunks),
            len(chunks) - len(new_chunks),
            len(chunks),
        )

        return {
            "loaded": len(all_docs),
            "chunks": len(chunks),
            "new": len(new_chunks),
            "skipped": len(chunks) - len(new_chunks),
        }

    def delete_documents(self, vector_store, filenames: List[str]) -> int:
        """删除指定文件名的所有文档块"""
        collection = vector_store.get()
        ids_to_delete = []
        for i, meta in enumerate(collection.get("metadatas", [])):
            if meta and meta.get("source") in filenames:
                ids_to_delete.append(collection["ids"][i])

        if ids_to_delete:
            vector_store.delete(ids=ids_to_delete)
            logger.info("删除了 %d 个块（来源: %s）", len(ids_to_delete), filenames)

        return len(ids_to_delete)
