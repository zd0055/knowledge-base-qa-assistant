"""多格式文档加载模块"""

import logging
from pathlib import Path
from typing import Dict, List, Set

# PDF magic bytes for basic content validation
_PDF_MAGIC = b"%PDF"


from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _make_loader(loader_cls, file_path: str):
    """创建加载器实例，TextLoader 使用 UTF-8 编码"""
    if loader_cls is TextLoader:
        return loader_cls(file_path, encoding="utf-8")
    return loader_cls(file_path)


_EXTENSION_MAP: Dict[str, type] = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".docx": Docx2txtLoader,
    ".csv": CSVLoader,
}


class DocumentLoader:
    """Unified document loader with extension-based parser selection and basic validation."""
    DEFAULT_EXTENSIONS: Set[str] = set(_EXTENSION_MAP.keys())

    def __init__(self, supported_extensions: Set[str] | None = None):
        self.supported_extensions = supported_extensions or set(_EXTENSION_MAP.keys())

    def load_file(self, file_path: str | Path) -> List[Document]:
        """加载单个文件，返回 Document 列表"""
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext not in self.supported_extensions:
            logger.warning("不支持的文件格式: %s (跳过: %s)", ext, path.name)
            return []

        loader_cls = _EXTENSION_MAP.get(ext)
        if loader_cls is None:
            logger.warning("未找到对应的加载器: %s (跳过: %s)", ext, path.name)
            return []

        # Basic file header validation
        if ext == ".pdf":
            with open(path, "rb") as fh:
                header = fh.read(4)
                if header != _PDF_MAGIC:
                    logger.warning("Invalid PDF header in %s (skipping)", path.name)
                    return []

        logger.info("加载文档: %s", path.name)
        try:
            loader = _make_loader(loader_cls, str(path))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = path.name
                doc.metadata["file_path"] = str(path)
                doc.metadata["file_type"] = ext
            return docs
        except Exception as e:
            logger.error("加载失败: %s - %s", path.name, e)
            return []

    def load_directory(self, directory: str | Path) -> List[Document]:
        """加载目录下所有支持的文档"""
        path = Path(directory)
        if not path.exists():
            logger.warning("文档目录不存在: %s", path)
            return []

        all_docs: List[Document] = []
        for ext in self.supported_extensions:
            for file_path in sorted(path.rglob(f"*{ext}")):
                docs = self.load_file(file_path)
                all_docs.extend(docs)

        logger.info("共加载 %d 个文档片段，来自 %s", len(all_docs), path)
        return all_docs

    @staticmethod
    def get_supported_descriptions() -> str:
        """返回支持的格式说明"""
        exts = sorted(_EXTENSION_MAP.keys())
        return ", ".join(exts)
