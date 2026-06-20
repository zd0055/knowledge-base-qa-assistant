"""嵌入模型管理模块"""

import logging
import os
from typing import Any, Dict, List

from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """嵌入模型管理器，支持本地 HuggingFace 模型"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._embeddings = None

    def _create_embeddings(self, model_name: str, device: str, cfg: dict):
        """创建 HuggingFaceEmbeddings 实例"""
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": cfg.get("batch_size", 32),
            },
        )

    def get_embeddings(self):
        """获取嵌入模型实例（惰性初始化，支持镜像回退）"""
        if self._embeddings is not None:
            return self._embeddings

        cfg = self.config.get("embedding", {})
        model_name = cfg.get("model_name", "BAAI/bge-small-zh-v1.5")
        device = cfg.get("device", "cpu")

        # 尝试主端点
        logger.info("加载嵌入模型: %s (device=%s)", model_name, device)
        try:
            self._embeddings = self._create_embeddings(model_name, device, cfg)
            return self._embeddings
        except Exception as e:
            logger.warning("从 HuggingFace 加载失败: %s", e)

        # 尝试镜像端点
        logger.info("尝试从镜像 hf-mirror.com 下载...")
        old_endpoint = os.environ.get("HF_ENDPOINT", "")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        try:
            self._embeddings = self._create_embeddings(model_name, device, cfg)
            if old_endpoint:
                os.environ["HF_ENDPOINT"] = old_endpoint
            return self._embeddings
        except Exception as e2:
            if old_endpoint:
                os.environ["HF_ENDPOINT"] = old_endpoint
            raise RuntimeError(
                f"嵌入模型 '{model_name}' 下载失败。请手动下载或检查网络连接。\n"
                f"手动下载地址: https://huggingface.co/{model_name}\n"
                f"国内镜像: https://hf-mirror.com/{model_name}\n"
                f"下载后放入: ~/.cache/huggingface/hub/\n"
                f"错误: {e2}"
            ) from e2

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文本"""
        emb = self.get_embeddings()
        return emb.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """嵌入查询文本"""
        emb = self.get_embeddings()
        return emb.embed_query(text)

    @property
    def dimension(self) -> int:
        """获取嵌入维度"""
        sample = self.embed_texts(["test"])
        return len(sample[0])

    def __repr__(self) -> str:
        cfg = self.config.get("embedding", {})
        return f"EmbeddingManager(model={cfg.get('model_name')}, device={cfg.get('device')})"
