"""文本嵌入：默认用 fastembed 在本地 CPU 上推理，也可以改用 OpenAI 兼容的 embeddings 接口。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import openai

from gleaner.config import Config

# GitHub Actions 中缓存这个目录，避免每次运行都重新下载模型
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "paper-gleaner" / "models"


class EmbeddingError(Exception):
    """加载嵌入模型或调用嵌入接口失败。"""


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """返回形状为 (len(texts), dim) 的矩阵，每行已做 L2 归一化。"""
        ...


class FastEmbedEmbedder:
    def __init__(self, model: str, *, cache_dir: Path | None = None, batch_size: int = 64) -> None:
        self.model = model
        self.cache_dir = cache_dir or Path(os.environ.get("GLEANER_MODEL_CACHE", DEFAULT_CACHE_DIR))
        self.batch_size = batch_size
        self._model: Any = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        if self._model is None:
            from fastembed import TextEmbedding  # 导入较慢，用到时再加载

            try:
                self._model = TextEmbedding(model_name=self.model, cache_dir=str(self.cache_dir))
            except Exception as e:  # fastembed 下载或加载失败时抛出的异常类型不固定
                raise EmbeddingError(f"加载嵌入模型 {self.model} 失败：{e}") from e
        vectors = list(self._model.embed(texts, batch_size=self.batch_size))
        return normalize(np.asarray(vectors, dtype=np.float32))


class ApiEmbedder:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any = None,
        batch_size: int = 64,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        if client is None:
            if not api_key:
                raise EmbeddingError("使用 API 嵌入需要设置 EMBEDDING_API_KEY 或 LLM_API_KEY")
            client = openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=3)
        self._client = client

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            try:
                response = self._client.embeddings.create(model=self.model, input=batch)
            except openai.OpenAIError as e:
                raise EmbeddingError(f"调用嵌入接口 {self.model} 失败：{e}") from e
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return normalize(np.asarray(vectors, dtype=np.float32))


def build_embedder(config: Config) -> Embedder:
    embedding = config.ranking.embedding
    if embedding.provider == "local":
        return FastEmbedEmbedder(embedding.model)
    api_key = embedding.api_key or config.llm.api_key
    return ApiEmbedder(
        embedding.model,
        api_key=api_key.get_secret_value() if api_key else None,
        base_url=embedding.base_url or config.llm.base_url,
    )


def normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
