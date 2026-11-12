"""Embedding boundaries for real API calls and deterministic offline tests."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

from app.core.config import EmbeddingSettings


class EmbeddingError(ValueError):
    """Raised when an Embedder returns unusable or mismatched data."""


class EmbeddingServiceError(RuntimeError):
    """A safe, user-readable remote Embedding service failure."""


class Embedder(Protocol):
    """The minimum boundary shared by document and query Embedding paths."""

    @property
    def model_name(self) -> str:
        """Return the exact configured model identity."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed an ordered batch of future index documents."""

    def embed_query(self, text: str) -> list[float]:
        """Embed exactly one similarity-check query."""


def create_embedding_client(settings: EmbeddingSettings) -> OpenAI:
    """Create a client that is independent from the chat-model settings."""
    return OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=settings.timeout_seconds,
        max_retries=0,
    )


class OpenAICompatibleEmbedder:
    """Batch documents and queries through an OpenAI-compatible endpoint."""

    def __init__(
        self,
        client: OpenAI,
        *,
        model_name: str,
        batch_size: int = 16,
        expected_dimension: int | None = None,
        document_task: str = "retrieval.passage",
        query_task: str = "retrieval.query",
        embedding_type: str = "float",
        normalized: bool = True,
        truncate: bool = False,
    ) -> None:
        cleaned_model = model_name.strip()
        if not cleaned_model:
            raise ValueError("Embedding 模型名称不能为空")
        if isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("Embedding batch_size 必须大于 0")
        if expected_dimension is not None and (
            isinstance(expected_dimension, bool) or expected_dimension <= 0
        ):
            raise ValueError("期望向量维度必须大于 0")
        if not document_task.strip() or not query_task.strip():
            raise ValueError("文档和查询 Embedding task 不能为空")
        if embedding_type != "float":
            raise ValueError("当前只支持 float Embedding")

        self._client = client
        self._model_name = cleaned_model
        self.batch_size = batch_size
        self.expected_dimension = expected_dimension
        self.document_task = document_task
        self.query_task = query_task
        self.embedding_type = embedding_type
        self.normalized = normalized
        self.truncate = truncate
        self.document_batch_count = 0
        self.query_call_count = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        documents = _validate_texts(texts, label="Chunk 文本")
        vectors: list[list[float]] = []
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start : start + self.batch_size]
            self.document_batch_count += 1
            vectors.extend(self._request(batch, task=self.document_task))
        if len(vectors) != len(documents):
            raise EmbeddingError(
                "Embedding 返回数量与输入 Chunk 数量不一致："
                f"输入 {len(documents)}，返回 {len(vectors)}"
            )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        query = _validate_text(text, label="查询")
        self.query_call_count += 1
        vectors = self._request([query], task=self.query_task)
        if len(vectors) != 1:
            raise EmbeddingError(
                f"单个查询必须返回 1 个向量，实际返回 {len(vectors)}"
            )
        return vectors[0]

    def _request(
        self,
        texts: Sequence[str],
        *,
        task: str,
    ) -> list[list[float]]:
        extra_body: dict[str, object] = {
            "task": task,
            "embedding_type": self.embedding_type,
            "normalized": self.normalized,
            "truncate": self.truncate,
        }
        if self.expected_dimension is not None:
            extra_body["dimensions"] = self.expected_dimension
        try:
            response = self._client.embeddings.create(
                model=self.model_name,
                input=list(texts),
                extra_body=extra_body,
            )
        except AuthenticationError:
            raise EmbeddingServiceError(
                "Embedding API 鉴权失败。请检查本地 Embedding API Key。"
            ) from None
        except APITimeoutError:
            raise EmbeddingServiceError("Embedding 请求超时。") from None
        except RateLimitError:
            raise EmbeddingServiceError("Embedding 服务暂时限流。") from None
        except APIConnectionError:
            raise EmbeddingServiceError(
                "无法连接 Embedding 服务。请检查网络和 Base URL。"
            ) from None
        except APIStatusError as error:
            raise EmbeddingServiceError(
                f"Embedding 服务返回 HTTP {error.status_code}。"
            ) from None
        except OpenAIError:
            raise EmbeddingServiceError("Embedding 服务调用失败。") from None

        indexed: dict[int, list[float]] = {}
        for item in response.data:
            index = item.index
            if index in indexed:
                raise EmbeddingError(f"Embedding 响应包含重复 index：{index}")
            if index < 0 or index >= len(texts):
                raise EmbeddingError(f"Embedding 响应 index 越界：{index}")
            indexed[index] = list(item.embedding)
        expected_indexes = list(range(len(texts)))
        if sorted(indexed) != expected_indexes:
            raise EmbeddingError(
                "Embedding 响应 index 不连续，无法安全还原输入顺序"
            )

        vectors = [indexed[index] for index in expected_indexes]
        _validate_response_dimensions(
            vectors,
            expected_dimension=self.expected_dimension,
        )
        return vectors


class SyntheticHashEmbedder:
    """A deterministic lexical test double, not a semantic Embedding model."""

    _TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]", re.IGNORECASE)

    def __init__(self, dimension: int = 256) -> None:
        if isinstance(dimension, bool) or dimension <= 0:
            raise ValueError("合成向量维度必须大于 0")
        self.dimension = dimension
        self.document_call_count = 0
        self.query_call_count = 0

    @property
    def model_name(self) -> str:
        return f"synthetic-hash-v1-d{self.dimension}"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        documents = _validate_texts(texts, label="Chunk 文本")
        self.document_call_count += 1
        return [self._embed(text) for text in documents]

    def embed_query(self, text: str) -> list[float]:
        query = _validate_text(text, label="查询")
        self.query_call_count += 1
        return self._embed(query)

    def _embed(self, text: str) -> list[float]:
        tokens = self._TOKEN_PATTERN.findall(text.lower())
        features = [*tokens]
        features.extend(
            f"{left}\u241f{right}"
            for left, right in zip(tokens, tokens[1:], strict=False)
        )
        vector = [0.0] * self.dimension
        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], "big") % self.dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[bucket] += sign
        return vector


def _validate_texts(texts: Sequence[str], *, label: str) -> list[str]:
    values = list(texts)
    if not values:
        raise EmbeddingError(f"{label}列表不能为空")
    return [_validate_text(value, label=label) for value in values]


def _validate_text(text: str, *, label: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise EmbeddingError(f"{label}不能为空")
    return text


def _validate_response_dimensions(
    vectors: Sequence[Sequence[float]],
    *,
    expected_dimension: int | None,
) -> None:
    if not vectors:
        raise EmbeddingError("Embedding 响应不能是空向量列表")
    dimension = len(vectors[0])
    if dimension == 0:
        raise EmbeddingError("Embedding 向量维度不能为 0")
    if any(len(vector) != dimension for vector in vectors):
        raise EmbeddingError("Embedding 返回了维度不一致的向量")
    if expected_dimension is not None and dimension != expected_dimension:
        raise EmbeddingError(
            "Embedding 实际维度与配置不一致："
            f"配置 {expected_dimension}，实际 {dimension}"
        )
    for vector in vectors:
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in vector
        ):
            raise EmbeddingError("Embedding 向量必须只包含有限数值")
