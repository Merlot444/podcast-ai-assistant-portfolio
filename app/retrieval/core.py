"""Formal Day 8 retrieval interface built on the persisted Day 7 index."""

from __future__ import annotations

from dataclasses import dataclass

from app.indexing.embedding import Embedder
from app.indexing.index import PersistentVectorIndex
from app.schemas.chunk import TranscriptChunk
from app.schemas.retrieval import RetrievalResult


class RetrievalError(ValueError):
    """Raised when a query or retriever/index pairing is not safe to use."""


@dataclass(frozen=True)
class Retriever:
    """Reuse one loaded index and embed exactly one vector per query call."""

    vector_index: PersistentVectorIndex
    embedder: Embedder

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Return stable cosine-ranked Chunks with complete source metadata."""
        self._validate_request(query, top_k)
        self._validate_index_mapping()
        self._validate_embedding_compatibility()

        candidates = self.vector_index.similarity_check(
            query,
            embedder=self.embedder,
            top_k=top_k,
        )
        return [
            RetrievalResult.from_chunk(
                rank=candidate.rank,
                score=candidate.score,
                row_index=candidate.row_index,
                chunk=candidate.chunk,
            )
            for candidate in candidates
        ]

    @staticmethod
    def _validate_request(query: str, top_k: int) -> None:
        if not isinstance(query, str) or not query.strip():
            raise RetrievalError("查询不能为空")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise RetrievalError("top_k 必须是大于 0 的整数")

    def _validate_index_mapping(self) -> None:
        index = self.vector_index.faiss_index
        manifest = self.vector_index.manifest
        if index.ntotal != manifest.vector_count:
            raise RetrievalError("FAISS 向量数量与 manifest 不一致")
        if index.d != manifest.vector_dimension:
            raise RetrievalError("FAISS 向量维度与 manifest 不一致")
        if len(manifest.chunks) != manifest.vector_count:
            raise RetrievalError("manifest Chunk 数量与向量数量不一致")
        if len(manifest.chunk_id_order) != manifest.vector_count:
            raise RetrievalError("chunk_id_order 数量与向量数量不一致")
        for row_index, chunk in enumerate(manifest.chunks):
            if chunk.chunk_index != row_index:
                raise RetrievalError("Chunk 顺序与 FAISS 行号不一致")
            if manifest.chunk_id_order[row_index] != chunk.chunk_id:
                raise RetrievalError("chunk_id_order 与 Chunk 映射不一致")

    def _validate_embedding_compatibility(self) -> None:
        manifest = self.vector_index.manifest
        if self.embedder.model_name != manifest.embedding_model:
            raise RetrievalError(
                "查询 Embedding 模型与索引不一致："
                f"索引 {manifest.embedding_model}，"
                f"查询 {self.embedder.model_name}"
            )
        self._match_optional_setting(
            "期望查询向量维度",
            manifest.vector_dimension,
            getattr(
                self.embedder,
                "expected_dimension",
                manifest.vector_dimension,
            ),
        )
        self._match_optional_setting(
            "文档 Embedding task",
            manifest.embedding_document_task,
            getattr(self.embedder, "document_task", None),
        )
        self._match_optional_setting(
            "查询 Embedding task",
            manifest.embedding_query_task,
            getattr(self.embedder, "query_task", None),
        )
        self._match_optional_setting(
            "查询 Embedding 输出类型",
            manifest.embedding_type,
            getattr(self.embedder, "embedding_type", None),
        )
        self._match_optional_setting(
            "供应商归一化配置",
            manifest.provider_normalized,
            getattr(self.embedder, "normalized", None),
        )
        self._match_optional_setting(
            "供应商截断配置",
            manifest.provider_truncate,
            getattr(self.embedder, "truncate", None),
        )

    @staticmethod
    def _match_optional_setting(
        label: str,
        expected: object | None,
        actual: object | None,
    ) -> None:
        if expected is not None and actual != expected:
            raise RetrievalError(
                f"{label}与索引不一致：索引 {expected}，查询 {actual}"
            )
