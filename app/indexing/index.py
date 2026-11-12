"""Build, persist, reload, and minimally inspect a Day 7 FAISS index."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from pydantic import ValidationError

from app.core.errors import safe_error_message
from app.indexing.embedding import Embedder, EmbeddingError
from app.schemas.chunk import ChunkedTranscript, TranscriptChunk
from app.schemas.index import VectorIndexManifest


INDEX_FILENAME = "index.faiss"
MANIFEST_FILENAME = "manifest.json"
INDEX_TYPE = "faiss_index_flat_ip"
DISTANCE_METRIC = "cosine"


class VectorIndexError(ValueError):
    """Raised when vectors, mappings, or query configuration are invalid."""


class IndexPersistenceError(RuntimeError):
    """Raised when a saved index cannot be trusted or loaded."""


@dataclass(frozen=True)
class SimilarityCheckResult:
    """One metadata-rich result from the deliberately small Day 7 check."""

    rank: int
    score: float
    row_index: int
    chunk: TranscriptChunk

    def metadata_dict(self) -> dict[str, object]:
        """Return provenance without printing the copyrighted Chunk text."""
        return {
            "rank": self.rank,
            "score": self.score,
            "row_index": self.row_index,
            "chunk_id": self.chunk.chunk_id,
            "source_segment_ids": self.chunk.source_segment_ids,
            "source_segment_start_index": self.chunk.source_segment_start_index,
            "source_segment_end_index": self.chunk.source_segment_end_index,
            "source_spans": [
                span.model_dump(mode="json") for span in self.chunk.source_spans
            ],
            "speakers": self.chunk.speakers,
            "section_titles": self.chunk.section_titles,
            "start_time": self.chunk.start_time,
            "end_time": self.chunk.end_time,
            "time_range_complete": self.chunk.time_range_complete,
        }


@dataclass
class PersistentVectorIndex:
    """A loaded FAISS index and the manifest that maps every row to a Chunk."""

    faiss_index: faiss.Index
    manifest: VectorIndexManifest

    def similarity_check(
        self,
        query: str,
        *,
        embedder: Embedder,
        top_k: int = 3,
    ) -> list[SimilarityCheckResult]:
        """Run the minimum Day 7 check without becoming a Day 8 retriever."""
        if not isinstance(query, str) or not query.strip():
            raise VectorIndexError("查询不能为空")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise VectorIndexError("top_k 必须是大于 0 的整数")
        if embedder.model_name != self.manifest.embedding_model:
            raise VectorIndexError(
                "查询 Embedding 模型与索引不一致："
                f"索引 {self.manifest.embedding_model}，"
                f"查询 {embedder.model_name}"
            )
        expected_query_task = self.manifest.embedding_query_task
        actual_query_task = getattr(embedder, "query_task", None)
        if (
            expected_query_task is not None
            and actual_query_task != expected_query_task
        ):
            raise VectorIndexError(
                "查询 Embedding task 与索引不一致："
                f"索引 {expected_query_task}，查询 {actual_query_task}"
            )
        expected_type = self.manifest.embedding_type
        actual_type = getattr(embedder, "embedding_type", None)
        if expected_type is not None and actual_type != expected_type:
            raise VectorIndexError("查询 Embedding 输出类型与索引不一致")
        expected_normalized = self.manifest.provider_normalized
        actual_normalized = getattr(embedder, "normalized", None)
        if expected_normalized is not None and actual_normalized != expected_normalized:
            raise VectorIndexError("查询 Embedding normalized 配置与索引不一致")
        expected_truncate = self.manifest.provider_truncate
        actual_truncate = getattr(embedder, "truncate", None)
        if expected_truncate is not None and actual_truncate != expected_truncate:
            raise VectorIndexError("查询 Embedding truncate 配置与索引不一致")

        vector = embedder.embed_query(query)
        matrix = _validated_normalized_matrix(
            [vector],
            expected_count=1,
            expected_dimension=self.manifest.vector_dimension,
            label="查询 Embedding",
        )

        # Search every row so equal-score ties can always be ordered by row index.
        scores, row_indexes = self.faiss_index.search(
            matrix,
            self.manifest.vector_count,
        )
        candidates = [
            (float(score), int(row_index))
            for score, row_index in zip(
                scores[0],
                row_indexes[0],
                strict=True,
            )
            if row_index >= 0
        ]
        candidates.sort(key=lambda item: (-item[0], item[1]))
        limit = min(top_k, self.manifest.vector_count)
        return [
            SimilarityCheckResult(
                rank=rank,
                score=score,
                row_index=row_index,
                chunk=self.manifest.chunks[row_index],
            )
            for rank, (score, row_index) in enumerate(
                candidates[:limit],
                start=1,
            )
        ]


def chunked_transcript_fingerprint(chunked: ChunkedTranscript) -> str:
    """Hash a canonical representation rather than one file's whitespace."""
    canonical = json.dumps(
        chunked.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_vector_index(
    chunked: ChunkedTranscript,
    *,
    embedder: Embedder,
    input_chunk_file: str,
    distance_metric: str = DISTANCE_METRIC,
) -> PersistentVectorIndex:
    """Embed ordered Day 6 Chunks and add them to an exact FAISS index."""
    if distance_metric != DISTANCE_METRIC:
        raise VectorIndexError(f"不支持的距离度量：{distance_metric}")
    if not chunked.chunks:
        raise VectorIndexError("Chunk 列表不能为空")
    if not embedder.model_name.strip():
        raise VectorIndexError("Embedding 模型名称不能为空")
    if not input_chunk_file.strip():
        raise VectorIndexError("输入 Chunk 文件名不能为空")

    vectors = embedder.embed_documents([chunk.text for chunk in chunked.chunks])
    matrix = _validated_normalized_matrix(
        vectors,
        expected_count=len(chunked.chunks),
        label="文档 Embedding",
    )
    dimension = int(matrix.shape[1])
    index = faiss.IndexFlatIP(dimension)
    index.add(matrix)

    manifest = VectorIndexManifest(
        embedding_model=embedder.model_name,
        embedding_document_task=getattr(embedder, "document_task", None),
        embedding_query_task=getattr(embedder, "query_task", None),
        embedding_type=getattr(embedder, "embedding_type", None),
        provider_normalized=getattr(embedder, "normalized", None),
        provider_truncate=getattr(embedder, "truncate", None),
        vector_dimension=dimension,
        vector_count=len(chunked.chunks),
        chunk_id_order=[chunk.chunk_id for chunk in chunked.chunks],
        input_chunk_sha256=chunked_transcript_fingerprint(chunked),
        input_chunk_file=input_chunk_file,
        chunking_algorithm=chunked.algorithm,
        episode_id=chunked.episode_id,
        source_file=chunked.source_file,
        input_segment_count=chunked.input_segment_count,
        chunk_size=chunked.chunk_size,
        overlap=chunked.overlap,
        chunks=chunked.chunks,
    )
    return PersistentVectorIndex(faiss_index=index, manifest=manifest)


def save_vector_index(
    vector_index: PersistentVectorIndex,
    directory: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Save FAISS bytes and a validated manifest beside them."""
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    index_path = destination / INDEX_FILENAME
    manifest_path = destination / MANIFEST_FILENAME
    existing = [path for path in (index_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise IndexPersistenceError(
            f"目标已有索引产物（{names}）；不会静默覆盖，请显式允许 overwrite"
        )

    _validate_index_matches_manifest(
        vector_index.faiss_index,
        vector_index.manifest,
    )
    with tempfile.TemporaryDirectory(
        dir=destination,
        prefix=".day7-index-",
    ) as temporary:
        temporary_dir = Path(temporary)
        temporary_index = temporary_dir / INDEX_FILENAME
        temporary_manifest = temporary_dir / MANIFEST_FILENAME
        try:
            faiss.write_index(
                vector_index.faiss_index,
                str(temporary_index),
            )
        except (OSError, RuntimeError) as error:
            raise IndexPersistenceError("FAISS 索引写入失败") from error

        index_sha256 = _file_sha256(temporary_index)
        saved_manifest = vector_index.manifest.model_copy(
            update={"index_sha256": index_sha256},
        )
        temporary_manifest.write_text(
            json.dumps(
                saved_manifest.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_index, index_path)
        os.replace(temporary_manifest, manifest_path)
    vector_index.manifest = saved_manifest
    return index_path, manifest_path


def load_vector_index(
    directory: str | Path,
    *,
    expected_chunks: ChunkedTranscript | None = None,
    expected_embedding_model: str | None = None,
    expected_dimension: int | None = None,
    expected_document_task: str | None = None,
    expected_query_task: str | None = None,
    expected_embedding_type: str | None = None,
    expected_provider_normalized: bool | None = None,
    expected_provider_truncate: bool | None = None,
) -> PersistentVectorIndex:
    """Load and cross-check persisted artifacts without an Embedder call."""
    source = Path(directory)
    manifest_path = source / MANIFEST_FILENAME
    index_path = source / INDEX_FILENAME
    if not manifest_path.is_file():
        raise IndexPersistenceError(f"缺少 manifest：{manifest_path}")
    if not index_path.is_file():
        raise IndexPersistenceError(f"缺少 FAISS 索引：{index_path}")

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise IndexPersistenceError("manifest 必须使用 UTF-8 编码") from error
    except json.JSONDecodeError as error:
        raise IndexPersistenceError(
            f"manifest JSON 损坏（第 {error.lineno} 行，第 {error.colno} 列）"
        ) from error
    try:
        manifest = VectorIndexManifest.model_validate(payload)
    except ValidationError as error:
        raise IndexPersistenceError(
            f"manifest 校验失败：{safe_error_message(error)}"
        ) from error
    if manifest.index_sha256 is None:
        raise IndexPersistenceError("manifest 缺少 index_sha256")
    if _file_sha256(index_path) != manifest.index_sha256:
        raise IndexPersistenceError("FAISS 索引校验和与 manifest 不一致")

    try:
        index = faiss.read_index(str(index_path))
    except (OSError, RuntimeError) as error:
        raise IndexPersistenceError("FAISS 索引文件损坏或无法读取") from error
    _validate_index_matches_manifest(index, manifest)

    if expected_chunks is not None:
        actual_fingerprint = chunked_transcript_fingerprint(expected_chunks)
        if actual_fingerprint != manifest.input_chunk_sha256:
            raise IndexPersistenceError("当前 Chunk 内容指纹与已保存索引不一致")
    if (
        expected_embedding_model is not None
        and expected_embedding_model != manifest.embedding_model
    ):
        raise IndexPersistenceError("期望 Embedding 模型与 manifest 不一致")
    if (
        expected_dimension is not None
        and expected_dimension != manifest.vector_dimension
    ):
        raise IndexPersistenceError("期望向量维度与 manifest 不一致")
    if (
        expected_document_task is not None
        and expected_document_task != manifest.embedding_document_task
    ):
        raise IndexPersistenceError("期望文档 Embedding task 与 manifest 不一致")
    if (
        expected_query_task is not None
        and expected_query_task != manifest.embedding_query_task
    ):
        raise IndexPersistenceError("期望查询 Embedding task 与 manifest 不一致")
    if (
        expected_embedding_type is not None
        and expected_embedding_type != manifest.embedding_type
    ):
        raise IndexPersistenceError("期望 Embedding 输出类型与 manifest 不一致")
    if (
        expected_provider_normalized is not None
        and expected_provider_normalized != manifest.provider_normalized
    ):
        raise IndexPersistenceError("供应商归一化配置与 manifest 不一致")
    if (
        expected_provider_truncate is not None
        and expected_provider_truncate != manifest.provider_truncate
    ):
        raise IndexPersistenceError("供应商截断配置与 manifest 不一致")

    return PersistentVectorIndex(faiss_index=index, manifest=manifest)


def _validated_normalized_matrix(
    vectors: Sequence[Sequence[float]],
    *,
    expected_count: int,
    label: str,
    expected_dimension: int | None = None,
) -> np.ndarray:
    rows = [list(vector) for vector in vectors]
    if not rows:
        raise EmbeddingError(f"{label} 不能为空")
    if len(rows) != expected_count:
        raise EmbeddingError(
            f"{label} 数量不一致：期望 {expected_count}，实际 {len(rows)}"
        )
    dimension = len(rows[0])
    if dimension == 0:
        raise EmbeddingError(f"{label} 维度不能为 0")
    if any(len(row) != dimension for row in rows):
        raise EmbeddingError(f"{label} 向量维度不一致")
    if expected_dimension is not None and dimension != expected_dimension:
        raise EmbeddingError(
            f"{label} 维度与索引不一致："
            f"索引 {expected_dimension}，实际 {dimension}"
        )
    for row in rows:
        for value in row:
            if isinstance(value, bool):
                raise EmbeddingError(f"{label} 必须只包含有限数值")
            try:
                finite = math.isfinite(float(value))
            except (TypeError, ValueError) as error:
                raise EmbeddingError(f"{label} 必须只包含有限数值") from error
            if not finite:
                raise EmbeddingError(f"{label} 必须只包含有限数值")

    try:
        matrix = np.asarray(rows, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise EmbeddingError(f"{label} 无法转换为 float32") from error
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(~np.isfinite(norms)):
        raise EmbeddingError(f"{label} 的 L2 范数无效")
    if np.any(norms == 0):
        raise EmbeddingError(f"{label} 不能包含全零向量")
    matrix /= norms[:, np.newaxis]
    return np.ascontiguousarray(matrix, dtype=np.float32)


def _validate_index_matches_manifest(
    index: faiss.Index,
    manifest: VectorIndexManifest,
) -> None:
    if not isinstance(index, faiss.IndexFlatIP):
        raise IndexPersistenceError("FAISS 文件不是预期的 IndexFlatIP")
    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise IndexPersistenceError("FAISS 索引距离度量不是内积")
    if index.d != manifest.vector_dimension:
        raise IndexPersistenceError("FAISS 索引维度与 manifest 不一致")
    if index.ntotal != manifest.vector_count:
        raise IndexPersistenceError("FAISS 向量数量与 manifest 不一致")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
