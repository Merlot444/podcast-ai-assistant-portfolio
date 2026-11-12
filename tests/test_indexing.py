"""Offline normal, failure, and boundary tests for Day 7 FAISS persistence."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import faiss
import numpy as np
import pytest
from pydantic import ValidationError

from app.indexing.embedding import EmbeddingError
from app.indexing.index import (
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    IndexPersistenceError,
    VectorIndexError,
    build_vector_index,
    load_vector_index,
    save_vector_index,
)
from app.schemas.chunk import ChunkedTranscript, ChunkSourceSpan, TranscriptChunk


EPISODE_ID = "ep_0123456789abcdef"


class FakeEmbedder:
    """Controllable test double that records document and query calls."""

    def __init__(
        self,
        document_vectors: Sequence[Sequence[float]],
        query_vector: Sequence[float],
        *,
        model_name: str = "fake-embedding-v1",
    ) -> None:
        self.document_vectors = [list(vector) for vector in document_vectors]
        self.query_vector = list(query_vector)
        self._model_name = model_name
        self.document_call_count = 0
        self.query_call_count = 0
        self.document_inputs: list[str] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_call_count += 1
        self.document_inputs = list(texts)
        return [list(vector) for vector in self.document_vectors]

    def embed_query(self, text: str) -> list[float]:
        self.query_call_count += 1
        return list(self.query_vector)


def _chunk(index: int, text: str, *, section: str | None = None) -> TranscriptChunk:
    segment_id = f"{EPISODE_ID}:{index:06d}"
    return TranscriptChunk(
        chunk_id=f"chunk_{index:016x}",
        chunk_index=index,
        episode_id=EPISODE_ID,
        source_file="episode.txt",
        text=text,
        source_segment_ids=[segment_id],
        source_segment_start_index=index,
        source_segment_end_index=index,
        source_spans=[
            ChunkSourceSpan(
                segment_id=segment_id,
                segment_index=index,
                start_char=0,
                end_char=len(text),
            )
        ],
        speakers=[f"Speaker {index}"],
        section_titles=[section or f"Chapter {index}"],
        start_time=float(index * 10),
        end_time=float(index * 10 + 5),
        time_range_complete=True,
    )


def _chunked(texts: Sequence[str] = ("alpha", "beta", "gamma")) -> ChunkedTranscript:
    chunks = [_chunk(index, text) for index, text in enumerate(texts)]
    return ChunkedTranscript(
        algorithm="fixed_character_windows_v1",
        episode_id=EPISODE_ID,
        source_file="episode.txt",
        input_segment_count=len(chunks),
        chunk_size=max(len(text) for text in texts),
        overlap=0,
        chunks=chunks,
    )


def _build(
    vectors: Sequence[Sequence[float]] = (
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
    ),
    query: Sequence[float] = (1.0, 0.0),
):
    chunked = _chunked()
    embedder = FakeEmbedder(vectors, query)
    index = build_vector_index(
        chunked,
        embedder=embedder,
        input_chunk_file="chunks.json",
    )
    return chunked, embedder, index


def test_build_save_reload_and_query_preserve_row_mapping_and_metadata(
    tmp_path: Path,
) -> None:
    chunked, embedder, index = _build()
    before = index.similarity_check("alpha question", embedder=embedder, top_k=3)
    destination = tmp_path / "nested" / "index"

    index_path, manifest_path = save_vector_index(index, destination)
    document_calls_before_load = embedder.document_call_count
    loaded = load_vector_index(
        destination,
        expected_chunks=chunked,
        expected_embedding_model=embedder.model_name,
        expected_dimension=2,
    )
    after = loaded.similarity_check("alpha question", embedder=embedder, top_k=3)

    assert index_path.is_file()
    assert manifest_path.is_file()
    assert embedder.document_inputs == ["alpha", "beta", "gamma"]
    assert embedder.document_call_count == document_calls_before_load == 1
    assert embedder.query_call_count == 2
    assert loaded.manifest.chunk_id_order == [
        chunk.chunk_id for chunk in chunked.chunks
    ]
    assert [(result.row_index, result.score) for result in before] == pytest.approx(
        [(result.row_index, result.score) for result in after]
    )
    first = after[0]
    assert first.row_index == 0
    assert first.chunk.chunk_id == chunked.chunks[0].chunk_id
    assert first.metadata_dict()["source_spans"] == [
        {
            "segment_id": f"{EPISODE_ID}:000000",
            "segment_index": 0,
            "start_char": 0,
            "end_char": 5,
        }
    ]
    assert first.metadata_dict()["speakers"] == ["Speaker 0"]

    with pytest.raises(ValidationError, match="frozen"):
        loaded.manifest.embedding_model = "mutated"


def test_equal_scores_are_stably_ordered_by_row_index() -> None:
    _, embedder, index = _build(
        vectors=((1.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        query=(1.0, 0.0),
    )

    results = index.similarity_check("tie", embedder=embedder, top_k=2)

    assert [result.row_index for result in results] == [0, 1]
    assert results[0].score == pytest.approx(results[1].score)


def test_one_chunk_one_dimension_exact_match_and_large_top_k() -> None:
    chunked = _chunked(("only",))
    embedder = FakeEmbedder([[2.0]], [9.0])
    index = build_vector_index(
        chunked,
        embedder=embedder,
        input_chunk_file="one.json",
    )

    results = index.similarity_check("same direction", embedder=embedder, top_k=9)

    assert len(results) == 1
    assert results[0].score == pytest.approx(1.0)
    assert results[0].chunk.chunk_id == chunked.chunks[0].chunk_id


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        ([[1.0, 0.0]], "数量不一致"),
        ([[1.0], [1.0, 2.0], [3.0]], "维度不一致"),
        ([[], [], []], "维度不能为 0"),
        ([[float("nan")], [1.0], [2.0]], "有限数值"),
        ([[float("inf")], [1.0], [2.0]], "有限数值"),
        ([[0.0], [1.0], [2.0]], "全零向量"),
    ],
)
def test_invalid_document_vectors_are_rejected(
    vectors: Sequence[Sequence[float]],
    message: str,
) -> None:
    chunked = _chunked()
    embedder = FakeEmbedder(vectors, [1.0])

    with pytest.raises(EmbeddingError, match=message):
        build_vector_index(
            chunked,
            embedder=embedder,
            input_chunk_file="chunks.json",
        )


def test_empty_chunk_list_is_defensively_rejected() -> None:
    invalid = _chunked().model_copy(update={"chunks": []})
    embedder = FakeEmbedder([], [1.0])

    with pytest.raises(VectorIndexError, match="Chunk 列表不能为空"):
        build_vector_index(
            invalid,
            embedder=embedder,
            input_chunk_file="chunks.json",
        )

    assert embedder.document_call_count == 0


@pytest.mark.parametrize(
    ("query", "top_k", "message"),
    [
        ("", 1, "查询不能为空"),
        ("question", 0, "top_k"),
        ("question", True, "top_k"),
    ],
)
def test_invalid_query_or_top_k_is_rejected_before_embedding(
    query: str,
    top_k: int,
    message: str,
) -> None:
    _, embedder, index = _build()

    with pytest.raises(VectorIndexError, match=message):
        index.similarity_check(query, embedder=embedder, top_k=top_k)

    assert embedder.query_call_count == 0


def test_query_model_and_dimension_must_match_the_loaded_index() -> None:
    _, _, index = _build()
    wrong_model = FakeEmbedder(
        [[1.0, 0.0]] * 3,
        [1.0, 0.0],
        model_name="other-model",
    )
    wrong_dimension = FakeEmbedder(
        [[1.0, 0.0]] * 3,
        [1.0, 0.0, 0.0],
    )

    with pytest.raises(VectorIndexError, match="模型与索引不一致"):
        index.similarity_check("question", embedder=wrong_model)
    with pytest.raises(EmbeddingError, match="维度与索引不一致"):
        index.similarity_check("question", embedder=wrong_dimension)


def test_jina_request_options_are_persisted_and_checked(tmp_path: Path) -> None:
    chunked = _chunked()
    embedder = FakeEmbedder(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        [1.0, 0.0],
    )
    embedder.document_task = "retrieval.passage"
    embedder.query_task = "retrieval.query"
    embedder.embedding_type = "float"
    embedder.normalized = True
    embedder.truncate = False
    index = build_vector_index(
        chunked,
        embedder=embedder,
        input_chunk_file="chunks.json",
    )
    destination = tmp_path / "jina-options"
    save_vector_index(index, destination)

    loaded = load_vector_index(
        destination,
        expected_document_task="retrieval.passage",
        expected_query_task="retrieval.query",
        expected_embedding_type="float",
        expected_provider_normalized=True,
        expected_provider_truncate=False,
    )

    assert loaded.manifest.embedding_document_task == "retrieval.passage"
    assert loaded.manifest.embedding_query_task == "retrieval.query"
    embedder.query_task = "text-matching"
    with pytest.raises(VectorIndexError, match="task 与索引不一致"):
        loaded.similarity_check("question", embedder=embedder)
    with pytest.raises(IndexPersistenceError, match="文档 Embedding task"):
        load_vector_index(
            destination,
            expected_document_task="text-matching",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("normalized", False, "normalized"),
        ("truncate", True, "truncate"),
    ],
)
def test_similarity_check_rejects_provider_option_mismatch(
    field: str,
    value: bool,
    message: str,
) -> None:
    chunked = _chunked()
    embedder = FakeEmbedder(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        [1.0, 0.0],
    )
    embedder.document_task = "retrieval.passage"
    embedder.query_task = "retrieval.query"
    embedder.embedding_type = "float"
    embedder.normalized = True
    embedder.truncate = False
    index = build_vector_index(
        chunked,
        embedder=embedder,
        input_chunk_file="chunks.json",
    )
    setattr(embedder, field, value)

    with pytest.raises(VectorIndexError, match=message):
        index.similarity_check("question", embedder=embedder)

    assert embedder.query_call_count == 0


def test_missing_or_corrupt_manifest_and_corrupt_index_are_rejected(
    tmp_path: Path,
) -> None:
    _, _, index = _build()
    first = tmp_path / "missing"
    first.mkdir()
    with pytest.raises(IndexPersistenceError, match="缺少 manifest"):
        load_vector_index(first)

    corrupt_manifest = tmp_path / "corrupt-manifest"
    save_vector_index(index, corrupt_manifest)
    (corrupt_manifest / MANIFEST_FILENAME).write_text("{broken", encoding="utf-8")
    with pytest.raises(IndexPersistenceError, match="manifest JSON 损坏"):
        load_vector_index(corrupt_manifest)

    corrupt_index = tmp_path / "corrupt-index"
    save_vector_index(index, corrupt_index)
    with (corrupt_index / INDEX_FILENAME).open("ab") as file:
        file.write(b"corruption")
    with pytest.raises(IndexPersistenceError, match="校验和"):
        load_vector_index(corrupt_index)


def test_manifest_and_faiss_count_mismatch_is_rejected(tmp_path: Path) -> None:
    _, _, index = _build()
    destination = tmp_path / "count-mismatch"
    save_vector_index(index, destination)

    smaller = faiss.IndexFlatIP(2)
    smaller.add(np.asarray([[1.0, 0.0]], dtype=np.float32))
    index_path = destination / INDEX_FILENAME
    faiss.write_index(smaller, str(index_path))
    payload = json.loads(
        (destination / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    payload["index_sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    (destination / MANIFEST_FILENAME).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(IndexPersistenceError, match="向量数量"):
        load_vector_index(destination)


def test_saved_index_rejects_current_chunk_model_and_dimension_mismatch(
    tmp_path: Path,
) -> None:
    _, _, index = _build()
    destination = tmp_path / "expected-mismatch"
    save_vector_index(index, destination)

    with pytest.raises(IndexPersistenceError, match="Chunk 内容指纹"):
        load_vector_index(destination, expected_chunks=_chunked(("x", "y", "z")))
    with pytest.raises(IndexPersistenceError, match="模型"):
        load_vector_index(destination, expected_embedding_model="other-model")
    with pytest.raises(IndexPersistenceError, match="维度"):
        load_vector_index(destination, expected_dimension=99)


def test_unsupported_metric_and_manifest_configuration_are_rejected(
    tmp_path: Path,
) -> None:
    chunked, embedder, index = _build()
    with pytest.raises(VectorIndexError, match="不支持的距离度量"):
        build_vector_index(
            chunked,
            embedder=embedder,
            input_chunk_file="chunks.json",
            distance_metric="euclidean",
        )

    destination = tmp_path / "unsupported-manifest"
    save_vector_index(index, destination)
    manifest_path = destination / MANIFEST_FILENAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["distance_metric"] = "euclidean"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(IndexPersistenceError, match="manifest 校验失败"):
        load_vector_index(destination)


def test_existing_artifacts_are_not_silently_overwritten(tmp_path: Path) -> None:
    _, _, first = _build()
    changed_chunked = _chunked(("different", "content", "set"))
    changed_embedder = FakeEmbedder(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        [1.0, 0.0],
    )
    changed = build_vector_index(
        changed_chunked,
        embedder=changed_embedder,
        input_chunk_file="other.json",
    )
    destination = tmp_path / "existing"
    save_vector_index(first, destination)
    original = (destination / MANIFEST_FILENAME).read_bytes()

    with pytest.raises(IndexPersistenceError, match="不会静默覆盖"):
        save_vector_index(changed, destination)

    assert (destination / MANIFEST_FILENAME).read_bytes() == original


def test_synthetic_cli_builds_reloads_and_queries_without_env(
    tmp_path: Path,
) -> None:
    source = tmp_path / "chunks.json"
    source.write_text(
        json.dumps(_chunked().model_dump(mode="json")),
        encoding="utf-8",
    )
    destination = tmp_path / "cli-index"

    built = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.indexing",
            "build",
            str(source),
            str(destination),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={},
    )
    queried = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.indexing",
            "query",
            str(destination),
            "alpha",
            "--top-k",
            "2",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={},
    )

    assert built.returncode == 0, built.stderr
    assert "不是语义 Embedding 模型" in built.stderr
    assert "重新加载：成功；未调用 embed_documents()" in built.stdout
    assert queried.returncode == 0, queried.stderr
    payload = json.loads(queried.stdout)
    assert payload["document_embeddings_recomputed"] is False
    assert len(payload["results"]) == 2
    assert "text" not in payload["results"][0]
