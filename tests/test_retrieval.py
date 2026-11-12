"""Offline normal, failure, and boundary tests for Day 8 retrieval."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from app.indexing.embedding import EmbeddingError
from app.indexing.index import build_vector_index, load_vector_index, save_vector_index
from app.retrieval import RetrievalError, Retriever
from app.schemas.chunk import ChunkedTranscript, ChunkSourceSpan, TranscriptChunk


EPISODE_ID = "ep_0123456789abcdef"


class FakeEmbedder:
    """Controllable Embedder that makes every document/query call observable."""

    def __init__(
        self,
        document_vectors: Sequence[Sequence[float]],
        query_vector: Sequence[float],
        *,
        model_name: str = "fake-retrieval-v1",
    ) -> None:
        self.document_vectors = [list(vector) for vector in document_vectors]
        self.query_vector = list(query_vector)
        self._model_name = model_name
        self.document_task = "retrieval.passage"
        self.query_task = "retrieval.query"
        self.embedding_type = "float"
        self.normalized = True
        self.truncate = False
        self.document_call_count = 0
        self.query_call_count = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_call_count += 1
        return [list(vector) for vector in self.document_vectors]

    def embed_query(self, text: str) -> list[float]:
        self.query_call_count += 1
        return list(self.query_vector)


class QueryOnlyFakeEmbedder(FakeEmbedder):
    """A query adapter that fails loudly if retrieval embeds documents."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise AssertionError("retrieve() 不得调用 embed_documents()")


def _chunk(
    index: int,
    text: str,
    *,
    speakers: list[str] | None = None,
    section_titles: list[str] | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    time_range_complete: bool = True,
) -> TranscriptChunk:
    segment_id = f"{EPISODE_ID}:{index:06d}"
    if start_time is None and time_range_complete:
        start_time = float(index * 10)
    if end_time is None and time_range_complete:
        end_time = float(index * 10 + 5)
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
        speakers=[f"Speaker {index}"] if speakers is None else speakers,
        section_titles=(
            [f"Chapter {index}"] if section_titles is None else section_titles
        ),
        start_time=start_time,
        end_time=end_time,
        time_range_complete=time_range_complete,
    )


def _chunked(
    count: int = 8,
    *,
    final_chunk: TranscriptChunk | None = None,
) -> ChunkedTranscript:
    chunks = [_chunk(index, f"complete chunk text {index}") for index in range(count)]
    if final_chunk is not None:
        chunks[-1] = final_chunk
    return ChunkedTranscript(
        algorithm="fixed_character_windows_v1",
        episode_id=EPISODE_ID,
        source_file="episode.txt",
        input_segment_count=count,
        chunk_size=max(len(chunk.text) for chunk in chunks),
        overlap=0,
        chunks=chunks,
    )


def _descending_vectors(count: int) -> list[list[float]]:
    return [[float(count - index), float(index)] for index in range(count)]


def _build(count: int = 8):
    chunked = _chunked(count)
    embedder = FakeEmbedder(_descending_vectors(count), [1.0, 0.0])
    vector_index = build_vector_index(
        chunked,
        embedder=embedder,
        input_chunk_file="chunks.json",
    )
    return chunked, embedder, vector_index


def test_retrieve_returns_sorted_full_text_and_source_metadata() -> None:
    chunked, embedder, vector_index = _build()
    document_calls_before = embedder.document_call_count

    results = Retriever(vector_index, embedder).retrieve("alpha question", 3)

    assert [result.rank for result in results] == [1, 2, 3]
    assert [result.row_index for result in results] == [0, 1, 2]
    assert [result.score for result in results] == sorted(
        (result.score for result in results),
        reverse=True,
    )
    assert embedder.query_call_count == 1
    assert embedder.document_call_count == document_calls_before == 1
    first = results[0]
    assert first.chunk_id == vector_index.manifest.chunk_id_order[first.row_index]
    assert first.text == chunked.chunks[0].text
    assert first.episode_id == EPISODE_ID
    assert first.source_file == "episode.txt"
    assert first.source_segment_ids == [f"{EPISODE_ID}:000000"]
    assert first.source_segment_start_index == 0
    assert first.source_segment_end_index == 0
    assert first.source_spans[0].model_dump() == {
        "segment_id": f"{EPISODE_ID}:000000",
        "segment_index": 0,
        "start_char": 0,
        "end_char": len(chunked.chunks[0].text),
    }
    assert first.speakers == ["Speaker 0"]
    assert first.section_titles == ["Chapter 0"]
    assert first.start_time == 0.0
    assert first.end_time == 5.0
    assert first.time_range_complete is True


def test_top_k_3_5_8_are_stable_prefixes_and_deterministic() -> None:
    _, embedder, vector_index = _build()
    retriever = Retriever(vector_index, embedder)

    top_3 = retriever.retrieve("same query", 3)
    top_5 = retriever.retrieve("same query", 5)
    top_8 = retriever.retrieve("same query", 8)
    repeated = retriever.retrieve("same query", 8)

    ids_3 = [result.chunk_id for result in top_3]
    ids_5 = [result.chunk_id for result in top_5]
    ids_8 = [result.chunk_id for result in top_8]
    assert ids_3 == ids_5[:3] == ids_8[:3]
    assert ids_5 == ids_8[:5]
    assert top_8 == repeated
    assert embedder.query_call_count == 4
    assert embedder.document_call_count == 1


def test_reloaded_retrieval_matches_in_memory_without_document_embedding(
    tmp_path: Path,
) -> None:
    _, embedder, vector_index = _build()
    before = Retriever(vector_index, embedder).retrieve("question", 5)
    destination = tmp_path / "index"
    save_vector_index(vector_index, destination)
    document_calls_before_load = embedder.document_call_count

    loaded = load_vector_index(destination)
    after = Retriever(loaded, embedder).retrieve("question", 5)

    assert after == before
    assert embedder.document_call_count == document_calls_before_load == 1
    assert embedder.query_call_count == 2


def test_retrieve_never_calls_embed_documents() -> None:
    _, _, vector_index = _build()
    query_only = QueryOnlyFakeEmbedder([], [1.0, 0.0])

    results = Retriever(vector_index, query_only).retrieve("question", 1)

    assert len(results) == 1
    assert query_only.query_call_count == 1


def test_equal_scores_use_row_index_as_stable_second_sort_key() -> None:
    chunked = _chunked(3)
    embedder = FakeEmbedder(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        [1.0, 0.0],
    )
    vector_index = build_vector_index(
        chunked,
        embedder=embedder,
        input_chunk_file="chunks.json",
    )

    results = Retriever(vector_index, embedder).retrieve("tie", 2)

    assert [result.row_index for result in results] == [0, 1]
    assert results[0].score == pytest.approx(results[1].score)


def test_one_chunk_exact_match_and_top_k_larger_than_index() -> None:
    chunked = _chunked(1)
    embedder = FakeEmbedder([[2.0]], [9.0])
    vector_index = build_vector_index(
        chunked,
        embedder=embedder,
        input_chunk_file="one.json",
    )

    results = Retriever(vector_index, embedder).retrieve("same direction", 8)

    assert len(results) == 1
    assert results[0].rank == 1
    assert results[0].score == pytest.approx(1.0)


def test_legal_missing_metadata_is_preserved_without_invention() -> None:
    incomplete = _chunk(
        0,
        "text with incomplete metadata",
        speakers=[],
        section_titles=[],
        start_time=None,
        end_time=None,
        time_range_complete=False,
    )
    chunked = _chunked(1, final_chunk=incomplete)
    embedder = FakeEmbedder([[1.0]], [1.0])
    vector_index = build_vector_index(
        chunked,
        embedder=embedder,
        input_chunk_file="incomplete.json",
    )

    result = Retriever(vector_index, embedder).retrieve("question", 1)[0]

    assert result.speakers == []
    assert result.section_titles == []
    assert result.start_time is None
    assert result.end_time is None
    assert result.time_range_complete is False


@pytest.mark.parametrize("query", ["", "   ", None, 12])
def test_empty_or_non_string_query_is_rejected_before_embedding(
    query: object,
) -> None:
    _, embedder, vector_index = _build()

    with pytest.raises(RetrievalError, match="查询不能为空"):
        Retriever(vector_index, embedder).retrieve(query, 3)  # type: ignore[arg-type]

    assert embedder.query_call_count == 0


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5, "3"])
def test_invalid_top_k_is_rejected_before_embedding(top_k: object) -> None:
    _, embedder, vector_index = _build()

    with pytest.raises(RetrievalError, match="top_k"):
        Retriever(vector_index, embedder).retrieve(  # type: ignore[arg-type]
            "question",
            top_k,
        )

    assert embedder.query_call_count == 0


@pytest.mark.parametrize(
    ("query_vector", "message"),
    [
        ([], "维度不能为 0"),
        ([float("nan"), 0.0], "有限数值"),
        ([float("inf"), 0.0], "有限数值"),
        ([1.0, 0.0, 0.0], "维度与索引不一致"),
        ([0.0, 0.0], "全零向量"),
    ],
)
def test_invalid_query_vectors_fail_safely(
    query_vector: Sequence[float],
    message: str,
) -> None:
    _, _, vector_index = _build()
    embedder = FakeEmbedder([], query_vector)

    with pytest.raises(EmbeddingError, match=message):
        Retriever(vector_index, embedder).retrieve("question", 3)

    assert embedder.query_call_count == 1
    assert embedder.document_call_count == 0


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("_model_name", "other-model", "模型与索引不一致"),
        ("expected_dimension", 3, "向量维度"),
        ("document_task", "text-matching", "文档 Embedding task"),
        ("query_task", "text-matching", "task"),
        ("embedding_type", "binary", "输出类型"),
        ("normalized", False, "归一化"),
        ("truncate", True, "截断"),
    ],
)
def test_query_configuration_must_match_manifest_before_embedding(
    attribute: str,
    value: object,
    message: str,
) -> None:
    _, _, vector_index = _build()
    query_embedder = FakeEmbedder([], [1.0, 0.0])
    setattr(query_embedder, attribute, value)

    with pytest.raises(RetrievalError, match=message):
        Retriever(vector_index, query_embedder).retrieve("question", 3)

    assert query_embedder.query_call_count == 0


def test_corrupted_chunk_id_mapping_fails_before_embedding() -> None:
    _, embedder, vector_index = _build()
    vector_index.manifest = vector_index.manifest.model_copy(
        update={
            "chunk_id_order": list(reversed(vector_index.manifest.chunk_id_order))
        }
    )
    query_calls_before = embedder.query_call_count

    with pytest.raises(RetrievalError, match="chunk_id_order"):
        Retriever(vector_index, embedder).retrieve("question", 3)

    assert embedder.query_call_count == query_calls_before


def test_synthetic_cli_retrieves_offline_and_returns_json(
    tmp_path: Path,
) -> None:
    from app.indexing.embedding import SyntheticHashEmbedder

    chunked = _chunked(3)
    vector_index = build_vector_index(
        chunked,
        embedder=SyntheticHashEmbedder(32),
        input_chunk_file="chunks.json",
    )
    destination = tmp_path / "synthetic-index"
    save_vector_index(vector_index, destination)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.retrieval",
            str(destination),
            "complete chunk text",
            "--top-k",
            "8",
            "--embedder",
            "synthetic-hash",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={},
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["query_embedding_calls"] == 1
    assert payload["document_embeddings_recomputed"] is False
    assert payload["requested_top_k"] == 8
    assert payload["returned_count"] == 3
    assert payload["score_semantics"] == "cosine_similarity_not_probability"
    assert payload["results"][0]["text"]
    assert payload["results"][0]["source_spans"]
