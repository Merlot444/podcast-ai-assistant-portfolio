"""Offline Day 14 BM25, corpus-identity and Hybrid-RRF tests."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from app.retrieval.bm25 import BM25Retriever, tokenize_unicode
from app.retrieval.corpus import RetrievalCorpusIdentity
from app.retrieval.hybrid import (
    CHILD_CANDIDATE_DEPTH,
    HybridRetrievalError,
    HybridRetriever,
    reciprocal_rank_fusion,
)
from app.schemas.chunk import ChunkSourceSpan, TranscriptChunk
from app.schemas.retriever import RetrievedCandidate


EPISODE_ID = "ep_b4f2962a8f12ba8d"


def _chunk(position: int, text: str, *, segment: int | None = None) -> TranscriptChunk:
    segment_index = position if segment is None else segment
    segment_id = f"{EPISODE_ID}:{segment_index:06d}"
    return TranscriptChunk(
        chunk_id=f"chunk_{position:016x}",
        chunk_index=position,
        episode_id=EPISODE_ID,
        source_file="synthetic.txt",
        text=text,
        source_segment_ids=[segment_id],
        source_segment_start_index=segment_index,
        source_segment_end_index=segment_index,
        source_spans=[
            ChunkSourceSpan(
                segment_id=segment_id,
                segment_index=segment_index,
                start_char=0,
                end_char=len(text),
            )
        ],
        speakers=[],
        section_titles=[],
        start_time=None,
        end_time=None,
        time_range_complete=False,
    )


def _identity(chunks: tuple[TranscriptChunk, ...]) -> RetrievalCorpusIdentity:
    return RetrievalCorpusIdentity(
        episode_id=EPISODE_ID,
        source_file="synthetic.txt",
        chunking_algorithm="fixed_character_windows_v1",
        chunk_size=1000,
        overlap=200,
        input_chunk_sha256="a" * 64,
        chunk_id_order=tuple(chunk.chunk_id for chunk in chunks),
    )


def _bm25(*texts: str) -> BM25Retriever:
    chunks = tuple(_chunk(position, text) for position, text in enumerate(texts))
    return BM25Retriever(chunks=chunks, corpus_identity=_identity(chunks))


def _candidate(
    rank: int,
    position: int,
    *,
    name: str,
    score: float = 1.0,
    segment: int | None = None,
) -> RetrievedCandidate:
    segment_index = position if segment is None else segment
    return RetrievedCandidate(
        rank=rank,
        chunk_id=f"chunk_{position:016x}",
        source_segment_ids=(f"{EPISODE_ID}:{segment_index:06d}",),
        score=score,
        score_semantics="cosine_similarity" if name == "vector" else "bm25_okapi",
        retriever_name=name,
    )


def test_tokenizer_is_deterministic_nfkc_casefold_and_unicode_aware() -> None:
    text = "Ｇｅｍｉｎｉ CAFÉ café"
    assert tokenize_unicode(text) == ("gemini", "café", "café")
    assert tokenize_unicode(text) == tokenize_unicode(text)


def test_tokenizer_preserves_acronyms_numbers_hyphens_and_apostrophes() -> None:
    assert tokenize_unicode("RT‑2 AGI 2026 Hassabis’s") == (
        "rt-2",
        "agi",
        "2026",
        "hassabis's",
    )


def test_bm25_empty_or_unmatched_query_returns_empty_safely() -> None:
    retriever = _bm25("Gemini scaling", "AlphaZero search")
    assert retriever.retrieve("---", 8) == []
    assert retriever.retrieve("unseen", 8) == []


def test_bm25_hand_calculated_single_term_example() -> None:
    retriever = _bm25("alpha", "beta")
    result = retriever.retrieve("alpha", 1)
    expected_idf = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    assert result[0].score == pytest.approx(expected_idf)
    assert result[0].chunk_id == "chunk_0000000000000000"


def test_bm25_exact_term_beats_nonmatching_document_and_normalizes_hyphen() -> None:
    retriever = _bm25("RT-2 robotics result", "robotics result only")
    result = retriever.retrieve("rt‑2", 8)
    assert [item.chunk_id for item in result] == ["chunk_0000000000000000"]
    assert result[0].score_semantics == "bm25_okapi"
    assert result[0].retriever_name == "bm25"


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5, "2"])
def test_bm25_rejects_invalid_top_k(top_k: object) -> None:
    with pytest.raises(ValueError, match="top_k"):
        _bm25("alpha").retrieve("alpha", top_k)  # type: ignore[arg-type]


def test_bm25_top_k_larger_than_corpus_and_stable_equal_score_order() -> None:
    retriever = _bm25("alpha", "alpha", "beta")
    first = retriever.retrieve("alpha", 20)
    second = retriever.retrieve("alpha", 20)
    assert [item.chunk_id for item in first] == [
        "chunk_0000000000000000",
        "chunk_0000000000000001",
    ]
    assert first == second


def test_rrf_hand_calculation_merges_duplicate_and_keeps_single_route() -> None:
    chunks = tuple(_chunk(position, f"text {position}") for position in range(3))
    fused = reciprocal_rank_fusion(
        [_candidate(1, 0, name="vector"), _candidate(2, 1, name="vector")],
        [_candidate(1, 0, name="bm25"), _candidate(2, 2, name="bm25")],
        corpus_identity=_identity(chunks),
        top_k=3,
    )
    assert fused[0].chunk_id == "chunk_0000000000000000"
    assert fused[0].score == pytest.approx(2 / 61)
    assert {item.chunk_id: item.score for item in fused[1:]} == pytest.approx(
        {
            "chunk_0000000000000001": 1 / 62,
            "chunk_0000000000000002": 1 / 62,
        }
    )
    assert all(item.score_semantics == "reciprocal_rank_fusion" for item in fused)
    assert all(item.retriever_name == "hybrid_rrf" for item in fused)


def test_rrf_equal_score_uses_best_rank_then_catalog_position_then_chunk_id() -> None:
    chunks = tuple(_chunk(position, f"text {position}") for position in range(3))
    fused = reciprocal_rank_fusion(
        [_candidate(1, 1, name="vector")],
        [_candidate(1, 0, name="bm25")],
        corpus_identity=_identity(chunks),
        top_k=2,
    )
    assert [item.chunk_id for item in fused] == [
        "chunk_0000000000000000",
        "chunk_0000000000000001",
    ]


def test_rrf_ignores_raw_cosine_and_bm25_scores() -> None:
    chunks = tuple(_chunk(position, f"text {position}") for position in range(2))
    ordinary = reciprocal_rank_fusion(
        [_candidate(1, 0, name="vector", score=0.9)],
        [_candidate(1, 1, name="bm25", score=20)],
        corpus_identity=_identity(chunks),
        top_k=2,
    )
    reversed_scales = reciprocal_rank_fusion(
        [_candidate(1, 0, name="vector", score=-999)],
        [_candidate(1, 1, name="bm25", score=0.0001)],
        corpus_identity=_identity(chunks),
        top_k=2,
    )
    assert [(item.chunk_id, item.score) for item in ordinary] == [
        (item.chunk_id, item.score) for item in reversed_scales
    ]


def test_rrf_rejects_conflicting_source_metadata_for_same_chunk() -> None:
    chunks = (_chunk(0, "alpha"),)
    with pytest.raises(HybridRetrievalError, match="source_segment_ids"):
        reciprocal_rank_fusion(
            [_candidate(1, 0, name="vector", segment=0)],
            [_candidate(1, 0, name="bm25", segment=1)],
            corpus_identity=_identity(chunks),
            top_k=1,
        )


def test_rrf_rejects_child_results_above_declared_depth_before_slicing() -> None:
    chunks = tuple(_chunk(position, f"text {position}") for position in range(3))

    with pytest.raises(HybridRetrievalError, match="child_candidate_depth"):
        reciprocal_rank_fusion(
            [
                _candidate(1, 0, name="vector"),
                _candidate(2, 1, name="vector"),
                _candidate(3, 2, name="vector"),
            ],
            [_candidate(1, 0, name="bm25")],
            corpus_identity=_identity(chunks),
            top_k=2,
            child_candidate_depth=2,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("episode_id", "ep_1111111111111111"),
        ("chunk_size", 999),
        ("overlap", 199),
        ("input_chunk_sha256", "b" * 64),
        ("chunk_id_order", ("chunk_ffffffffffffffff",)),
    ],
)
def test_hybrid_rejects_any_corpus_identity_mismatch(field: str, value: object) -> None:
    chunks = (_chunk(0, "alpha"),)
    identity = _identity(chunks)

    class Fake:
        name = "vector"

        def retrieve(self, query: str, top_k: int):
            return []

    with pytest.raises(HybridRetrievalError, match="identity"):
        HybridRetriever(
            vector_retriever=Fake(),
            bm25_retriever=Fake(),
            vector_corpus_identity=identity,
            bm25_corpus_identity=replace(identity, **{field: value}),
        )


def test_hybrid_always_requests_fixed_child_depth_not_top_k_times_two() -> None:
    chunks = (_chunk(0, "alpha"),)
    identity = _identity(chunks)

    class Fake:
        def __init__(self, name: str) -> None:
            self.name = name
            self.calls: list[int] = []

        def retrieve(self, query: str, top_k: int):
            self.calls.append(top_k)
            return [_candidate(1, 0, name=self.name)]

    vector = Fake("vector")
    bm25 = Fake("bm25")
    hybrid = HybridRetriever(
        vector_retriever=vector,
        bm25_retriever=bm25,
        vector_corpus_identity=identity,
        bm25_corpus_identity=identity,
    )
    hybrid.retrieve("alpha", 1)
    assert vector.calls == [CHILD_CANDIDATE_DEPTH]
    assert bm25.calls == [CHILD_CANDIDATE_DEPTH]
