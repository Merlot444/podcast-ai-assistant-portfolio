"""Composition-based reciprocal-rank fusion for evaluation retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.base import EvaluationRetriever
from app.retrieval.corpus import RetrievalCorpusIdentity
from app.schemas.retriever import RetrievedCandidate


RRF_K = 60
CHILD_CANDIDATE_DEPTH = 20


class HybridRetrievalError(ValueError):
    """Raised when child corpora or candidate metadata cannot be safely fused."""


def reciprocal_rank_fusion(
    vector_candidates: list[RetrievedCandidate],
    bm25_candidates: list[RetrievedCandidate],
    *,
    corpus_identity: RetrievalCorpusIdentity,
    top_k: int,
    rrf_k: int = RRF_K,
    child_candidate_depth: int = CHILD_CANDIDATE_DEPTH,
) -> list[RetrievedCandidate]:
    """Fuse two fixed-depth rankings without mixing their raw scores."""
    _validate_positive_integer(top_k, "top_k")
    _validate_positive_integer(rrf_k, "rrf_k")
    _validate_positive_integer(child_candidate_depth, "child_candidate_depth")
    _validate_child_candidates(
        vector_candidates,
        expected_name="vector",
        maximum_count=child_candidate_depth,
    )
    _validate_child_candidates(
        bm25_candidates,
        expected_name="bm25",
        maximum_count=child_candidate_depth,
    )
    vector = vector_candidates[:child_candidate_depth]
    bm25 = bm25_candidates[:child_candidate_depth]

    catalog_positions = corpus_identity.chunk_position_by_id
    combined: dict[str, dict[str, object]] = {}
    for child_name, candidates in (("vector", vector), ("bm25", bm25)):
        for candidate in candidates:
            if candidate.chunk_id not in catalog_positions:
                raise HybridRetrievalError(
                    f"{child_name} 返回了 corpus identity 中不存在的 Chunk"
                )
            current = combined.get(candidate.chunk_id)
            if current is None:
                current = {
                    "source_segment_ids": candidate.source_segment_ids,
                    "score": 0.0,
                    "best_rank": candidate.rank,
                }
                combined[candidate.chunk_id] = current
            elif current["source_segment_ids"] != candidate.source_segment_ids:
                raise HybridRetrievalError(
                    f"Chunk {candidate.chunk_id} 的 source_segment_ids 在子结果中冲突"
                )
            current["score"] = float(current["score"]) + 1 / (rrf_k + candidate.rank)
            current["best_rank"] = min(int(current["best_rank"]), candidate.rank)

    ordered = sorted(
        combined.items(),
        key=lambda item: (
            -float(item[1]["score"]),
            int(item[1]["best_rank"]),
            catalog_positions[item[0]],
            item[0],
        ),
    )
    return [
        RetrievedCandidate(
            rank=rank,
            chunk_id=chunk_id,
            source_segment_ids=details["source_segment_ids"],
            score=float(details["score"]),
            score_semantics="reciprocal_rank_fusion",
            retriever_name="hybrid_rrf",
        )
        for rank, (chunk_id, details) in enumerate(ordered[:top_k], start=1)
    ]


def component_ranks(
    candidates: list[RetrievedCandidate],
) -> dict[str, int]:
    """Expose one ordered child ranking for persisted fusion auditing."""
    return {candidate.chunk_id: candidate.rank for candidate in candidates}


@dataclass(frozen=True)
class HybridRetriever:
    """Compose independent vector and BM25 children at a fixed candidate depth."""

    vector_retriever: EvaluationRetriever
    bm25_retriever: EvaluationRetriever
    vector_corpus_identity: RetrievalCorpusIdentity
    bm25_corpus_identity: RetrievalCorpusIdentity
    rrf_k: int = RRF_K
    child_candidate_depth: int = CHILD_CANDIDATE_DEPTH
    name: str = "hybrid_rrf"

    def __post_init__(self) -> None:
        if self.vector_corpus_identity != self.bm25_corpus_identity:
            raise HybridRetrievalError(
                "Vector 与 BM25 的 corpus identity（episode、Chunk catalog、"
                "配置、顺序或哈希）不一致"
            )
        _validate_positive_integer(self.rrf_k, "rrf_k")
        _validate_positive_integer(self.child_candidate_depth, "child_candidate_depth")

    def retrieve(self, query: str, top_k: int) -> list[RetrievedCandidate]:
        _validate_positive_integer(top_k, "top_k")
        vector = self.vector_retriever.retrieve(query, self.child_candidate_depth)
        bm25 = self.bm25_retriever.retrieve(query, self.child_candidate_depth)
        return reciprocal_rank_fusion(
            vector,
            bm25,
            corpus_identity=self.vector_corpus_identity,
            top_k=top_k,
            rrf_k=self.rrf_k,
            child_candidate_depth=self.child_candidate_depth,
        )


def _validate_child_candidates(
    candidates: list[RetrievedCandidate],
    *,
    expected_name: str,
    maximum_count: int,
) -> None:
    if len(candidates) > maximum_count:
        raise HybridRetrievalError(
            "子检索结果数量超过 child_candidate_depth"
        )
    if [item.rank for item in candidates] != list(range(1, len(candidates) + 1)):
        raise HybridRetrievalError("子检索结果 rank 必须从 1 连续并与顺序一致")
    if any(item.retriever_name != expected_name for item in candidates):
        raise HybridRetrievalError("子检索结果 provenance 与预期检索器不一致")
    chunk_ids = [item.chunk_id for item in candidates]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise HybridRetrievalError("同一子检索结果不能重复返回 Chunk")


def _validate_positive_integer(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HybridRetrievalError(f"{label} 必须是大于 0 的整数")
