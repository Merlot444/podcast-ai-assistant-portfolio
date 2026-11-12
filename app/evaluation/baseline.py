"""Assemble a complete vector-baseline artifact after an injected run."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from app.schemas.evaluation import EvaluationDataset
from app.schemas.evaluation_result import (
    IndexConfiguration,
    InputArtifactHashes,
    RetrievalBaselineResult,
    RetrievalEvaluationRun,
    VectorRetrieverConfiguration,
)
from app.schemas.index import VectorIndexManifest


def file_sha256(path: str | Path) -> str:
    """Hash one file without mutating or loading its contents into the result."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_vector_baseline_result(
    *,
    dataset: EvaluationDataset,
    run: RetrievalEvaluationRun,
    manifest: VectorIndexManifest,
    input_artifact_hashes: InputArtifactHashes,
    query_embedding_calls: int,
    document_embedding_calls: int,
    created_at: datetime | None = None,
) -> RetrievalBaselineResult:
    """Join provider-independent metrics with observed vector-run provenance."""
    if query_embedding_calls != run.query_count:
        raise ValueError(
            "正式向量基线必须对每题恰好调用一次查询 Embedding："
            f"题目 {run.query_count}，调用 {query_embedding_calls}"
        )
    if document_embedding_calls != 0:
        raise ValueError("正式基线不得重新计算文档 Embedding")
    if run.k_values != (1, 3, 5, 8):
        raise ValueError("Day 12 正式基线固定使用 K=1,3,5,8")

    timestamp = (created_at or datetime.now(UTC)).astimezone(UTC)
    answerable_count = sum(case.answerable for case in run.cases)
    return RetrievalBaselineResult(
        schema_version=1,
        evaluation_id=(
            f"{dataset.dataset_id}-vector-"
            f"{timestamp.strftime('%Y%m%dt%H%M%Sz').lower()}"
        ),
        created_at_utc=timestamp,
        dataset_id=dataset.dataset_id,
        dataset_cases_sha256=dataset.cases_sha256,
        episode_id=dataset.episode_id,
        retriever_name=run.retriever_name,
        retriever_config=VectorRetrieverConfiguration(
            adapter="vector_retriever_adapter_v1",
            underlying_retriever="day8_vector_retriever",
            score_semantics="cosine_similarity",
            query_strategy="one_top_8_call_then_prefixes",
            equal_score_tiebreaker="faiss_row_index_ascending",
        ),
        index_config=IndexConfiguration(
            index_type=manifest.index_type,
            distance_metric=manifest.distance_metric,
            vector_count=manifest.vector_count,
            vector_dimension=manifest.vector_dimension,
            chunking_algorithm=manifest.chunking_algorithm,
            chunk_size=manifest.chunk_size,
            overlap=manifest.overlap,
        ),
        input_artifact_hashes=input_artifact_hashes,
        embedding_model=manifest.embedding_model,
        k_values=run.k_values,
        answerable_case_count=answerable_count,
        unanswerable_case_count=len(run.cases) - answerable_count,
        query_count=run.query_count,
        query_embedding_calls=query_embedding_calls,
        document_embedding_calls=document_embedding_calls,
        answer_model_calls=0,
        elapsed_seconds=run.elapsed_seconds,
        aggregate_metrics=run.aggregate_metrics,
        category_metrics=run.category_metrics,
        cases=run.cases,
        failure_reviews=(),
    )
