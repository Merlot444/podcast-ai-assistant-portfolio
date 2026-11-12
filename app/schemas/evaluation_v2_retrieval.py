"""Strict metadata-only schema for the 80-case retrieval comparison."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator

from app.schemas.evaluation import EvaluationCategory
from app.schemas.evaluation_result import EvaluationCaseResult, MetricSummaryAtK


MethodName = Literal["vector", "bm25", "hybrid_rrf"]


class ExpandedDatasetSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base_dataset_id: Literal["eval_dataset_v1"]
    base_cases_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pilot_dataset_id: Literal["eval_dataset_v2_pilot"]
    pilot_collection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expanded_case_count: Literal[80]
    answerable_case_count: Literal[64]
    unanswerable_case_count: Literal[16]


class ExpandedRetrievalConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    embedding_model: str = Field(min_length=1)
    embedding_dimension: StrictInt = Field(gt=0)
    document_task: Literal["retrieval.passage"]
    query_task: Literal["retrieval.query"]
    normalized: Literal[True]
    truncate: Literal[False]
    chunk_size: Literal[1000]
    overlap: Literal[200]
    k_values: tuple[Literal[1, 3, 5, 8], ...]
    candidate_depth: Literal[20]
    bm25_k1: Literal[1.5]
    bm25_b: Literal[0.75]
    rrf_k: Literal[60]


class ExpandedEpisodeIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_slug: Literal["demis", "ilya25", "mark25"]
    episode_id: str = Field(pattern=r"^ep_[0-9a-f]{16}$")
    case_count: StrictInt = Field(gt=0)
    answerable_case_count: StrictInt = Field(gt=0)
    chunk_count: StrictInt = Field(gt=0)
    chunk_size: Literal[1000]
    overlap: Literal[200]
    input_chunk_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    faiss_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExpandedCallCounts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    index_build_document_embedding_batches: StrictInt = Field(ge=0)
    index_build_document_count: StrictInt = Field(ge=0)
    experiment_query_embedding_calls: Literal[80]
    vector_retrieval_calls: Literal[80]
    bm25_retrieval_calls: Literal[80]
    rrf_fusion_calls: Literal[80]
    experiment_document_embedding_calls: Literal[0]
    answer_model_calls: Literal[0]


class MetricDelta(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hit_count: StrictInt
    hit_at_k: float
    macro_recall_sum: float
    macro_recall_at_k: float


class MethodLatency(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    total: float = Field(ge=0)
    median_per_query: float = Field(ge=0)


class ExpandedMethodResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    retriever_name: MethodName
    aggregate_metrics: dict[int, MetricSummaryAtK]
    category_metrics: dict[EvaluationCategory, dict[int, MetricSummaryAtK]]
    episode_metrics: dict[str, dict[int, MetricSummaryAtK]]
    delta_vs_vector: dict[int, MetricDelta]
    latency_seconds: MethodLatency
    cases: tuple[EvaluationCaseResult, ...] = Field(min_length=80, max_length=80)

    @model_validator(mode="after")
    def validate_method(self) -> Self:
        expected_k = (1, 3, 5, 8)
        if tuple(self.aggregate_metrics) != expected_k:
            raise ValueError("总体指标必须覆盖 K=1,3,5,8")
        if tuple(self.delta_vs_vector) != expected_k:
            raise ValueError("相对 Vector 差值必须覆盖 K=1,3,5,8")
        if set(self.episode_metrics) != {"demis", "ilya25", "mark25"}:
            raise ValueError("每种方法必须包含三期节目指标")
        return self


class BM25CorpusSize(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_tokens: StrictInt = Field(gt=0)
    vocabulary_sum: StrictInt = Field(gt=0)
    postings: StrictInt = Field(gt=0)
    memory_bytes: StrictInt = Field(gt=0)


class EvaluationV2RetrievalExperiment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    experiment_id: str = Field(pattern=r"^eval_dataset_v2_pilot-retrieval-[0-9]{8}t[0-9]{6}z$")
    created_at_utc: datetime
    dataset: ExpandedDatasetSummary
    configuration: ExpandedRetrievalConfiguration
    episode_identities: tuple[ExpandedEpisodeIdentity, ...] = Field(
        min_length=3, max_length=3
    )
    input_hashes: dict[str, str]
    call_counts: ExpandedCallCounts
    bm25_corpus_size: BM25CorpusSize
    methods: tuple[ExpandedMethodResult, ...] = Field(min_length=3, max_length=3)
    total_experiment_seconds: float = Field(gt=0)
    protected_inputs_unchanged: Literal[True]

    @model_validator(mode="after")
    def validate_complete_experiment(self) -> Self:
        if tuple(item.episode_slug for item in self.episode_identities) != (
            "demis",
            "ilya25",
            "mark25",
        ):
            raise ValueError("episode identities 必须按固定顺序保存")
        if tuple(item.retriever_name for item in self.methods) != (
            "vector",
            "bm25",
            "hybrid_rrf",
        ):
            raise ValueError("三种检索方法必须按固定顺序保存")
        case_orders = [tuple(case.case_id for case in method.cases) for method in self.methods]
        if len(set(case_orders)) != 1:
            raise ValueError("三种检索方法必须覆盖完全相同的案例顺序")
        if any(
            summary.hit_denominator != self.dataset.answerable_case_count
            for method in self.methods
            for summary in method.aggregate_metrics.values()
        ):
            raise ValueError("总体指标分母必须等于 64 道可回答题")
        if any(not _is_sha256(value) for value in self.input_hashes.values()):
            raise ValueError("所有输入文件摘要必须为 SHA-256")
        return self


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
