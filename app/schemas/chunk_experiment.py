"""Strict, secret-free schemas for the Day 13 Chunk parameter experiment."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator

from app.schemas.evaluation import EvaluationCategory
from app.schemas.evaluation_result import (
    EvaluationCaseResult,
    MetricSummaryAtK,
    VectorRetrieverConfiguration,
)


ChunkExperimentConfigId = Literal["small", "baseline", "large"]


class ChunkExperimentControlledVariables(BaseModel):
    """Inputs and behavior that every configuration must share."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transcript_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_segment_truth: Literal["expected_source_segment_ids"]
    expected_chunk_ids_scope: Literal["baseline_1000_200_only"]
    embedding_document_task: str
    embedding_query_task: str
    embedding_type: str
    provider_normalized: StrictBool
    provider_truncate: StrictBool
    index_type: Literal["faiss_index_flat_ip"]
    distance_metric: Literal["cosine"]
    retriever_config: VectorRetrieverConfiguration
    k_values: tuple[Literal[1, 3, 5, 8], ...]
    query_strategy: Literal["one_top_8_call_per_case_then_prefixes"]
    answerable_case_count: StrictInt = Field(ge=0)
    unanswerable_case_count: StrictInt = Field(ge=0)
    unanswerable_cases_scored: Literal[False]
    answer_model_calls: Literal[0]


class ChunkExperimentSelectionPolicy(BaseModel):
    """The deterministic policy fixed before alternative results are observed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_hit_at_3: float = Field(ge=0, le=1)
    ordered_quality_criteria: tuple[str, ...] = Field(min_length=5)
    ordered_tiebreakers: tuple[str, ...] = Field(min_length=3)
    small_sample_warning: str = Field(min_length=1)
    overlap_causal_claim_allowed: Literal[False]


class MetricDeltaAtK(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hit_count: StrictInt
    hit_at_k: float
    macro_recall_sum: float
    macro_recall_at_k: float


class ChunkExperimentDelta(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    by_k: dict[int, MetricDeltaAtK]
    chunk_count: StrictInt
    shared_source_candidate_count: StrictInt


class SourceDiversityDiagnostics(BaseModel):
    """Top-8 overlap proxy based only on stable source-segment metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_count: StrictInt = Field(ge=0)
    shared_source_candidate_count: StrictInt = Field(ge=0)
    shared_source_candidate_rate: float = Field(ge=0, le=1)
    unique_source_segment_count_sum: StrictInt = Field(ge=0)
    mean_unique_source_segments_per_case: float = Field(ge=0)


class ChunkExperimentConfigurationResult(BaseModel):
    """One complete configuration result with index and provider provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    config_id: ChunkExperimentConfigId
    chunk_size: StrictInt = Field(gt=0)
    overlap: StrictInt = Field(ge=0)
    overlap_ratio: float = Field(gt=0, lt=1)
    chunk_count: StrictInt = Field(gt=0)
    vector_count: StrictInt = Field(gt=0)
    vector_dimension: StrictInt = Field(gt=0)
    index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_chunk_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    index_reused: StrictBool
    document_embedding_calls: StrictInt = Field(ge=0)
    query_embedding_calls: StrictInt = Field(ge=0)
    experiment_document_embedding_calls: StrictInt = Field(ge=0)
    experiment_query_embedding_calls: StrictInt = Field(ge=0)
    answer_model_calls: Literal[0]
    build_elapsed_seconds: float = Field(ge=0)
    query_elapsed_seconds: float = Field(ge=0)
    aggregate_metrics: dict[int, MetricSummaryAtK]
    category_metrics: dict[EvaluationCategory, dict[int, MetricSummaryAtK]]
    source_diversity: SourceDiversityDiagnostics
    cases: tuple[EvaluationCaseResult, ...]
    delta_vs_baseline: ChunkExperimentDelta

    @model_validator(mode="after")
    def validate_configuration(self) -> Self:
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap 必须小于 chunk_size")
        if abs(self.overlap_ratio - self.overlap / self.chunk_size) > 1e-12:
            raise ValueError("overlap_ratio 必须等于 overlap/chunk_size")
        if self.chunk_count != self.vector_count:
            raise ValueError("Chunk 数量必须等于向量数量")
        if self.query_embedding_calls != len(self.cases):
            raise ValueError("每组必须为每道题记录一次查询 Embedding")
        if tuple(self.aggregate_metrics) != (1, 3, 5, 8):
            raise ValueError("每组聚合指标必须按 K=1,3,5,8 保存")
        return self


class ChunkExperimentComparisonRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    config_id: ChunkExperimentConfigId
    eligible_by_hit_at_3: StrictBool
    new_hit_at_1_failures: tuple[str, ...]
    new_hit_at_3_failures: tuple[str, ...]
    selection_rank: StrictInt = Field(ge=1, le=3)


class ChunkExperimentSelectedDefault(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    config_id: ChunkExperimentConfigId
    chunk_size: StrictInt = Field(gt=0)
    overlap: StrictInt = Field(ge=0)
    retained_day12_baseline: StrictBool
    rationale: str = Field(min_length=1, max_length=1500)
    uncertainty: str = Field(min_length=1, max_length=1000)


class CaseConfigurationReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    config_id: ChunkExperimentConfigId
    correct_candidate_ranks: dict[str, tuple[StrictInt, ...]]
    retrieved_expected_segment_ids: tuple[str, ...]
    missing_expected_segment_ids: tuple[str, ...]
    shared_source_candidate_ranks: tuple[StrictInt, ...]
    hit_at_1: StrictBool
    hit_at_3: StrictBool
    recall_at_5: float = Field(ge=0, le=1)
    recall_at_8: float = Field(ge=0, le=1)


class ChunkExperimentFailureReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    review_reason: Literal["preselected_focus", "new_regression"]
    configurations: tuple[CaseConfigurationReview, ...] = Field(min_length=3, max_length=3)
    summary: str = Field(min_length=1, max_length=1200)


class ChunkExperimentInputHashes(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transcript_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_faiss_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ChunkParameterExperimentResult(BaseModel):
    """Complete Day 13 artifact; it deliberately contains no question or text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    experiment_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{2,127}$")
    created_at_utc: datetime
    dataset_id: Literal["eval_dataset_v1"]
    dataset_cases_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    episode_id: str = Field(pattern=r"^ep_[0-9a-f]{16}$")
    embedding_model: str
    distance_metric: Literal["cosine"]
    k_values: tuple[Literal[1, 3, 5, 8], ...]
    input_artifact_hashes: ChunkExperimentInputHashes
    controlled_variables: ChunkExperimentControlledVariables
    selection_policy: ChunkExperimentSelectionPolicy
    baseline_evaluation_id: str
    configurations: tuple[ChunkExperimentConfigurationResult, ...] = Field(
        min_length=3,
        max_length=3,
    )
    comparison: tuple[ChunkExperimentComparisonRow, ...] = Field(
        min_length=3,
        max_length=3,
    )
    selected_default: ChunkExperimentSelectedDefault
    failure_reviews: tuple[ChunkExperimentFailureReview, ...]
    elapsed_seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_complete_matrix(self) -> Self:
        ids = tuple(item.config_id for item in self.configurations)
        if ids != ("small", "baseline", "large"):
            raise ValueError("正式结果必须按 small/baseline/large 保存三组配置")
        if tuple(item.config_id for item in self.comparison) != ids:
            raise ValueError("comparison 必须与 configurations 使用同一顺序")
        if self.selected_default.config_id not in ids:
            raise ValueError("selected_default 必须来自正式参数矩阵")
        return self
