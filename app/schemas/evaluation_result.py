"""Strict versioned schemas for one Day 12 retrieval baseline."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from app.schemas.evaluation import EvaluationCategory


class SavedCandidate(BaseModel):
    """Metadata-only candidate safe to persist in a baseline artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: StrictInt = Field(ge=1)
    chunk_id: str = Field(pattern=r"^chunk_[0-9a-f]{16}$")
    source_segment_ids: tuple[str, ...] = Field(min_length=1)
    score: float
    score_semantics: str = Field(min_length=1, max_length=64)

    @field_validator("score")
    @classmethod
    def require_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("检索分数必须是有限数值")
        return value


class CaseMetricsAtK(BaseModel):
    """Hit/Recall for one case and one K, or explicit nulls when unscored."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scored: StrictBool
    hit_at_k: StrictBool | None
    recall_at_k: float | None = Field(default=None, ge=0, le=1)
    correct_segment_count: StrictInt | None = Field(default=None, ge=0)
    expected_segment_count: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def validate_scored_state(self) -> Self:
        values = (
            self.hit_at_k,
            self.recall_at_k,
            self.correct_segment_count,
        )
        if self.scored and any(value is None for value in values):
            raise ValueError("计分案例必须包含完整 Hit/Recall 统计")
        if not self.scored and any(value is not None for value in values):
            raise ValueError("不计分案例的 Hit/Recall 必须为 null")
        return self


class MetricSummaryAtK(BaseModel):
    """Auditable aggregate numerators and denominators for one K."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hit_count: StrictInt = Field(ge=0)
    hit_denominator: StrictInt = Field(ge=0)
    hit_at_k: float | None = Field(default=None, ge=0, le=1)
    macro_recall_sum: float = Field(ge=0)
    macro_recall_denominator: StrictInt = Field(ge=0)
    macro_recall_at_k: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_denominators(self) -> Self:
        if self.hit_denominator != self.macro_recall_denominator:
            raise ValueError("Hit 与 Macro Recall 必须使用同一可回答题分母")
        if self.hit_count > self.hit_denominator:
            raise ValueError("hit_count 不能大于分母")
        if self.hit_denominator == 0:
            if self.hit_at_k is not None or self.macro_recall_at_k is not None:
                raise ValueError("零分母聚合指标必须为 null")
        elif self.hit_at_k is None or self.macro_recall_at_k is None:
            raise ValueError("非零分母聚合指标不能为 null")
        return self


class EvaluationCaseResult(BaseModel):
    """One question's safe retrieval diagnostics and prefix metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    category: EvaluationCategory
    answerable: StrictBool
    scored: StrictBool
    expected_source_segment_ids: tuple[str, ...]
    retrieved_candidates: tuple[SavedCandidate, ...]
    metrics_by_k: dict[int, CaseMetricsAtK]
    elapsed_seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_case_state(self) -> Self:
        if self.scored != self.answerable:
            raise ValueError("本次检索基线只对可回答题计分")
        if self.answerable != bool(self.expected_source_segment_ids):
            raise ValueError("answerable 必须与稳定来源真值一致")
        expected_ranks = list(range(1, len(self.retrieved_candidates) + 1))
        actual_ranks = [item.rank for item in self.retrieved_candidates]
        if actual_ranks != expected_ranks:
            raise ValueError("保存的候选 rank 必须从 1 连续且保持原始顺序")
        if any(metric.scored != self.scored for metric in self.metrics_by_k.values()):
            raise ValueError("逐 K 指标的计分状态必须与案例一致")
        return self


class RetrievalEvaluationRun(BaseModel):
    """Provider-independent output returned by the injected evaluator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    retriever_name: str
    k_values: tuple[StrictInt, ...]
    query_count: StrictInt = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)
    aggregate_metrics: dict[int, MetricSummaryAtK]
    category_metrics: dict[EvaluationCategory, dict[int, MetricSummaryAtK]]
    cases: tuple[EvaluationCaseResult, ...]


class VectorRetrieverConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    adapter: Literal["vector_retriever_adapter_v1"]
    underlying_retriever: Literal["day8_vector_retriever"]
    score_semantics: Literal["cosine_similarity"]
    query_strategy: Literal["one_top_8_call_then_prefixes"]
    equal_score_tiebreaker: Literal["faiss_row_index_ascending"]


class IndexConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    index_type: Literal["faiss_index_flat_ip"]
    distance_metric: Literal["cosine"]
    vector_count: StrictInt = Field(gt=0)
    vector_dimension: StrictInt = Field(gt=0)
    chunking_algorithm: Literal["fixed_character_windows_v1"]
    chunk_size: StrictInt = Field(gt=0)
    overlap: StrictInt = Field(ge=0)


class InputArtifactHashes(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    faiss_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


FailureReviewType = Literal[
    "failure",
    "low_recall",
    "late_hit",
    "partial_recall",
    "dangerous_distractor",
]


class FailureReview(BaseModel):
    """Human-authored review of a failure or explicitly labelled low-risk case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    review_type: FailureReviewType
    expected_source_segment_ids: tuple[str, ...]
    correct_candidate_ranks: dict[str, tuple[StrictInt, ...]]
    retrieved_expected_segment_ids: tuple[str, ...]
    missing_expected_segment_ids: tuple[str, ...]
    diagnosis: Literal["recall", "ranking", "partial_recall", "distractor_risk"]
    possible_factors: tuple[
        Literal[
            "overlap",
            "chunk_boundary",
            "semantic_confusion",
            "query_wording",
            "none_observed",
        ],
        ...,
    ]
    summary: str = Field(min_length=1, max_length=1000)
    next_step: str = Field(min_length=1, max_length=500)


class RetrievalBaselineResult(BaseModel):
    """Complete, reproducible and secret-free Day 12 baseline artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    evaluation_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{2,127}$")
    created_at_utc: datetime
    dataset_id: Literal["eval_dataset_v1"]
    dataset_cases_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    episode_id: str = Field(pattern=r"^ep_[0-9a-f]{16}$")
    retriever_name: str
    retriever_config: VectorRetrieverConfiguration
    index_config: IndexConfiguration
    input_artifact_hashes: InputArtifactHashes
    embedding_model: str
    k_values: tuple[StrictInt, ...]
    answerable_case_count: StrictInt = Field(ge=0)
    unanswerable_case_count: StrictInt = Field(ge=0)
    query_count: StrictInt = Field(ge=0)
    query_embedding_calls: StrictInt = Field(ge=0)
    document_embedding_calls: StrictInt = Field(ge=0)
    answer_model_calls: StrictInt = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)
    aggregate_metrics: dict[int, MetricSummaryAtK]
    category_metrics: dict[EvaluationCategory, dict[int, MetricSummaryAtK]]
    cases: tuple[EvaluationCaseResult, ...]
    failure_reviews: tuple[FailureReview, ...] = ()

    @model_validator(mode="after")
    def validate_counts_and_k_values(self) -> Self:
        if self.query_count != len(self.cases):
            raise ValueError("query_count 必须等于逐题结果数量")
        if self.answerable_case_count + self.unanswerable_case_count != len(
            self.cases
        ):
            raise ValueError("可回答与无答案题数量必须覆盖全部案例")
        if tuple(self.aggregate_metrics) != self.k_values:
            raise ValueError("aggregate_metrics 必须覆盖有序 k_values")
        return self
