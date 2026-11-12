"""Strict metadata-only schemas for the frozen Day 15 rerank experiment."""

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
from app.schemas.evaluation_result import EvaluationCaseResult, MetricSummaryAtK
from app.schemas.retrieval_experiment import (
    Day14CorpusIdentity,
    DiagnosticSubsetCase,
    MethodMetricDeltaAtK,
)


Day15MethodName = Literal[
    "vector",
    "vector_rerank",
    "hybrid_rrf",
    "hybrid_rrf_rerank",
]


class Day15InputArtifactHashes(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transcript_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunk_catalog_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    faiss_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    day12_baseline_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    day13_experiment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    day14_experiment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RerankerConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: Literal["jina_reranker_api"]
    provider: Literal["Jina AI"]
    endpoint: Literal["https://api.jina.ai/v1/rerank"]
    model_name: Literal["jina-reranker-v3"]
    model_revision: None
    model_parameter_count: Literal[597000000]
    score_semantics: Literal["jina_relevance_score_higher_is_better"]
    candidate_depth: Literal[20]
    returned_depth: Literal[20]
    product_focus_k: Literal[5]
    evaluation_k_values: tuple[Literal[1, 3, 5, 8], ...]
    maximum_input_tokens: Literal[131072]
    maximum_documents_per_model_pass: Literal[64]
    application_document_character_limit: Literal[1000]
    truncation_rule: Literal["first_1000_unicode_code_points"]
    provider_auto_truncation: Literal[True]
    timeout_seconds: Literal[60.0]
    maximum_retries: Literal[0]
    batching: Literal["one_query_with_20_documents_per_request"]
    concurrency: Literal["sequential_single_request"]
    return_documents: Literal[False]
    input_order: Literal["original_rank_ascending"]
    equal_score_tiebreakers: tuple[
        Literal[
            "original_rank_ascending",
            "chunk_catalog_position_ascending",
            "chunk_id_ascending",
        ],
        ...,
    ]
    official_model_url: str


class Day15SelectionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fixed_before_results: Literal[True]
    vector_rerank_requirements: tuple[str, ...] = Field(min_length=8)
    hybrid_rerank_additional_requirements: tuple[str, ...] = Field(min_length=3)
    minimum_distinct_recall_at_1_improvement_cases: Literal[2]
    maximum_acceptable_p95_request_seconds: Literal[5.0]
    one_case_percentage_points: Literal[6.25]
    production_change_deferred_to_day17: Literal[True]
    small_sample_warning: str


class CandidatePoolCaseLimit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    answerable: StrictBool
    expected_source_segment_ids: tuple[str, ...]
    present_expected_source_segment_ids: tuple[str, ...]
    missing_expected_source_segment_ids: tuple[str, ...]
    hit_at_20: StrictBool | None
    recall_at_20: float | None = Field(default=None, ge=0, le=1)


class CandidatePoolLimit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_method: Literal["vector", "hybrid_rrf"]
    candidate_depth: Literal[20]
    aggregate_at_20: MetricSummaryAtK
    multi_segment_at_20: MetricSummaryAtK
    cases: tuple[CandidatePoolCaseLimit, ...] = Field(min_length=20, max_length=20)


class Day15MethodResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    method_name: Day15MethodName
    score_semantics: Literal[
        "cosine_similarity",
        "reciprocal_rank_fusion",
        "jina_relevance_score_higher_is_better",
    ]
    day15_model_calls: StrictInt = Field(ge=0)
    aggregate_metrics: dict[int, MetricSummaryAtK]
    category_metrics: dict[EvaluationCategory, dict[int, MetricSummaryAtK]]
    diagnostic_subset_metrics: dict[int, MetricSummaryAtK]
    delta_vs_vector: dict[int, MethodMetricDeltaAtK]
    cases: tuple[EvaluationCaseResult, ...] = Field(min_length=20, max_length=20)

    @model_validator(mode="after")
    def validate_complete_method(self) -> Self:
        expected_k = (1, 3, 5, 8)
        if tuple(self.aggregate_metrics) != expected_k:
            raise ValueError("Day 15 方法必须保存 K=1,3,5,8 聚合指标")
        if tuple(self.diagnostic_subset_metrics) != expected_k:
            raise ValueError("Day 15 诊断子集必须保存 K=1,3,5,8")
        if tuple(self.delta_vs_vector) != expected_k:
            raise ValueError("Day 15 相对 Vector 差值必须覆盖 K=1,3,5,8")
        return self


class RerankCandidateAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(pattern=r"^chunk_[0-9a-f]{16}$")
    source_segment_ids: tuple[str, ...]
    original_retriever: Literal["vector", "hybrid_rrf"]
    original_rank: StrictInt = Field(ge=1, le=20)
    original_score: float
    original_score_semantics: Literal["cosine_similarity", "reciprocal_rank_fusion"]
    rerank_rank: StrictInt = Field(ge=1, le=20)
    rerank_score: float
    rerank_score_semantics: Literal["jina_relevance_score_higher_is_better"]

    @field_validator("original_score", "rerank_score")
    @classmethod
    def require_finite_scores(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Day 15 审计分数必须有限")
        return value


class RerankCaseAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    candidates: tuple[RerankCandidateAudit, ...] = Field(min_length=20, max_length=20)

    @model_validator(mode="after")
    def validate_ranks_and_identity(self) -> Self:
        if [item.rerank_rank for item in self.candidates] != list(range(1, 21)):
            raise ValueError("Rerank 审计必须按 rerank rank 1..20 保存")
        if {item.original_rank for item in self.candidates} != set(range(1, 21)):
            raise ValueError("Rerank 审计必须保留完整原始 rank 1..20")
        ids = [item.chunk_id for item in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("Rerank 审计不能重复 Chunk")
        return self


class RerankRunAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_method: Literal["vector", "hybrid_rrf"]
    output_method: Literal["vector_rerank", "hybrid_rrf_rerank"]
    cases: tuple[RerankCaseAudit, ...] = Field(min_length=20, max_length=20)


class RegressionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    method_name: Day15MethodName
    new_hit_at_1_regressions_vs_vector: tuple[str, ...]
    fixed_hit_at_1_failures_vs_vector: tuple[str, ...]
    new_hit_at_3_regressions_vs_vector: tuple[str, ...]
    fixed_hit_at_3_failures_vs_vector: tuple[str, ...]
    recall_at_1_improvement_case_ids_vs_vector: tuple[str, ...]
    source_method_hit_at_1_repairs: tuple[str, ...]
    source_method_hit_at_1_regressions: tuple[str, ...]


class Day15MethodCaseReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    method_name: Day15MethodName
    correct_candidate_ranks: dict[str, tuple[StrictInt, ...]]
    hit_at_1: StrictBool | None
    hit_at_3: StrictBool | None
    recall_at_5: float | None = Field(default=None, ge=0, le=1)
    recall_at_8: float | None = Field(default=None, ge=0, le=1)


class Day15CaseReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    review_reason: Literal["predeclared_focus", "new_hit_regression"]
    methods: tuple[Day15MethodCaseReview, ...] = Field(min_length=4, max_length=4)


class RerankCallCounts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    vector_query_embedding_calls: Literal[0]
    vector_retrieval_calls: Literal[0]
    bm25_retrieval_calls: Literal[0]
    rrf_fusion_calls: Literal[0]
    rerank_logical_calls: Literal[40]
    rerank_http_attempts: Literal[40]
    successful_provider_responses: Literal[40]
    pre_formal_failed_rerank_attempts: StrictInt = Field(ge=0)
    pre_formal_successful_synthetic_diagnostic_requests: StrictInt = Field(ge=0)
    query_document_pairs: Literal[800]
    document_embedding_calls: Literal[0]
    answer_model_calls: Literal[0]


class LatencySummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_count: StrictInt = Field(ge=0)
    total_seconds: float = Field(ge=0)
    median_seconds: float = Field(ge=0)
    p95_seconds: float = Field(ge=0)
    maximum_seconds: float = Field(ge=0)


class RerankLatencyAndCost(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_load_seconds: None
    cold_start_note: Literal["remote_api_not_measured_separately"]
    vector_rerank: LatencySummary
    hybrid_rrf_rerank: LatencySummary
    all_requests: LatencySummary
    total_experiment_seconds: float = Field(ge=0)
    query_characters: StrictInt = Field(ge=0)
    document_characters: StrictInt = Field(ge=0)
    provider_reported_total_tokens: StrictInt | None = Field(default=None, ge=0)
    provider_billing_unit: Literal["input_tokens"]
    confirmed_monetary_cost_usd: None
    monetary_cost_status: Literal["unknown_provider_did_not_return_currency_cost"]
    local_peak_memory_bytes: None
    local_model_disk_bytes: None
    remote_model_parameter_count: Literal[597000000]
    latency_evidence_limit: str


Day15RecommendationValue = Literal[
    "retain_vector",
    "vector_rerank_candidate_default",
    "hybrid_rerank_candidate_default",
    "rerank_query_specific",
    "insufficient_sample_no_switch",
]


class Day15Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: Day15RecommendationValue
    vector_rerank_quality_qualified: StrictBool
    hybrid_rerank_quality_qualified: StrictBool
    latency_acceptable_by_predeclared_threshold: StrictBool
    monetary_cost_confirmed: Literal[False]
    production_default_changed: Literal[False]
    rationale: str
    uncertainty: str


class RerankExperimentResult(BaseModel):
    """Complete Day 15 result without questions, text, prompts, answers or secrets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    experiment_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{2,127}$")
    created_at_utc: datetime
    dataset_id: Literal["eval_dataset_v1"]
    dataset_cases_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    episode_id: str = Field(pattern=r"^ep_[0-9a-f]{16}$")
    k_values: tuple[Literal[1, 3, 5, 8], ...]
    corpus_identity: Day14CorpusIdentity
    day14_experiment_id: str
    input_artifact_hashes: Day15InputArtifactHashes
    reranker_configuration: RerankerConfiguration
    selection_policy: Day15SelectionPolicy
    diagnostic_subset: tuple[DiagnosticSubsetCase, ...]
    candidate_pool_limits: tuple[CandidatePoolLimit, ...] = Field(min_length=2, max_length=2)
    methods: tuple[Day15MethodResult, ...] = Field(min_length=4, max_length=4)
    rerank_audits: tuple[RerankRunAudit, ...] = Field(min_length=2, max_length=2)
    regression_summaries: tuple[RegressionSummary, ...] = Field(min_length=4, max_length=4)
    focus_and_regression_reviews: tuple[Day15CaseReview, ...]
    call_counts: RerankCallCounts
    latency_and_cost: RerankLatencyAndCost
    recommendation: Day15Recommendation
    limitations: tuple[str, ...] = Field(min_length=4)

    @model_validator(mode="after")
    def validate_complete_experiment(self) -> Self:
        method_order = (
            "vector",
            "vector_rerank",
            "hybrid_rrf",
            "hybrid_rrf_rerank",
        )
        if tuple(item.method_name for item in self.methods) != method_order:
            raise ValueError("Day 15 必须按四种固定方法顺序保存")
        if tuple(item.method_name for item in self.regression_summaries) != method_order:
            raise ValueError("Day 15 退化摘要必须覆盖四种固定方法")
        case_ids = tuple(item.case_id for item in self.methods[0].cases)
        if len(case_ids) != 20 or len(set(case_ids)) != 20:
            raise ValueError("Day 15 必须保存同一组 20 道冻结题")
        if any(tuple(case.case_id for case in method.cases) != case_ids for method in self.methods):
            raise ValueError("Day 15 四种方法必须共享完全相同的 case 顺序")
        for audit in self.rerank_audits:
            if tuple(case.case_id for case in audit.cases) != case_ids:
                raise ValueError("Rerank 审计必须覆盖相同顺序的 20 道题")
        return self
