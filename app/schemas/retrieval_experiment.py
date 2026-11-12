"""Strict, metadata-only schemas for the Day 14 retrieval comparison."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator

from app.schemas.evaluation import EvaluationCategory
from app.schemas.evaluation_result import (
    EvaluationCaseResult,
    MetricSummaryAtK,
)


RetrievalMethodName = Literal["vector", "bm25", "hybrid_rrf"]


class Day14CorpusIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: str = Field(pattern=r"^ep_[0-9a-f]{16}$")
    source_file: str
    chunking_algorithm: Literal["fixed_character_windows_v1"]
    chunk_size: Literal[1000]
    overlap: Literal[200]
    input_chunk_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunk_count: StrictInt = Field(gt=0)
    chunk_id_order_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BM25TokenizerConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["unicode_nfkc_casefold_alnum_joiners_v1"]
    unicode_normalization: Literal["NFKC"]
    casing: Literal["casefold"]
    token_characters: Literal["unicode_letters_and_numbers"]
    hyphen_handling: Literal["preserve_internal_and_normalize_variants"]
    apostrophe_handling: Literal["preserve_internal_and_normalize_variants"]
    stemming: Literal[False]
    stopword_removal: Literal[False]


class BM25Configuration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    implementation: Literal["auditable_bm25_okapi_v1"]
    score_semantics: Literal["bm25_okapi"]
    idf_formula: Literal["ln(1+(N-df+0.5)/(df+0.5))"]
    k1: Literal[1.5]
    b: Literal[0.75]
    query_term_duplicates: Literal["deduplicated_in_first_occurrence_order"]
    equal_score_tiebreaker: Literal["chunk_catalog_position_ascending"]
    tokenizer: BM25TokenizerConfiguration


class RRFConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    implementation: Literal["reciprocal_rank_fusion_v1"]
    score_semantics: Literal["reciprocal_rank_fusion"]
    formula: Literal["sum(1/(rrf_k+rank_i))"]
    rank_origin: Literal[1]
    rrf_k: Literal[60]
    child_candidate_depth: Literal[20]
    final_k_values: tuple[Literal[1, 3, 5, 8], ...]
    merge_key: Literal["chunk_id"]
    equal_score_tiebreakers: tuple[
        Literal[
            "best_child_rank_ascending",
            "chunk_catalog_position_ascending",
            "chunk_id_ascending",
        ],
        ...,
    ]


class Day14SelectionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fixed_before_results: Literal[True]
    ordered_criteria: tuple[str, ...] = Field(min_length=8)
    hybrid_candidate_requirements: tuple[str, ...] = Field(min_length=4)
    one_case_percentage_points: Literal[6.25]
    small_sample_warning: str


class DiagnosticSubsetCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    reason: str = Field(min_length=1, max_length=300)


class Day14InputArtifactHashes(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transcript_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunk_catalog_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    faiss_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    day12_baseline_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    day13_experiment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class MethodMetricDeltaAtK(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hit_count: StrictInt
    hit_at_k: float
    macro_recall_sum: float
    macro_recall_at_k: float


class RetrievalMethodResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    retriever_name: RetrievalMethodName
    score_semantics: Literal[
        "cosine_similarity", "bm25_okapi", "reciprocal_rank_fusion"
    ]
    retrieval_calls: StrictInt = Field(ge=0)
    aggregate_metrics: dict[int, MetricSummaryAtK]
    category_metrics: dict[EvaluationCategory, dict[int, MetricSummaryAtK]]
    diagnostic_subset_metrics: dict[int, MetricSummaryAtK]
    delta_vs_vector: dict[int, MethodMetricDeltaAtK]
    cases: tuple[EvaluationCaseResult, ...]

    @model_validator(mode="after")
    def validate_complete_method(self) -> Self:
        if self.retrieval_calls != len(self.cases):
            raise ValueError("每种方法的 retrieval_calls 必须等于题目数")
        expected_k = (1, 3, 5, 8)
        if tuple(self.aggregate_metrics) != expected_k:
            raise ValueError("每种方法必须保存 K=1,3,5,8 聚合指标")
        if tuple(self.diagnostic_subset_metrics) != expected_k:
            raise ValueError("诊断子集必须保存 K=1,3,5,8 指标")
        if tuple(self.delta_vs_vector) != expected_k:
            raise ValueError("相对 Vector 差值必须覆盖 K=1,3,5,8")
        return self


class HybridFusionCandidateAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    final_rank: StrictInt = Field(ge=1)
    chunk_id: str = Field(pattern=r"^chunk_[0-9a-f]{16}$")
    source_segment_ids: tuple[str, ...]
    rrf_score: float = Field(gt=0)
    vector_rank: StrictInt | None = Field(default=None, ge=1, le=20)
    bm25_rank: StrictInt | None = Field(default=None, ge=1, le=20)

    @model_validator(mode="after")
    def require_component_rank(self) -> Self:
        if self.vector_rank is None and self.bm25_rank is None:
            raise ValueError("Hybrid 候选必须至少来自一个子排名")
        return self


class HybridFusionCaseAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    candidates: tuple[HybridFusionCandidateAudit, ...] = Field(max_length=20)


class MethodCaseReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    retriever_name: RetrievalMethodName
    correct_candidate_ranks: dict[str, tuple[StrictInt, ...]]
    retrieved_expected_segment_ids: tuple[str, ...]
    missing_expected_segment_ids: tuple[str, ...]
    hit_at_1: StrictBool | None
    hit_at_3: StrictBool | None
    recall_at_5: float | None = Field(default=None, ge=0, le=1)
    recall_at_8: float | None = Field(default=None, ge=0, le=1)


class Day14CaseReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    review_reason: Literal["preselected_focus", "new_hit_regression"]
    methods: tuple[MethodCaseReview, ...] = Field(min_length=3, max_length=3)


class VectorDriftComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    day12_evaluation_id: str
    metrics_match_day12: StrictBool
    top8_exact_order_match_count: StrictInt = Field(ge=0, le=20)
    top8_case_count: Literal[20]
    ranking_drift_case_ids: tuple[str, ...]
    metric_delta_vs_day12: dict[int, MethodMetricDeltaAtK]
    interpretation: str


class Day14CallCounts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    vector_retrieval_calls: Literal[20]
    vector_query_embedding_calls: Literal[20]
    bm25_retrieval_calls: Literal[20]
    rrf_fusion_calls: Literal[20]
    document_embedding_calls: Literal[0]
    answer_model_calls: Literal[0]


class Day14LatencyAndSize(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    bm25_build_seconds: float = Field(ge=0)
    vector_query_seconds: float = Field(ge=0)
    vector_query_median_seconds: float = Field(ge=0)
    bm25_query_seconds: float = Field(ge=0)
    bm25_query_median_seconds: float = Field(ge=0)
    rrf_fusion_seconds: float = Field(ge=0)
    rrf_fusion_median_seconds: float = Field(ge=0)
    total_experiment_seconds: float = Field(ge=0)
    bm25_document_token_count: StrictInt = Field(ge=0)
    bm25_vocabulary_size: StrictInt = Field(ge=0)
    bm25_posting_count: StrictInt = Field(ge=0)
    bm25_python_memory_bytes_estimate: StrictInt = Field(ge=0)
    faiss_file_bytes: StrictInt = Field(gt=0)
    latency_evidence_limit: str


Day14RecommendationValue = Literal[
    "hybrid_rrf_candidate_default",
    "retain_vector",
    "insufficient_evidence_no_switch",
]


class Day14Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: Day14RecommendationValue
    hybrid_eligible_by_hit_at_3: StrictBool
    new_hybrid_hit_at_1_regressions: tuple[str, ...]
    new_hybrid_hit_at_3_regressions: tuple[str, ...]
    rationale: str = Field(min_length=1, max_length=2000)
    uncertainty: str = Field(min_length=1, max_length=1200)
    production_default_changed: Literal[False]


class RetrievalExperimentResult(BaseModel):
    """Complete Day 14 result; questions, Chunk text and model responses are absent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    experiment_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{2,127}$")
    created_at_utc: datetime
    dataset_id: Literal["eval_dataset_v1"]
    dataset_cases_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    episode_id: str = Field(pattern=r"^ep_[0-9a-f]{16}$")
    k_values: tuple[Literal[1, 3, 5, 8], ...]
    corpus_identity: Day14CorpusIdentity
    input_artifact_hashes: Day14InputArtifactHashes
    bm25_configuration: BM25Configuration
    rrf_configuration: RRFConfiguration
    selection_policy: Day14SelectionPolicy
    diagnostic_subset: tuple[DiagnosticSubsetCase, ...]
    methods: tuple[RetrievalMethodResult, ...] = Field(min_length=3, max_length=3)
    hybrid_fusion_audit: tuple[HybridFusionCaseAudit, ...] = Field(min_length=20, max_length=20)
    focus_and_regression_reviews: tuple[Day14CaseReview, ...]
    vector_drift: VectorDriftComparison
    call_counts: Day14CallCounts
    latency_and_size: Day14LatencyAndSize
    recommendation: Day14Recommendation

    @model_validator(mode="after")
    def validate_complete_comparison(self) -> Self:
        if tuple(item.retriever_name for item in self.methods) != (
            "vector",
            "bm25",
            "hybrid_rrf",
        ):
            raise ValueError("正式结果必须按 Vector/BM25/Hybrid-RRF 保存")
        case_ids = tuple(item.case_id for item in self.methods[0].cases)
        if len(case_ids) != 20 or len(set(case_ids)) != 20:
            raise ValueError("正式结果必须包含同一组 20 道冻结题")
        if any(tuple(item.case_id for item in method.cases) != case_ids for method in self.methods):
            raise ValueError("三种方法的 case 顺序必须完全一致")
        if tuple(item.case_id for item in self.hybrid_fusion_audit) != case_ids:
            raise ValueError("Hybrid 审计必须覆盖相同顺序的 20 道题")
        return self
