"""Strict metadata-only schema for the Day 17 RAG 2.0 freeze."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator


FailureCategory = Literal[
    "retrieval_no_recall",
    "ranking_below_top_k",
    "partial_multi_source_recall",
    "chunk_boundary_or_duplication",
    "question_ambiguity",
    "insufficient_top_k_context",
    "generation_incorrect",
    "generation_incomplete",
    "citation_wrong_or_incomplete",
    "incorrect_refusal",
    "unsafe_answer_on_unanswerable",
]

FAILURE_CATEGORY_ORDER: tuple[FailureCategory, ...] = (
    "retrieval_no_recall",
    "ranking_below_top_k",
    "partial_multi_source_recall",
    "chunk_boundary_or_duplication",
    "question_ambiguity",
    "insufficient_top_k_context",
    "generation_incorrect",
    "generation_incomplete",
    "citation_wrong_or_incomplete",
    "incorrect_refusal",
    "unsafe_answer_on_unanswerable",
)

MethodName = Literal[
    "vector",
    "bm25",
    "hybrid_rrf",
    "vector_rerank",
    "hybrid_rrf_rerank",
    "answer_generation",
]


class Day17InputArtifactHashes(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    eval_dataset_v1_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    day12_vector_baseline_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    day13_chunk_experiment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    day14_retrieval_experiment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    day15_rerank_experiment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    day16_answer_quality_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    eval_dataset_v2_pilot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    eval_dataset_v2_retrieval_experiment_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )


class RetrievalDefault(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["vector"]
    embedding_model: Literal["jina-embeddings-v5-text-small"]
    score_semantics: Literal["cosine_similarity"]
    evidence_depth: Literal[5]
    rationale: str = Field(min_length=1, max_length=1600)


class CandidateRetriever(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["vector", "bm25", "hybrid_rrf"]
    role: Literal["production_default", "hybrid_component", "candidate_experiment"]
    production_enabled: StrictBool
    answer_quality_evaluated: StrictBool
    retrieval_evidence_scope: str = Field(min_length=1, max_length=500)
    improvement_summary: str = Field(min_length=1, max_length=1000)
    regression_summary: str = Field(min_length=1, max_length=1000)
    next_gate: str = Field(min_length=1, max_length=1000)


class RerankDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["query_specific_candidate_not_enabled"]
    evaluated_methods: tuple[
        Literal["vector_rerank", "hybrid_rrf_rerank"], ...
    ] = Field(min_length=2, max_length=2)
    stable_quality_gain: Literal[False]
    latency_p95_seconds: float = Field(ge=0)
    monetary_cost_confirmed: Literal[False]
    answer_quality_evaluated: Literal[False]
    rationale: str = Field(min_length=1, max_length=1400)


class ChunkConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    algorithm: Literal["fixed_character_windows_v1"]
    chunk_size: Literal[1000]
    overlap: Literal[200]
    selection_source: Literal["day13_controlled_comparison"]
    episode_input_chunk_sha256: dict[
        Literal["demis", "ilya25", "mark25"], str
    ]
    globally_optimal_claim: Literal[False]

    @model_validator(mode="after")
    def validate_episode_hashes(self) -> Self:
        if tuple(self.episode_input_chunk_sha256) != ("demis", "ilya25", "mark25"):
            raise ValueError("Chunk 冻结必须按固定顺序覆盖三期节目")
        if any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in self.episode_input_chunk_sha256.values()
        ):
            raise ValueError("input_chunk_sha256 必须是 SHA-256")
        return self


class AnswerConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: Literal["deepseek-v4-pro"]
    temperature: Literal[0.2]
    thinking: Literal["disabled"]
    maximum_retries: Literal[0]
    evidence_depth: Literal[5]
    grounded_system_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_changed: Literal[False]
    rationale: str = Field(min_length=1, max_length=1400)


class RankEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    method: MethodName
    expected_source_ranks: tuple[StrictInt, ...]
    evaluated_depth: StrictInt = Field(ge=1, le=20)
    finding: str = Field(min_length=1, max_length=900)

    @model_validator(mode="after")
    def validate_ranks(self) -> Self:
        if any(rank < 1 or rank > self.evaluated_depth for rank in self.expected_source_ranks):
            raise ValueError("证据排名必须位于声明的评测深度内")
        if tuple(sorted(set(self.expected_source_ranks))) != self.expected_source_ranks:
            raise ValueError("证据排名必须去重并升序")
        return self


class FailureTaxonomyEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    category: FailureCategory
    observed: StrictBool
    case_ids: tuple[str, ...]
    methods: tuple[MethodName, ...]
    evidence_ranks: tuple[RankEvidence, ...]
    answer_impact: str = Field(min_length=1, max_length=1000)
    root_cause: str = Field(min_length=1, max_length=1200)
    current_handling: str = Field(min_length=1, max_length=1000)
    next_action: str = Field(min_length=1, max_length=1000)
    blocks_rag_v2_freeze: StrictBool

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        if self.observed:
            if not self.case_ids or not self.methods or not self.evidence_ranks:
                raise ValueError("已观察失败类别必须包含真实 case、方法和排名证据")
            if set(item.case_id for item in self.evidence_ranks) - set(self.case_ids):
                raise ValueError("排名证据 case 必须属于该失败类别")
            if set(item.method for item in self.evidence_ranks) - set(self.methods):
                raise ValueError("排名证据 method 必须属于该失败类别")
        elif self.case_ids or self.methods or self.evidence_ranks:
            raise ValueError("本轮未观察到的类别不得伪造案例或排名")
        if len(set(self.case_ids)) != len(self.case_ids):
            raise ValueError("失败类别 case ID 不能重复")
        return self


class MethodCaseFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    method: MethodName
    expected_source_ranks: dict[str, tuple[StrictInt, ...]]
    hit_at_5: StrictBool | None
    recall_at_5: float | None = Field(default=None, ge=0, le=1)
    answer_quality_available: StrictBool
    finding: str = Field(min_length=1, max_length=1200)


class Day16ScoreSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    correctness: StrictInt | None = Field(default=None, ge=0, le=2)
    evidence_alignment: StrictInt | None = Field(default=None, ge=0, le=2)
    citation_correctness: StrictInt | None = Field(default=None, ge=0, le=2)
    completeness: StrictInt | None = Field(default=None, ge=0, le=2)
    correct_refusal: StrictBool | None

    @model_validator(mode="after")
    def validate_score_kind(self) -> Self:
        scores = (
            self.correctness,
            self.evidence_alignment,
            self.citation_correctness,
            self.completeness,
        )
        if self.correct_refusal is None and any(value is None for value in scores):
            raise ValueError("可回答题必须保存完整四项评分")
        if self.correct_refusal is not None and any(value is not None for value in scores):
            raise ValueError("无答案题只保存拒答结果，不伪造四项评分")
        return self


class Day17CaseReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    evidence_source: Literal["day16_answer_quality", "v2_retrieval_comparison"]
    review_roles: tuple[
        Literal[
            "required_focus",
            "complete_answer_and_citations",
            "retrieval_success_answer_not_full_score",
            "multi_source_incomplete",
            "correct_unanswerable_refusal",
            "highest_risk_failure",
        ],
        ...,
    ]
    method_findings: tuple[MethodCaseFinding, ...]
    day16_scores: Day16ScoreSummary | None
    root_cause: str = Field(min_length=1, max_length=1200)
    answer_impact: str = Field(min_length=1, max_length=1200)
    disposition: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        if self.evidence_source == "day16_answer_quality":
            if self.day16_scores is None:
                raise ValueError("Day 16 案例必须保存评分或拒答摘要")
            if any(not item.answer_quality_available for item in self.method_findings):
                raise ValueError("Day 16 案例的 Vector finding 必须有回答质量证据")
        elif self.day16_scores is not None:
            raise ValueError("纯检索案例不得伪造 Day 16 回答质量")
        return self


class RetrievalMetricFreeze(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal["vector", "bm25", "hybrid_rrf"]
    hit_at_5_count: StrictInt = Field(ge=0, le=64)
    hit_at_5_denominator: Literal[64]
    hit_at_5: float = Field(ge=0, le=1)
    macro_recall_at_5: float = Field(ge=0, le=1)


class AnswerMetricFreeze(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    answerable_case_count: Literal[16]
    unanswerable_case_count: Literal[4]
    correctness_mean: float = Field(ge=0, le=2)
    evidence_alignment_mean: float = Field(ge=0, le=2)
    citation_correctness_mean: float = Field(ge=0, le=2)
    completeness_mean: float = Field(ge=0, le=2)
    correct_refusal_count: StrictInt = Field(ge=0, le=4)
    core_claim_support_rate: float = Field(ge=0, le=1)


class EvidenceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    expanded_retrieval_case_count: Literal[80]
    expanded_answerable_case_count: Literal[64]
    expanded_unanswerable_case_count: Literal[16]
    retrieval_top5: tuple[RetrievalMetricFreeze, ...] = Field(min_length=3, max_length=3)
    day16_answer_quality: AnswerMetricFreeze
    hybrid_top3_hit_lower_than_vector: Literal[True]
    hybrid_top5_answer_quality_evaluated: Literal[False]
    v2_independent_human_review_complete: Literal[False]


class RagVersionComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dimension: Literal[
        "online_pipeline",
        "evaluation_scope",
        "retrieval_comparison",
        "answer_and_refusal_evidence",
        "failure_analysis",
        "configuration_freeze",
    ]
    rag_1_0: str = Field(min_length=1, max_length=800)
    rag_2_0: str = Field(min_length=1, max_length=1200)
    production_behavior_changed: StrictBool


class Day17ProviderCallCounts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    embedding_calls: Literal[0]
    vector_query_calls: Literal[0]
    bm25_calls: Literal[0]
    rrf_calls: Literal[0]
    rerank_calls: Literal[0]
    answer_model_calls: Literal[0]
    judge_model_calls: Literal[0]


class RagV2FreezeResult(BaseModel):
    """Complete Day 17 decision artifact without questions, text, prompts or answers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    freeze_id: str = Field(pattern=r"^rag-v2-day17-[0-9]{8}t[0-9]{6}z$")
    created_at_utc: datetime
    input_artifact_hashes: Day17InputArtifactHashes
    retrieval_default: RetrievalDefault
    candidate_retrievers: tuple[CandidateRetriever, ...] = Field(
        min_length=3, max_length=3
    )
    rerank_enabled: Literal[False]
    rerank_decision: RerankDecision
    chunk_configuration: ChunkConfiguration
    answer_configuration: AnswerConfiguration
    failure_taxonomy: tuple[FailureTaxonomyEntry, ...] = Field(
        min_length=11, max_length=11
    )
    case_reviews: tuple[Day17CaseReview, ...] = Field(min_length=9)
    rag_1_vs_rag_2_comparison: tuple[RagVersionComparison, ...] = Field(
        min_length=6, max_length=6
    )
    evidence_summary: EvidenceSummary
    decision_rationale: tuple[str, ...] = Field(min_length=5)
    known_limitations: tuple[str, ...] = Field(min_length=7)
    future_evaluation_requirements: tuple[str, ...] = Field(min_length=5)
    provider_call_counts: Day17ProviderCallCounts
    production_default_changed: Literal[False]

    @model_validator(mode="after")
    def validate_complete_freeze(self) -> Self:
        if (
            self.created_at_utc.tzinfo is None
            or self.created_at_utc.utcoffset() is None
            or self.created_at_utc.utcoffset().total_seconds() != 0
        ):
            raise ValueError("created_at_utc 必须是 UTC 时间")
        if tuple(item.name for item in self.candidate_retrievers) != (
            "vector",
            "bm25",
            "hybrid_rrf",
        ):
            raise ValueError("候选检索器必须按 Vector/BM25/Hybrid-RRF 完整覆盖")
        vector, bm25, hybrid = self.candidate_retrievers
        if (
            not vector.production_enabled
            or vector.role != "production_default"
            or bm25.production_enabled
            or bm25.role != "hybrid_component"
            or hybrid.production_enabled
            or hybrid.role != "candidate_experiment"
        ):
            raise ValueError("候选检索器角色必须与冻结的 Vector 默认一致")
        if hybrid.answer_quality_evaluated:
            raise ValueError("Hybrid Top-5 尚未完成回答质量评测")
        if tuple(item.category for item in self.failure_taxonomy) != FAILURE_CATEGORY_ORDER:
            raise ValueError("失败分类必须按固定顺序完整覆盖")
        case_ids = tuple(item.case_id for item in self.case_reviews)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("案例复核不能重复 case ID")
        required_focus = {
            "alignment_safeguards_pause_trigger",
            "mark25_b1_chapter_intelligence_explosion_start",
            "ilya25_b2_ssi_compute",
            "ilya25_b1_timestamp_second_student",
            "mark25_b3_productivity_and_customer_support",
        }
        if not required_focus.issubset(case_ids):
            raise ValueError("Day 17 重点案例覆盖不完整")
        required_roles = {
            "complete_answer_and_citations",
            "retrieval_success_answer_not_full_score",
            "multi_source_incomplete",
            "correct_unanswerable_refusal",
            "highest_risk_failure",
        }
        roles = {role for item in self.case_reviews for role in item.review_roles}
        if not required_roles.issubset(roles):
            raise ValueError("Day 16 成功、失败、多来源、拒答与危险案例覆盖不完整")
        if tuple(item.method for item in self.evidence_summary.retrieval_top5) != (
            "vector",
            "bm25",
            "hybrid_rrf",
        ):
            raise ValueError("80 题证据必须完整覆盖三种检索方法")
        if tuple(item.dimension for item in self.rag_1_vs_rag_2_comparison) != (
            "online_pipeline",
            "evaluation_scope",
            "retrieval_comparison",
            "answer_and_refusal_evidence",
            "failure_analysis",
            "configuration_freeze",
        ):
            raise ValueError("RAG 1.0/2.0 对照维度不完整")
        return self
