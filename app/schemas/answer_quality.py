"""Strict schemas for the frozen Day 16 answer-quality evaluation."""

from __future__ import annotations

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

from app.schemas.chunk import ChunkSourceSpan
from app.schemas.evaluation import EvaluationCategory
from app.schemas.rag import GroundedCitation
from app.schemas.retrieval_experiment import Day14CorpusIdentity


ReviewerMethod = Literal["主执行代理单审阅者语义复核"]
FailureCategory = Literal[
    "retrieval_missing",
    "ranking_outside_top5",
    "context_incomplete",
    "context_overlap_noise",
    "question_ambiguous",
    "generation_incorrect",
    "generation_unsupported",
    "false_refusal",
    "citation_wrong",
    "citation_incomplete",
    "provider_or_contract_failure",
]

FAILURE_CATEGORIES: tuple[FailureCategory, ...] = (
    "retrieval_missing",
    "ranking_outside_top5",
    "context_incomplete",
    "context_overlap_noise",
    "question_ambiguous",
    "generation_incorrect",
    "generation_unsupported",
    "false_refusal",
    "citation_wrong",
    "citation_incomplete",
    "provider_or_contract_failure",
)


class Day16InputArtifactHashes(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transcript_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunk_catalog_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_chunk_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    faiss_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    day12_baseline_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    day13_experiment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    day14_experiment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    day15_experiment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    grounded_prompt_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    grounded_system_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Day16GitState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    head_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    branch: str = Field(min_length=1, max_length=200)
    working_tree_dirty: StrictBool
    protected_untracked_files: tuple[
        Literal["AGENTS.md", "docs/day12-retrieval-boundary-draft.md"], ...
    ]


class AnswerModelConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["OpenAI-compatible DeepSeek service"]
    model_name: Literal["deepseek-v4-pro"]
    model_revision: None
    response_model_name: str | None = Field(default=None, min_length=1)
    response_format: Literal["json_object"]
    output_schema: Literal[
        "answer,citations.evidence_id,has_sufficient_evidence"
    ]
    output_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    temperature: Literal[0.2]
    thinking: Literal["disabled"]
    max_tokens: Literal[1200]
    timeout_seconds: Literal[30.0]
    maximum_retries: Literal[0]
    evidence_depth: Literal[5]
    batching: Literal["one_question_per_request"]
    concurrency: Literal["sequential"]


class VectorTop5Identity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_artifact: Literal[
        "data/evaluation/eval_dataset_v1-retrieval-experiment-day14.json"
    ]
    source_experiment_id: str = Field(min_length=1)
    source_retriever: Literal["vector"]
    source_candidate_depth: Literal[20]
    evidence_depth: Literal[5]
    case_count: Literal[20]
    ordered_top5_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AnswerEvidenceMetadata(BaseModel):
    """One local, application-owned evidence record without Chunk text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(pattern=r"^E[1-5]$")
    rank: StrictInt = Field(ge=1, le=5)
    score: float
    row_index: StrictInt = Field(ge=0)
    chunk_id: str = Field(pattern=r"^chunk_[0-9a-f]{16}$")
    episode_id: str = Field(pattern=r"^ep_[0-9a-f]{16}$")
    source_file: str = Field(min_length=1)
    speakers: tuple[str, ...]
    section_titles: tuple[str, ...]
    start_time: float | None = Field(default=None, ge=0)
    end_time: float | None = Field(default=None, ge=0)
    time_range_complete: StrictBool
    source_segment_ids: tuple[str, ...] = Field(min_length=1)
    source_segment_start_index: StrictInt = Field(ge=0)
    source_segment_end_index: StrictInt = Field(ge=0)
    source_spans: tuple[ChunkSourceSpan, ...] = Field(min_length=1)


class AnswerTokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: StrictInt | None = Field(default=None, ge=0)
    output_tokens: StrictInt | None = Field(default=None, ge=0)
    total_tokens: StrictInt | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_totals(self) -> Self:
        values = (self.input_tokens, self.output_tokens, self.total_tokens)
        if any(value is None for value in values) and not all(
            value is None for value in values
        ):
            raise ValueError("Token usage 必须全部存在或全部未知")
        if (
            self.total_tokens is not None
            and self.total_tokens != self.input_tokens + self.output_tokens
        ):
            raise ValueError("total_tokens 必须等于输入与输出 Token 之和")
        return self


class AnswerCallTelemetry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    elapsed_seconds: float = Field(ge=0)
    question_characters: StrictInt = Field(ge=1)
    evidence_characters: StrictInt = Field(ge=1)
    token_usage: AnswerTokenUsage


class GeneratedAnswerCase(BaseModel):
    """One answer frozen before reference answers and truth are used for review."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    evidence: tuple[AnswerEvidenceMetadata, ...] = Field(min_length=5, max_length=5)
    answer: str = Field(min_length=1)
    citations: tuple[GroundedCitation, ...]
    has_sufficient_evidence: StrictBool
    telemetry: AnswerCallTelemetry

    @field_validator("answer")
    @classmethod
    def reject_blank_answer(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer 不能为空")
        return value

    @model_validator(mode="after")
    def validate_evidence_and_citations(self) -> Self:
        if tuple(item.evidence_id for item in self.evidence) != (
            "E1",
            "E2",
            "E3",
            "E4",
            "E5",
        ):
            raise ValueError("Day 16 证据必须严格按 E1..E5 保存")
        if tuple(item.rank for item in self.evidence) != (1, 2, 3, 4, 5):
            raise ValueError("Day 16 证据必须是原始 Vector rank 1..5")
        if len({item.chunk_id for item in self.evidence}) != 5:
            raise ValueError("Day 16 Top-5 不能包含重复 Chunk")
        evidence_by_id = {item.evidence_id: item for item in self.evidence}
        if len({item.evidence_id for item in self.citations}) != len(self.citations):
            raise ValueError("最终引用不能重复 evidence_id")
        for citation in self.citations:
            evidence = evidence_by_id.get(citation.evidence_id)
            if evidence is None:
                raise ValueError("最终引用必须来自本次 Vector Top-5")
            if (
                citation.chunk_id != evidence.chunk_id
                or citation.rank != evidence.rank
                or citation.row_index != evidence.row_index
                or tuple(citation.source_segment_ids) != evidence.source_segment_ids
            ):
                raise ValueError("引用元数据必须与应用证据映射完全一致")
        if self.has_sufficient_evidence != bool(self.citations):
            raise ValueError("充分性与最终引用状态不一致")
        return self


class Day16CallCounts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    answer_model_calls: StrictInt = Field(ge=0)
    successful_answer_model_calls: StrictInt = Field(ge=0)
    failed_answer_model_calls: StrictInt = Field(ge=0)
    vector_query_embedding_calls: Literal[0]
    document_embedding_calls: Literal[0]
    vector_retrieval_calls: Literal[0]
    bm25_retrieval_calls: Literal[0]
    rrf_fusion_calls: Literal[0]
    rerank_calls: Literal[0]
    judge_model_calls: Literal[0]


class Day16LatencyAndCost(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_count: StrictInt = Field(ge=1)
    total_seconds: float = Field(ge=0)
    median_seconds: float = Field(ge=0)
    p95_seconds: float = Field(ge=0)
    maximum_seconds: float = Field(ge=0)
    input_tokens: StrictInt | None = Field(default=None, ge=0)
    output_tokens: StrictInt | None = Field(default=None, ge=0)
    total_tokens: StrictInt | None = Field(default=None, ge=0)
    question_characters: StrictInt = Field(ge=1)
    evidence_characters: StrictInt = Field(ge=1)
    provider_billing_unit: Literal["input_and_output_tokens"]
    confirmed_monetary_cost_usd: None
    monetary_cost_status: Literal[
        "unknown_provider_did_not_return_currency_cost"
    ]


class AnswerGenerationArtifact(BaseModel):
    """Complete 20-answer freeze; intentionally contains no questions or truth."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    artifact_type: Literal["day16_answer_generation_freeze"]
    generation_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{2,127}$")
    created_at_utc: datetime
    dataset_id: Literal["eval_dataset_v1"]
    dataset_cases_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    episode_id: str = Field(pattern=r"^ep_[0-9a-f]{16}$")
    git_state: Day16GitState
    corpus_identity: Day14CorpusIdentity
    input_artifact_hashes: Day16InputArtifactHashes
    vector_top5_identity: VectorTop5Identity
    answer_model_configuration: AnswerModelConfiguration
    generation_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cases: tuple[GeneratedAnswerCase, ...] = Field(min_length=20, max_length=20)
    call_counts: Day16CallCounts
    latency_and_cost: Day16LatencyAndCost

    @model_validator(mode="after")
    def validate_complete_generation(self) -> Self:
        case_ids = tuple(item.case_id for item in self.cases)
        if len(set(case_ids)) != 20:
            raise ValueError("生成冻结必须包含 20 道唯一案例")
        calls = self.call_counts
        if (
            calls.answer_model_calls != 20
            or calls.successful_answer_model_calls != 20
            or calls.failed_answer_model_calls != 0
        ):
            raise ValueError("正式生成必须恰好完成 20 次成功回答调用")
        if self.latency_and_cost.request_count != 20:
            raise ValueError("延迟统计必须覆盖全部 20 次回答调用")
        return self


class ManualDimensionScore(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    score: StrictInt = Field(ge=0, le=2)
    reason: str = Field(min_length=1, max_length=800)

    @field_validator("reason")
    @classmethod
    def strip_reason(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or cleaned != value:
            raise ValueError("评分理由必须非空且不能包含首尾空白")
        return value


class CitationSemanticReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(pattern=r"^E[1-5]$")
    chunk_id: str = Field(pattern=r"^chunk_[0-9a-f]{16}$")
    supports_core_claim: StrictBool
    truth_segment_alignment: Literal["match", "partial", "mismatch"]
    reason: str = Field(min_length=1, max_length=600)


class AnswerableManualReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    reviewer_method: ReviewerMethod
    correctness: ManualDimensionScore
    evidence_alignment: ManualDimensionScore
    citation_correctness: ManualDimensionScore
    completeness: ManualDimensionScore
    core_claim_count: StrictInt = Field(ge=0)
    supported_core_claim_count: StrictInt = Field(ge=0)
    all_core_claims_supported: StrictBool
    citation_reviews: tuple[CitationSemanticReview, ...]
    citation_segments_match_truth: StrictBool
    duplicate_chunk_displaced_distinct_source: StrictBool
    top5_context_sufficient: StrictBool
    primary_failure_category: FailureCategory | None
    failure_categories: tuple[FailureCategory, ...]
    overall_reason: str = Field(min_length=1, max_length=1200)

    @model_validator(mode="after")
    def validate_review(self) -> Self:
        if self.supported_core_claim_count > self.core_claim_count:
            raise ValueError("有支持的核心结论数不能大于核心结论总数")
        expected_all = (
            self.core_claim_count > 0
            and self.supported_core_claim_count == self.core_claim_count
        )
        if self.all_core_claims_supported != expected_all:
            raise ValueError("全部核心结论支持标记与计数不一致")
        evidence_ids = tuple(item.evidence_id for item in self.citation_reviews)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("引用语义复核不能重复 evidence_id")
        if len(self.failure_categories) != len(set(self.failure_categories)):
            raise ValueError("失败类别不能重复")
        if self.primary_failure_category is not None and (
            self.primary_failure_category not in self.failure_categories
        ):
            raise ValueError("主要失败类别必须包含在失败类别列表中")
        if self.primary_failure_category is None and self.failure_categories:
            raise ValueError("存在失败类别时必须指定主要失败类别")
        return self


class UnanswerableManualReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    reviewer_method: ReviewerMethod
    correct_refusal: StrictBool
    incorrect_answer: StrictBool
    used_fixed_refusal: StrictBool
    citations_empty: StrictBool
    invented_fact: StrictBool
    primary_failure_category: FailureCategory | None
    failure_categories: tuple[FailureCategory, ...]
    reason: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_review(self) -> Self:
        if self.correct_refusal == self.incorrect_answer:
            raise ValueError("无答案题必须且只能是正确拒答或错误作答之一")
        if self.correct_refusal and (
            not self.used_fixed_refusal or not self.citations_empty or self.invented_fact
        ):
            raise ValueError("正确拒答必须使用固定文本、空引用且不能编造事实")
        if len(self.failure_categories) != len(set(self.failure_categories)):
            raise ValueError("失败类别不能重复")
        if self.primary_failure_category is not None and (
            self.primary_failure_category not in self.failure_categories
        ):
            raise ValueError("主要失败类别必须包含在失败类别列表中")
        if self.primary_failure_category is None and self.failure_categories:
            raise ValueError("存在失败类别时必须指定主要失败类别")
        return self


class Day16ManualReviewArtifact(BaseModel):
    """Human-style review entered only after the generation freeze exists."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    artifact_type: Literal["day16_single_reviewer_manual_review"]
    generation_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rubric_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer_method: ReviewerMethod
    answerable_reviews: tuple[AnswerableManualReview, ...] = Field(
        min_length=16, max_length=16
    )
    unanswerable_reviews: tuple[UnanswerableManualReview, ...] = Field(
        min_length=4, max_length=4
    )

    @model_validator(mode="after")
    def validate_complete_review(self) -> Self:
        reviews = self.answerable_reviews + self.unanswerable_reviews
        case_ids = tuple(item.case_id for item in reviews)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("单审阅者评分必须覆盖 20 道唯一案例")
        if any(item.reviewer_method != self.reviewer_method for item in reviews):
            raise ValueError("每题 reviewer_method 必须与顶层一致")
        return self


class DimensionAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    score_sum: StrictInt = Field(ge=0)
    maximum_score: StrictInt = Field(ge=0)
    case_count: StrictInt = Field(ge=0)
    mean_score: float = Field(ge=0, le=2)


class ScoreAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    correctness: DimensionAggregate
    evidence_alignment: DimensionAggregate
    citation_correctness: DimensionAggregate
    completeness: DimensionAggregate


class AnswerabilityMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    correct_refusal_count: StrictInt = Field(ge=0, le=4)
    unanswerable_denominator: Literal[4]
    incorrect_answer_count: StrictInt = Field(ge=0, le=4)
    false_refusal_count: StrictInt = Field(ge=0, le=16)
    answerable_denominator: Literal[16]
    successful_answer_count: StrictInt = Field(ge=0, le=16)


class CitationMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mapping_legal_citation_count: StrictInt = Field(ge=0)
    total_citation_count: StrictInt = Field(ge=0)
    mapping_legal_rate: float | None = Field(default=None, ge=0, le=1)
    semantically_supported_citation_count: StrictInt = Field(ge=0)
    semantic_support_rate: float | None = Field(default=None, ge=0, le=1)
    truth_aligned_citation_count: StrictInt = Field(ge=0)
    supported_core_claim_count: StrictInt = Field(ge=0)
    total_core_claim_count: StrictInt = Field(ge=0)
    core_claim_support_rate: float | None = Field(default=None, ge=0, le=1)
    fully_supported_answer_count: StrictInt = Field(ge=0, le=16)
    answerable_denominator: Literal[16]
    duplicate_displacement_answer_count: StrictInt = Field(ge=0, le=16)
    citation_granularity_limit: Literal[
        "answer_level_citations_without_sentence_level_alignment"
    ]


class Day16Metrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    answerable_scores: ScoreAggregate
    category_scores: dict[EvaluationCategory, ScoreAggregate]
    multi_segment_scores: ScoreAggregate
    proper_noun_subset_case_ids: tuple[str, ...]
    proper_noun_subset_scores: ScoreAggregate
    answerability: AnswerabilityMetrics
    citations: CitationMetrics
    failure_category_counts: dict[FailureCategory, StrictInt]
    primary_failure_category_counts: dict[FailureCategory, StrictInt]


class AnswerQualityCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    category: EvaluationCategory
    answerable: StrictBool
    evidence: tuple[AnswerEvidenceMetadata, ...] = Field(min_length=5, max_length=5)
    answer: str
    citations: tuple[GroundedCitation, ...]
    has_sufficient_evidence: StrictBool
    telemetry: AnswerCallTelemetry
    answerable_review: AnswerableManualReview | None
    unanswerable_review: UnanswerableManualReview | None

    @model_validator(mode="after")
    def validate_review_kind(self) -> Self:
        if self.answerable:
            if self.answerable_review is None or self.unanswerable_review is not None:
                raise ValueError("可回答题必须且只能保存四项语义评分")
        elif self.unanswerable_review is None or self.answerable_review is not None:
            raise ValueError("无答案题必须且只能保存拒答复核")
        return self


class AnswerQualityExperimentResult(BaseModel):
    """Complete Day 16 result without questions, reference answers, text or prompts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    experiment_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{2,127}$")
    created_at_utc: datetime
    dataset_id: Literal["eval_dataset_v1"]
    dataset_cases_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    episode_id: str = Field(pattern=r"^ep_[0-9a-f]{16}$")
    reviewer_method: ReviewerMethod
    rubric_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    git_state: Day16GitState
    corpus_identity: Day14CorpusIdentity
    input_artifact_hashes: Day16InputArtifactHashes
    vector_top5_identity: VectorTop5Identity
    answer_model_configuration: AnswerModelConfiguration
    cases: tuple[AnswerQualityCaseResult, ...] = Field(min_length=20, max_length=20)
    metrics: Day16Metrics
    call_counts: Day16CallCounts
    latency_and_cost: Day16LatencyAndCost
    limitations: tuple[str, ...] = Field(min_length=5)
    future_evaluation_expansion_priorities: tuple[str, ...] = Field(min_length=3)
    day17_required_inputs: tuple[str, ...] = Field(min_length=4)

    @model_validator(mode="after")
    def validate_complete_result(self) -> Self:
        if len({item.case_id for item in self.cases}) != 20:
            raise ValueError("正式 Day 16 结果必须包含 20 道唯一案例")
        if sum(item.answerable for item in self.cases) != 16:
            raise ValueError("正式 Day 16 结果必须包含 16 道可回答题")
        return self
