"""Validated Day 10 RAG 1.0 run and regression-suite contracts."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.rag import GroundedAnswer


RAG_STAGE_NAMES = frozenset(
    {"ingestion", "chunking", "index", "retrieval", "answer"}
)
REGRESSION_CATEGORIES = frozenset(
    {
        "direct_fact_or_view",
        "multi_chunk",
        "speaker_or_time_provenance",
        "imperfect_ranking",
        "unanswerable",
    }
)


class RagTokenUsage(BaseModel):
    """Provider-reported token counts captured from the actual answer call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("total_tokens 必须等于 prompt_tokens + completion_tokens")
        return self


class RagRunMetrics(BaseModel):
    """Observable counts and timings from one Day 10 pipeline invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage_elapsed_seconds: dict[
        str,
        Annotated[float, Field(ge=0)],
    ]
    total_elapsed_seconds: float = Field(ge=0)
    query_embedding_calls: int = Field(ge=0)
    document_embedding_calls: int = Field(ge=0)
    document_embedding_batches: int | None = Field(default=None, ge=0)
    answer_model_calls: int = Field(ge=0)
    answer_token_usage: RagTokenUsage | None = None

    @model_validator(mode="after")
    def validate_stage_names(self) -> Self:
        if set(self.stage_elapsed_seconds) != RAG_STAGE_NAMES:
            missing = sorted(RAG_STAGE_NAMES - set(self.stage_elapsed_seconds))
            extra = sorted(set(self.stage_elapsed_seconds) - RAG_STAGE_NAMES)
            raise ValueError(
                f"阶段耗时字段不完整；missing={missing}, extra={extra}"
            )
        return self


class RagRunResult(BaseModel):
    """Safe, serializable output from one complete Day 10 RAG run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: str = Field(pattern=r"^ep_[0-9a-f]{16}$")
    source_file: str = Field(min_length=1)
    input_file: str = Field(min_length=1)
    input_kind: Literal["raw", "processed"]
    input_segment_count: int = Field(ge=1)
    chunk_count: int = Field(ge=1)
    index_directory: str = Field(min_length=1)
    index_action: Literal["reused", "built"]
    embedding_model: str = Field(min_length=1)
    answer_model: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=2_000)
    vector_count: int = Field(ge=1)
    vector_dimension: int = Field(ge=1)
    top_k: int = Field(ge=1)
    retrieved_count: int = Field(ge=1)
    retrieved_chunk_ids: list[str] = Field(min_length=1)
    answer: GroundedAnswer
    metrics: RagRunMetrics

    @field_validator(
        "source_file",
        "input_file",
        "index_directory",
        "embedding_model",
        "answer_model",
        "question",
    )
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("字段不能为空")
        return cleaned

    @field_validator("retrieved_chunk_ids")
    @classmethod
    def validate_retrieved_chunk_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("retrieved_chunk_ids 不能重复")
        for chunk_id in value:
            if not _is_chunk_id(chunk_id):
                raise ValueError(f"非法 Chunk ID：{chunk_id}")
        return value

    @model_validator(mode="after")
    def validate_counts_and_citations(self) -> Self:
        if self.vector_count != self.chunk_count:
            raise ValueError("vector_count 必须等于 chunk_count")
        if self.retrieved_count != len(self.retrieved_chunk_ids):
            raise ValueError("retrieved_count 必须等于 retrieved_chunk_ids 数量")
        if self.vector_count < self.retrieved_count:
            raise ValueError("retrieved_count 不能超过 vector_count")
        if self.retrieved_count > self.top_k:
            raise ValueError("retrieved_count 不能超过 top_k")
        if self.metrics.query_embedding_calls != 1:
            raise ValueError("成功运行必须恰好调用一次 query Embedding")
        if self.metrics.answer_model_calls != 1:
            raise ValueError("成功运行必须恰好调用一次回答模型")
        expected_document_calls = 0 if self.index_action == "reused" else 1
        if self.metrics.document_embedding_calls != expected_document_calls:
            raise ValueError("文档 Embedding 调用次数与索引动作不一致")
        retrieved = set(self.retrieved_chunk_ids)
        for citation in self.answer.citations:
            if citation.chunk_id not in retrieved:
                raise ValueError("最终引用必须来自本次检索候选")
            if citation.episode_id != self.episode_id:
                raise ValueError("最终引用必须属于本次节目")
            if citation.source_file != self.source_file:
                raise ValueError("最终引用 source_file 与本次节目不一致")
            expected_rank = self.retrieved_chunk_ids.index(citation.chunk_id) + 1
            if citation.rank != expected_rank:
                raise ValueError("引用 rank 与本次候选顺序不一致")
            if citation.evidence_id != f"E{expected_rank}":
                raise ValueError("引用 evidence_id 与本次候选顺序不一致")
        return self


class Day10RegressionCase(BaseModel):
    """One stable smoke-regression question, without generated answer text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    category: Literal[
        "direct_fact_or_view",
        "multi_chunk",
        "speaker_or_time_provenance",
        "imperfect_ranking",
        "unanswerable",
    ]
    question: str = Field(min_length=1, max_length=500)
    top_k: int = Field(ge=1)
    expected_has_sufficient_evidence: bool
    expected_chunk_ids: list[str]
    notes: str = Field(min_length=1, max_length=500)

    @field_validator("question", "notes")
    @classmethod
    def strip_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("字段不能为空")
        return cleaned

    @field_validator("expected_chunk_ids")
    @classmethod
    def validate_expected_chunk_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("expected_chunk_ids 不能重复")
        for chunk_id in value:
            if not _is_chunk_id(chunk_id):
                raise ValueError(
                    "expected_chunk_ids 只能保存稳定 chunk_<16 hex>，"
                    f"不能保存临时证据编号：{chunk_id}"
                )
        return value

    @model_validator(mode="after")
    def validate_expected_evidence_direction(self) -> Self:
        if len(self.expected_chunk_ids) > self.top_k:
            raise ValueError("expected_chunk_ids 数量不能超过 top_k")
        if self.expected_has_sufficient_evidence and not self.expected_chunk_ids:
            raise ValueError("可回答案例必须至少保存一个预期 Chunk ID")
        if not self.expected_has_sufficient_evidence and self.expected_chunk_ids:
            raise ValueError("无答案案例的 expected_chunk_ids 必须为空")
        if (
            self.category == "unanswerable"
            and self.expected_has_sufficient_evidence
        ):
            raise ValueError("unanswerable 案例必须期望拒答")
        if (
            self.category != "unanswerable"
            and not self.expected_has_sufficient_evidence
        ):
            raise ValueError("可回答类别不能期望拒答")
        return self


class Day10RegressionSuite(BaseModel):
    """Exactly five Day 10 smoke cases, one for each required category."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    episode_id: str = Field(pattern=r"^ep_[0-9a-f]{16}$")
    cases: list[Day10RegressionCase] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def validate_suite_coverage(self) -> Self:
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("case_id 必须唯一")
        categories = {case.category for case in self.cases}
        if categories != REGRESSION_CATEGORIES:
            missing = sorted(REGRESSION_CATEGORIES - categories)
            extra = sorted(categories - REGRESSION_CATEGORIES)
            raise ValueError(
                f"五题类别不完整；missing={missing}, extra={extra}"
            )
        return self


def _is_chunk_id(value: str) -> bool:
    if not isinstance(value, str) or len(value) != len("chunk_") + 16:
        return False
    if not value.startswith("chunk_"):
        return False
    return all(character in "0123456789abcdef" for character in value[6:])
