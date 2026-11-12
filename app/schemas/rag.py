"""Validated model-facing and application-facing Day 9 answer schemas."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.chunk import ChunkSourceSpan


class ModelCitation(BaseModel):
    """One evidence selection returned by the model.

    The model may select only an application-assigned evidence ID. It is not
    trusted to return source metadata.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    evidence_id: str = Field(pattern=r"^E[1-9][0-9]*$")


class ModelGroundedAnswer(BaseModel):
    """The deliberately small JSON contract returned by the chat model."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    answer: str = Field(min_length=1)
    citations: list[ModelCitation]
    has_sufficient_evidence: bool

    @field_validator("answer")
    @classmethod
    def reject_blank_answer(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("answer 不能为空")
        return cleaned


class GroundedCitation(BaseModel):
    """One trusted citation mapped from a Day 8 RetrievalResult."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(pattern=r"^E[1-9][0-9]*$")
    rank: int = Field(ge=1)
    score: float
    row_index: int = Field(ge=0)
    chunk_id: str = Field(pattern=r"^chunk_[0-9a-f]{16}$")
    episode_id: str = Field(pattern=r"^ep_[0-9a-f]{16}$")
    source_file: str = Field(min_length=1)
    speakers: list[str]
    section_titles: list[str]
    start_time: float | None = Field(default=None, ge=0)
    end_time: float | None = Field(default=None, ge=0)
    time_range_complete: bool
    source_segment_ids: list[str] = Field(min_length=1)
    source_segment_start_index: int = Field(ge=0)
    source_segment_end_index: int = Field(ge=0)
    source_spans: list[ChunkSourceSpan] = Field(min_length=1)


class GroundedAnswer(BaseModel):
    """Final Day 9 business result with application-owned citations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: str = Field(min_length=1)
    citations: list[GroundedCitation]
    has_sufficient_evidence: bool

    @field_validator("answer")
    @classmethod
    def reject_blank_answer(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("answer 不能为空")
        return cleaned

    @model_validator(mode="after")
    def validate_sufficiency_invariants(self) -> Self:
        if self.has_sufficient_evidence and not self.citations:
            raise ValueError("证据充分的答案必须至少包含一个引用")
        if not self.has_sufficient_evidence and self.citations:
            raise ValueError("证据不足的答案不能包含引用")
        return self
