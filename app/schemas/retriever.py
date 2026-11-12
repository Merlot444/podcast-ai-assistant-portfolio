"""Backend-agnostic, metadata-only retrieval contract for evaluation."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RetrievedCandidate(BaseModel):
    """The smallest ranked candidate needed by retrieval metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=1, strict=True)
    chunk_id: str = Field(pattern=r"^chunk_[0-9a-f]{16}$")
    source_segment_ids: tuple[str, ...] = Field(min_length=1)
    score: float
    score_semantics: str = Field(min_length=1, max_length=64)
    retriever_name: str = Field(min_length=1, max_length=100)

    @field_validator("source_segment_ids")
    @classmethod
    def require_unique_source_segments(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("候选的 source_segment_ids 不能重复")
        return value

    @field_validator("score")
    @classmethod
    def require_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("检索分数必须是有限数值")
        return value

    @field_validator("score_semantics", "retriever_name")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("字段不能为空")
        if cleaned != value:
            raise ValueError("字段不能包含首尾空白")
        return value
