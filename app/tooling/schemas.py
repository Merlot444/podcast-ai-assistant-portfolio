"""Strict input, output, error, and audit schemas for Day 21 tools."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


EPISODE_ID_PATTERN = r"^ep_[0-9a-f]{16}$"
SEGMENT_ID_PATTERN = r"^ep_[0-9a-f]{16}:\d{6}$"
CHUNK_ID_PATTERN = r"^chunk_[0-9a-f]{16}$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
CALL_ID_PATTERN = r"^call_[0-9a-f]{32}$"
SAFE_LABEL_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$"
SourceSegmentId = Annotated[str, Field(pattern=SEGMENT_ID_PATTERN)]

ToolName = Literal[
    "search_transcript",
    "get_episode_metadata",
    "get_segment_context",
]
RecordedToolName = Literal[
    "search_transcript",
    "get_episode_metadata",
    "get_segment_context",
    "unknown",
]
ToolErrorCode = Literal[
    "unknown_tool",
    "invalid_arguments",
    "not_found",
    "tool_failed",
]


class ToolSchema(BaseModel):
    """Project-wide strict base for data crossing the tool boundary."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
    )


class ToolArguments(ToolSchema):
    """Normalize string arguments before applying their field constraints."""

    @field_validator("*", mode="before")
    @classmethod
    def strip_string_arguments(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class SearchTranscriptArguments(ToolArguments):
    query: str = Field(min_length=1, max_length=500)
    episode_id: str = Field(pattern=EPISODE_ID_PATTERN)
    top_k: int = Field(ge=1, le=8)


class GetEpisodeMetadataArguments(ToolArguments):
    episode_id: str = Field(pattern=EPISODE_ID_PATTERN)


class GetSegmentContextArguments(ToolArguments):
    episode_id: str = Field(pattern=EPISODE_ID_PATTERN)
    segment_id: str = Field(pattern=SEGMENT_ID_PATTERN)
    context_before: int = Field(ge=0, le=2)
    context_after: int = Field(ge=0, le=2)

    @model_validator(mode="after")
    def require_matching_episode(self) -> Self:
        if not self.segment_id.startswith(f"{self.episode_id}:"):
            raise ValueError("segment_id 不属于 episode_id")
        return self


class SearchHit(ToolSchema):
    rank: int = Field(ge=1)
    chunk_id: str = Field(pattern=CHUNK_ID_PATTERN)
    episode_id: str = Field(pattern=EPISODE_ID_PATTERN)
    source_segment_ids: tuple[SourceSegmentId, ...] = Field(min_length=1)
    score: float
    score_semantics: str = Field(
        min_length=1,
        max_length=64,
        pattern=SAFE_LABEL_PATTERN,
    )
    retriever_name: str = Field(
        min_length=1,
        max_length=100,
        pattern=SAFE_LABEL_PATTERN,
    )

    @field_validator("source_segment_ids")
    @classmethod
    def require_unique_source_segments(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("source_segment_ids 不能重复")
        return value

    @field_validator("score")
    @classmethod
    def require_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("score 必须是有限数值")
        return value

    @model_validator(mode="after")
    def require_episode_scoped_segments(self) -> Self:
        expected_prefix = f"{self.episode_id}:"
        if any(
            not segment_id.startswith(expected_prefix)
            for segment_id in self.source_segment_ids
        ):
            raise ValueError("检索结果不能跨 episode")
        return self


class SearchTranscriptData(ToolSchema):
    episode_id: str = Field(pattern=EPISODE_ID_PATTERN)
    hits: tuple[SearchHit, ...] = Field(max_length=8)
    result_count: int = Field(ge=0, le=8)

    @model_validator(mode="after")
    def require_consistent_hits(self) -> Self:
        if self.result_count != len(self.hits):
            raise ValueError("result_count 与 hits 数量不一致")
        if [item.rank for item in self.hits] != list(range(1, len(self.hits) + 1)):
            raise ValueError("hits rank 必须从 1 连续递增")
        if any(item.episode_id != self.episode_id for item in self.hits):
            raise ValueError("hits 必须属于请求的 episode")
        return self


class EpisodeMetadataData(ToolSchema):
    episode_id: str = Field(pattern=EPISODE_ID_PATTERN)
    source_format: Literal["txt", "json"]
    segment_count: int = Field(ge=1)
    timed_segment_count: int = Field(ge=0)
    speaker_count: int = Field(ge=0)
    section_count: int = Field(ge=0)
    section_names: tuple[str, ...]

    @model_validator(mode="after")
    def require_consistent_counts(self) -> Self:
        if self.timed_segment_count > self.segment_count:
            raise ValueError("timed_segment_count 不能超过 segment_count")
        if self.speaker_count > self.segment_count:
            raise ValueError("speaker_count 不能超过 segment_count")
        if self.section_count > self.segment_count:
            raise ValueError("section_count 不能超过 segment_count")
        if self.section_count != len(self.section_names):
            raise ValueError("section_count 与 section_names 数量不一致")
        if len(self.section_names) != len(set(self.section_names)):
            raise ValueError("section_names 不能重复")
        return self


class SegmentContextItem(ToolSchema):
    segment_id: str = Field(pattern=SEGMENT_ID_PATTERN)
    segment_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    speaker: str | None = None
    section_title: str | None = None
    start_time: float | None = Field(default=None, ge=0)
    end_time: float | None = Field(default=None, ge=0)
    is_target: bool

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text 不能为空")
        return value


class SegmentContextData(ToolSchema):
    episode_id: str = Field(pattern=EPISODE_ID_PATTERN)
    target_segment_id: str = Field(pattern=SEGMENT_ID_PATTERN)
    context_before: int = Field(ge=0, le=2)
    context_after: int = Field(ge=0, le=2)
    segments: tuple[SegmentContextItem, ...] = Field(min_length=1, max_length=5)
    result_count: int = Field(ge=1, le=5)

    @model_validator(mode="after")
    def require_consistent_context(self) -> Self:
        if self.result_count != len(self.segments):
            raise ValueError("result_count 与 segments 数量不一致")
        if sum(item.is_target for item in self.segments) != 1:
            raise ValueError("上下文必须且只能包含一个目标片段")
        target = next(item for item in self.segments if item.is_target)
        if target.segment_id != self.target_segment_id:
            raise ValueError("目标片段标记与 target_segment_id 不一致")
        expected_prefix = f"{self.episode_id}:"
        if any(
            not item.segment_id.startswith(expected_prefix) for item in self.segments
        ):
            raise ValueError("上下文片段不能跨 episode")
        indices = [item.segment_index for item in self.segments]
        if indices != sorted(indices) or len(indices) != len(set(indices)):
            raise ValueError("上下文片段必须按唯一 segment_index 排序")
        return self


ToolData = SearchTranscriptData | EpisodeMetadataData | SegmentContextData


class ToolError(ToolSchema):
    code: ToolErrorCode
    message: str = Field(min_length=1, max_length=200)


class ToolSuccess(ToolSchema):
    ok: Literal[True] = True
    tool_name: ToolName
    call_id: str = Field(pattern=CALL_ID_PATTERN)
    data: ToolData

    @model_validator(mode="after")
    def require_matching_result_type(self) -> Self:
        expected_type = {
            "search_transcript": SearchTranscriptData,
            "get_episode_metadata": EpisodeMetadataData,
            "get_segment_context": SegmentContextData,
        }[self.tool_name]
        if not isinstance(self.data, expected_type):
            raise ValueError("tool_name 与 data 类型不一致")
        return self


class ToolFailure(ToolSchema):
    ok: Literal[False] = False
    tool_name: RecordedToolName
    call_id: str = Field(pattern=CALL_ID_PATTERN)
    error: ToolError


ToolDispatchResult = ToolSuccess | ToolFailure


class ToolCallRecord(ToolSchema):
    """One content-free audit summary for exactly one dispatch attempt."""

    call_id: str = Field(pattern=CALL_ID_PATTERN)
    tool_name: RecordedToolName
    requested_tool_sha256: str = Field(pattern=SHA256_PATTERN)
    started_at: datetime
    duration_ms: float = Field(ge=0)
    outcome: Literal["success", "failure"]
    error_code: ToolErrorCode | None = None
    result_count: int = Field(ge=0, le=8)
    episode_id: str | None = Field(default=None, pattern=EPISODE_ID_PATTERN)
    segment_id: str | None = Field(default=None, pattern=SEGMENT_ID_PATTERN)
    result_ids: tuple[str, ...] = Field(max_length=8)
    argument_sha256: str = Field(pattern=SHA256_PATTERN)
    result_sha256: str = Field(pattern=SHA256_PATTERN)
    query_length: int | None = Field(default=None, ge=0)
    query_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def require_consistent_outcome(self) -> Self:
        if self.started_at.tzinfo is None:
            raise ValueError("started_at 必须包含时区")
        if self.outcome == "success" and self.error_code is not None:
            raise ValueError("成功记录不能包含 error_code")
        if self.outcome == "failure" and self.error_code is None:
            raise ValueError("失败记录必须包含 error_code")
        return self
