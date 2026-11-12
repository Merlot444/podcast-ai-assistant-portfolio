"""Validated, citation-ready schemas for Day 6 retrieval chunks."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ChunkSourceSpan(BaseModel):
    """An exact half-open character range in one Day 3 source segment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    segment_id: str = Field(pattern=r"^ep_[0-9a-f]{16}:\d{6}$")
    segment_index: int = Field(ge=0)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_character_range(self) -> Self:
        if self.end_char <= self.start_char:
            raise ValueError("end_char 必须大于 start_char")
        expected_suffix = f"{self.segment_index:06d}"
        if self.segment_id.rsplit(":", maxsplit=1)[-1] != expected_suffix:
            raise ValueError("segment_id 与 segment_index 不一致")
        return self


class TranscriptChunk(BaseModel):
    """One retrieval unit with deterministic source and time metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(pattern=r"^chunk_[0-9a-f]{16}$")
    chunk_index: int = Field(ge=0)
    episode_id: str = Field(pattern=r"^ep_[0-9a-f]{16}$")
    source_file: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_segment_ids: list[str] = Field(min_length=1)
    source_segment_start_index: int = Field(ge=0)
    source_segment_end_index: int = Field(ge=0)
    source_spans: list[ChunkSourceSpan] = Field(min_length=1)
    speakers: list[str]
    section_titles: list[str]
    start_time: float | None = Field(default=None, ge=0)
    end_time: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Latest known source coverage time. When an original segment has "
            "no end_time, its start_time is the last known coverage point."
        ),
    )
    time_range_complete: bool = Field(
        description=(
            "True only when every contributing source segment has both a "
            "start_time and an end_time."
        )
    )

    @field_validator("source_file")
    @classmethod
    def strip_source_file(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("source_file 不能为空")
        return cleaned

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text 不能为空")
        return value

    @field_validator("source_segment_ids")
    @classmethod
    def clean_source_ids(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("source_segment_ids 不能包含空值")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("source_segment_ids 不能重复")
        return cleaned

    @field_validator("speakers", "section_titles")
    @classmethod
    def clean_optional_metadata_lists(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(item.strip() for item in value if item.strip()))

    @model_validator(mode="after")
    def validate_traceability(self) -> Self:
        span_ids = list(dict.fromkeys(span.segment_id for span in self.source_spans))
        if self.source_segment_ids != span_ids:
            raise ValueError("source_segment_ids 必须与 source_spans 顺序一致")

        span_indices = [span.segment_index for span in self.source_spans]
        if span_indices != sorted(span_indices):
            raise ValueError("source_spans 必须按 segment_index 排序")
        if self.source_segment_start_index != span_indices[0]:
            raise ValueError("source_segment_start_index 与 source_spans 不一致")
        if self.source_segment_end_index != span_indices[-1]:
            raise ValueError("source_segment_end_index 与 source_spans 不一致")

        expected_prefix = f"{self.episode_id}:"
        if any(
            not span.segment_id.startswith(expected_prefix)
            for span in self.source_spans
        ):
            raise ValueError("source_spans 中的 segment_id 不属于当前 episode_id")

        if (
            self.start_time is not None
            and self.end_time is not None
            and self.end_time < self.start_time
        ):
            raise ValueError("end_time 不能早于 start_time")
        if self.time_range_complete and (
            self.start_time is None or self.end_time is None
        ):
            raise ValueError("完整时间范围必须同时包含 start_time 和 end_time")
        return self


class ChunkedTranscript(BaseModel):
    """One deterministic Day 6 chunking result and its reproducibility inputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    algorithm: Literal["fixed_character_windows_v1"]
    episode_id: str = Field(pattern=r"^ep_[0-9a-f]{16}$")
    source_file: str = Field(min_length=1)
    input_segment_count: int = Field(ge=1)
    chunk_size: int = Field(gt=0)
    overlap: int = Field(ge=0)
    chunks: list[TranscriptChunk] = Field(min_length=1)

    @field_validator("source_file")
    @classmethod
    def strip_source_file(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("source_file 不能为空")
        return cleaned

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap 必须小于 chunk_size")

        if [chunk.chunk_index for chunk in self.chunks] != list(
            range(len(self.chunks))
        ):
            raise ValueError("chunk_index 必须从 0 开始连续递增")
        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("chunk_id 不能重复")

        for chunk in self.chunks:
            if chunk.episode_id != self.episode_id:
                raise ValueError("Chunk 的 episode_id 与文件级 episode_id 不一致")
            if chunk.source_file != self.source_file:
                raise ValueError("Chunk 的 source_file 与文件级 source_file 不一致")
            if len(chunk.text) > self.chunk_size:
                raise ValueError("Chunk 文本不能超过 chunk_size")
        return self
