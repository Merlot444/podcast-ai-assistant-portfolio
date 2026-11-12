"""Validated, citation-ready transcript data models."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TranscriptSegment(BaseModel):
    """One cleaned transcript segment with the metadata needed for citation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    segment_id: str = Field(
        min_length=1,
        pattern=r"^ep_[0-9a-f]{16}:\d{6}$",
    )
    segment_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    speaker: str | None = None
    section_title: str | None = None
    start_time: float | None = Field(default=None, ge=0)
    end_time: float | None = Field(default=None, ge=0)

    @field_validator("text", "source_file")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        """Trim required strings and reject values that become empty."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("字段不能为空")
        return cleaned

    @field_validator("speaker", "section_title")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        """Represent missing or whitespace-only optional text consistently."""
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @model_validator(mode="after")
    def validate_time_range(self) -> Self:
        """Reject an end timestamp that occurs before its start timestamp."""
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.end_time < self.start_time
        ):
            raise ValueError("end_time 不能早于 start_time")
        return self


class ProcessedTranscript(BaseModel):
    """A deterministic, unified representation of one source transcript."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: str = Field(
        min_length=1,
        pattern=r"^ep_[0-9a-f]{16}$",
    )
    source_file: str = Field(min_length=1)
    source_format: Literal["txt", "json"]
    segments: list[TranscriptSegment] = Field(min_length=1)

    @field_validator("source_file")
    @classmethod
    def strip_source_file(cls, value: str) -> str:
        """Normalize the source filename stored at document level."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("source_file 不能为空")
        return cleaned
