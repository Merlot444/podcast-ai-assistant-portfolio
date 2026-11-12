"""Validated structures for the Day 4 content-analysis baseline."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)


class SectionSummaryOutput(BaseModel):
    """Model output for one bounded analysis batch."""

    model_config = ConfigDict(extra="forbid")

    summary: Annotated[str, Field(min_length=1, max_length=300)]
    key_points: list[str] = Field(min_length=2, max_length=5)

    @field_validator("summary")
    @classmethod
    def strip_summary(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("summary 不能为空")
        return cleaned

    @field_validator("key_points")
    @classmethod
    def clean_key_points(cls, value: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(item.strip() for item in value if item.strip()))
        if not 2 <= len(cleaned) <= 5:
            raise ValueError("key_points 数量必须在 2 到 5 之间")
        return cleaned


class GeneratedChapterOverview(BaseModel):
    """Chapter text returned by the model before source metadata is attached."""

    model_config = ConfigDict(extra="forbid")

    section_title: Annotated[str, Field(min_length=1)]
    overview: Annotated[str, Field(min_length=1, max_length=300)]

    @field_validator("section_title", "overview")
    @classmethod
    def strip_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("字段不能为空")
        return cleaned


class GeneratedAnalysisOutput(BaseModel):
    """Direct or reduce-stage model output."""

    model_config = ConfigDict(extra="forbid")

    summary: Annotated[str, Field(min_length=1, max_length=500)]
    keywords: list[str] = Field(min_length=3, max_length=8)
    chapter_overviews: list[GeneratedChapterOverview] = Field(min_length=1)
    recommendation: Annotated[str, Field(min_length=1, max_length=180)]

    @field_validator("summary", "recommendation")
    @classmethod
    def strip_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("字段不能为空")
        return cleaned

    @field_validator("keywords")
    @classmethod
    def clean_keywords(cls, value: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(item.strip() for item in value if item.strip()))
        if not 3 <= len(cleaned) <= 8:
            raise ValueError("keywords 数量必须在 3 到 8 之间")
        return cleaned

    @model_validator(mode="after")
    def chapter_titles_are_unique(self) -> Self:
        titles = [item.section_title for item in self.chapter_overviews]
        if len(titles) != len(set(titles)):
            raise ValueError("chapter_overviews 中的 section_title 不能重复")
        return self


class ChapterOverview(BaseModel):
    """One generated chapter overview with deterministic source metadata."""

    model_config = ConfigDict(extra="forbid")

    section_title: Annotated[str, Field(min_length=1)]
    overview: Annotated[str, Field(min_length=1, max_length=300)]
    source_segment_ids: list[str] = Field(min_length=1)
    speakers: list[str]
    start_time: float | None = Field(
        default=None,
        ge=0,
        description="Earliest source segment start time.",
    )
    end_time: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Latest known coverage timestamp; when source end times are absent, "
            "this is the last source segment start time, not an exact speech end."
        ),
    )

    @field_validator("section_title", "overview")
    @classmethod
    def strip_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("字段不能为空")
        return cleaned

    @field_validator("source_segment_ids", "speakers")
    @classmethod
    def clean_lists(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(item.strip() for item in value if item.strip()))

    @model_validator(mode="after")
    def validate_time_range(self) -> Self:
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.end_time < self.start_time
        ):
            raise ValueError("end_time 不能早于 start_time")
        return self


class BatchSummary(BaseModel):
    """One saved map-stage summary with its source coverage."""

    model_config = ConfigDict(extra="forbid")

    section_title: Annotated[str, Field(min_length=1)]
    batch_index: int = Field(ge=1)
    batch_count: int = Field(ge=1)
    summary: Annotated[str, Field(min_length=1, max_length=300)]
    key_points: list[str] = Field(min_length=2, max_length=5)
    source_segment_ids: list[str] = Field(min_length=1)
    speakers: list[str]
    start_time: float | None = Field(
        default=None,
        ge=0,
        description="Earliest source segment start time.",
    )
    end_time: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Latest known coverage timestamp; when source end times are absent, "
            "this is the last source segment start time, not an exact speech end."
        ),
    )

    @field_validator("section_title", "summary")
    @classmethod
    def strip_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("字段不能为空")
        return cleaned

    @field_validator("key_points", "source_segment_ids", "speakers")
    @classmethod
    def clean_lists(cls, value: list[str], info: ValidationInfo) -> list[str]:
        cleaned = list(dict.fromkeys(item.strip() for item in value if item.strip()))
        if info.field_name == "key_points" and not 2 <= len(cleaned) <= 5:
            raise ValueError("key_points 数量必须在 2 到 5 之间")
        return cleaned

    @model_validator(mode="after")
    def validate_batch_and_time_ranges(self) -> Self:
        if self.batch_index > self.batch_count:
            raise ValueError("batch_index 不能大于 batch_count")
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.end_time < self.start_time
        ):
            raise ValueError("end_time 不能早于 start_time")
        return self


class ModelCallMetric(BaseModel):
    """Observable facts for one real or mocked model call."""

    model_config = ConfigDict(extra="forbid")

    call_index: int = Field(ge=1)
    stage: Literal["direct", "segment_summary", "synthesis"]
    model: Annotated[str, Field(min_length=1)]
    section_title: str | None = None
    batch_index: int | None = Field(default=None, ge=1)
    prompt_characters: int = Field(ge=0)
    output_characters: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)


class AnalysisMetrics(BaseModel):
    """Required Day 4 measurements for one complete analysis run."""

    model_config = ConfigDict(extra="forbid")

    input_characters: int = Field(ge=0)
    segment_count: int = Field(ge=1)
    batch_count: int = Field(ge=1)
    model_call_count: int = Field(ge=1)
    output_characters: int = Field(ge=0)
    calls: list[ModelCallMetric] = Field(min_length=1)
    total_duration_seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def counts_match_calls(self) -> Self:
        if self.model_call_count != len(self.calls):
            raise ValueError("model_call_count 必须等于 calls 数量")
        if self.output_characters != sum(
            call.output_characters for call in self.calls
        ):
            raise ValueError("output_characters 必须等于各调用输出字符数之和")
        return self


class Day4AnalysisResult(BaseModel):
    """Saved result for either direct analysis or map-reduce summarization."""

    model_config = ConfigDict(extra="forbid")

    episode_id: Annotated[str, Field(pattern=r"^ep_[0-9a-f]{16}$")]
    source_file: Annotated[str, Field(min_length=1)]
    mode: Literal["direct", "map_reduce"]
    summary: Annotated[str, Field(min_length=1, max_length=500)]
    keywords: list[str] = Field(min_length=3, max_length=8)
    batch_summaries: list[BatchSummary]
    chapter_overviews: list[ChapterOverview] = Field(min_length=1)
    recommendation: Annotated[str, Field(min_length=1, max_length=180)]
    metrics: AnalysisMetrics
