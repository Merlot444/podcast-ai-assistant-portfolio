"""Pydantic structure for podcast analysis output."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class SummaryOutput(BaseModel):
    """Validated output for the standalone summary Prompt."""

    model_config = ConfigDict(extra="forbid")

    summary: Annotated[str, Field(min_length=1, max_length=80)]

    @field_validator("summary")
    @classmethod
    def clean_summary(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("summary 不能为空")
        return cleaned


class TagOutput(BaseModel):
    """Validated output for Zero-shot and Few-shot tag Prompts."""

    model_config = ConfigDict(extra="forbid")

    tags: list[str]

    @field_validator("tags")
    @classmethod
    def clean_tags(cls, value: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(item.strip() for item in value if item.strip()))
        if not 3 <= len(cleaned) <= 5:
            raise ValueError("tags 数量必须在 3 到 5 之间")
        if any(len(item) > 8 for item in cleaned):
            raise ValueError("每个 tag 不能超过 8 个字符")
        return cleaned


class TitleIntroOutput(BaseModel):
    """Validated output for the standalone title and introduction Prompt."""

    model_config = ConfigDict(extra="forbid")

    title: Annotated[str, Field(min_length=1, max_length=24)]
    introduction: Annotated[str, Field(min_length=1, max_length=100)]

    @field_validator("title", "introduction")
    @classmethod
    def clean_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("字段不能为空")
        return cleaned


class PodcastAnalysis(BaseModel):
    """A small, validated analysis result for one transcript excerpt."""

    model_config = ConfigDict(extra="forbid")

    summary: Annotated[str, Field(min_length=1, max_length=120)]
    keywords: list[str]
    tags: list[str]

    @field_validator("summary")
    @classmethod
    def clean_summary(cls, value: str) -> str:
        """Reject summaries that become empty after trimming."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("summary 不能为空")
        return cleaned

    @field_validator("keywords", "tags")
    @classmethod
    def clean_labels(cls, value: list[str], info: ValidationInfo) -> list[str]:
        """Trim, deduplicate, and constrain keywords and tags."""
        cleaned = list(dict.fromkeys(item.strip() for item in value if item.strip()))
        field_name = info.field_name
        minimum, maximum = (3, 5) if field_name == "keywords" else (2, 4)
        if not minimum <= len(cleaned) <= maximum:
            raise ValueError(f"{field_name} 数量必须在 {minimum} 到 {maximum} 之间")
        return cleaned
