"""Validated, metadata-rich results returned by the Day 8 retriever."""

from __future__ import annotations

import math
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.chunk import ChunkSourceSpan, TranscriptChunk


class RetrievalResult(BaseModel):
    """One ranked Chunk with the complete text and its source provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=1)
    score: float
    row_index: int = Field(ge=0)
    chunk_id: str = Field(pattern=r"^chunk_[0-9a-f]{16}$")
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
    end_time: float | None = Field(default=None, ge=0)
    time_range_complete: bool

    @field_validator("score")
    @classmethod
    def require_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("检索分数必须是有限数值")
        return value


    @classmethod
    def from_chunk(
        cls,
        *,
        rank: int,
        score: float,
        row_index: int,
        chunk: TranscriptChunk,
    ) -> Self:
        return cls(
            rank=rank,
            score=score,
            row_index=row_index,
            chunk_id=chunk.chunk_id,
            episode_id=chunk.episode_id,
            source_file=chunk.source_file,
            text=chunk.text,
            source_segment_ids=chunk.source_segment_ids,
            source_segment_start_index=chunk.source_segment_start_index,
            source_segment_end_index=chunk.source_segment_end_index,
            source_spans=chunk.source_spans,
            speakers=chunk.speakers,
            section_titles=chunk.section_titles,
            start_time=chunk.start_time,
            end_time=chunk.end_time,
            time_range_complete=chunk.time_range_complete,
        )
