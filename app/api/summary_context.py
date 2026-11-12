"""Bounded full-episode evidence, without provider calls or source paths."""

from typing import Literal

from pydantic import Field

from app.api.adapters import EvaluationRetrieveService
from app.api.schemas import ApiSchema, RetrievedChunkResponse
from app.tooling.schemas import EPISODE_ID_PATTERN


class SummaryContextRequest(ApiSchema):
    episode_id: str = Field(pattern=EPISODE_ID_PATTERN)


class SummaryContextResponse(ApiSchema):
    ok: Literal[True] = True
    episode_id: str = Field(pattern=EPISODE_ID_PATTERN)
    result_count: int = Field(ge=1, le=256)
    retrieval_call_count: Literal[0] = 0
    duration_ms: float = Field(default=0.0, ge=0)
    input_chunk_sha256: str
    coverage: Literal["full_episode"] = "full_episode"
    chunks: tuple[RetrievedChunkResponse, ...] = Field(min_length=1, max_length=256)


def build_summary_context(service: EvaluationRetrieveService) -> SummaryContextResponse:
    chunks = sorted(service.chunks, key=lambda chunk: chunk.chunk_index)
    # Fail closed rather than silently truncating the episode.
    if len(chunks) > 256 or sum(len(chunk.text) for chunk in chunks) > 200_000:
        raise ValueError("Episode exceeds summary context limit")
    return SummaryContextResponse(
        episode_id=service.episode_id,
        result_count=len(chunks),
        input_chunk_sha256=service.input_chunk_sha256,
        chunks=tuple(
            RetrievedChunkResponse(
                rank=rank,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source_segment_ids=tuple(chunk.source_segment_ids),
                score=0.0,
                score_semantics="not_applicable_full_episode",
                retriever_name="full_episode_catalog",
                speakers=tuple(chunk.speakers),
                section_titles=tuple(chunk.section_titles),
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                time_range_complete=chunk.time_range_complete,
            )
            for rank, chunk in enumerate(chunks, 1)
        ),
    )
