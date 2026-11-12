"""Read-only handlers over explicitly injected retrievers and transcripts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from app.retrieval.base import EvaluationRetriever
from app.schemas.transcript import ProcessedTranscript, TranscriptSegment
from app.tooling.schemas import (
    EPISODE_ID_PATTERN,
    EpisodeMetadataData,
    GetEpisodeMetadataArguments,
    GetSegmentContextArguments,
    SearchHit,
    SearchTranscriptArguments,
    SearchTranscriptData,
    SegmentContextData,
    SegmentContextItem,
)


class ToolNotFoundError(LookupError):
    """Signal an expected missing episode or segment without exposing internals."""


@dataclass(frozen=True)
class ToolDependencies:
    """Trusted, preloaded capabilities that model arguments cannot replace."""

    transcripts: Mapping[str, ProcessedTranscript]
    retrievers: Mapping[str, EvaluationRetriever] = field(default_factory=dict)

    def __post_init__(self) -> None:
        transcripts = dict(self.transcripts)
        retrievers = dict(self.retrievers)
        for episode_id, transcript in transcripts.items():
            if not re.fullmatch(EPISODE_ID_PATTERN, episode_id):
                raise ValueError("transcript catalog 包含非法 episode_id")
            if transcript.episode_id != episode_id:
                raise ValueError("transcript catalog key 与 episode_id 不一致")
            _validate_transcript_segments(transcript)
        if any(episode_id not in transcripts for episode_id in retrievers):
            raise ValueError("retriever 必须绑定到已加载的 transcript")
        object.__setattr__(self, "transcripts", MappingProxyType(transcripts))
        object.__setattr__(self, "retrievers", MappingProxyType(retrievers))


@dataclass(frozen=True)
class ToolHandlers:
    dependencies: ToolDependencies

    def search_transcript(
        self,
        arguments: SearchTranscriptArguments,
    ) -> SearchTranscriptData:
        transcript = self._transcript(arguments.episode_id)
        retriever = self.dependencies.retrievers.get(arguments.episode_id)
        if retriever is None:
            raise ToolNotFoundError("episode 没有已加载 retriever")

        candidates = retriever.retrieve(arguments.query, arguments.top_k)
        expected_prefix = f"{arguments.episode_id}:"
        known_segment_ids = {item.segment_id for item in transcript.segments}
        hits: list[SearchHit] = []
        seen_chunk_ids: set[str] = set()
        for candidate in candidates:
            source_ids = tuple(candidate.source_segment_ids)
            is_allowed = bool(source_ids) and all(
                segment_id.startswith(expected_prefix)
                and segment_id in known_segment_ids
                for segment_id in source_ids
            )
            if not is_allowed or candidate.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(candidate.chunk_id)
            hits.append(
                SearchHit(
                    rank=len(hits) + 1,
                    chunk_id=candidate.chunk_id,
                    episode_id=arguments.episode_id,
                    source_segment_ids=source_ids,
                    score=candidate.score,
                    score_semantics=candidate.score_semantics,
                    retriever_name=candidate.retriever_name,
                )
            )
            if len(hits) == arguments.top_k:
                break
        return SearchTranscriptData(
            episode_id=arguments.episode_id,
            hits=tuple(hits),
            result_count=len(hits),
        )

    def get_episode_metadata(
        self,
        arguments: GetEpisodeMetadataArguments,
    ) -> EpisodeMetadataData:
        transcript = self._transcript(arguments.episode_id)
        speakers = tuple(
            dict.fromkeys(
                segment.speaker
                for segment in transcript.segments
                if segment.speaker is not None
            )
        )
        section_names = tuple(
            dict.fromkeys(
                segment.section_title
                for segment in transcript.segments
                if segment.section_title is not None
            )
        )
        return EpisodeMetadataData(
            episode_id=transcript.episode_id,
            source_format=transcript.source_format,
            segment_count=len(transcript.segments),
            timed_segment_count=sum(
                segment.start_time is not None for segment in transcript.segments
            ),
            speaker_count=len(speakers),
            section_count=len(section_names),
            section_names=section_names,
        )

    def get_segment_context(
        self,
        arguments: GetSegmentContextArguments,
    ) -> SegmentContextData:
        transcript = self._transcript(arguments.episode_id)
        target_position = next(
            (
                position
                for position, segment in enumerate(transcript.segments)
                if segment.segment_id == arguments.segment_id
            ),
            None,
        )
        if target_position is None:
            raise ToolNotFoundError("segment 不存在")
        start = max(0, target_position - arguments.context_before)
        end = min(
            len(transcript.segments),
            target_position + arguments.context_after + 1,
        )
        context = tuple(
            _context_item(
                segment,
                is_target=segment.segment_id == arguments.segment_id,
            )
            for segment in transcript.segments[start:end]
        )
        return SegmentContextData(
            episode_id=arguments.episode_id,
            target_segment_id=arguments.segment_id,
            context_before=arguments.context_before,
            context_after=arguments.context_after,
            segments=context,
            result_count=len(context),
        )

    def _transcript(self, episode_id: str) -> ProcessedTranscript:
        transcript = self.dependencies.transcripts.get(episode_id)
        if transcript is None:
            raise ToolNotFoundError("episode 不存在")
        return transcript


def _validate_transcript_segments(transcript: ProcessedTranscript) -> None:
    expected_prefix = f"{transcript.episode_id}:"
    ids = [segment.segment_id for segment in transcript.segments]
    indices = [segment.segment_index for segment in transcript.segments]
    if len(ids) != len(set(ids)) or len(indices) != len(set(indices)):
        raise ValueError("transcript segment identity 必须唯一")
    if indices != sorted(indices):
        raise ValueError("transcript segments 必须按 segment_index 排序")
    for segment in transcript.segments:
        if not segment.segment_id.startswith(expected_prefix):
            raise ValueError("transcript segment 不属于 episode")
        if segment.segment_id.rsplit(":", 1)[-1] != f"{segment.segment_index:06d}":
            raise ValueError("segment_id 与 segment_index 不一致")


def _context_item(
    segment: TranscriptSegment,
    *,
    is_target: bool,
) -> SegmentContextItem:
    return SegmentContextItem(
        segment_id=segment.segment_id,
        segment_index=segment.segment_index,
        text=segment.text,
        speaker=segment.speaker,
        section_title=segment.section_title,
        start_time=segment.start_time,
        end_time=segment.end_time,
        is_target=is_target,
    )
