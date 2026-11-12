"""Deterministic fixed-character chunking with exact source spans."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.schemas.chunk import ChunkedTranscript, ChunkSourceSpan, TranscriptChunk
from app.schemas.transcript import ProcessedTranscript, TranscriptSegment


DEFAULT_CHUNK_SIZE = 1_000
DEFAULT_OVERLAP = 200
ALGORITHM = "fixed_character_windows_v1"


class ChunkingError(ValueError):
    """Raised when Day 6 input or configuration cannot preserve traceability."""


@dataclass(frozen=True)
class _LayoutSegment:
    segment: TranscriptSegment
    block_start: int
    block_end: int


@dataclass(frozen=True)
class _SectionBlock:
    section_title: str | None
    text: str
    layout: tuple[_LayoutSegment, ...]


def _validate_configuration(chunk_size: int, overlap: int) -> None:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise ValueError("chunk_size 必须是整数")
    if isinstance(overlap, bool) or not isinstance(overlap, int):
        raise ValueError("overlap 必须是整数")
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0:
        raise ValueError("overlap 不能小于 0")
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")


def _validate_traceability(
    *,
    episode_id: str,
    source_file: str,
    segments: Sequence[TranscriptSegment],
) -> None:
    if not re.fullmatch(r"ep_[0-9a-f]{16}", episode_id):
        raise ChunkingError("episode_id 格式无效，无法生成稳定 Chunk ID")
    if not source_file.strip():
        raise ChunkingError("source_file 不能为空")
    if not segments:
        raise ChunkingError("待切分的转写片段不能为空")

    previous_index = -1
    for segment in segments:
        if segment.segment_index <= previous_index:
            raise ChunkingError("segment_index 必须严格递增且不能重复")
        expected_id = f"{episode_id}:{segment.segment_index:06d}"
        if segment.segment_id != expected_id:
            raise ChunkingError(
                f"片段 {segment.segment_id} 与 episode_id/segment_index 不一致；"
                f"预期 {expected_id}"
            )
        if segment.source_file != source_file:
            raise ChunkingError(
                f"片段 {segment.segment_id} 的 source_file 与节目来源不一致"
            )
        previous_index = segment.segment_index


def _make_block(
    section_title: str | None,
    segments: Sequence[TranscriptSegment],
) -> _SectionBlock:
    parts: list[str] = []
    layout: list[_LayoutSegment] = []
    cursor = 0
    for segment in segments:
        if parts:
            parts.append("\n")
            cursor += 1
        start = cursor
        parts.append(segment.text)
        cursor += len(segment.text)
        layout.append(
            _LayoutSegment(
                segment=segment,
                block_start=start,
                block_end=cursor,
            )
        )
    return _SectionBlock(
        section_title=section_title,
        text="".join(parts),
        layout=tuple(layout),
    )


def _section_blocks(
    segments: Sequence[TranscriptSegment],
) -> list[_SectionBlock]:
    blocks: list[_SectionBlock] = []
    current_title = segments[0].section_title
    current_segments: list[TranscriptSegment] = []
    for segment in segments:
        if current_segments and segment.section_title != current_title:
            blocks.append(_make_block(current_title, current_segments))
            current_segments = []
            current_title = segment.section_title
        current_segments.append(segment)
    blocks.append(_make_block(current_title, current_segments))
    return blocks


def _window_spans(
    block: _SectionBlock,
    *,
    start: int,
    end: int,
) -> list[ChunkSourceSpan]:
    spans: list[ChunkSourceSpan] = []
    for item in block.layout:
        intersection_start = max(start, item.block_start)
        intersection_end = min(end, item.block_end)
        if intersection_start >= intersection_end:
            continue
        spans.append(
            ChunkSourceSpan(
                segment_id=item.segment.segment_id,
                segment_index=item.segment.segment_index,
                start_char=intersection_start - item.block_start,
                end_char=intersection_end - item.block_start,
            )
        )
    return spans


def _stable_chunk_id(
    episode_id: str,
    section_titles: Sequence[str],
    spans: Sequence[ChunkSourceSpan],
) -> str:
    identity = {
        "episode_id": episode_id,
        "section_titles": list(section_titles),
        "source_spans": [
            [
                span.segment_id,
                span.start_char,
                span.end_char,
            ]
            for span in spans
        ],
    }
    canonical = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"chunk_{digest}"


def _ordered_unique(values: Sequence[str | None]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value is not None))


def _build_chunk(
    *,
    episode_id: str,
    source_file: str,
    chunk_index: int,
    text: str,
    spans: Sequence[ChunkSourceSpan],
    segment_by_index: dict[int, TranscriptSegment],
) -> TranscriptChunk:
    source_segments = [segment_by_index[span.segment_index] for span in spans]
    source_ids = [span.segment_id for span in spans]
    speakers = _ordered_unique([segment.speaker for segment in source_segments])
    section_titles = _ordered_unique(
        [segment.section_title for segment in source_segments]
    )
    starts = [
        segment.start_time
        for segment in source_segments
        if segment.start_time is not None
    ]
    known_ends = [
        segment.end_time
        if segment.end_time is not None
        else segment.start_time
        for segment in source_segments
        if segment.end_time is not None or segment.start_time is not None
    ]
    time_range_complete = all(
        segment.start_time is not None and segment.end_time is not None
        for segment in source_segments
    )
    return TranscriptChunk(
        chunk_id=_stable_chunk_id(episode_id, section_titles, spans),
        chunk_index=chunk_index,
        episode_id=episode_id,
        source_file=source_file,
        text=text,
        source_segment_ids=source_ids,
        source_segment_start_index=spans[0].segment_index,
        source_segment_end_index=spans[-1].segment_index,
        source_spans=list(spans),
        speakers=speakers,
        section_titles=section_titles,
        start_time=min(starts) if starts else None,
        end_time=max(known_ends) if known_ends else None,
        time_range_complete=time_range_complete,
    )


def create_chunks(
    *,
    episode_id: str,
    source_file: str,
    segments: Sequence[TranscriptSegment],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[TranscriptChunk]:
    """Create stable fixed-character windows from ordered source segments."""
    _validate_configuration(chunk_size, overlap)
    source_segments = tuple(segments)
    _validate_traceability(
        episode_id=episode_id,
        source_file=source_file,
        segments=source_segments,
    )

    chunks: list[TranscriptChunk] = []
    step = chunk_size - overlap
    segment_by_index = {
        segment.segment_index: segment for segment in source_segments
    }
    for block in _section_blocks(source_segments):
        start = 0
        while start < len(block.text):
            end = min(start + chunk_size, len(block.text))
            spans = _window_spans(block, start=start, end=end)
            if spans:
                chunks.append(
                    _build_chunk(
                        episode_id=episode_id,
                        source_file=source_file,
                        chunk_index=len(chunks),
                        text=block.text[start:end],
                        spans=spans,
                        segment_by_index=segment_by_index,
                    )
                )
            if end == len(block.text):
                break
            start += step
    if not chunks:
        raise ChunkingError("切分后没有生成可追溯的 Chunk")
    return chunks


def chunk_transcript(
    transcript: ProcessedTranscript,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> ChunkedTranscript:
    """Create a validated file-level Day 6 result from a Day 3 transcript."""
    chunks = create_chunks(
        episode_id=transcript.episode_id,
        source_file=transcript.source_file,
        segments=transcript.segments,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    return ChunkedTranscript(
        algorithm=ALGORITHM,
        episode_id=transcript.episode_id,
        source_file=transcript.source_file,
        input_segment_count=len(transcript.segments),
        chunk_size=chunk_size,
        overlap=overlap,
        chunks=chunks,
    )
