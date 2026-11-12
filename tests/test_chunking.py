"""Offline tests for deterministic Day 6 transcript chunking."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from pydantic import ValidationError

from app.chunking.core import ChunkingError, chunk_transcript, create_chunks
from app.schemas.transcript import ProcessedTranscript, TranscriptSegment


EPISODE_ID = "ep_0123456789abcdef"


def _segment(
    index: int,
    text: str,
    *,
    source_file: str = "episode.txt",
    speaker: str | None = "Speaker",
    section_title: str | None = "Chapter",
    start_time: float | None = None,
    end_time: float | None = None,
    episode_id: str = EPISODE_ID,
) -> TranscriptSegment:
    return TranscriptSegment(
        segment_id=f"{episode_id}:{index:06d}",
        segment_index=index,
        text=text,
        source_file=source_file,
        speaker=speaker,
        section_title=section_title,
        start_time=start_time,
        end_time=end_time,
    )


def _transcript(segments: list[TranscriptSegment]) -> ProcessedTranscript:
    return ProcessedTranscript(
        episode_id=EPISODE_ID,
        source_file="episode.txt",
        source_format="txt",
        segments=segments,
    )


def test_adjacent_segments_merge_with_ordered_source_metadata() -> None:
    transcript = _transcript(
        [
            _segment(
                0,
                "Alpha",
                speaker="Alice",
                start_time=0,
                end_time=4,
            ),
            _segment(
                1,
                "Beta",
                speaker="Bob",
                start_time=5,
                end_time=8,
            ),
            _segment(
                2,
                "Gamma",
                speaker="Alice",
                start_time=9,
            ),
        ]
    )

    result = chunk_transcript(transcript, chunk_size=100, overlap=0)

    assert len(result.chunks) == 1
    chunk = result.chunks[0]
    assert chunk.chunk_index == 0
    assert chunk.text == "Alpha\nBeta\nGamma"
    assert chunk.source_segment_ids == [
        f"{EPISODE_ID}:000000",
        f"{EPISODE_ID}:000001",
        f"{EPISODE_ID}:000002",
    ]
    assert chunk.source_segment_start_index == 0
    assert chunk.source_segment_end_index == 2
    assert chunk.speakers == ["Alice", "Bob"]
    assert chunk.section_titles == ["Chapter"]
    assert chunk.start_time == 0
    assert chunk.end_time == 9
    assert chunk.time_range_complete is False

    with pytest.raises(ValidationError, match="frozen"):
        chunk.text = "mutated"

    with pytest.raises(ValidationError, match="frozen"):
        result.chunk_size = 1


def test_character_overlap_is_exact_stable_and_does_not_lose_long_segment() -> None:
    text = "abcdefghijklmnopqrstuvwxyz"
    transcript = _transcript(
        [
            _segment(
                0,
                text,
                speaker=None,
                section_title=None,
                start_time=None,
                end_time=None,
            )
        ]
    )

    first = chunk_transcript(transcript, chunk_size=10, overlap=3)
    second = chunk_transcript(transcript, chunk_size=10, overlap=3)

    assert first == second
    assert [chunk.text for chunk in first.chunks] == [
        "abcdefghij",
        "hijklmnopq",
        "opqrstuvwx",
        "vwxyz",
    ]
    assert [chunk.chunk_index for chunk in first.chunks] == [0, 1, 2, 3]
    assert first.chunks[0].text[-3:] == first.chunks[1].text[:3]
    assert first.chunks[1].text[-3:] == first.chunks[2].text[:3]
    assert len({chunk.chunk_id for chunk in first.chunks}) == 4
    assert first.chunks[0].source_spans[0].model_dump() == {
        "segment_id": f"{EPISODE_ID}:000000",
        "segment_index": 0,
        "start_char": 0,
        "end_char": 10,
    }
    assert first.chunks[1].source_spans[0].start_char == 7
    covered_positions = {
        position
        for chunk in first.chunks
        for span in chunk.source_spans
        for position in range(span.start_char, span.end_char)
    }
    assert covered_positions == set(range(len(text)))
    assert first.chunks[0].speakers == []
    assert first.chunks[0].start_time is None
    assert first.chunks[0].end_time is None
    assert first.chunks[0].time_range_complete is False


def test_chapter_boundary_starts_a_new_window_and_keeps_short_final_chunk() -> None:
    transcript = _transcript(
        [
            _segment(0, "12345678", section_title="Chapter A"),
            _segment(1, "abcdefgh", section_title="Chapter B"),
        ]
    )

    result = chunk_transcript(transcript, chunk_size=10, overlap=2)

    assert [chunk.text for chunk in result.chunks] == ["12345678", "abcdefgh"]
    assert [chunk.section_titles for chunk in result.chunks] == [
        ["Chapter A"],
        ["Chapter B"],
    ]
    assert all(len(chunk.section_titles) == 1 for chunk in result.chunks)


@pytest.mark.parametrize(
    ("chunk_size", "overlap", "message"),
    [
        (0, 0, "chunk_size 必须大于 0"),
        (-1, 0, "chunk_size 必须大于 0"),
        (10, -1, "overlap 不能小于 0"),
        (10, 10, "overlap 必须小于 chunk_size"),
        (10, 11, "overlap 必须小于 chunk_size"),
    ],
)
def test_invalid_chunk_configuration_is_rejected(
    chunk_size: int,
    overlap: int,
    message: str,
) -> None:
    transcript = _transcript([_segment(0, "content")])

    with pytest.raises(ValueError, match=message):
        chunk_transcript(
            transcript,
            chunk_size=chunk_size,
            overlap=overlap,
        )


def test_empty_segment_list_is_rejected() -> None:
    with pytest.raises(ChunkingError, match="不能为空"):
        create_chunks(
            episode_id=EPISODE_ID,
            source_file="episode.txt",
            segments=[],
            chunk_size=10,
            overlap=0,
        )


@pytest.mark.parametrize(
    ("segments", "message"),
    [
        (
            [
                _segment(
                    0,
                    "wrong episode",
                    episode_id="ep_fedcba9876543210",
                )
            ],
            "episode_id/segment_index 不一致",
        ),
        (
            [_segment(0, "wrong file", source_file="other.txt")],
            "source_file 与节目来源不一致",
        ),
        (
            [_segment(1, "later"), _segment(0, "earlier")],
            "segment_index 必须严格递增",
        ),
    ],
)
def test_broken_source_traceability_has_readable_error(
    segments: list[TranscriptSegment],
    message: str,
) -> None:
    with pytest.raises(ChunkingError, match=message):
        create_chunks(
            episode_id=EPISODE_ID,
            source_file="episode.txt",
            segments=segments,
            chunk_size=100,
            overlap=0,
        )


def test_exact_budget_next_segment_and_zero_overlap_are_deterministic() -> None:
    transcript = _transcript(
        [
            _segment(0, "A" * 10),
            _segment(1, "B"),
        ]
    )

    result = chunk_transcript(transcript, chunk_size=10, overlap=0)

    assert [chunk.text for chunk in result.chunks] == ["A" * 10, "\nB"]
    assert result.chunks[0].source_segment_ids == [f"{EPISODE_ID}:000000"]
    assert result.chunks[1].source_segment_ids == [f"{EPISODE_ID}:000001"]
    assert result.chunks[0].source_spans[0].end_char == 10
    assert result.chunks[1].source_spans[0].start_char == 0


def test_single_segment_with_zero_length_time_range_is_complete() -> None:
    transcript = _transcript(
        [
            _segment(
                0,
                "Moment",
                speaker=None,
                section_title=None,
                start_time=5,
                end_time=5,
            )
        ]
    )

    chunk = chunk_transcript(transcript, chunk_size=6, overlap=0).chunks[0]

    assert chunk.source_segment_start_index == chunk.source_segment_end_index == 0
    assert chunk.start_time == chunk.end_time == 5
    assert chunk.time_range_complete is True
    assert chunk.speakers == []
    assert chunk.section_titles == []


def test_module_cli_reads_processed_json_and_saves_chunks_offline(tmp_path) -> None:
    transcript = _transcript(
        [
            _segment(0, "First", start_time=1, end_time=2),
            _segment(1, "Second", start_time=3, end_time=4),
        ]
    )
    source = tmp_path / "processed.json"
    output = tmp_path / "chunks.json"
    source.write_text(
        json.dumps(transcript.model_dump(mode="json"), ensure_ascii=False),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.chunking",
            str(source),
            "--output",
            str(output),
            "--chunk-size",
            "8",
            "--overlap",
            "2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "切分完成：2 个原始片段" in completed.stdout
    assert "生成 Chunk：2 个" in completed.stdout
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["algorithm"] == "fixed_character_windows_v1"
    assert saved["chunk_size"] == 8
    assert saved["overlap"] == 2
    assert saved["chunks"][0]["source_segment_ids"] == [
        f"{EPISODE_ID}:000000",
        f"{EPISODE_ID}:000001",
    ]
