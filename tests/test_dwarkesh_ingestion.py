"""Focused tests for public Dwarkesh visible-article transcript extraction."""

from __future__ import annotations

import json

import pytest

from app.ingestion.dwarkesh import (
    DwarkeshExtractionError,
    extract_dwarkesh_article,
    save_extracted_transcript,
)
from app.ingestion.transcripts import load_transcript


def test_timed_article_keeps_turn_times_sections_and_discards_page_chrome(
    tmp_path,
) -> None:
    article = """Sponsors
Do not ingest this sponsor.
Timestamps
(00:00:00) – First chapter
(00:01:00) – Second chapter
Transcript
00:00:00 – First chapter
Host Person 00:00:00

Opening question.

Guest Person 00:00:05

First answer.

00:01:00 – Second chapter
Host Person 00:01:00

Second question.

Guest Person 00:01:05

Second answer.

Host Person 00:01:10

Closing question.

Guest Person 00:01:15

Closing answer.
12 Likes
∙
3 Restacks
Discussion about this video
Comments
"""

    payload = extract_dwarkesh_article(
        article,
        source_url="https://www.dwarkesh.com/p/example",
    )

    assert payload["timestamp_semantics"] == "speaker_turn"
    assert len(payload["segments"]) == 6
    assert payload["segments"][0] == {
        "speaker": "Host Person",
        "section_title": "First chapter",
        "start_time": 0.0,
        "end_time": None,
        "text": "Opening question.",
    }
    assert payload["segments"][2]["section_title"] == "Second chapter"
    assert "sponsor" not in json.dumps(payload).casefold()
    assert "likes" not in json.dumps(payload).casefold()

    raw_path = tmp_path / "raw.json"
    save_extracted_transcript(payload, raw_path)
    transcript = load_transcript(raw_path)
    assert transcript.segments[2].section_title == "Second chapter"
    assert transcript.segments[3].start_time == 65.0


def test_speaker_block_article_uses_only_honest_chapter_start_times() -> None:
    article = """Timestamps
(00:00:00) – Model design
(00:10:00) – Product design
Transcript
Model design
Host Person

What changed?

Guest Person

The model changed.

Host Person

Why?

Guest Person

Because the goal changed.

Product design
Host Person

What do users need?

Guest Person

They need a fast response.

Host Person

Anything else?

Guest Person

They need a useful response.
Discussion about this video
"""

    payload = extract_dwarkesh_article(
        article,
        source_url="https://dwarkesh.com/p/example-two",
    )

    assert payload["timestamp_semantics"] == "chapter_start_only"
    assert len(payload["segments"]) == 8
    assert [item["start_time"] for item in payload["segments"]] == [
        0.0,
        None,
        None,
        None,
        600.0,
        None,
        None,
        None,
    ]
    assert {item["speaker"] for item in payload["segments"]} == {
        "Host Person",
        "Guest Person",
    }
    assert payload["identified_speakers"] == ["Guest Person", "Host Person"]


def test_repeated_sentence_is_not_silently_promoted_to_speaker() -> None:
    article = """Timestamps
(00:00:00) – First chapter
(00:10:00) – Second chapter
Transcript
First chapter
Host Person

That is right.

Guest Person

That is right.

Host Person

Real question.

Guest Person

Real answer.

Second chapter
Host Person

Another question.

Guest Person

Another answer.
Discussion about this video
"""

    payload = extract_dwarkesh_article(
        article,
        source_url="https://dwarkesh.com/p/repeated-sentence",
    )

    assert payload["identified_speakers"] == ["Guest Person", "Host Person"]
    assert "That is right." in payload["segments"][0]["text"]
    assert "That is right." in payload["segments"][1]["text"]


def test_timed_body_rejects_sentence_shaped_speaker_headers() -> None:
    article = """Timestamps
(00:00:00) – First chapter
Transcript
00:00:00 – First chapter
That is right. 00:00:00

Body text.

Another sentence! 00:00:05

More body text.
Discussion about this video
"""

    with pytest.raises(DwarkeshExtractionError, match="说话人名称形状"):
        extract_dwarkesh_article(
            article,
            source_url="https://dwarkesh.com/p/timed-sentence-header",
        )


def test_source_and_boundaries_fail_closed() -> None:
    with pytest.raises(DwarkeshExtractionError, match="source_url"):
        extract_dwarkesh_article(
            "Transcript\nmissing",
            source_url="https://example.com/p/not-allowed",
        )
    with pytest.raises(DwarkeshExtractionError, match="章节时间戳"):
        extract_dwarkesh_article(
            "Transcript\nHost Person 00:00:00\nText",
            source_url="https://www.dwarkesh.com/p/missing-timestamps",
        )


def test_save_refuses_overwrite(tmp_path) -> None:
    destination = tmp_path / "existing.json"
    destination.write_text("user data", encoding="utf-8")

    with pytest.raises(DwarkeshExtractionError, match="不会静默覆盖"):
        save_extracted_transcript({"segments": []}, destination)

    assert destination.read_text(encoding="utf-8") == "user data"
