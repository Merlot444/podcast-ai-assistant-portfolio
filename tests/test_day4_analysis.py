"""Mocked tests for the Day 4 analysis baseline."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.core.config import Settings
from app.core.llm import create_json_completion
from app.generation.day4 import (
    Day4AnalysisError,
    analyze_processed_transcript,
    create_analysis_batches,
    load_processed_transcript,
)
from app.schemas.transcript import ProcessedTranscript, TranscriptSegment


EPISODE_ID = "ep_0123456789abcdef"


def _segment(
    index: int,
    text: str,
    section_title: str,
    *,
    speaker: str = "Speaker",
    start_time: float | None = None,
) -> TranscriptSegment:
    return TranscriptSegment(
        segment_id=f"{EPISODE_ID}:{index:06d}",
        segment_index=index,
        text=text,
        source_file="episode.txt",
        speaker=speaker,
        section_title=section_title,
        start_time=start_time if start_time is not None else float(index * 10),
    )


def _transcript(segments: list[TranscriptSegment]) -> ProcessedTranscript:
    return ProcessedTranscript(
        episode_id=EPISODE_ID,
        source_file="episode.txt",
        source_format="txt",
        segments=segments,
    )


def _settings() -> Settings:
    return Settings(
        api_key="test-key-not-a-secret",
        base_url="https://example.test",
        model_fast="fast-test-model",
        model_strong="strong-test-model",
    )


def _response(content: str, *, finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content),
            )
        ]
    )


def _clock():
    value = -0.25

    def tick() -> float:
        nonlocal value
        value += 0.25
        return value

    return tick


def test_processed_json_is_loaded_and_validated(tmp_path) -> None:
    transcript = _transcript([_segment(0, "Hello.", "Opening")])
    source = tmp_path / "processed.json"
    source.write_text(
        json.dumps(transcript.model_dump(mode="json")),
        encoding="utf-8",
    )

    loaded = load_processed_transcript(source)

    assert loaded == transcript


def test_invalid_processed_json_does_not_echo_transcript_input(tmp_path) -> None:
    sensitive_text = "PRIVATE-TRANSCRIPT-PAYLOAD"
    source = tmp_path / "invalid-processed.json"
    source.write_text(
        json.dumps(
            {
                "episode_id": EPISODE_ID,
                "source_file": "episode.txt",
                "source_format": "txt",
                "segments": sensitive_text,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(Day4AnalysisError) as captured:
        load_processed_transcript(source)

    assert sensitive_text not in str(captured.value)


def test_analysis_batches_keep_chapters_and_citation_metadata() -> None:
    transcript = _transcript(
        [
            _segment(0, "A" * 150, "Chapter A", speaker="Alice"),
            _segment(1, "B" * 150, "Chapter A", speaker="Bob"),
            _segment(2, "C", "Chapter B", speaker="Alice"),
        ]
    )

    batches = create_analysis_batches(
        transcript.segments,
        max_characters=240,
    )

    assert [batch.section_title for batch in batches] == [
        "Chapter A",
        "Chapter A",
        "Chapter B",
    ]
    assert [batch.batch_index for batch in batches] == [1, 2, 1]
    assert [batch.batch_count for batch in batches] == [2, 2, 1]
    assert transcript.segments[0].segment_id in batches[0].rendered_text
    assert "Alice" in batches[0].rendered_text
    assert "00:00:00" in batches[0].rendered_text


def test_single_segment_larger_than_budget_is_rejected() -> None:
    transcript = _transcript([_segment(0, "A" * 300, "Chapter A")])

    with pytest.raises(Day4AnalysisError, match="不会拆开一个原始片段"):
        create_analysis_batches(
            transcript.segments,
            max_characters=200,
        )


def test_short_content_uses_one_mocked_direct_call() -> None:
    transcript = _transcript(
        [
            _segment(0, "The guest discusses planning.", "Planning", speaker="Host"),
            _segment(
                1,
                "Scaling alone may not be enough.",
                "Scaling",
                speaker="Guest",
            ),
        ]
    )
    raw_output = json.dumps(
        {
            "summary": "节目讨论规划与扩展。",
            "keywords": ["规划", "扩展", "人工智能"],
            "chapter_overviews": [
                {"section_title": "Planning", "overview": "讨论规划能力。"},
                {"section_title": "Scaling", "overview": "讨论扩展的限制。"},
            ],
            "recommendation": "适合关注人工智能研究路径的听众。",
        },
        ensure_ascii=False,
    )
    client = Mock()
    client.chat.completions.create.return_value = _response(raw_output)

    result = analyze_processed_transcript(
        client,
        _settings(),
        transcript,
        direct_max_characters=10_000,
        clock=_clock(),
    )

    assert result.mode == "direct"
    assert result.metrics.model_call_count == 1
    assert result.metrics.batch_count == 1
    assert result.metrics.output_characters == len(raw_output)
    assert result.batch_summaries == []
    assert result.chapter_overviews[0].source_segment_ids == [
        f"{EPISODE_ID}:000000"
    ]
    assert result.chapter_overviews[1].speakers == ["Guest"]
    request = client.chat.completions.create.call_args.kwargs
    assert request["model"] == "strong-test-model"
    assert request["max_tokens"] == 1_600
    assert "section_title 必须逐字原样返回" in request["messages"][1]["content"]


def test_direct_analysis_rejects_missing_or_unknown_chapter() -> None:
    transcript = _transcript([_segment(0, "Short text.", "Expected Chapter")])
    raw_output = json.dumps(
        {
            "summary": "摘要。",
            "keywords": ["一", "二", "三"],
            "chapter_overviews": [
                {"section_title": "Unknown Chapter", "overview": "错误章节。"}
            ],
            "recommendation": "推荐。",
        },
        ensure_ascii=False,
    )
    client = Mock()
    client.chat.completions.create.return_value = _response(raw_output)

    with pytest.raises(Day4AnalysisError, match="没有原样保留章节标题"):
        analyze_processed_transcript(
            client,
            _settings(),
            transcript,
            clock=_clock(),
        )


def test_schema_rejects_output_longer_than_prompt_contract() -> None:
    transcript = _transcript([_segment(0, "Short text.", "Opening")])
    raw_output = json.dumps(
        {
            "summary": "超" * 501,
            "keywords": ["一", "二", "三"],
            "chapter_overviews": [
                {"section_title": "Opening", "overview": "章节概述。"}
            ],
            "recommendation": "推荐。",
        },
        ensure_ascii=False,
    )
    client = Mock()
    client.chat.completions.create.return_value = _response(raw_output)

    with pytest.raises(Day4AnalysisError, match="业务结构") as captured:
        analyze_processed_transcript(
            client,
            _settings(),
            transcript,
            clock=_clock(),
        )

    assert "超" * 501 not in str(captured.value)


def test_long_content_uses_mocked_segment_summaries_then_synthesis() -> None:
    transcript = _transcript(
        [
            _segment(0, "A" * 250, "Chapter A", speaker="Alice"),
            _segment(1, "B" * 250, "Chapter A", speaker="Bob"),
            _segment(2, "C" * 250, "Chapter B", speaker="Alice"),
        ]
    )
    partial_outputs = [
        {"summary": "A1 摘要", "key_points": ["A1 要点", "A1 条件"]},
        {"summary": "A2 摘要", "key_points": ["A2 要点", "A2 条件"]},
        {"summary": "B 摘要", "key_points": ["B 要点", "B 条件"]},
    ]
    final_output = {
        "summary": "整期汇总摘要。",
        "keywords": ["主题A", "主题B", "条件"],
        "chapter_overviews": [
            {"section_title": "Chapter A", "overview": "A 章节概述。"},
            {"section_title": "Chapter B", "overview": "B 章节概述。"},
        ],
        "recommendation": "一段克制的推荐文案。",
    }
    client = Mock()
    client.chat.completions.create.side_effect = [
        *[
            _response(json.dumps(output, ensure_ascii=False))
            for output in partial_outputs
        ],
        _response(json.dumps(final_output, ensure_ascii=False)),
    ]

    result = analyze_processed_transcript(
        client,
        _settings(),
        transcript,
        direct_max_characters=1,
        batch_max_characters=340,
        clock=_clock(),
    )

    assert result.mode == "map_reduce"
    assert result.metrics.batch_count == 3
    assert result.metrics.model_call_count == 4
    assert len(result.batch_summaries) == 3
    assert result.batch_summaries[0].source_segment_ids == [
        f"{EPISODE_ID}:000000"
    ]
    assert result.batch_summaries[1].speakers == ["Bob"]
    assert [call.stage for call in result.metrics.calls] == [
        "segment_summary",
        "segment_summary",
        "segment_summary",
        "synthesis",
    ]
    requests = client.chat.completions.create.call_args_list
    assert [request.kwargs["model"] for request in requests] == [
        "fast-test-model",
        "fast-test-model",
        "fast-test-model",
        "strong-test-model",
    ]
    synthesis_input = requests[-1].kwargs["messages"][1]["content"]
    assert "A1 摘要" in synthesis_input
    assert "B 摘要" in synthesis_input


def test_truncated_model_output_is_rejected_without_real_api_call() -> None:
    transcript = _transcript([_segment(0, "Short text.", "Opening")])
    client = Mock()
    client.chat.completions.create.return_value = _response(
        '{"summary":"unfinished"',
        finish_reason="length",
    )

    with pytest.raises(Day4AnalysisError, match="长度上限"):
        analyze_processed_transcript(
            client,
            _settings(),
            transcript,
            clock=_clock(),
        )


def test_non_positive_limits_are_rejected_before_model_call() -> None:
    transcript = _transcript([_segment(0, "Short text.", "Opening")])
    client = Mock()

    with pytest.raises(ValueError, match="batch_max_characters"):
        analyze_processed_transcript(
            client,
            _settings(),
            transcript,
            batch_max_characters=0,
        )

    client.chat.completions.create.assert_not_called()


def test_model_wrapper_rejects_non_positive_max_tokens() -> None:
    client = Mock()

    with pytest.raises(ValueError, match="max_tokens"):
        create_json_completion(
            client,
            model="test-model",
            messages=[],
            max_tokens=0,
        )

    client.chat.completions.create.assert_not_called()
