"""Day 3 tests for transcript reading, cleaning, metadata, and persistence."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from pydantic import ValidationError

from app.core.files import atomic_write_text
from app.ingestion.pipeline import process_transcript
from app.ingestion.transcripts import TranscriptFormatError, load_transcript
from app.schemas.transcript import TranscriptSegment


def test_txt_is_cleaned_without_losing_citation_metadata(tmp_path) -> None:
    source = tmp_path / "episode.txt"
    source.write_text(
        "\n".join(
            [
                "[00:01 - 00:04] 主持人：  欢迎   收听。",
                "[音乐]",
                "[00:05] 嘉宾: 今天讨论 RAG。 [杂音]",
                "",
            ]
        ),
        encoding="utf-8",
    )

    transcript = load_transcript(source)

    assert transcript.source_file == "episode.txt"
    assert transcript.source_format == "txt"
    assert transcript.episode_id.startswith("ep_")
    assert len(transcript.segments) == 2
    assert transcript.segments[0].model_dump() == {
        "segment_id": f"{transcript.episode_id}:000000",
        "segment_index": 0,
        "text": "欢迎 收听。",
        "source_file": "episode.txt",
        "speaker": "主持人",
        "section_title": None,
        "start_time": 1.0,
        "end_time": 4.0,
    }
    assert transcript.segments[1].speaker == "嘉宾"
    assert transcript.segments[1].start_time == 5.0
    assert transcript.segments[1].end_time is None
    assert transcript.segments[1].text == "今天讨论 RAG。"
    assert transcript.segments[1].segment_id == f"{transcript.episode_id}:000001"


def test_txt_decimal_seconds_are_parsed_as_timestamps(tmp_path) -> None:
    source = tmp_path / "decimal-time.txt"
    source.write_text(
        "[00:06.5 - 00:08.25] 嘉宾：带小数秒的内容。\n"
        "[01:02:03.75] 主持人：三段式时间。\n",
        encoding="utf-8",
    )

    transcript = load_transcript(source)

    assert transcript.segments[0].speaker == "嘉宾"
    assert transcript.segments[0].start_time == 6.5
    assert transcript.segments[0].end_time == 8.25
    assert transcript.segments[1].speaker == "主持人"
    assert transcript.segments[1].start_time == 3723.75


def test_webpage_copy_layout_is_combined_and_keeps_section_titles(tmp_path) -> None:
    source = tmp_path / "web-copy.txt"
    source.write_text(
        "\n".join(
            [
                "Transcript",
                "Nature of intelligence",
                "Edited by Example Editor",
                "",
                "Dwarkesh Patel 00:00:44",
                "",
                "Welcome to the podcast.",
                "",
                "Demis Hassabis 00:00:51",
                "",
                "Thanks for having me.",
                "",
                "A second paragraph in the same turn.",
                "",
                "RL atop LLMs",
                "Dwarkesh Patel 00:05:56",
                "",
                "How will planning interact with language models?",
            ]
        ),
        encoding="utf-8",
    )

    transcript = load_transcript(source)

    assert len(transcript.segments) == 3
    assert transcript.segments[0].speaker == "Dwarkesh Patel"
    assert transcript.segments[0].start_time == 44.0
    assert transcript.segments[0].section_title == "Nature of intelligence"
    assert transcript.segments[1].text == (
        "Thanks for having me. A second paragraph in the same turn."
    )
    assert transcript.segments[1].section_title == "Nature of intelligence"
    assert transcript.segments[2].section_title == "RL atop LLMs"
    assert transcript.segments[2].start_time == 356.0


def test_single_turn_webpage_copy_uses_trailing_time_parser(tmp_path) -> None:
    source = tmp_path / "single-turn-web-copy.txt"
    source.write_text(
        "\n".join(
            [
                "Transcript",
                "One chapter",
                "Solo Speaker 00:00:03",
                "",
                "A single copied turn.",
            ]
        ),
        encoding="utf-8",
    )

    transcript = load_transcript(source)

    assert len(transcript.segments) == 1
    assert transcript.segments[0].speaker == "Solo Speaker"
    assert transcript.segments[0].start_time == 3.0
    assert transcript.segments[0].section_title == "One chapter"
    assert transcript.segments[0].text == "A single copied turn."


def test_json_aliases_and_nested_metadata_are_normalized(tmp_path) -> None:
    source = tmp_path / "episode.json"
    source.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "speaker": " 主持人 ",
                        "start": 1.5,
                        "end": 3,
                        "text": " 第一段   内容 ",
                    },
                    {
                        "content": "第二段内容",
                        "metadata": {
                            "speaker_name": "嘉宾",
                            "chapter_title": "第二章",
                            "start_time": "00:04",
                            "end_time": "00:06.5",
                        },
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    transcript = load_transcript(source)

    assert [segment.text for segment in transcript.segments] == [
        "第一段 内容",
        "第二段内容",
    ]
    assert transcript.segments[0].source_file == "episode.json"
    assert transcript.segments[1].speaker == "嘉宾"
    assert transcript.segments[1].section_title == "第二章"
    assert transcript.segments[1].start_time == 4.0
    assert transcript.segments[1].end_time == 6.5


def test_repeated_processing_produces_identical_json(tmp_path) -> None:
    source = tmp_path / "stable.txt"
    source.write_text(
        "[01:02] A: 相同输入应该得到相同输出。\n",
        encoding="utf-8",
    )
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"

    first, _ = process_transcript(source, first_output)
    second, _ = process_transcript(source, second_output)

    assert first == second
    assert first.episode_id == second.episode_id
    assert first.segments[0].segment_id == second.segments[0].segment_id
    assert first_output.read_bytes() == second_output.read_bytes()
    saved = json.loads(first_output.read_text(encoding="utf-8"))
    assert saved["segments"][0]["start_time"] == 62.0


def test_transcript_models_are_frozen() -> None:
    segment = TranscriptSegment(
        segment_id="ep_0123456789abcdef:000000",
        segment_index=0,
        text="original",
        source_file="episode.txt",
    )

    with pytest.raises(ValidationError, match="frozen"):
        segment.text = "mutated"


def test_atomic_text_write_preserves_existing_file_on_replace_failure(
    tmp_path, monkeypatch
) -> None:
    destination = tmp_path / "result.json"
    destination.write_text("original", encoding="utf-8")

    def fail_replace(source, target) -> None:
        del source, target
        raise OSError("synthetic replace failure")

    monkeypatch.setattr("app.core.files.os.replace", fail_replace)

    with pytest.raises(OSError, match="synthetic replace failure"):
        atomic_write_text(destination, "replacement")

    assert destination.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.iterdir()) == [destination]


def test_invalid_or_empty_input_has_readable_error(tmp_path) -> None:
    empty_source = tmp_path / "empty.txt"
    empty_source.write_text("\n [掌声] \n", encoding="utf-8")

    with pytest.raises(TranscriptFormatError, match="没有可用"):
        load_transcript(empty_source)

    unsupported = tmp_path / "episode.csv"
    unsupported.write_text("speaker,text", encoding="utf-8")
    with pytest.raises(TranscriptFormatError, match="只支持"):
        load_transcript(unsupported)


def test_invalid_timestamp_type_has_a_readable_format_error(tmp_path) -> None:
    source = tmp_path / "invalid-time.json"
    source.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "text": "时间字段不应接受对象。",
                        "start_time": {"seconds": 3},
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(TranscriptFormatError, match="必须是秒数"):
        load_transcript(source)


def test_invalid_timestamp_does_not_echo_source_value(tmp_path) -> None:
    sensitive_value = "PRIVATE-TRANSCRIPT-VALUE"
    source = tmp_path / "invalid-time-value.json"
    source.write_text(
        json.dumps(
            {"segments": [{"text": "safe text", "start_time": sensitive_value}]}
        ),
        encoding="utf-8",
    )

    with pytest.raises(TranscriptFormatError) as captured:
        load_transcript(source)

    assert sensitive_value not in str(captured.value)


def test_malformed_json_reports_source_location(tmp_path) -> None:
    source = tmp_path / "malformed.json"
    source.write_text(
        '{"segments": [{"text": "缺少结束符号"}]',
        encoding="utf-8",
    )

    with pytest.raises(
        TranscriptFormatError,
        match=r"JSON 格式错误（第 1 行，第 \d+ 列）",
    ):
        load_transcript(source)


def test_end_time_before_start_time_is_a_readable_format_error(tmp_path) -> None:
    source = tmp_path / "backwards-time.json"
    source.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "text": "结束时间不能早于开始时间。",
                        "start_time": 10,
                        "end_time": 5,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(TranscriptFormatError, match="输入契约校验失败"):
        load_transcript(source)


def test_module_cli_processes_a_file_end_to_end(tmp_path) -> None:
    source = tmp_path / "cli-input.txt"
    destination = tmp_path / "cli-output.json"
    source.write_text("[00:03] 主持人: CLI 验收。\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.ingestion",
            str(source),
            "--output",
            str(destination),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "处理完成：1 个片段" in completed.stdout
    assert destination.is_file()
    assert json.loads(destination.read_text(encoding="utf-8"))["segments"][0][
        "speaker"
    ] == "主持人"
