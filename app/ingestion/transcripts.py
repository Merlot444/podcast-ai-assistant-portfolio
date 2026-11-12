"""Load TXT and JSON transcripts into one validated representation."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from app.core.errors import safe_error_message
from app.ingestion.cleaning import clean_speaker, clean_text, parse_timestamp
from app.schemas.transcript import ProcessedTranscript, TranscriptSegment


class TranscriptFormatError(ValueError):
    """Raised when a source transcript has an unsupported or invalid format."""


_TIME_TOKEN = (
    r"(?:\d{1,3}:\d{2}:\d{2}(?:\.\d+)?|"
    r"\d{1,3}:\d{2}(?:\.\d+)?)"
)
_BRACKETED_TIME = re.compile(
    rf"^\s*\[(?P<start>{_TIME_TOKEN})"
    rf"(?:\s*(?:-->|-|–|—|~|～|至)\s*(?P<end>{_TIME_TOKEN}))?\]\s*"
)
_PLAIN_TIME_RANGE = re.compile(
    rf"^\s*(?P<start>{_TIME_TOKEN})"
    rf"\s*(?:-->|-|–|—|~|～|至)\s*(?P<end>{_TIME_TOKEN})\s+"
)
_SPEAKER_PREFIX = re.compile(r"^(?P<speaker>[^:：\n]{1,40})\s*[:：]\s*(?P<text>.*)$")
_TRAILING_SPEAKER_TIME = re.compile(
    rf"^(?P<speaker>.+?)\s+(?P<start>{_TIME_TOKEN})$"
)


def _segment(
    *,
    index: int,
    episode_id: str,
    text: str,
    source_file: str,
    speaker: Any = None,
    section_title: Any = None,
    start_time: Any = None,
    end_time: Any = None,
) -> TranscriptSegment | None:
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return None
    try:
        return TranscriptSegment(
            segment_id=f"{episode_id}:{index:06d}",
            segment_index=index,
            text=cleaned_text,
            source_file=source_file,
            speaker=clean_speaker(speaker),
            section_title=clean_speaker(section_title),
            start_time=parse_timestamp(start_time, field_name="start_time"),
            end_time=parse_timestamp(end_time, field_name="end_time"),
        )
    except (ValueError, ValidationError) as error:
        raise TranscriptFormatError(
            f"第 {index + 1} 段数据无效：{safe_error_message(error)}"
        ) from error


def _parse_txt_line(
    line: str,
    *,
    index: int,
    episode_id: str,
    source_file: str,
) -> TranscriptSegment | None:
    normalized = unicodedata.normalize("NFKC", line).strip()
    if not normalized:
        return None

    start_time: str | None = None
    end_time: str | None = None
    timestamp_match = _BRACKETED_TIME.match(normalized)
    if timestamp_match is None:
        timestamp_match = _PLAIN_TIME_RANGE.match(normalized)
    if timestamp_match is not None:
        start_time = timestamp_match.group("start")
        end_time = timestamp_match.group("end")
        normalized = normalized[timestamp_match.end() :].strip()

    speaker: str | None = None
    speaker_match = _SPEAKER_PREFIX.match(normalized)
    if speaker_match is not None:
        speaker = speaker_match.group("speaker")
        normalized = speaker_match.group("text")

    return _segment(
        index=index,
        episode_id=episode_id,
        text=normalized,
        source_file=source_file,
        speaker=speaker,
        start_time=start_time,
        end_time=end_time,
    )


def _load_txt(path: Path, episode_id: str) -> ProcessedTranscript:
    try:
        raw_text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        raise TranscriptFormatError("TXT 文件必须使用 UTF-8 编码") from error

    lines = raw_text.splitlines()
    trailing_headers = [
        (line_index, match)
        for line_index, line in enumerate(lines)
        if (match := _TRAILING_SPEAKER_TIME.fullmatch(line.strip())) is not None
    ]
    webpage_preamble = lines[: trailing_headers[0][0]] if trailing_headers else []
    has_webpage_marker = any(
        clean_text(line).casefold() == "transcript"
        or clean_text(line).casefold().startswith("edited by ")
        for line in webpage_preamble
    )
    if len(trailing_headers) >= 2 or (trailing_headers and has_webpage_marker):
        return _load_trailing_time_txt(
            path,
            episode_id,
            lines,
            trailing_headers,
        )

    segments: list[TranscriptSegment] = []
    for line in lines:
        segment = _parse_txt_line(
            line,
            index=len(segments),
            episode_id=episode_id,
            source_file=path.name,
        )
        if segment is not None:
            segments.append(segment)
    return _document(path, episode_id, "txt", segments)


def _initial_section_title(lines: list[str]) -> str | None:
    """Extract a useful section title from a copied webpage preamble."""
    candidates = []
    for line in lines:
        cleaned = clean_text(line)
        if not cleaned:
            continue
        lowered = cleaned.casefold()
        if lowered == "transcript" or lowered.startswith("edited by "):
            continue
        candidates.append(cleaned)
    return candidates[-1] if candidates else None


def _load_trailing_time_txt(
    path: Path,
    episode_id: str,
    lines: list[str],
    headers: list[tuple[int, re.Match[str]]],
) -> ProcessedTranscript:
    """Parse webpage text laid out as ``Speaker HH:MM:SS`` then body text."""
    segments: list[TranscriptSegment] = []
    current_section = _initial_section_title(lines[: headers[0][0]])

    for header_number, (line_index, match) in enumerate(headers):
        next_line_index = (
            headers[header_number + 1][0]
            if header_number + 1 < len(headers)
            else len(lines)
        )
        body_end = next_line_index
        next_section: str | None = None

        # On the source webpage, a chapter heading touches the following
        # speaker header, while normal dialogue is separated from it by a blank
        # line. Preserve that heading as metadata for subsequent dialogue.
        if (
            header_number + 1 < len(headers)
            and next_line_index - 1 > line_index
            and lines[next_line_index - 1].strip()
        ):
            next_section = clean_text(lines[next_line_index - 1])
            body_end -= 1

        body = "\n".join(
            line.strip()
            for line in lines[line_index + 1 : body_end]
            if line.strip()
        )
        segment = _segment(
            index=len(segments),
            episode_id=episode_id,
            text=body,
            source_file=path.name,
            speaker=match.group("speaker"),
            section_title=current_section,
            start_time=match.group("start"),
        )
        if segment is None:
            raise TranscriptFormatError(
                f"网页格式第 {header_number + 1} 个说话人后缺少正文"
            )
        segments.append(segment)
        if next_section:
            current_section = next_section

    return _document(path, episode_id, "txt", segments)


def _first_value(item: dict[str, Any], metadata: dict[str, Any], *names: str) -> Any:
    for container in (item, metadata):
        for name in names:
            if name in container:
                return container[name]
    return None


def _json_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise TranscriptFormatError("JSON 顶层必须是对象或数组")

    for key in ("segments", "utterances", "items", "transcript"):
        value = payload.get(key)
        if isinstance(value, list):
            return value

    if any(key in payload for key in ("text", "content", "transcript")):
        return [payload]
    raise TranscriptFormatError("JSON 中未找到 segments、utterances、items 或文本字段")


def _load_json(path: Path, episode_id: str) -> ProcessedTranscript:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except UnicodeDecodeError as error:
        raise TranscriptFormatError("JSON 文件必须使用 UTF-8 编码") from error
    except json.JSONDecodeError as error:
        raise TranscriptFormatError(
            f"JSON 格式错误（第 {error.lineno} 行，第 {error.colno} 列）"
        ) from error

    segments: list[TranscriptSegment] = []
    for item_number, item in enumerate(_json_items(payload), start=1):
        if isinstance(item, str):
            segment = _segment(
                index=len(segments),
                episode_id=episode_id,
                text=item,
                source_file=path.name,
            )
        elif isinstance(item, dict):
            metadata_value = item.get("metadata", {})
            if metadata_value is None:
                metadata_value = {}
            if not isinstance(metadata_value, dict):
                raise TranscriptFormatError(f"JSON 第 {item_number} 项的 metadata 必须是对象")
            text = _first_value(item, metadata_value, "text", "content", "transcript")
            if not isinstance(text, str):
                raise TranscriptFormatError(f"JSON 第 {item_number} 项缺少字符串文本字段")
            segment = _segment(
                index=len(segments),
                episode_id=episode_id,
                text=text,
                source_file=path.name,
                speaker=_first_value(item, metadata_value, "speaker", "speaker_name", "name"),
                section_title=_first_value(
                    item,
                    metadata_value,
                    "section_title",
                    "section",
                    "chapter",
                    "chapter_title",
                ),
                start_time=_first_value(
                    item, metadata_value, "start_time", "start", "startTime"
                ),
                end_time=_first_value(item, metadata_value, "end_time", "end", "endTime"),
            )
        else:
            raise TranscriptFormatError(f"JSON 第 {item_number} 项必须是对象或字符串")

        if segment is not None:
            segments.append(segment)
    return _document(path, episode_id, "json", segments)


def _document(
    path: Path,
    episode_id: str,
    source_format: Literal["txt", "json"],
    segments: list[TranscriptSegment],
) -> ProcessedTranscript:
    if not segments:
        raise TranscriptFormatError("清洗后没有可用的转写内容")
    return ProcessedTranscript(
        episode_id=episode_id,
        source_file=path.name,
        source_format=source_format,
        segments=segments,
    )


def load_transcript(path: str | Path) -> ProcessedTranscript:
    """Load and clean one UTF-8 TXT or JSON transcript."""
    source_path = Path(path)
    if not source_path.is_file():
        raise TranscriptFormatError(f"找不到输入文件：{source_path}")
    content_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()[:16]
    episode_id = f"ep_{content_digest}"
    suffix = source_path.suffix.lower()
    if suffix == ".txt":
        return _load_txt(source_path, episode_id)
    if suffix == ".json":
        return _load_json(source_path, episode_id)
    raise TranscriptFormatError("只支持 .txt 和 .json 文件")
