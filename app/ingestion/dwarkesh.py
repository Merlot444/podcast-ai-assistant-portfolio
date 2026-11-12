"""Extract public Dwarkesh article text into the existing transcript boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.core.errors import safe_error_message
from app.ingestion.cleaning import clean_text, parse_timestamp
from app.ingestion.transcripts import TranscriptFormatError


_CHAPTER_LINE = re.compile(
    r"^\(?(?P<time>\d{2}:\d{2}:\d{2})\)?\s*[-–—]\s*(?P<title>.+)$"
)
_SPEAKER_TIME_LINE = re.compile(
    r"^(?P<speaker>[^\n]{1,80}?)\s+(?P<time>\d{2}:\d{2}:\d{2})$"
)
_SOCIAL_COUNT_LINE = re.compile(r"^\d[\d,]*\s+(?:Likes?|Restacks?)$")
_NAME_WORD = re.compile(r"^[^\W\d_]+(?:[-'][^\W\d_]+)*$", re.UNICODE)
_NAME_PARTICLES = frozenset({"al", "bin", "da", "de", "del", "di", "la", "van", "von"})


class DwarkeshExtractionError(ValueError):
    """Raised when visible article text cannot be safely interpreted."""


def extract_dwarkesh_article(
    article_text: str,
    *,
    source_url: str,
) -> dict[str, Any]:
    """Return a metadata-bearing raw JSON payload with no webpage chrome."""
    _validate_source_url(source_url)
    lines = [unicodedata.normalize("NFKC", line).strip() for line in article_text.splitlines()]
    transcript_index, boundary_index = _transcript_bounds(lines)
    chapters = _chapters(lines[:transcript_index])
    if not chapters:
        raise DwarkeshExtractionError("页面没有可验证的章节时间戳")
    body = _trim_social_suffix(lines[transcript_index + 1 : boundary_index])
    timed_headers = sum(_SPEAKER_TIME_LINE.fullmatch(line) is not None for line in body)
    if timed_headers >= 2:
        segments = _parse_timed_body(body, chapters)
        timestamp_semantics = "speaker_turn"
    else:
        segments = _parse_speaker_block_body(body, chapters)
        timestamp_semantics = "chapter_start_only"
    _validate_segments(segments)
    canonical = json.dumps(
        segments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "schema_version": 1,
        "source_type": "dwarkesh_public_transcript",
        "source_url": source_url,
        "article_text_sha256": hashlib.sha256(article_text.encode("utf-8")).hexdigest(),
        "transcript_segments_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "timestamp_semantics": timestamp_semantics,
        "identified_speakers": sorted(
            {segment["speaker"] for segment in segments}, key=str.casefold
        ),
        "segments": segments,
    }


def save_extracted_transcript(
    payload: dict[str, Any],
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically save extracted raw JSON and refuse silent replacement."""
    destination = Path(output_path)
    if destination.exists() and not overwrite:
        raise DwarkeshExtractionError(
            f"目标转写已存在：{destination}；不会静默覆盖"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, destination)
    except OSError as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise DwarkeshExtractionError("转写提取结果写入失败") from error
    return destination


def _validate_source_url(source_url: str) -> None:
    parsed = urlparse(source_url)
    if (
        parsed.scheme != "https"
        or parsed.netloc.casefold() not in {"dwarkesh.com", "www.dwarkesh.com"}
        or not parsed.path.startswith("/p/")
    ):
        raise DwarkeshExtractionError("source_url 必须是公开的 Dwarkesh 节目页面")


def _transcript_bounds(lines: list[str]) -> tuple[int, int]:
    boundaries = [
        index for index, line in enumerate(lines) if line == "Discussion about this video"
    ]
    boundary = boundaries[0] if boundaries else len(lines)
    markers = [
        index for index, line in enumerate(lines[:boundary]) if line == "Transcript"
    ]
    if not markers:
        raise DwarkeshExtractionError("页面中未找到 Transcript 正文边界")
    return markers[-1], boundary


def _chapters(lines: list[str]) -> tuple[tuple[str, float], ...]:
    timestamp_markers = [index for index, line in enumerate(lines) if line == "Timestamps"]
    if not timestamp_markers:
        return ()
    result: list[tuple[str, float]] = []
    for line in lines[timestamp_markers[-1] + 1 :]:
        match = _CHAPTER_LINE.fullmatch(line)
        if match is None:
            continue
        title = _strip_quotes(clean_text(match.group("title")))
        seconds = parse_timestamp(match.group("time"), field_name="chapter_time")
        if seconds is None:
            raise DwarkeshExtractionError("章节时间戳不能为空")
        result.append((title, seconds))
    if len(result) != len({title.casefold() for title, _ in result}):
        raise DwarkeshExtractionError("章节标题必须唯一")
    return tuple(result)


def _trim_social_suffix(lines: list[str]) -> list[str]:
    trimmed = list(lines)
    while trimmed and (not trimmed[-1] or trimmed[-1] == "∙" or _SOCIAL_COUNT_LINE.fullmatch(trimmed[-1])):
        trimmed.pop()
    return trimmed


def _parse_timed_body(
    lines: list[str],
    chapters: tuple[tuple[str, float], ...],
) -> list[dict[str, Any]]:
    chapter_by_time = {seconds: title for title, seconds in chapters}
    current_section: str | None = None
    current_speaker: str | None = None
    current_start: float | None = None
    body: list[str] = []
    segments: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal body
        if current_speaker is None:
            body = []
            return
        text = clean_text(" ".join(part for part in body if part))
        if not text:
            raise DwarkeshExtractionError(f"说话人 {current_speaker} 后没有正文")
        segments.append(
            {
                "speaker": current_speaker,
                "section_title": current_section,
                "start_time": current_start,
                "end_time": None,
                "text": text,
            }
        )
        body = []

    for line in lines:
        chapter_match = _CHAPTER_LINE.fullmatch(line)
        if chapter_match is not None:
            flush()
            current_speaker = None
            current_start = None
            chapter_time = parse_timestamp(
                chapter_match.group("time"), field_name="chapter_time"
            )
            if chapter_time not in chapter_by_time:
                raise DwarkeshExtractionError("正文章节时间与 Timestamps 清单不一致")
            current_section = chapter_by_time[chapter_time]
            continue
        speaker_match = _SPEAKER_TIME_LINE.fullmatch(line)
        if speaker_match is not None:
            flush()
            candidate = clean_text(speaker_match.group("speaker"))
            if not _looks_like_speaker(candidate):
                raise DwarkeshExtractionError(
                    "逐轮时间头中的说话人名称形状无效"
                )
            current_speaker = candidate
            current_start = parse_timestamp(
                speaker_match.group("time"), field_name="speaker_time"
            )
            continue
        if current_speaker is not None:
            body.append(line)
    flush()
    return segments


def _parse_speaker_block_body(
    lines: list[str],
    chapters: tuple[tuple[str, float], ...],
) -> list[dict[str, Any]]:
    counts = Counter(line for line in lines if line)
    chapter_titles = {title for title, _ in chapters}
    speakers = {
        line
        for line, count in counts.items()
        if count >= 3 and _looks_like_speaker(line) and not _match_chapter(line, chapters)
    }
    if len(speakers) < 2:
        raise DwarkeshExtractionError("无法从无逐轮时间格式中确定至少两位说话人")

    current_section: str | None = None
    current_chapter_start: float | None = None
    chapter_start_available = False
    current_speaker: str | None = None
    body: list[str] = []
    segments: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal body, chapter_start_available
        if current_speaker is None:
            body = []
            return
        text = clean_text(" ".join(part for part in body if part))
        if not text:
            raise DwarkeshExtractionError(f"说话人 {current_speaker} 后没有正文")
        segments.append(
            {
                "speaker": current_speaker,
                "section_title": current_section,
                "start_time": current_chapter_start if chapter_start_available else None,
                "end_time": None,
                "text": text,
            }
        )
        chapter_start_available = False
        body = []

    for line in lines:
        chapter = _match_chapter(line, chapters)
        if chapter is not None:
            flush()
            current_speaker = None
            current_section, current_chapter_start = chapter
            chapter_start_available = True
            continue
        if line in speakers:
            flush()
            current_speaker = line
            continue
        if current_speaker is not None:
            body.append(line)
    flush()
    if set(segment["speaker"] for segment in segments) != speakers:
        raise DwarkeshExtractionError("并非所有识别出的说话人都产生了正文片段")
    if not chapter_titles.issuperset(
        segment["section_title"] for segment in segments if segment["section_title"]
    ):
        raise DwarkeshExtractionError("正文包含未知章节")
    return segments


def _looks_like_speaker(value: str) -> bool:
    words = value.split()
    if not 2 <= len(words) <= 6 or len(value) > 80:
        return False
    if any(character in value for character in ".,!?;:。，！？；："):
        return False
    for word in words:
        if not _NAME_WORD.fullmatch(word):
            return False
        if word.casefold() in _NAME_PARTICLES:
            continue
        parts = re.split(r"[-']", word)
        if not all(
            part.isupper()
            or (part[0].isupper() and (len(part) == 1 or part[1:].islower()))
            for part in parts
        ):
            return False
    return True


def _match_chapter(
    value: str,
    chapters: tuple[tuple[str, float], ...],
) -> tuple[str, float] | None:
    if not value or len(value) > 120 or len(value.split()) > 14:
        return None
    normalized = _normalize_title(value)
    best: tuple[float, str, float] | None = None
    for title, seconds in chapters:
        ratio = SequenceMatcher(None, normalized, _normalize_title(title)).ratio()
        if best is None or ratio > best[0]:
            best = (ratio, title, seconds)
    if best is not None and best[0] >= 0.88:
        return best[1], best[2]
    return None


def _normalize_title(value: str) -> str:
    stripped = _strip_quotes(clean_text(value)).casefold().replace("&", "and")
    return " ".join(re.findall(r"[a-z0-9]+", stripped))


def _strip_quotes(value: str) -> str:
    return value.strip().strip("\"'“”‘’")


def _validate_segments(segments: list[dict[str, Any]]) -> None:
    if len(segments) < 4:
        raise DwarkeshExtractionError("提取后的正文片段过少")
    speakers = {segment["speaker"] for segment in segments}
    sections = {segment["section_title"] for segment in segments if segment["section_title"]}
    if len(speakers) < 2 or len(sections) < 2:
        raise DwarkeshExtractionError("提取结果缺少足够的说话人或章节覆盖")
    if any(not segment["text"] for segment in segments):
        raise DwarkeshExtractionError("提取结果包含空正文")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从 Dwarkesh 公开节目页面的可见 article 文本提取转写 JSON。"
    )
    parser.add_argument("input", type=Path, help="UTF-8 article 可见文本")
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        article_text = args.input.read_text(encoding="utf-8")
        payload = extract_dwarkesh_article(article_text, source_url=args.source_url)
        destination = save_extracted_transcript(
            payload,
            args.output,
            overwrite=args.overwrite,
        )
    except (OSError, UnicodeDecodeError, DwarkeshExtractionError, TranscriptFormatError) as error:
        print(f"提取失败：{safe_error_message(error)}", file=sys.stderr)
        raise SystemExit(1) from None
    print(f"提取完成：{len(payload['segments'])} 个片段")
    print(f"时间语义：{payload['timestamp_semantics']}")
    print(f"输出文件：{destination}")


if __name__ == "__main__":
    main()
