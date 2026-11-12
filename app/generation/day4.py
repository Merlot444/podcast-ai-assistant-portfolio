"""Day 4 direct and map-reduce podcast content-analysis baseline."""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.core.errors import safe_error_message
from app.core.llm import create_json_completion
from app.schemas.day4 import (
    AnalysisMetrics,
    BatchSummary,
    ChapterOverview,
    Day4AnalysisResult,
    GeneratedAnalysisOutput,
    ModelCallMetric,
    SectionSummaryOutput,
)
from app.schemas.transcript import ProcessedTranscript, TranscriptSegment


DIRECT_SYSTEM_PROMPT = """【角色】
你是一名严谨的中文播客内容编辑。

【任务】
直接分析用户提供的短篇转写稿，生成摘要、关键词、章节概述和推荐文案。

【约束】
1. 只使用输入中的信息，不补充外部事实，不猜测。
2. summary 应准确概括主要观点，不超过 500 个汉字。
3. keywords 返回 3 到 8 个简洁字符串。
4. chapter_overviews 必须覆盖用户列出的全部章节，section_title 必须原样复制。
5. 每个 overview 不超过 300 个汉字。
6. recommendation 是不夸大事实的中文推荐文案，不超过 180 个汉字。
7. 不要输出解释、前缀或 Markdown 代码块。

【输出格式】
只返回以下结构的合法 JSON 对象：
{
  "summary": "总体摘要",
  "keywords": ["关键词1", "关键词2", "关键词3"],
  "chapter_overviews": [
    {"section_title": "输入中的章节标题", "overview": "章节概述"}
  ],
  "recommendation": "推荐文案"
}
"""


SEGMENT_SUMMARY_SYSTEM_PROMPT = """【角色】
你是一名严谨的中文播客内容编辑。

【任务】
总结一批来自同一章节的连续播客片段，保留关键论点、条件和不确定性。

【约束】
1. 只使用输入中的信息，不补充外部事实，不猜测。
2. summary 不超过 300 个汉字。
3. key_points 返回 2 到 5 条具体要点。
4. 不要把提问者的问题误写成嘉宾已经确认的观点。
5. 不要输出解释、前缀或 Markdown 代码块。

【输出格式】
只返回以下结构的合法 JSON 对象：
{"summary": "分段摘要", "key_points": ["要点1", "要点2"]}
"""


SYNTHESIS_SYSTEM_PROMPT = """【角色】
你是一名严谨的中文播客主编。

【任务】
仅根据用户提供的分段摘要，汇总整期播客的摘要、关键词、章节概述和推荐文案。

【约束】
1. 不使用原始分段摘要以外的事实，不猜测，不补充背景知识。
2. 区分受访者观点、主持人问题和仍带不确定性的判断。
3. summary 不超过 500 个汉字，keywords 返回 3 到 8 个字符串。
4. chapter_overviews 必须覆盖用户列出的全部章节，section_title 必须原样复制。
5. 每个 overview 不超过 300 个汉字。
6. recommendation 不超过 180 个汉字，不使用标题党或夸大表述。
7. 不要输出解释、前缀或 Markdown 代码块。

【输出格式】
只返回以下结构的合法 JSON 对象：
{
  "summary": "整期汇总摘要",
  "keywords": ["关键词1", "关键词2", "关键词3"],
  "chapter_overviews": [
    {"section_title": "输入中的章节标题", "overview": "章节概述"}
  ],
  "recommendation": "推荐文案"
}
"""


class Day4AnalysisError(RuntimeError):
    """Raised when Day 4 input or model output cannot be validated."""


@dataclass(frozen=True)
class AnalysisBatch:
    """A bounded, chapter-aware batch used only for summarization."""

    section_title: str
    batch_index: int
    batch_count: int
    segments: tuple[TranscriptSegment, ...]
    rendered_text: str


@dataclass(frozen=True)
class _PartialResult:
    batch: AnalysisBatch
    output: SectionSummaryOutput


OutputModel = TypeVar("OutputModel", bound=BaseModel)
Clock = Callable[[], float]
_UNTITLED_SECTION = "未分章"


def load_processed_transcript(path: str | Path) -> ProcessedTranscript:
    """Read and validate one Day 3 processed transcript JSON file."""
    source = Path(path)
    if not source.is_file():
        raise Day4AnalysisError(f"找不到处理后的转写文件：{source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise Day4AnalysisError("处理后的转写文件必须使用 UTF-8 编码。") from error
    except json.JSONDecodeError as error:
        raise Day4AnalysisError(
            f"处理后的转写 JSON 格式错误（第 {error.lineno} 行，第 {error.colno} 列）。"
        ) from error
    try:
        return ProcessedTranscript.model_validate(payload)
    except ValidationError as error:
        raise Day4AnalysisError(
            f"处理后的转写数据校验失败：{safe_error_message(error)}"
        ) from error


def _format_timestamp(seconds: float | None) -> str:
    if seconds is None:
        return "--:--:--"
    rounded = int(seconds)
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def render_segment(segment: TranscriptSegment) -> str:
    """Render one segment without discarding citation metadata."""
    start = _format_timestamp(segment.start_time)
    end = _format_timestamp(segment.end_time)
    time_text = start if segment.end_time is None else f"{start}-{end}"
    speaker = segment.speaker or "未知说话人"
    return f"[{time_text}] [{segment.segment_id}] {speaker}: {segment.text}"


def _section_title(segment: TranscriptSegment) -> str:
    return segment.section_title or _UNTITLED_SECTION


def create_analysis_batches(
    segments: Sequence[TranscriptSegment],
    *,
    max_characters: int = 10_000,
) -> list[AnalysisBatch]:
    """Group consecutive segments by chapter and a simple character budget."""
    if not segments:
        raise Day4AnalysisError("待分析的转写片段不能为空。")
    if max_characters <= 0:
        raise ValueError("max_characters 必须大于 0。")

    raw_batches: list[tuple[str, tuple[TranscriptSegment, ...], str]] = []
    current_title: str | None = None
    current_segments: list[TranscriptSegment] = []
    current_lines: list[str] = []
    current_length = 0

    def flush() -> None:
        nonlocal current_segments, current_lines, current_length
        if current_title is not None and current_segments:
            raw_batches.append(
                (
                    current_title,
                    tuple(current_segments),
                    "\n".join(current_lines),
                )
            )
        current_segments = []
        current_lines = []
        current_length = 0

    for segment in segments:
        title = _section_title(segment)
        line = render_segment(segment)
        if len(line) > max_characters:
            raise Day4AnalysisError(
                f"片段 {segment.segment_id} 渲染后为 {len(line)} 个字符，"
                f"超过单批预算 {max_characters}。Day 4 不会拆开一个原始片段；"
                "请提高预算或先检查 Day 3 的片段粒度。"
            )
        separator_length = 1 if current_lines else 0
        exceeds_budget = (
            bool(current_lines)
            and current_length + separator_length + len(line) > max_characters
        )
        if current_lines and (title != current_title or exceeds_budget):
            flush()
        current_title = title
        current_segments.append(segment)
        current_lines.append(line)
        current_length += separator_length + len(line)
    flush()

    counts: dict[str, int] = {}
    for title, _, _ in raw_batches:
        counts[title] = counts.get(title, 0) + 1
    seen: dict[str, int] = {}
    batches: list[AnalysisBatch] = []
    for title, batch_segments, rendered_text in raw_batches:
        seen[title] = seen.get(title, 0) + 1
        batches.append(
            AnalysisBatch(
                section_title=title,
                batch_index=seen[title],
                batch_count=counts[title],
                segments=batch_segments,
                rendered_text=rendered_text,
            )
        )
    return batches


def _chapter_groups(
    segments: Sequence[TranscriptSegment],
) -> OrderedDict[str, list[TranscriptSegment]]:
    groups: OrderedDict[str, list[TranscriptSegment]] = OrderedDict()
    for segment in segments:
        groups.setdefault(_section_title(segment), []).append(segment)
    return groups


def _message_characters(messages: Sequence[ChatCompletionMessageParam]) -> int:
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += len(content)
    return total


def _parse_output(
    raw_content: str,
    output_model: type[OutputModel],
) -> OutputModel:
    if not raw_content.strip():
        raise Day4AnalysisError("模型返回了空内容。")
    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError:
        raise Day4AnalysisError("模型返回的内容不是合法 JSON。") from None
    try:
        return output_model.model_validate(payload)
    except ValidationError as error:
        raise Day4AnalysisError(
            f"模型输出不符合 Day 4 业务结构：{safe_error_message(error)}"
        ) from error


def _call_model(
    client: OpenAI,
    *,
    model: str,
    messages: list[ChatCompletionMessageParam],
    output_model: type[OutputModel],
    max_tokens: int,
    call_index: int,
    stage: str,
    timer: Clock,
    section_title: str | None = None,
    batch_index: int | None = None,
) -> tuple[OutputModel, ModelCallMetric]:
    started = timer()
    response = create_json_completion(
        client,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )
    elapsed = max(0.0, timer() - started)
    if not response.choices:
        raise Day4AnalysisError("模型响应中没有可用结果。")
    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise Day4AnalysisError("模型输出达到长度上限，JSON 可能被截断。")
    raw_content = choice.message.content or ""
    parsed = _parse_output(raw_content, output_model)
    metric = ModelCallMetric(
        call_index=call_index,
        stage=stage,
        model=model,
        section_title=section_title,
        batch_index=batch_index,
        prompt_characters=_message_characters(messages),
        output_characters=len(raw_content),
        elapsed_seconds=elapsed,
    )
    return parsed, metric


def _chapter_titles_text(
    groups: OrderedDict[str, list[TranscriptSegment]],
) -> str:
    return "\n".join(
        f"{index}. {title}" for index, title in enumerate(groups, start=1)
    )


def _attach_chapter_metadata(
    generated: GeneratedAnalysisOutput,
    groups: OrderedDict[str, list[TranscriptSegment]],
) -> list[ChapterOverview]:
    generated_by_title = {
        item.section_title: item.overview for item in generated.chapter_overviews
    }
    expected = list(groups)
    returned = list(generated_by_title)
    missing = [title for title in expected if title not in generated_by_title]
    extra = [title for title in returned if title not in groups]
    if missing or extra:
        details = []
        if missing:
            details.append("缺少章节：" + "、".join(missing))
        if extra:
            details.append("出现未知章节：" + "、".join(extra))
        raise Day4AnalysisError("模型没有原样保留章节标题；" + "；".join(details))

    overviews: list[ChapterOverview] = []
    for title, section_segments in groups.items():
        start_values = [
            segment.start_time
            for segment in section_segments
            if segment.start_time is not None
        ]
        end_values = [
            segment.end_time
            if segment.end_time is not None
            else segment.start_time
            for segment in section_segments
            if segment.end_time is not None or segment.start_time is not None
        ]
        overviews.append(
            ChapterOverview(
                section_title=title,
                overview=generated_by_title[title],
                source_segment_ids=[
                    segment.segment_id for segment in section_segments
                ],
                speakers=[
                    segment.speaker
                    for segment in section_segments
                    if segment.speaker is not None
                ],
                start_time=min(start_values) if start_values else None,
                end_time=max(end_values) if end_values else None,
            )
        )
    return overviews


def _coverage_time_range(
    segments: Sequence[TranscriptSegment],
) -> tuple[float | None, float | None]:
    """Return earliest start and latest known coverage timestamp.

    Most Day 3 webpage segments have no explicit ``end_time``. In that case the
    second value is the last segment's start time, not an invented speech end.
    """
    start_values = [
        segment.start_time for segment in segments if segment.start_time is not None
    ]
    end_values = [
        segment.end_time if segment.end_time is not None else segment.start_time
        for segment in segments
        if segment.end_time is not None or segment.start_time is not None
    ]
    return (
        min(start_values) if start_values else None,
        max(end_values) if end_values else None,
    )


def _saved_batch_summaries(
    partials: Sequence[_PartialResult],
) -> list[BatchSummary]:
    saved = []
    for partial in partials:
        batch = partial.batch
        start_time, end_time = _coverage_time_range(batch.segments)
        saved.append(
            BatchSummary(
                section_title=batch.section_title,
                batch_index=batch.batch_index,
                batch_count=batch.batch_count,
                summary=partial.output.summary,
                key_points=partial.output.key_points,
                source_segment_ids=[
                    segment.segment_id for segment in batch.segments
                ],
                speakers=[
                    segment.speaker
                    for segment in batch.segments
                    if segment.speaker is not None
                ],
                start_time=start_time,
                end_time=end_time,
            )
        )
    return saved


def _result(
    transcript: ProcessedTranscript,
    *,
    mode: str,
    generated: GeneratedAnalysisOutput,
    groups: OrderedDict[str, list[TranscriptSegment]],
    batch_count: int,
    partials: Sequence[_PartialResult],
    calls: list[ModelCallMetric],
    total_duration: float,
) -> Day4AnalysisResult:
    return Day4AnalysisResult(
        episode_id=transcript.episode_id,
        source_file=transcript.source_file,
        mode=mode,
        summary=generated.summary,
        keywords=generated.keywords,
        batch_summaries=_saved_batch_summaries(partials),
        chapter_overviews=_attach_chapter_metadata(generated, groups),
        recommendation=generated.recommendation,
        metrics=AnalysisMetrics(
            input_characters=sum(len(segment.text) for segment in transcript.segments),
            segment_count=len(transcript.segments),
            batch_count=batch_count,
            model_call_count=len(calls),
            output_characters=sum(call.output_characters for call in calls),
            calls=calls,
            total_duration_seconds=total_duration,
        ),
    )


def _direct_user_prompt(
    transcript: ProcessedTranscript,
    groups: OrderedDict[str, list[TranscriptSegment]],
) -> str:
    body = "\n".join(render_segment(segment) for segment in transcript.segments)
    return (
        "章节列表（section_title 必须逐字原样返回）：\n"
        f"{_chapter_titles_text(groups)}\n\n"
        "待分析转写稿：\n"
        f"{body}"
    )


def _segment_user_prompt(batch: AnalysisBatch) -> str:
    speakers = list(
        dict.fromkeys(
            segment.speaker
            for segment in batch.segments
            if segment.speaker is not None
        )
    )
    return (
        f"章节：{batch.section_title}\n"
        f"本章节分析批次：{batch.batch_index}/{batch.batch_count}\n"
        f"说话人：{', '.join(speakers) if speakers else '未知'}\n\n"
        "连续转写片段：\n"
        f"{batch.rendered_text}"
    )


def _synthesis_user_prompt(
    groups: OrderedDict[str, list[TranscriptSegment]],
    partials: Sequence[_PartialResult],
) -> str:
    blocks = []
    for partial in partials:
        batch = partial.batch
        key_points = "\n".join(
            f"- {point}" for point in partial.output.key_points
        )
        blocks.append(
            f"章节：{batch.section_title}\n"
            f"批次：{batch.batch_index}/{batch.batch_count}\n"
            f"分段摘要：{partial.output.summary}\n"
            f"关键点：\n{key_points}"
        )
    return (
        "章节列表（section_title 必须逐字原样返回，并保持顺序）：\n"
        f"{_chapter_titles_text(groups)}\n\n"
        "分段摘要：\n\n"
        + "\n\n".join(blocks)
    )


def analyze_processed_transcript(
    client: OpenAI,
    settings: Settings,
    transcript: ProcessedTranscript,
    *,
    direct_max_characters: int = 10_000,
    batch_max_characters: int = 10_000,
    clock: Clock | None = None,
) -> Day4AnalysisResult:
    """Analyze short content directly or long content with map-reduce."""
    if direct_max_characters <= 0:
        raise ValueError("direct_max_characters 必须大于 0。")
    if batch_max_characters <= 0:
        raise ValueError("batch_max_characters 必须大于 0。")
    timer = clock or time.perf_counter
    overall_started = timer()
    groups = _chapter_groups(transcript.segments)
    rendered_length = sum(
        len(render_segment(segment)) + 1 for segment in transcript.segments
    )

    if rendered_length <= direct_max_characters:
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
            {"role": "user", "content": _direct_user_prompt(transcript, groups)},
        ]
        generated, metric = _call_model(
            client,
            model=settings.model_strong,
            messages=messages,
            output_model=GeneratedAnalysisOutput,
            max_tokens=1_600,
            call_index=1,
            stage="direct",
            timer=timer,
        )
        total_duration = max(0.0, timer() - overall_started)
        return _result(
            transcript,
            mode="direct",
            generated=generated,
            groups=groups,
            batch_count=1,
            partials=[],
            calls=[metric],
            total_duration=total_duration,
        )

    batches = create_analysis_batches(
        transcript.segments,
        max_characters=batch_max_characters,
    )
    partials: list[_PartialResult] = []
    calls: list[ModelCallMetric] = []
    for batch in batches:
        messages = [
            {"role": "system", "content": SEGMENT_SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": _segment_user_prompt(batch)},
        ]
        partial, metric = _call_model(
            client,
            model=settings.model_fast,
            messages=messages,
            output_model=SectionSummaryOutput,
            max_tokens=700,
            call_index=len(calls) + 1,
            stage="segment_summary",
            timer=timer,
            section_title=batch.section_title,
            batch_index=batch.batch_index,
        )
        partials.append(_PartialResult(batch=batch, output=partial))
        calls.append(metric)

    synthesis_messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": _synthesis_user_prompt(groups, partials)},
    ]
    generated, metric = _call_model(
        client,
        model=settings.model_strong,
        messages=synthesis_messages,
        output_model=GeneratedAnalysisOutput,
        max_tokens=2_400,
        call_index=len(calls) + 1,
        stage="synthesis",
        timer=timer,
    )
    calls.append(metric)
    total_duration = max(0.0, timer() - overall_started)
    return _result(
        transcript,
        mode="map_reduce",
        generated=generated,
        groups=groups,
        batch_count=len(batches),
        partials=partials,
        calls=calls,
        total_duration=total_duration,
    )


def save_day4_analysis(
    result: Day4AnalysisResult,
    output_path: str | Path,
) -> Path:
    """Save a validated Day 4 result as readable UTF-8 JSON."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            result.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination
