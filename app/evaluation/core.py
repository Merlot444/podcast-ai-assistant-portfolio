"""Offline loading and source-integrity checks for eval_dataset_v1."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from app.core.errors import safe_error_message
from app.chunking.core import chunk_transcript
from app.indexing.index import chunked_transcript_fingerprint
from app.schemas.evaluation import (
    EvaluationDataset,
    EvaluationTimeRange,
    EvaluationValidationReport,
)
from app.schemas.index import VectorIndexManifest
from app.schemas.transcript import ProcessedTranscript, TranscriptSegment


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION_DATASET = (
    _PROJECT_ROOT / "data" / "evaluation" / "eval_dataset_v1.json"
)
DEFAULT_EVALUATION_TRANSCRIPT = (
    _PROJECT_ROOT / "data" / "processed" / "demis-hassabis.json"
)
DEFAULT_EVALUATION_INDEX_DIRECTORY = (
    _PROJECT_ROOT
    / "data"
    / "indexes"
    / "demis-hassabis-jina-v5-small-1024"
)
DEFAULT_RETRIEVAL_BASELINE = (
    _PROJECT_ROOT
    / "data"
    / "evaluation"
    / "eval_dataset_v1-vector-baseline.json"
)


class EvaluationDatasetError(ValueError):
    """Raised when the frozen dataset file cannot be trusted."""


class EvaluationSourceError(ValueError):
    """Raised when a human annotation does not match its real source."""


def load_evaluation_dataset(
    path: str | Path = DEFAULT_EVALUATION_DATASET,
) -> EvaluationDataset:
    """Load the strict, versioned JSON without network or environment access."""
    source = Path(path)
    if not source.is_file():
        raise EvaluationDatasetError(f"找不到 Day 11 评测集：{source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise EvaluationDatasetError("Day 11 评测集必须使用 UTF-8 编码") from error
    except json.JSONDecodeError as error:
        raise EvaluationDatasetError(
            f"Day 11 评测集 JSON 格式错误（第 {error.lineno} 行，第 {error.colno} 列）"
        ) from error
    try:
        return EvaluationDataset.model_validate(payload)
    except ValidationError as error:
        raise EvaluationDatasetError(
            f"Day 11 评测集校验失败：{safe_error_message(error)}"
        ) from error


def validate_evaluation_sources(
    dataset: EvaluationDataset,
    *,
    transcript: ProcessedTranscript,
    manifest: VectorIndexManifest,
) -> EvaluationValidationReport:
    """Cross-check every annotation and return non-metric integrity statistics."""
    _validate_dataset_level_sources(dataset, transcript, manifest)

    segment_by_id = _unique_segments(transcript)
    chunk_by_id = {chunk.chunk_id: chunk for chunk in manifest.chunks}

    for case in dataset.cases:
        if not case.answerable:
            continue

        segments = []
        for segment_id in case.expected_source_segment_ids:
            segment = segment_by_id.get(segment_id)
            if segment is None:
                raise EvaluationSourceError(
                    f"案例 {case.case_id} 引用了不存在的来源片段：{segment_id}"
                )
            segments.append(segment)

        chunks = []
        for chunk_id in case.expected_chunk_ids:
            chunk = chunk_by_id.get(chunk_id)
            if chunk is None:
                raise EvaluationSourceError(
                    f"案例 {case.case_id} 引用了不存在的基线 Chunk：{chunk_id}"
                )
            chunks.append(chunk)

        for segment_id in case.expected_source_segment_ids:
            if not any(segment_id in chunk.source_segment_ids for chunk in chunks):
                raise EvaluationSourceError(
                    f"案例 {case.case_id} 的基线 Chunk 未覆盖来源片段：{segment_id}"
                )
        for chunk in chunks:
            if not set(chunk.source_segment_ids).intersection(
                case.expected_source_segment_ids
            ):
                raise EvaluationSourceError(
                    f"案例 {case.case_id} 的基线 Chunk 与稳定来源无交集："
                    f"{chunk.chunk_id}"
                )

        actual_speakers = _ordered_metadata(segments, "speaker")
        if case.expected_speakers and actual_speakers != case.expected_speakers:
            raise EvaluationSourceError(
                f"案例 {case.case_id} 的说话人标注与来源片段不一致"
            )
        actual_sections = _ordered_metadata(segments, "section_title")
        if (
            case.expected_section_titles
            and actual_sections != case.expected_section_titles
        ):
            raise EvaluationSourceError(
                f"案例 {case.case_id} 的章节标注与来源片段不一致"
            )

        if case.expected_time_ranges:
            annotated_time_ids = {
                item.source_segment_id for item in case.expected_time_ranges
            }
            actual_time_ranges = tuple(
                EvaluationTimeRange(
                    source_segment_id=segment.segment_id,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    time_range_complete=(
                        segment.start_time is not None
                        and segment.end_time is not None
                    ),
                )
                for segment in segments
                if segment.segment_id in annotated_time_ids
            )
            if actual_time_ranges != case.expected_time_ranges:
                raise EvaluationSourceError(
                    f"案例 {case.case_id} 的时间范围标注与来源片段不一致"
                )

    _reject_embedded_full_chunk_text(dataset, manifest)

    category_counts = Counter(case.category for case in dataset.cases)
    answerable = [case for case in dataset.cases if case.answerable]
    return EvaluationValidationReport(
        dataset_id=dataset.dataset_id,
        schema_version=dataset.schema_version,
        episode_id=dataset.episode_id,
        cases_sha256=dataset.cases_sha256,
        total_cases=len(dataset.cases),
        category_counts={
            category: category_counts[category]
            for category in sorted(category_counts)
        },
        answerable_cases=len(answerable),
        unanswerable_cases=len(dataset.cases) - len(answerable),
        single_source_cases=sum(
            len(case.expected_source_segment_ids) == 1 for case in answerable
        ),
        multi_source_cases=sum(
            len(case.expected_source_segment_ids) > 1 for case in answerable
        ),
        timestamp_cases=category_counts["timestamp"],
        incomplete_time_range_cases=sum(
            any(not item.time_range_complete for item in case.expected_time_ranges)
            for case in answerable
        ),
        source_segment_annotations=sum(
            len(case.expected_source_segment_ids) for case in answerable
        ),
        baseline_chunk_annotations=sum(
            len(case.expected_chunk_ids) for case in answerable
        ),
        all_source_segment_ids_exist=True,
        all_baseline_chunk_ids_exist=True,
        episode_ids_match=True,
        all_provided_source_metadata_matches=True,
        baseline_configuration_matches=True,
    )


def _validate_dataset_level_sources(
    dataset: EvaluationDataset,
    transcript: ProcessedTranscript,
    manifest: VectorIndexManifest,
) -> None:
    if dataset.episode_id != transcript.episode_id:
        raise EvaluationSourceError("评测集与处理后转写稿的 episode_id 不一致")
    if dataset.episode_id != manifest.episode_id:
        raise EvaluationSourceError("评测集与索引 manifest 的 episode_id 不一致")
    if transcript.source_file != manifest.source_file:
        raise EvaluationSourceError("处理后转写稿与索引 manifest 的 source_file 不一致")
    if len(transcript.segments) != manifest.input_segment_count:
        raise EvaluationSourceError("处理后转写片段数量与索引 manifest 不一致")

    baseline = dataset.chunk_baseline
    actual_baseline = (
        manifest.chunking_algorithm,
        manifest.chunk_size,
        manifest.overlap,
        manifest.input_chunk_sha256,
    )
    expected_baseline = (
        baseline.algorithm,
        baseline.chunk_size,
        baseline.overlap,
        baseline.input_chunk_sha256,
    )
    if actual_baseline != expected_baseline:
        raise EvaluationSourceError(
            "expected_chunk_ids 只适用于数据集声明的当前 Chunk 基线，"
            "但索引 manifest 配置或内容指纹不匹配"
        )

    try:
        current_chunks = chunk_transcript(
            transcript,
            chunk_size=manifest.chunk_size,
            overlap=manifest.overlap,
        )
    except ValueError as error:
        raise EvaluationSourceError(
            "无法从当前处理后转写稿重建 Chunk 基线："
            f"{safe_error_message(error)}"
        ) from error
    current_fingerprint = chunked_transcript_fingerprint(current_chunks)
    if current_fingerprint != manifest.input_chunk_sha256:
        raise EvaluationSourceError(
            "当前处理后转写稿重建的 Chunk 内容指纹与索引 manifest 不一致"
        )


def _unique_segments(
    transcript: ProcessedTranscript,
) -> dict[str, TranscriptSegment]:
    segment_by_id = {segment.segment_id: segment for segment in transcript.segments}
    if len(segment_by_id) != len(transcript.segments):
        raise EvaluationSourceError("处理后转写稿包含重复 segment_id")
    return segment_by_id


def _ordered_metadata(
    segments: list[TranscriptSegment],
    attribute: Literal["speaker", "section_title"],
) -> tuple[str, ...]:
    values = (
        getattr(segment, attribute)
        for segment in segments
        if getattr(segment, attribute) is not None
    )
    return tuple(dict.fromkeys(values))


def _reject_embedded_full_chunk_text(
    dataset: EvaluationDataset,
    manifest: VectorIndexManifest,
) -> None:
    annotation_texts = [
        _normalized_text(text)
        for case in dataset.cases
        for text in (case.question, case.reference_answer, case.notes)
        if text is not None
    ]
    for chunk in manifest.chunks:
        chunk_text = _normalized_text(chunk.text)
        if chunk_text and any(chunk_text in text for text in annotation_texts):
            raise EvaluationSourceError(
                f"案例元数据不能保存完整 Chunk 正文：{chunk.chunk_id}"
            )


def _normalized_text(value: str) -> str:
    return " ".join(value.split())
