"""Offline loading and source validation for the additive v2 pilot."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from app.core.errors import safe_error_message
from app.evaluation.core import load_evaluation_dataset
from app.indexing.index import chunked_transcript_fingerprint
from app.schemas.chunk import ChunkedTranscript
from app.schemas.evaluation_v2 import EvaluationDatasetV2Pilot, PilotEpisodeDataset
from app.schemas.transcript import ProcessedTranscript


class EvaluationV2PilotError(ValueError):
    """Raised when the collection or any local source cannot be trusted."""


@dataclass(frozen=True)
class PilotSourceFiles:
    """Local ignored artifacts required to validate one public-source episode."""

    raw_transcript: Path
    processed_transcript: Path
    chunk_catalog: Path


def load_v2_pilot(path: str | Path) -> EvaluationDatasetV2Pilot:
    """Load the strict collection without network or environment access."""
    source = Path(path)
    if not source.is_file():
        raise EvaluationV2PilotError(f"找不到 v2 试点评测集：{source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        return EvaluationDatasetV2Pilot.model_validate(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
        raise EvaluationV2PilotError(
            f"v2 试点评测集校验失败：{safe_error_message(error)}"
        ) from error


def validate_v2_pilot_sources(
    dataset: EvaluationDatasetV2Pilot,
    *,
    base_dataset_path: str | Path,
    episode_sources: dict[str, PilotSourceFiles],
) -> dict[str, object]:
    """Cross-check hashes, stable IDs, metadata, and Chunk coverage offline."""
    base_path = Path(base_dataset_path)
    base = load_evaluation_dataset(base_path)
    if _sha256(base_path) != dataset.parent.artifact_sha256:
        raise EvaluationV2PilotError("v1 文件摘要与 v2 parent 引用不一致")
    if base.cases_sha256 != dataset.parent.cases_sha256:
        raise EvaluationV2PilotError("v1 案例摘要与 v2 parent 引用不一致")

    expected_slugs = {item.episode_slug for item in dataset.episode_datasets}
    if set(episode_sources) != expected_slugs:
        raise EvaluationV2PilotError("本地节目来源集合与 v2 声明不一致")

    episode_reports = []
    for episode in dataset.episode_datasets:
        episode_reports.append(
            _validate_episode(episode, episode_sources[episode.episode_slug])
        )
    return {
        "dataset_id": dataset.dataset_id,
        "expanded_case_count": dataset.expanded_case_count,
        "new_case_count": dataset.new_case_count,
        "base_dataset_unchanged": True,
        "provider_calls_during_validation": 0,
        "all_source_hashes_match": True,
        "all_source_segment_ids_exist": True,
        "all_expected_chunks_cover_truth": True,
        "all_source_metadata_match": True,
        "episodes": episode_reports,
    }


def _validate_episode(
    expected: PilotEpisodeDataset,
    paths: PilotSourceFiles,
) -> dict[str, object]:
    for path in (paths.raw_transcript, paths.processed_transcript, paths.chunk_catalog):
        if not path.is_file():
            raise EvaluationV2PilotError(f"缺少本地来源文件：{path}")
    hashes = (
        _sha256(paths.raw_transcript),
        _sha256(paths.processed_transcript),
        _sha256(paths.chunk_catalog),
    )
    expected_hashes = (
        expected.raw_transcript_sha256,
        expected.processed_transcript_sha256,
        expected.chunk_catalog_sha256,
    )
    if hashes != expected_hashes:
        raise EvaluationV2PilotError(f"{expected.episode_slug} 本地来源摘要不一致")

    try:
        raw = json.loads(paths.raw_transcript.read_text(encoding="utf-8"))
        transcript = ProcessedTranscript.model_validate_json(
            paths.processed_transcript.read_text(encoding="utf-8")
        )
        chunked = ChunkedTranscript.model_validate_json(
            paths.chunk_catalog.read_text(encoding="utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
        raise EvaluationV2PilotError(
            f"{expected.episode_slug} 本地来源格式错误"
        ) from error

    if raw.get("article_text_sha256") != expected.article_text_sha256:
        raise EvaluationV2PilotError(f"{expected.episode_slug} 网页正文摘要不一致")
    if raw.get("source_url") != expected.source_url:
        raise EvaluationV2PilotError(f"{expected.episode_slug} 来源 URL 不一致")
    if raw.get("timestamp_semantics") != expected.timestamp_semantics:
        raise EvaluationV2PilotError(f"{expected.episode_slug} 时间语义不一致")
    if transcript.episode_id != expected.episode_id or chunked.episode_id != expected.episode_id:
        raise EvaluationV2PilotError(f"{expected.episode_slug} episode_id 不一致")
    baseline = expected.chunk_baseline
    if (
        chunked.algorithm,
        chunked.chunk_size,
        chunked.overlap,
        chunked_transcript_fingerprint(chunked),
    ) != (
        baseline.algorithm,
        baseline.chunk_size,
        baseline.overlap,
        baseline.input_chunk_sha256,
    ):
        raise EvaluationV2PilotError(f"{expected.episode_slug} Chunk 基线不一致")

    segments = {item.segment_id: item for item in transcript.segments}
    chunks = {item.chunk_id: item for item in chunked.chunks}
    for case in expected.cases:
        if not case.answerable:
            continue
        source_segments = []
        for source_id in case.expected_source_segment_ids:
            if source_id not in segments:
                raise EvaluationV2PilotError(
                    f"{case.case_id} 引用了不存在的来源片段：{source_id}"
                )
            source_segments.append(segments[source_id])
        expected_chunks = []
        for chunk_id in case.expected_chunk_ids:
            if chunk_id not in chunks:
                raise EvaluationV2PilotError(
                    f"{case.case_id} 引用了不存在的 Chunk：{chunk_id}"
                )
            expected_chunks.append(chunks[chunk_id])
        for source_id in case.expected_source_segment_ids:
            if not any(source_id in chunk.source_segment_ids for chunk in expected_chunks):
                raise EvaluationV2PilotError(
                    f"{case.case_id} 的 Chunk 未覆盖来源片段：{source_id}"
                )
        for chunk in expected_chunks:
            if not set(chunk.source_segment_ids).intersection(
                case.expected_source_segment_ids
            ):
                raise EvaluationV2PilotError(
                    f"{case.case_id} 的 Chunk 与稳定来源没有交集：{chunk.chunk_id}"
                )
        if tuple(dict.fromkeys(item.speaker for item in source_segments)) != case.expected_speakers:
            raise EvaluationV2PilotError(f"{case.case_id} 说话人元数据不一致")
        sections = tuple(
            dict.fromkeys(
                item.section_title
                for item in source_segments
                if item.section_title is not None
            )
        )
        if sections != case.expected_section_titles:
            raise EvaluationV2PilotError(f"{case.case_id} 章节元数据不一致")
        if case.category == "timestamp" and not case.expected_time_ranges:
            raise EvaluationV2PilotError(f"{case.case_id} 缺少显式时间真值")
        for time_range in case.expected_time_ranges:
            segment = segments[time_range.source_segment_id]
            actual = (
                segment.start_time,
                segment.end_time,
                segment.start_time is not None and segment.end_time is not None,
            )
            annotated = (
                time_range.start_time,
                time_range.end_time,
                time_range.time_range_complete,
            )
            if actual != annotated:
                raise EvaluationV2PilotError(f"{case.case_id} 时间元数据不一致")

    return {
        "episode_slug": expected.episode_slug,
        "episode_id": expected.episode_id,
        "case_count": len(expected.cases),
        "segment_count": len(transcript.segments),
        "chunk_count": len(chunked.chunks),
        "timestamp_semantics": expected.timestamp_semantics,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
