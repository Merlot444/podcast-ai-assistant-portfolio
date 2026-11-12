"""Day 13 controlled Chunk-parameter experiment orchestration and comparison."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from pydantic import ValidationError

from app.core.errors import safe_error_message
from app.chunking.core import chunk_transcript
from app.chunking.pipeline import load_processed_transcript, save_chunked_transcript
from app.core.config import EmbeddingSettings
from app.evaluation.baseline import file_sha256
from app.evaluation.core import (
    DEFAULT_EVALUATION_DATASET,
    DEFAULT_EVALUATION_INDEX_DIRECTORY,
    DEFAULT_EVALUATION_TRANSCRIPT,
    DEFAULT_RETRIEVAL_BASELINE,
    load_evaluation_dataset,
    validate_evaluation_sources,
)
from app.evaluation.runner import DEFAULT_K_VALUES, run_retrieval_evaluation
from app.indexing.embedding import OpenAICompatibleEmbedder, create_embedding_client
from app.indexing.index import (
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    build_vector_index,
    load_vector_index,
    save_vector_index,
)
from app.retrieval import Retriever
from app.retrieval.vector import VectorRetrieverAdapter
from app.schemas.chunk import ChunkedTranscript
from app.schemas.chunk_experiment import (
    CaseConfigurationReview,
    ChunkExperimentComparisonRow,
    ChunkExperimentConfigurationResult,
    ChunkExperimentControlledVariables,
    ChunkExperimentDelta,
    ChunkExperimentFailureReview,
    ChunkExperimentInputHashes,
    ChunkExperimentSelectedDefault,
    ChunkExperimentSelectionPolicy,
    ChunkParameterExperimentResult,
    MetricDeltaAtK,
    SourceDiversityDiagnostics,
)
from app.schemas.evaluation import EvaluationDataset
from app.schemas.evaluation_result import (
    EvaluationCaseResult,
    RetrievalBaselineResult,
    RetrievalEvaluationRun,
    VectorRetrieverConfiguration,
)
from app.schemas.index import VectorIndexManifest
from app.schemas.transcript import ProcessedTranscript


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNK_EXPERIMENT_OUTPUT = (
    _PROJECT_ROOT
    / "data"
    / "evaluation"
    / "eval_dataset_v1-chunk-experiment-day13.json"
)
BASELINE_BUILD_ELAPSED_SECONDS = 4.533502
BASELINE_DOCUMENT_EMBEDDING_CALLS = 5
FOCUS_CASE_IDS = (
    "gemini_scaling_limits",
    "alignment_safeguards_pause_trigger",
    "agi_architecture_original_bets",
)


class ChunkExperimentError(RuntimeError):
    """Raised when a complete and trustworthy Day 13 result cannot be produced."""


class ChunkExperimentPersistenceError(RuntimeError):
    """Raised when the formal result cannot be saved without unsafe replacement."""


@dataclass(frozen=True)
class ChunkExperimentSpec:
    config_id: str
    chunk_size: int
    overlap: int
    chunk_file: Path | None
    index_directory: Path


DEFAULT_CHUNK_EXPERIMENT_SPECS = (
    ChunkExperimentSpec(
        "small",
        600,
        120,
        _PROJECT_ROOT / "data" / "processed" / "demis-hassabis-chunks-cs600-o120.json",
        _PROJECT_ROOT
        / "data"
        / "indexes"
        / "demis-hassabis-jina-v5-small-1024-cs600-o120",
    ),
    ChunkExperimentSpec(
        "baseline",
        1000,
        200,
        None,
        DEFAULT_EVALUATION_INDEX_DIRECTORY,
    ),
    ChunkExperimentSpec(
        "large",
        1400,
        280,
        _PROJECT_ROOT / "data" / "processed" / "demis-hassabis-chunks-cs1400-o280.json",
        _PROJECT_ROOT
        / "data"
        / "indexes"
        / "demis-hassabis-jina-v5-small-1024-cs1400-o280",
    ),
)


SELECTION_POLICY = ChunkExperimentSelectionPolicy(
    minimum_hit_at_3=1.0,
    ordered_quality_criteria=(
        "Hit@3 must not fall below the Day 12 baseline",
        "maximize overall Macro Recall@8",
        "maximize multi_segment Macro Recall@8",
        "maximize Hit@1",
        "maximize overall Macro Recall@3",
        "minimize newly regressed Hit@1 or Hit@3 cases",
    ),
    ordered_tiebreakers=(
        "fewer Chunks and vectors",
        "lower build elapsed time",
        "lower query elapsed time",
        "fewer Top-8 candidates sharing an already retrieved source segment",
    ),
    small_sample_warning=(
        "The frozen set has only 16 scored cases and four cases per category; "
        "the selected configuration is only the current default for this dataset."
    ),
    overlap_causal_claim_allowed=False,
)


def validate_chunk_experiment_matrix(
    specs: tuple[ChunkExperimentSpec, ...] = DEFAULT_CHUNK_EXPERIMENT_SPECS,
) -> None:
    """Lock the three configurations before any provider result is observed."""
    identities = tuple((item.config_id, item.chunk_size, item.overlap) for item in specs)
    expected = (
        ("small", 600, 120),
        ("baseline", 1000, 200),
        ("large", 1400, 280),
    )
    if identities != expected:
        raise ChunkExperimentError(
            "Day 13 参数矩阵必须固定为 small=600/120、baseline=1000/200、"
            "large=1400/280，且顺序不能改变"
        )
    if len({item.config_id for item in specs}) != 3:
        raise ChunkExperimentError("Chunk 实验 config_id 必须唯一")
    if any(item.overlap / item.chunk_size != 0.2 for item in specs):
        raise ChunkExperimentError("三组配置必须保持 20% overlap 比例")


def load_retrieval_baseline(path: str | Path) -> RetrievalBaselineResult:
    source = Path(path)
    if not source.is_file():
        raise ChunkExperimentError(f"找不到 Day 12 正式基线：{source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        return RetrievalBaselineResult.model_validate(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
        raise ChunkExperimentError(
            "Day 12 正式基线无法通过严格校验："
            f"{safe_error_message(error)}"
        ) from error


def validate_alternative_chunk_provenance(
    dataset: EvaluationDataset,
    *,
    transcript: ProcessedTranscript,
    chunked: ChunkedTranscript,
) -> None:
    """Validate stable segment provenance without reading expected_chunk_ids."""
    if dataset.episode_id != transcript.episode_id or dataset.episode_id != chunked.episode_id:
        raise ChunkExperimentError("alternative Chunk、评测集和转写稿 episode_id 不一致")
    if transcript.source_file != chunked.source_file:
        raise ChunkExperimentError("alternative Chunk 与转写稿 source_file 不一致")
    if len(transcript.segments) != chunked.input_segment_count:
        raise ChunkExperimentError("alternative Chunk 的输入 segment 数量不一致")

    segment_by_id = {item.segment_id: item for item in transcript.segments}
    expected_segment_ids = {
        segment_id
        for case in dataset.cases
        for segment_id in case.expected_source_segment_ids
    }
    if not expected_segment_ids.issubset(segment_by_id):
        raise ChunkExperimentError("冻结评测集包含转写稿中不存在的稳定 segment 真值")

    covered_segment_ids: set[str] = set()
    for chunk in chunked.chunks:
        for span in chunk.source_spans:
            segment = segment_by_id.get(span.segment_id)
            if segment is None:
                raise ChunkExperimentError(
                    f"alternative Chunk 引用了不存在的 segment：{span.segment_id}"
                )
            if segment.segment_index != span.segment_index:
                raise ChunkExperimentError("alternative Chunk 的 segment index 映射不一致")
            if span.end_char > len(segment.text):
                raise ChunkExperimentError("alternative Chunk 的 source span 越过原始片段")
            covered_segment_ids.add(span.segment_id)
    if covered_segment_ids != set(segment_by_id):
        raise ChunkExperimentError("alternative Chunk 未完整保留全部稳定 segment provenance")


def run_chunk_parameter_experiment(
    *,
    settings: EmbeddingSettings,
    dataset_path: str | Path = DEFAULT_EVALUATION_DATASET,
    transcript_path: str | Path = DEFAULT_EVALUATION_TRANSCRIPT,
    baseline_index_directory: str | Path = DEFAULT_EVALUATION_INDEX_DIRECTORY,
    baseline_result_path: str | Path = DEFAULT_RETRIEVAL_BASELINE,
    output_path: str | Path = DEFAULT_CHUNK_EXPERIMENT_OUTPUT,
    specs: tuple[ChunkExperimentSpec, ...] = DEFAULT_CHUNK_EXPERIMENT_SPECS,
    overwrite: bool = False,
) -> ChunkParameterExperimentResult:
    """Build two new indexes, run 40 live queries, compare, and save atomically."""
    validate_chunk_experiment_matrix(specs)
    started = perf_counter()
    dataset_path = Path(dataset_path)
    transcript_path = Path(transcript_path)
    baseline_index_directory = Path(baseline_index_directory)
    baseline_result_path = Path(baseline_result_path)
    output_path = Path(output_path)
    manifest_path = baseline_index_directory / MANIFEST_FILENAME
    faiss_path = baseline_index_directory / INDEX_FILENAME
    protected = {
        item.resolve()
        for item in (
            dataset_path,
            transcript_path,
            manifest_path,
            faiss_path,
            baseline_result_path,
        )
    }
    if output_path.resolve() in protected:
        raise ChunkExperimentPersistenceError("实验结果不能覆盖任何冻结输入产物")
    if output_path.exists() and not overwrite:
        raise ChunkExperimentPersistenceError(
            f"目标实验结果已存在：{output_path}；不会静默覆盖"
        )
    _preflight_alternative_destinations(specs)

    before_hashes = _input_hashes(
        dataset_path,
        transcript_path,
        manifest_path,
        faiss_path,
        baseline_result_path,
    )
    dataset = load_evaluation_dataset(dataset_path)
    transcript = load_processed_transcript(transcript_path)
    baseline_index = load_vector_index(baseline_index_directory)
    validate_evaluation_sources(
        dataset,
        transcript=transcript,
        manifest=baseline_index.manifest,
    )
    baseline = load_retrieval_baseline(baseline_result_path)
    _validate_baseline_reference(dataset, baseline_index.manifest, baseline, settings)

    configurations: list[ChunkExperimentConfigurationResult] = []
    baseline_configuration = _baseline_configuration(
        baseline,
        manifest=baseline_index.manifest,
        manifest_path=manifest_path,
    )
    alternative_runs: dict[str, ChunkExperimentConfigurationResult] = {}
    for spec in (specs[0], specs[2]):
        alternative_runs[spec.config_id] = _run_alternative_configuration(
            spec,
            dataset=dataset,
            transcript=transcript,
            settings=settings,
            baseline=baseline_configuration,
        )
    configurations.extend(
        (
            alternative_runs["small"],
            baseline_configuration,
            alternative_runs["large"],
        )
    )

    after_hashes = _input_hashes(
        dataset_path,
        transcript_path,
        manifest_path,
        faiss_path,
        baseline_result_path,
    )
    if after_hashes != before_hashes:
        raise ChunkExperimentError("正式实验前后冻结输入产物哈希发生变化，拒绝保存")

    comparison, selected = compare_and_select(tuple(configurations))
    timestamp = datetime.now(UTC)
    result = ChunkParameterExperimentResult(
        schema_version=1,
        experiment_id=(
            f"{dataset.dataset_id}-chunk-day13-"
            f"{timestamp.strftime('%Y%m%dt%H%M%Sz').lower()}"
        ),
        created_at_utc=timestamp,
        dataset_id=dataset.dataset_id,
        dataset_cases_sha256=dataset.cases_sha256,
        episode_id=dataset.episode_id,
        embedding_model=baseline.embedding_model,
        distance_metric="cosine",
        k_values=DEFAULT_K_VALUES,
        input_artifact_hashes=before_hashes,
        controlled_variables=_controlled_variables(
            transcript_sha256=before_hashes.transcript_file_sha256,
            manifest=baseline_index.manifest,
            dataset=dataset,
        ),
        selection_policy=SELECTION_POLICY,
        baseline_evaluation_id=baseline.evaluation_id,
        configurations=tuple(configurations),
        comparison=comparison,
        selected_default=selected,
        failure_reviews=build_failure_reviews(tuple(configurations)),
        elapsed_seconds=perf_counter() - started,
    )
    save_chunk_experiment(result, output_path, overwrite=overwrite)
    return result


def _run_alternative_configuration(
    spec: ChunkExperimentSpec,
    *,
    dataset: EvaluationDataset,
    transcript: ProcessedTranscript,
    settings: EmbeddingSettings,
    baseline: ChunkExperimentConfigurationResult,
) -> ChunkExperimentConfigurationResult:
    if spec.chunk_file is None:
        raise ChunkExperimentError("alternative 配置必须有独立 Chunk 输出路径")
    build_started = perf_counter()
    chunked = chunk_transcript(
        transcript,
        chunk_size=spec.chunk_size,
        overlap=spec.overlap,
    )
    validate_alternative_chunk_provenance(dataset, transcript=transcript, chunked=chunked)
    embedder = _build_embedder(settings)
    vector_index = build_vector_index(
        chunked,
        embedder=embedder,
        input_chunk_file=spec.chunk_file.name,
    )
    save_vector_index(vector_index, spec.index_directory)
    save_chunked_transcript(chunked, spec.chunk_file)
    reloaded = load_vector_index(
        spec.index_directory,
        expected_chunks=chunked,
        expected_embedding_model=settings.model,
        expected_dimension=settings.expected_dimension,
        expected_document_task=settings.document_task,
        expected_query_task=settings.query_task,
        expected_embedding_type=settings.embedding_type,
        expected_provider_normalized=settings.normalized,
        expected_provider_truncate=settings.truncate,
    )
    build_elapsed = perf_counter() - build_started
    retriever = VectorRetrieverAdapter(Retriever(reloaded, embedder))
    run = run_retrieval_evaluation(dataset, retriever=retriever)
    if embedder.query_call_count != len(dataset.cases):
        raise ChunkExperimentError("alternative 配置没有对每道题恰好调用一次查询 Embedding")
    if embedder.document_batch_count <= 0:
        raise ChunkExperimentError("alternative 配置没有真实调用文档 Embedding")
    manifest_file = spec.index_directory / MANIFEST_FILENAME
    index_file = spec.index_directory / INDEX_FILENAME
    return _configuration_from_run(
        config_id=spec.config_id,
        manifest=reloaded.manifest,
        manifest_sha256=file_sha256(manifest_file),
        index_sha256=file_sha256(index_file),
        run=run,
        index_reused=False,
        document_embedding_calls=embedder.document_batch_count,
        query_embedding_calls=embedder.query_call_count,
        experiment_document_embedding_calls=embedder.document_batch_count,
        experiment_query_embedding_calls=embedder.query_call_count,
        build_elapsed_seconds=build_elapsed,
        baseline=baseline,
    )


def _build_embedder(settings: EmbeddingSettings) -> OpenAICompatibleEmbedder:
    return OpenAICompatibleEmbedder(
        create_embedding_client(settings),
        model_name=settings.model,
        batch_size=settings.batch_size,
        expected_dimension=settings.expected_dimension,
        document_task=settings.document_task,
        query_task=settings.query_task,
        embedding_type=settings.embedding_type,
        normalized=settings.normalized,
        truncate=settings.truncate,
    )


def _baseline_configuration(
    baseline: RetrievalBaselineResult,
    *,
    manifest: VectorIndexManifest,
    manifest_path: Path,
) -> ChunkExperimentConfigurationResult:
    run = RetrievalEvaluationRun(
        retriever_name=baseline.retriever_name,
        k_values=baseline.k_values,
        query_count=baseline.query_count,
        elapsed_seconds=baseline.elapsed_seconds,
        aggregate_metrics=baseline.aggregate_metrics,
        category_metrics=baseline.category_metrics,
        cases=baseline.cases,
    )
    placeholder = _configuration_from_run(
        config_id="baseline",
        manifest=manifest,
        manifest_sha256=file_sha256(manifest_path),
        index_sha256=manifest.index_sha256 or "0" * 64,
        run=run,
        index_reused=True,
        document_embedding_calls=BASELINE_DOCUMENT_EMBEDDING_CALLS,
        query_embedding_calls=baseline.query_embedding_calls,
        experiment_document_embedding_calls=0,
        experiment_query_embedding_calls=0,
        build_elapsed_seconds=BASELINE_BUILD_ELAPSED_SECONDS,
        baseline=None,
    )
    return placeholder.model_copy(
        update={"delta_vs_baseline": _zero_delta(placeholder)}
    )


def _configuration_from_run(
    *,
    config_id: str,
    manifest: VectorIndexManifest,
    manifest_sha256: str,
    index_sha256: str,
    run: RetrievalEvaluationRun,
    index_reused: bool,
    document_embedding_calls: int,
    query_embedding_calls: int,
    experiment_document_embedding_calls: int,
    experiment_query_embedding_calls: int,
    build_elapsed_seconds: float,
    baseline: ChunkExperimentConfigurationResult | None,
) -> ChunkExperimentConfigurationResult:
    diversity = source_diversity_diagnostics(run.cases)
    temporary = ChunkExperimentConfigurationResult(
        config_id=config_id,
        chunk_size=manifest.chunk_size,
        overlap=manifest.overlap,
        overlap_ratio=manifest.overlap / manifest.chunk_size,
        chunk_count=manifest.vector_count,
        vector_count=manifest.vector_count,
        vector_dimension=manifest.vector_dimension,
        index_sha256=index_sha256,
        manifest_sha256=manifest_sha256,
        input_chunk_sha256=manifest.input_chunk_sha256,
        index_reused=index_reused,
        document_embedding_calls=document_embedding_calls,
        query_embedding_calls=query_embedding_calls,
        experiment_document_embedding_calls=experiment_document_embedding_calls,
        experiment_query_embedding_calls=experiment_query_embedding_calls,
        answer_model_calls=0,
        build_elapsed_seconds=build_elapsed_seconds,
        query_elapsed_seconds=run.elapsed_seconds,
        aggregate_metrics=run.aggregate_metrics,
        category_metrics=run.category_metrics,
        source_diversity=diversity,
        cases=run.cases,
        delta_vs_baseline=ChunkExperimentDelta(
            by_k={
                k: MetricDeltaAtK(
                    hit_count=0,
                    hit_at_k=0,
                    macro_recall_sum=0,
                    macro_recall_at_k=0,
                )
                for k in DEFAULT_K_VALUES
            },
            chunk_count=0,
            shared_source_candidate_count=0,
        ),
    )
    if baseline is None:
        return temporary
    return temporary.model_copy(update={"delta_vs_baseline": _delta(temporary, baseline)})


def _zero_delta(configuration: ChunkExperimentConfigurationResult) -> ChunkExperimentDelta:
    return ChunkExperimentDelta(
        by_k={
            k: MetricDeltaAtK(
                hit_count=0,
                hit_at_k=0,
                macro_recall_sum=0,
                macro_recall_at_k=0,
            )
            for k in configuration.aggregate_metrics
        },
        chunk_count=0,
        shared_source_candidate_count=0,
    )


def _delta(
    current: ChunkExperimentConfigurationResult,
    baseline: ChunkExperimentConfigurationResult,
) -> ChunkExperimentDelta:
    return ChunkExperimentDelta(
        by_k={
            k: MetricDeltaAtK(
                hit_count=(
                    current.aggregate_metrics[k].hit_count
                    - baseline.aggregate_metrics[k].hit_count
                ),
                hit_at_k=(
                    (current.aggregate_metrics[k].hit_at_k or 0)
                    - (baseline.aggregate_metrics[k].hit_at_k or 0)
                ),
                macro_recall_sum=(
                    current.aggregate_metrics[k].macro_recall_sum
                    - baseline.aggregate_metrics[k].macro_recall_sum
                ),
                macro_recall_at_k=(
                    (current.aggregate_metrics[k].macro_recall_at_k or 0)
                    - (baseline.aggregate_metrics[k].macro_recall_at_k or 0)
                ),
            )
            for k in DEFAULT_K_VALUES
        },
        chunk_count=current.chunk_count - baseline.chunk_count,
        shared_source_candidate_count=(
            current.source_diversity.shared_source_candidate_count
            - baseline.source_diversity.shared_source_candidate_count
        ),
    )


def source_diversity_diagnostics(
    cases: tuple[EvaluationCaseResult, ...],
) -> SourceDiversityDiagnostics:
    candidate_count = 0
    shared_count = 0
    unique_sum = 0
    for case in cases:
        seen: set[str] = set()
        for candidate in case.retrieved_candidates:
            candidate_count += 1
            current = set(candidate.source_segment_ids)
            if current.intersection(seen):
                shared_count += 1
            seen.update(current)
        unique_sum += len(seen)
    return SourceDiversityDiagnostics(
        candidate_count=candidate_count,
        shared_source_candidate_count=shared_count,
        shared_source_candidate_rate=(shared_count / candidate_count if candidate_count else 0),
        unique_source_segment_count_sum=unique_sum,
        mean_unique_source_segments_per_case=(unique_sum / len(cases) if cases else 0),
    )


def compare_and_select(
    configurations: tuple[ChunkExperimentConfigurationResult, ...],
) -> tuple[tuple[ChunkExperimentComparisonRow, ...], ChunkExperimentSelectedDefault]:
    baseline = next(item for item in configurations if item.config_id == "baseline")
    baseline_cases = {item.case_id: item for item in baseline.cases}

    def regressions(item: ChunkExperimentConfigurationResult, k: int) -> tuple[str, ...]:
        return tuple(
            case.case_id
            for case in item.cases
            if baseline_cases[case.case_id].metrics_by_k[k].hit_at_k is True
            and case.metrics_by_k[k].hit_at_k is False
        )

    def sort_key(item: ChunkExperimentConfigurationResult) -> tuple[object, ...]:
        hit3 = item.aggregate_metrics[3].hit_at_k or 0
        recall8 = item.aggregate_metrics[8].macro_recall_at_k or 0
        multi8 = item.category_metrics["multi_segment"][8].macro_recall_at_k or 0
        hit1 = item.aggregate_metrics[1].hit_at_k or 0
        recall3 = item.aggregate_metrics[3].macro_recall_at_k or 0
        new1 = regressions(item, 1)
        new3 = regressions(item, 3)
        return (
            hit3 < SELECTION_POLICY.minimum_hit_at_3,
            -recall8,
            -multi8,
            -hit1,
            -recall3,
            len(new1) + len(new3),
            item.vector_count,
            item.build_elapsed_seconds,
            item.query_elapsed_seconds,
            item.source_diversity.shared_source_candidate_count,
        )

    ordered = sorted(configurations, key=sort_key)
    ranks = {item.config_id: rank for rank, item in enumerate(ordered, start=1)}
    comparison = tuple(
        ChunkExperimentComparisonRow(
            config_id=item.config_id,
            eligible_by_hit_at_3=(
                (item.aggregate_metrics[3].hit_at_k or 0)
                >= SELECTION_POLICY.minimum_hit_at_3
            ),
            new_hit_at_1_failures=regressions(item, 1),
            new_hit_at_3_failures=regressions(item, 3),
            selection_rank=ranks[item.config_id],
        )
        for item in configurations
    )
    winner = ordered[0]
    selected = ChunkExperimentSelectedDefault(
        config_id=winner.config_id,
        chunk_size=winner.chunk_size,
        overlap=winner.overlap,
        retained_day12_baseline=winner.config_id == "baseline",
        rationale=(
            "按预先固定的顺序选择：先要求 Hit@3 不低于 100%，再依次比较整体 "
            "Macro Recall@8、multi_segment Macro Recall@8、Hit@1、Macro Recall@3、"
            "新增退化，最后才比较索引规模、建库/查询耗时与重复来源候选。"
        ),
        uncertainty=(
            "结论只适用于当前一期英文访谈、同一 Jina 模型和 16 道计分题；"
            "三组均保持 20% overlap，不能独立推出 overlap 比例的因果效果，"
            "也不能声称全局最优。"
        ),
    )
    return comparison, selected


def build_failure_reviews(
    configurations: tuple[ChunkExperimentConfigurationResult, ...],
) -> tuple[ChunkExperimentFailureReview, ...]:
    by_config = {
        item.config_id: {case.case_id: case for case in item.cases}
        for item in configurations
    }
    baseline_cases = by_config["baseline"]
    review_ids = list(FOCUS_CASE_IDS)
    for configuration in configurations:
        if configuration.config_id == "baseline":
            continue
        for case in configuration.cases:
            baseline = baseline_cases[case.case_id]
            regressed = any(
                baseline.metrics_by_k[k].hit_at_k is True
                and case.metrics_by_k[k].hit_at_k is False
                for k in (1, 3)
            )
            if regressed and case.case_id not in review_ids:
                review_ids.append(case.case_id)

    reviews: list[ChunkExperimentFailureReview] = []
    for case_id in review_ids:
        reason = "preselected_focus" if case_id in FOCUS_CASE_IDS else "new_regression"
        details = tuple(
            _case_review(configuration.config_id, by_config[configuration.config_id][case_id])
            for configuration in configurations
        )
        reviews.append(
            ChunkExperimentFailureReview(
                case_id=case_id,
                review_reason=reason,
                configurations=details,
                summary=(
                    "Day 12 预先指定的跨配置重点案例；比较正确来源排名、Top-8 "
                    "缺失和共享来源候选。"
                    if reason == "preselected_focus"
                    else "新配置使 Day 12 原本成功的 Hit@1 或 Hit@3 发生退化。"
                ),
            )
        )
    return tuple(reviews)


def _case_review(config_id: str, case: EvaluationCaseResult) -> CaseConfigurationReview:
    expected = tuple(case.expected_source_segment_ids)
    correct_ranks = {
        segment_id: tuple(
            candidate.rank
            for candidate in case.retrieved_candidates
            if segment_id in candidate.source_segment_ids
        )
        for segment_id in expected
    }
    retrieved = tuple(segment_id for segment_id in expected if correct_ranks[segment_id])
    missing = tuple(segment_id for segment_id in expected if not correct_ranks[segment_id])
    seen: set[str] = set()
    shared_ranks: list[int] = []
    for candidate in case.retrieved_candidates:
        current = set(candidate.source_segment_ids)
        if current.intersection(seen):
            shared_ranks.append(candidate.rank)
        seen.update(current)
    return CaseConfigurationReview(
        config_id=config_id,
        correct_candidate_ranks=correct_ranks,
        retrieved_expected_segment_ids=retrieved,
        missing_expected_segment_ids=missing,
        shared_source_candidate_ranks=tuple(shared_ranks),
        hit_at_1=case.metrics_by_k[1].hit_at_k is True,
        hit_at_3=case.metrics_by_k[3].hit_at_k is True,
        recall_at_5=case.metrics_by_k[5].recall_at_k or 0,
        recall_at_8=case.metrics_by_k[8].recall_at_k or 0,
    )


def _controlled_variables(
    *,
    transcript_sha256: str,
    manifest: VectorIndexManifest,
    dataset: EvaluationDataset,
) -> ChunkExperimentControlledVariables:
    return ChunkExperimentControlledVariables(
        transcript_file_sha256=transcript_sha256,
        source_segment_truth="expected_source_segment_ids",
        expected_chunk_ids_scope="baseline_1000_200_only",
        embedding_document_task=manifest.embedding_document_task or "",
        embedding_query_task=manifest.embedding_query_task or "",
        embedding_type=manifest.embedding_type or "",
        provider_normalized=manifest.provider_normalized is True,
        provider_truncate=manifest.provider_truncate is True,
        index_type="faiss_index_flat_ip",
        distance_metric="cosine",
        retriever_config=VectorRetrieverConfiguration(
            adapter="vector_retriever_adapter_v1",
            underlying_retriever="day8_vector_retriever",
            score_semantics="cosine_similarity",
            query_strategy="one_top_8_call_then_prefixes",
            equal_score_tiebreaker="faiss_row_index_ascending",
        ),
        k_values=DEFAULT_K_VALUES,
        query_strategy="one_top_8_call_per_case_then_prefixes",
        answerable_case_count=sum(item.answerable for item in dataset.cases),
        unanswerable_case_count=sum(not item.answerable for item in dataset.cases),
        unanswerable_cases_scored=False,
        answer_model_calls=0,
    )


def _validate_baseline_reference(
    dataset: EvaluationDataset,
    manifest: VectorIndexManifest,
    baseline: RetrievalBaselineResult,
    settings: EmbeddingSettings,
) -> None:
    expected = (
        dataset.dataset_id,
        dataset.cases_sha256,
        dataset.episode_id,
        manifest.embedding_model,
        manifest.vector_count,
        manifest.vector_dimension,
        manifest.chunk_size,
        manifest.overlap,
        tuple(DEFAULT_K_VALUES),
    )
    actual = (
        baseline.dataset_id,
        baseline.dataset_cases_sha256,
        baseline.episode_id,
        baseline.embedding_model,
        baseline.index_config.vector_count,
        baseline.index_config.vector_dimension,
        baseline.index_config.chunk_size,
        baseline.index_config.overlap,
        tuple(baseline.k_values),
    )
    if actual != expected:
        raise ChunkExperimentError("Day 12 正式结果与冻结数据集或当前基线索引不一致")
    if settings.model != manifest.embedding_model:
        raise ChunkExperimentError(".env 的 Embedding 模型与 Day 12 基线不一致")
    if settings.expected_dimension != manifest.vector_dimension:
        raise ChunkExperimentError(".env 的 Embedding 维度与 Day 12 基线不一致")
    if baseline.query_embedding_calls != len(dataset.cases):
        raise ChunkExperimentError("Day 12 基线查询调用数不等于冻结题数")
    if baseline.document_embedding_calls != 0 or baseline.answer_model_calls != 0:
        raise ChunkExperimentError("Day 12 检索基线调用计数不符合控制变量")


def _preflight_alternative_destinations(specs: tuple[ChunkExperimentSpec, ...]) -> None:
    for spec in specs:
        if spec.config_id == "baseline":
            continue
        if spec.chunk_file is None:
            raise ChunkExperimentError("alternative 配置缺少独立 Chunk 文件路径")
        occupied = [
            path
            for path in (
                spec.chunk_file,
                spec.index_directory / INDEX_FILENAME,
                spec.index_directory / MANIFEST_FILENAME,
            )
            if path.exists()
        ]
        if occupied:
            names = ", ".join(str(path) for path in occupied)
            raise ChunkExperimentPersistenceError(
                f"alternative 产物已存在，不会静默覆盖：{names}"
            )


def _input_hashes(
    dataset_path: Path,
    transcript_path: Path,
    manifest_path: Path,
    faiss_path: Path,
    baseline_result_path: Path,
) -> ChunkExperimentInputHashes:
    return ChunkExperimentInputHashes(
        dataset_file_sha256=file_sha256(dataset_path),
        transcript_file_sha256=file_sha256(transcript_path),
        baseline_manifest_sha256=file_sha256(manifest_path),
        baseline_faiss_sha256=file_sha256(faiss_path),
        baseline_result_sha256=file_sha256(baseline_result_path),
    )


def save_chunk_experiment(
    result: ChunkParameterExperimentResult,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise ChunkExperimentPersistenceError(
            f"目标实验结果已存在：{destination}；不会静默覆盖"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
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
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, destination)
    except OSError as error:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise ChunkExperimentPersistenceError("Chunk 实验结果写入失败") from error
    return destination
