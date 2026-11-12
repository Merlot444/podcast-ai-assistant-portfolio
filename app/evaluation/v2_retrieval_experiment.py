"""Reproducible three-episode runner for the frozen 80-case retrieval result."""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from pydantic import ValidationError

from app.core.config import EmbeddingSettings
from app.core.errors import safe_error_message
from app.core.files import atomic_write_text
from app.evaluation.baseline import file_sha256
from app.evaluation.core import DEFAULT_EVALUATION_DATASET, load_evaluation_dataset
from app.evaluation.payload_policy import (
    V2_RETRIEVAL_METADATA_KEY_EXCEPTIONS,
    find_unsafe_payload_path,
)
from app.evaluation.runner import (
    DEFAULT_K_VALUES,
    aggregate_retrieval_cases,
    run_precomputed_case_evaluation,
)
from app.evaluation.v2_pilot import load_v2_pilot
from app.indexing.embedding import (
    Embedder,
    OpenAICompatibleEmbedder,
    create_embedding_client,
)
from app.indexing.index import (
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    PersistentVectorIndex,
    chunked_transcript_fingerprint,
    load_vector_index,
)
from app.retrieval.base import EvaluationRetriever
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.core import Retriever
from app.retrieval.hybrid import (
    CHILD_CANDIDATE_DEPTH,
    RRF_K,
    reciprocal_rank_fusion,
)
from app.retrieval.vector import VectorRetrieverAdapter
from app.schemas.chunk import ChunkedTranscript
from app.schemas.evaluation import EvaluationCase, EvaluationChunkBaseline
from app.schemas.evaluation_result import RetrievalEvaluationRun
from app.schemas.evaluation_v2 import PilotEpisodeDataset
from app.schemas.evaluation_v2_retrieval import (
    BM25CorpusSize,
    EvaluationV2RetrievalExperiment,
    ExpandedCallCounts,
    ExpandedDatasetSummary,
    ExpandedEpisodeIdentity,
    ExpandedMethodResult,
    ExpandedRetrievalConfiguration,
    MethodLatency,
    MetricDelta,
)
from app.schemas.retriever import RetrievedCandidate
from app.schemas.transcript import ProcessedTranscript


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V2_DATASET = (
    _PROJECT_ROOT / "data" / "evaluation" / "eval_dataset_v2-pilot.json"
)
DEFAULT_V2_RETRIEVAL_OUTPUT = (
    _PROJECT_ROOT
    / "data"
    / "evaluation"
    / "eval_dataset_v2-pilot-retrieval-experiment.json"
)


@dataclass(frozen=True)
class ExpandedEpisodeFiles:
    """All local inputs needed to evaluate one episode without rebuilding its index."""

    episode_slug: str
    processed_transcript: Path
    chunk_catalog: Path
    index_directory: Path


DEFAULT_EXPANDED_EPISODE_FILES = (
    ExpandedEpisodeFiles(
        episode_slug="demis",
        processed_transcript=_PROJECT_ROOT / "data" / "processed" / "demis-hassabis.json",
        chunk_catalog=_PROJECT_ROOT
        / "data"
        / "processed"
        / "demis-hassabis-chunks.json",
        index_directory=_PROJECT_ROOT
        / "data"
        / "indexes"
        / "demis-hassabis-jina-v5-small-1024",
    ),
    ExpandedEpisodeFiles(
        episode_slug="ilya25",
        processed_transcript=_PROJECT_ROOT
        / "data"
        / "processed"
        / "ilya-sutskever-2025.json",
        chunk_catalog=_PROJECT_ROOT
        / "data"
        / "processed"
        / "ilya-sutskever-2025-chunks.json",
        index_directory=_PROJECT_ROOT
        / "data"
        / "indexes"
        / "ilya-sutskever-2025-jina-v5-small-1024",
    ),
    ExpandedEpisodeFiles(
        episode_slug="mark25",
        processed_transcript=_PROJECT_ROOT
        / "data"
        / "processed"
        / "mark-zuckerberg-2025.json",
        chunk_catalog=_PROJECT_ROOT
        / "data"
        / "processed"
        / "mark-zuckerberg-2025-chunks.json",
        index_directory=_PROJECT_ROOT
        / "data"
        / "indexes"
        / "mark-zuckerberg-2025-jina-v5-small-1024",
    ),
)

VectorRetrieverFactory = Callable[
    [str, PersistentVectorIndex, Embedder], EvaluationRetriever
]


class V2RetrievalExperimentError(RuntimeError):
    """Raised when the expanded comparison cannot produce a complete result."""


class V2RetrievalExperimentPersistenceError(RuntimeError):
    """Raised when the expanded result cannot be saved without ambiguity."""


@dataclass(frozen=True)
class _EpisodeInput:
    slug: str
    cases: tuple[EvaluationCase, ...]
    baseline: EvaluationChunkBaseline
    expected_episode_id: str
    files: ExpandedEpisodeFiles


EpisodeArtifactsLoader = Callable[
    [_EpisodeInput, EmbeddingSettings],
    tuple[ChunkedTranscript, PersistentVectorIndex],
]
BM25RetrieverFactory = Callable[[str, PersistentVectorIndex], BM25Retriever]


def run_v2_retrieval_experiment(
    *,
    settings: EmbeddingSettings,
    base_dataset_path: str | Path = DEFAULT_EVALUATION_DATASET,
    v2_dataset_path: str | Path = DEFAULT_V2_DATASET,
    episode_files: Sequence[ExpandedEpisodeFiles] = DEFAULT_EXPANDED_EPISODE_FILES,
    output_path: str | Path = DEFAULT_V2_RETRIEVAL_OUTPUT,
    overwrite: bool = False,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    clock: Callable[[], float] = perf_counter,
    vector_retriever_factory: VectorRetrieverFactory | None = None,
    episode_artifacts_loader: EpisodeArtifactsLoader | None = None,
    bm25_retriever_factory: BM25RetrieverFactory | None = None,
) -> EvaluationV2RetrievalExperiment:
    """Run Vector, BM25 and Hybrid-RRF once per expanded case and persist it."""
    started = clock()
    if settings.expected_dimension is None:
        raise V2RetrievalExperimentError(
            "80 题检索实验必须显式配置 Embedding 维度"
        )
    base_path = Path(base_dataset_path)
    pilot_path = Path(v2_dataset_path)
    destination = Path(output_path)
    files_by_slug = _validate_episode_file_set(episode_files)
    input_paths = _input_paths(base_path, pilot_path, episode_files)
    _preflight_output(destination, input_paths, overwrite=overwrite)
    hashes_before = _input_hashes(input_paths)

    base = load_evaluation_dataset(base_path)
    pilot = load_v2_pilot(pilot_path)
    if (
        pilot.parent.artifact_sha256 != hashes_before[_artifact_key(base_path)]
        or pilot.parent.cases_sha256 != base.cases_sha256
    ):
        raise V2RetrievalExperimentError("v2 parent 与当前 v1 冻结评测集不一致")

    episodes = _episode_inputs(base, pilot.episode_datasets, files_by_slug)
    embedder = _build_embedder(settings)
    vector_factory = vector_retriever_factory or _default_vector_retriever
    artifacts_loader = episode_artifacts_loader or _default_episode_artifacts_loader
    bm25_factory = bm25_retriever_factory or _default_bm25_retriever
    candidates: dict[str, dict[str, list[RetrievedCandidate]]] = {
        name: {} for name in ("vector", "bm25", "hybrid_rrf")
    }
    latency: dict[str, list[float]] = {
        name: [] for name in ("vector", "bm25", "hybrid_rrf")
    }
    case_slug: dict[str, str] = {}
    episode_identities: list[ExpandedEpisodeIdentity] = []
    bm25_indexes: list[BM25Retriever] = []

    for episode in episodes:
        chunked, index = artifacts_loader(episode, settings)
        vector = vector_factory(episode.slug, index, embedder)
        bm25 = bm25_factory(episode.slug, index)
        bm25_indexes.append(bm25)

        for case in episode.cases:
            case_slug[case.case_id] = episode.slug
            try:
                vector_started = clock()
                vector_candidates = vector.retrieve(
                    case.question, CHILD_CANDIDATE_DEPTH
                )
                latency["vector"].append(clock() - vector_started)

                bm25_started = clock()
                bm25_candidates = bm25.retrieve(case.question, CHILD_CANDIDATE_DEPTH)
                latency["bm25"].append(clock() - bm25_started)

                fusion_started = clock()
                hybrid_candidates = reciprocal_rank_fusion(
                    vector_candidates,
                    bm25_candidates,
                    corpus_identity=bm25.corpus_identity,
                    top_k=CHILD_CANDIDATE_DEPTH,
                    rrf_k=RRF_K,
                    child_candidate_depth=CHILD_CANDIDATE_DEPTH,
                )
                latency["hybrid_rrf"].append(clock() - fusion_started)
            except Exception as error:
                raise V2RetrievalExperimentError(
                    f"案例 {case.case_id} 三路检索失败：{type(error).__name__}"
                ) from error
            candidates["vector"][case.case_id] = vector_candidates
            candidates["bm25"][case.case_id] = bm25_candidates
            candidates["hybrid_rrf"][case.case_id] = hybrid_candidates

        episode_identities.append(
            _episode_identity(episode, chunked, index, hashes_before)
        )

    ordered_cases = tuple(case for episode in episodes for case in episode.cases)
    runs = tuple(
        run_precomputed_case_evaluation(
            ordered_cases,
            retriever_name=name,
            candidates_by_case_id=candidates[name],
            candidate_depth=CHILD_CANDIDATE_DEPTH,
        )
        for name in ("vector", "bm25", "hybrid_rrf")
    )
    methods = _method_results(runs, case_slug=case_slug, latency=latency)

    created_at = now()
    if created_at.tzinfo is None:
        raise V2RetrievalExperimentError("created_at 必须带时区")
    created_at = created_at.astimezone(UTC)
    hashes_after = _input_hashes(input_paths)
    if hashes_after != hashes_before:
        raise V2RetrievalExperimentError("实验期间冻结输入发生变化")

    result = EvaluationV2RetrievalExperiment(
        schema_version=1,
        experiment_id=(
            "eval_dataset_v2_pilot-retrieval-"
            + created_at.strftime("%Y%m%dt%H%M%Sz")
        ),
        created_at_utc=created_at,
        dataset=ExpandedDatasetSummary(
            base_dataset_id=base.dataset_id,
            base_cases_sha256=base.cases_sha256,
            pilot_dataset_id=pilot.dataset_id,
            pilot_collection_sha256=pilot.collection_sha256,
            expanded_case_count=len(ordered_cases),
            answerable_case_count=sum(case.answerable for case in ordered_cases),
            unanswerable_case_count=sum(not case.answerable for case in ordered_cases),
        ),
        configuration=ExpandedRetrievalConfiguration(
            embedding_model=settings.model,
            embedding_dimension=settings.expected_dimension,
            document_task=settings.document_task,
            query_task=settings.query_task,
            normalized=settings.normalized,
            truncate=settings.truncate,
            chunk_size=1000,
            overlap=200,
            k_values=DEFAULT_K_VALUES,
            candidate_depth=CHILD_CANDIDATE_DEPTH,
            bm25_k1=1.5,
            bm25_b=0.75,
            rrf_k=RRF_K,
        ),
        episode_identities=tuple(episode_identities),
        input_hashes=hashes_before,
        call_counts=ExpandedCallCounts(
            index_build_document_embedding_batches=0,
            index_build_document_count=0,
            experiment_query_embedding_calls=len(ordered_cases),
            vector_retrieval_calls=len(ordered_cases),
            bm25_retrieval_calls=len(ordered_cases),
            rrf_fusion_calls=len(ordered_cases),
            experiment_document_embedding_calls=0,
            answer_model_calls=0,
        ),
        bm25_corpus_size=BM25CorpusSize(
            document_tokens=sum(item.document_token_count for item in bm25_indexes),
            vocabulary_sum=sum(item.vocabulary_size for item in bm25_indexes),
            postings=sum(item.posting_count for item in bm25_indexes),
            memory_bytes=sum(item.approximate_memory_bytes for item in bm25_indexes),
        ),
        methods=methods,
        total_experiment_seconds=max(clock() - started, 1e-12),
        protected_inputs_unchanged=True,
    )
    save_v2_retrieval_experiment(result, destination, overwrite=overwrite)
    return result


def save_v2_retrieval_experiment(
    result: EvaluationV2RetrievalExperiment,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically save a complete expanded result without silent overwrite."""
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise V2RetrievalExperimentPersistenceError(
            f"目标 v2 检索结果已存在：{destination}；不会静默覆盖"
        )
    result_payload = result.model_dump(mode="json")
    unsafe_path = find_unsafe_payload_path(
        result_payload,
        allowed_keys=V2_RETRIEVAL_METADATA_KEY_EXCEPTIONS,
    )
    if unsafe_path is not None:
        raise V2RetrievalExperimentPersistenceError(
            f"v2 检索结果包含禁止持久化字段：{unsafe_path}"
        )
    payload = json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n"
    try:
        return atomic_write_text(destination, payload)
    except OSError as error:
        raise V2RetrievalExperimentPersistenceError("v2 检索结果写入失败") from error


def _validate_episode_file_set(
    episode_files: Sequence[ExpandedEpisodeFiles],
) -> dict[str, ExpandedEpisodeFiles]:
    files = tuple(episode_files)
    if tuple(item.episode_slug for item in files) != ("demis", "ilya25", "mark25"):
        raise V2RetrievalExperimentError(
            "episode_files 必须按 demis、ilya25、mark25 顺序完整提供"
        )
    return {item.episode_slug: item for item in files}


def _episode_inputs(base, pilot_episodes, files_by_slug) -> tuple[_EpisodeInput, ...]:
    pilot_by_slug: Mapping[str, PilotEpisodeDataset] = {
        item.episode_slug: item for item in pilot_episodes
    }
    if set(pilot_by_slug) != {"ilya25", "mark25"}:
        raise V2RetrievalExperimentError("v2 pilot 必须包含 ilya25 与 mark25")
    return (
        _EpisodeInput(
            slug="demis",
            cases=base.cases,
            baseline=base.chunk_baseline,
            expected_episode_id=base.episode_id,
            files=files_by_slug["demis"],
        ),
        *(
            _EpisodeInput(
                slug=slug,
                cases=pilot_by_slug[slug].cases,
                baseline=pilot_by_slug[slug].chunk_baseline,
                expected_episode_id=pilot_by_slug[slug].episode_id,
                files=files_by_slug[slug],
            )
            for slug in ("ilya25", "mark25")
        ),
    )


def _validate_episode_sources(
    episode: _EpisodeInput,
    transcript: ProcessedTranscript,
    chunked: ChunkedTranscript,
    index: PersistentVectorIndex,
) -> None:
    manifest = index.manifest
    actual = (
        transcript.episode_id,
        chunked.episode_id,
        manifest.episode_id,
        chunked.algorithm,
        chunked.chunk_size,
        chunked.overlap,
        chunked_transcript_fingerprint(chunked),
    )
    expected = (
        episode.expected_episode_id,
        episode.expected_episode_id,
        episode.expected_episode_id,
        episode.baseline.algorithm,
        episode.baseline.chunk_size,
        episode.baseline.overlap,
        episode.baseline.input_chunk_sha256,
    )
    if actual != expected or (manifest.chunk_size, manifest.overlap) != (1000, 200):
        raise V2RetrievalExperimentError(
            f"{episode.slug} 的 episode、Chunk 配置或内容指纹不一致"
        )


def _episode_identity(episode, chunked, index, hashes) -> ExpandedEpisodeIdentity:
    return ExpandedEpisodeIdentity(
        episode_slug=episode.slug,
        episode_id=episode.expected_episode_id,
        case_count=len(episode.cases),
        answerable_case_count=sum(case.answerable for case in episode.cases),
        chunk_count=len(chunked.chunks),
        chunk_size=chunked.chunk_size,
        overlap=chunked.overlap,
        input_chunk_sha256=index.manifest.input_chunk_sha256,
        manifest_sha256=hashes[_artifact_key(episode.files.index_directory / MANIFEST_FILENAME)],
        faiss_sha256=hashes[_artifact_key(episode.files.index_directory / INDEX_FILENAME)],
    )


def _method_results(
    runs: tuple[RetrievalEvaluationRun, ...],
    *,
    case_slug: dict[str, str],
    latency: dict[str, list[float]],
) -> tuple[ExpandedMethodResult, ...]:
    vector = runs[0]
    results = []
    for run in runs:
        timings = latency[run.retriever_name]
        results.append(
            ExpandedMethodResult(
                retriever_name=run.retriever_name,
                aggregate_metrics=run.aggregate_metrics,
                category_metrics=run.category_metrics,
                episode_metrics={
                    slug: aggregate_retrieval_cases(
                        [case for case in run.cases if case_slug[case.case_id] == slug]
                    )
                    for slug in ("demis", "ilya25", "mark25")
                },
                delta_vs_vector=_metric_delta(
                    run.aggregate_metrics, vector.aggregate_metrics
                ),
                latency_seconds=MethodLatency(
                    total=sum(timings),
                    median_per_query=statistics.median(timings),
                ),
                cases=run.cases,
            )
        )
    return tuple(results)  # type: ignore[return-value]


def _metric_delta(current, vector) -> dict[int, MetricDelta]:
    return {
        k: MetricDelta(
            hit_count=current[k].hit_count - vector[k].hit_count,
            hit_at_k=(current[k].hit_at_k or 0) - (vector[k].hit_at_k or 0),
            macro_recall_sum=(
                current[k].macro_recall_sum - vector[k].macro_recall_sum
            ),
            macro_recall_at_k=(current[k].macro_recall_at_k or 0)
            - (vector[k].macro_recall_at_k or 0),
        )
        for k in DEFAULT_K_VALUES
    }


def _input_paths(base, pilot, episode_files) -> tuple[Path, ...]:
    paths = [base, pilot]
    for item in episode_files:
        paths.extend(
            (
                item.processed_transcript,
                item.chunk_catalog,
                item.index_directory / MANIFEST_FILENAME,
                item.index_directory / INDEX_FILENAME,
            )
        )
    return tuple(paths)


def _input_hashes(paths: Sequence[Path]) -> dict[str, str]:
    result = {}
    for path in paths:
        if not path.is_file():
            raise V2RetrievalExperimentError(f"缺少 v2 检索输入：{path}")
        result[_artifact_key(path)] = file_sha256(path)
    return result


def _artifact_key(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(_PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _preflight_output(destination, input_paths, *, overwrite) -> None:
    if destination.resolve() in {path.resolve() for path in input_paths}:
        raise V2RetrievalExperimentPersistenceError("v2 输出不能覆盖冻结输入")
    if destination.exists() and not overwrite:
        raise V2RetrievalExperimentPersistenceError(
            f"目标 v2 检索结果已存在：{destination}；不会静默覆盖"
        )


def _load_model(path: Path, model, label: str):
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError) as error:
        raise V2RetrievalExperimentError(
            f"{label} 无法严格加载：{safe_error_message(error)}"
        ) from error


def _default_episode_artifacts_loader(
    episode: _EpisodeInput,
    settings: EmbeddingSettings,
) -> tuple[ChunkedTranscript, PersistentVectorIndex]:
    """Strictly load and cross-check one production episode's protected inputs."""
    transcript = _load_model(
        episode.files.processed_transcript,
        ProcessedTranscript,
        f"{episode.slug} 处理后转写稿",
    )
    chunked = _load_model(
        episode.files.chunk_catalog,
        ChunkedTranscript,
        f"{episode.slug} Chunk catalog",
    )
    index = load_vector_index(
        episode.files.index_directory,
        expected_chunks=chunked,
        expected_embedding_model=settings.model,
        expected_dimension=settings.expected_dimension,
        expected_document_task=settings.document_task,
        expected_query_task=settings.query_task,
        expected_embedding_type=settings.embedding_type,
        expected_provider_normalized=settings.normalized,
        expected_provider_truncate=settings.truncate,
    )
    _validate_episode_sources(episode, transcript, chunked, index)
    return chunked, index


def _default_bm25_retriever(
    slug: str,
    index: PersistentVectorIndex,
) -> BM25Retriever:
    del slug
    return BM25Retriever.from_manifest(index.manifest)


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


def _default_vector_retriever(
    slug: str,
    index: PersistentVectorIndex,
    embedder: Embedder,
) -> EvaluationRetriever:
    del slug
    return VectorRetrieverAdapter(Retriever(index, embedder))
