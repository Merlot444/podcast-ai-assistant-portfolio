"""Day 14 Vector/BM25/Hybrid-RRF controlled comparison and persistence."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from pydantic import ValidationError

from app.core.errors import safe_error_message
from app.chunking.pipeline import load_processed_transcript
from app.core.config import EmbeddingSettings
from app.evaluation.baseline import file_sha256
from app.evaluation.chunk_experiment import load_retrieval_baseline
from app.evaluation.core import (
    DEFAULT_EVALUATION_DATASET,
    DEFAULT_EVALUATION_INDEX_DIRECTORY,
    DEFAULT_EVALUATION_TRANSCRIPT,
    DEFAULT_RETRIEVAL_BASELINE,
    load_evaluation_dataset,
    validate_evaluation_sources,
)
from app.evaluation.runner import (
    DEFAULT_K_VALUES,
    aggregate_retrieval_cases,
    run_precomputed_retrieval_evaluation,
)
from app.indexing import OpenAICompatibleEmbedder, load_vector_index
from app.indexing.embedding import create_embedding_client
from app.indexing.index import (
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    chunked_transcript_fingerprint,
)
from app.retrieval import Retriever
from app.retrieval.bm25 import (
    BM25_B,
    BM25_K1,
    TOKENIZER_VERSION,
    BM25Retriever,
)
from app.retrieval.hybrid import (
    CHILD_CANDIDATE_DEPTH,
    RRF_K,
    component_ranks,
    reciprocal_rank_fusion,
)
from app.retrieval.vector import VectorRetrieverAdapter
from app.schemas.chunk import ChunkedTranscript
from app.schemas.chunk_experiment import ChunkParameterExperimentResult
from app.schemas.evaluation_result import EvaluationCaseResult, RetrievalEvaluationRun
from app.schemas.retrieval_experiment import (
    BM25Configuration,
    BM25TokenizerConfiguration,
    Day14CallCounts,
    Day14CaseReview,
    Day14CorpusIdentity,
    Day14InputArtifactHashes,
    Day14LatencyAndSize,
    Day14Recommendation,
    Day14SelectionPolicy,
    DiagnosticSubsetCase,
    HybridFusionCandidateAudit,
    HybridFusionCaseAudit,
    MethodCaseReview,
    MethodMetricDeltaAtK,
    RetrievalExperimentResult,
    RetrievalMethodResult,
    RRFConfiguration,
    VectorDriftComparison,
)
from app.schemas.retriever import RetrievedCandidate


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNK_CATALOG = (
    _PROJECT_ROOT / "data" / "processed" / "demis-hassabis-chunks.json"
)
DEFAULT_DAY13_EXPERIMENT = (
    _PROJECT_ROOT
    / "data"
    / "evaluation"
    / "eval_dataset_v1-chunk-experiment-day13.json"
)
DEFAULT_RETRIEVAL_EXPERIMENT_OUTPUT = (
    _PROJECT_ROOT
    / "data"
    / "evaluation"
    / "eval_dataset_v1-retrieval-experiment-day14.json"
)

DIAGNOSTIC_SUBSET = (
    DiagnosticSubsetCase(
        case_id="gemini_scaling_limits",
        reason="Gemini is an exact model/product name and Hassabis is a person name.",
    ),
    DiagnosticSubsetCase(
        case_id="agi_timeline_uncertainty",
        reason="AGI is an English acronym and the question names Hassabis.",
    ),
    DiagnosticSubsetCase(
        case_id="robotics_data_poor_research",
        reason="DeepMind is an exact organization name.",
    ),
    DiagnosticSubsetCase(
        case_id="tree_search_rl_synthetic_reward",
        reason="LLM, RL and AlphaZero are exact acronyms or proper names.",
    ),
    DiagnosticSubsetCase(
        case_id="agi_architecture_original_bets",
        reason="AGI and DeepMind are exact acronym/proper-name signals.",
    ),
    DiagnosticSubsetCase(
        case_id="gato_exchange_provenance",
        reason="Gato and the two speaker names are exact entities.",
    ),
)

SELECTION_POLICY = Day14SelectionPolicy(
    fixed_before_results=True,
    ordered_criteria=(
        "Hybrid Hit@3 must not be lower than the same-run Vector Hit@3",
        "prefer higher overall Macro Recall@8",
        "inspect multi_segment Macro Recall@8",
        "then compare Hit@1 and Macro Recall@3/5",
        "reject unexplained new Hit@1 or Hit@3 regressions",
        "inspect the predeclared proper-name/acronym/person subset",
        "inspect similar-term distractor regressions",
        "when quality is close, prefer lower local cost and complexity",
        "do not switch if Hybrid has no stable quality benefit",
    ),
    hybrid_candidate_requirements=(
        "same-run Hit@3 is not lower than Vector",
        "overall Macro Recall@8 is strictly higher than Vector",
        "multi_segment Macro Recall@8 is not lower than Vector",
        "no new Hit@1 or Hit@3 regression relative to Vector",
    ),
    one_case_percentage_points=6.25,
    small_sample_warning=(
        "Only 16 scored questions from one English episode are available; one case "
        "equals 6.25 percentage points and no result establishes broad generality."
    ),
)

FOCUS_CASE_IDS = (
    "alignment_safeguards_pause_trigger",
    "agi_architecture_original_bets",
    "gemini_scaling_limits",
    "tree_search_rl_synthetic_reward",
)


class RetrievalExperimentError(RuntimeError):
    """Raised when a complete trustworthy Day 14 result cannot be produced."""


class RetrievalExperimentPersistenceError(RuntimeError):
    """Raised when the formal Day 14 result cannot be written safely."""


def run_retrieval_experiment(
    *,
    settings: EmbeddingSettings,
    dataset_path: str | Path = DEFAULT_EVALUATION_DATASET,
    transcript_path: str | Path = DEFAULT_EVALUATION_TRANSCRIPT,
    chunk_catalog_path: str | Path = DEFAULT_CHUNK_CATALOG,
    index_directory: str | Path = DEFAULT_EVALUATION_INDEX_DIRECTORY,
    day12_baseline_path: str | Path = DEFAULT_RETRIEVAL_BASELINE,
    day13_experiment_path: str | Path = DEFAULT_DAY13_EXPERIMENT,
    output_path: str | Path = DEFAULT_RETRIEVAL_EXPERIMENT_OUTPUT,
    overwrite: bool = False,
) -> RetrievalExperimentResult:
    """Retrieve each query once per child, reuse rankings, compare and save."""
    started = perf_counter()
    paths = _resolve_paths(
        dataset_path=dataset_path,
        transcript_path=transcript_path,
        chunk_catalog_path=chunk_catalog_path,
        index_directory=index_directory,
        day12_baseline_path=day12_baseline_path,
        day13_experiment_path=day13_experiment_path,
        output_path=output_path,
    )
    _preflight_output(paths, overwrite=overwrite)
    before_hashes = _input_hashes(paths)

    dataset = load_evaluation_dataset(paths["dataset"])
    transcript = load_processed_transcript(paths["transcript"])
    vector_index = load_vector_index(paths["index_directory"])
    validate_evaluation_sources(
        dataset,
        transcript=transcript,
        manifest=vector_index.manifest,
    )
    chunked = _load_chunk_catalog(paths["chunk_catalog"])
    _validate_chunk_catalog(chunked, vector_index.manifest)
    day12 = load_retrieval_baseline(paths["day12"])
    day13 = _load_day13_experiment(paths["day13"])
    _validate_formal_inputs(
        dataset=dataset,
        manifest=vector_index.manifest,
        day12=day12,
        day13=day13,
        settings=settings,
        hashes=before_hashes,
    )
    _validate_predeclared_case_ids(dataset)

    embedder = _build_embedder(settings)
    vector = VectorRetrieverAdapter(Retriever(vector_index, embedder))
    bm25 = BM25Retriever.from_manifest(vector_index.manifest)
    if vector.corpus_identity != bm25.corpus_identity:
        raise RetrievalExperimentError("Vector 与 BM25 的完整 corpus identity 不一致")

    vector_by_case: dict[str, list[RetrievedCandidate]] = {}
    bm25_by_case: dict[str, list[RetrievedCandidate]] = {}
    hybrid_by_case: dict[str, list[RetrievedCandidate]] = {}
    vector_times: list[float] = []
    bm25_times: list[float] = []
    fusion_times: list[float] = []
    fusion_audit: list[HybridFusionCaseAudit] = []

    for case in dataset.cases:
        try:
            child_started = perf_counter()
            vector_candidates = vector.retrieve(case.question, CHILD_CANDIDATE_DEPTH)
            vector_times.append(perf_counter() - child_started)

            child_started = perf_counter()
            bm25_candidates = bm25.retrieve(case.question, CHILD_CANDIDATE_DEPTH)
            bm25_times.append(perf_counter() - child_started)

            fusion_started = perf_counter()
            hybrid_candidates = reciprocal_rank_fusion(
                vector_candidates,
                bm25_candidates,
                corpus_identity=vector.corpus_identity,
                top_k=CHILD_CANDIDATE_DEPTH,
                rrf_k=RRF_K,
                child_candidate_depth=CHILD_CANDIDATE_DEPTH,
            )
            fusion_times.append(perf_counter() - fusion_started)
        except Exception as error:
            raise RetrievalExperimentError(
                f"案例 {case.case_id} 的 Vector/BM25/RRF 阶段失败，"
                f"未生成完整 Day 14 结果：{type(error).__name__}"
            ) from error

        vector_by_case[case.case_id] = vector_candidates
        bm25_by_case[case.case_id] = bm25_candidates
        hybrid_by_case[case.case_id] = hybrid_candidates
        vector_ranks = component_ranks(vector_candidates)
        bm25_ranks = component_ranks(bm25_candidates)
        fusion_audit.append(
            HybridFusionCaseAudit(
                case_id=case.case_id,
                candidates=tuple(
                    HybridFusionCandidateAudit(
                        final_rank=item.rank,
                        chunk_id=item.chunk_id,
                        source_segment_ids=item.source_segment_ids,
                        rrf_score=item.score,
                        vector_rank=vector_ranks.get(item.chunk_id),
                        bm25_rank=bm25_ranks.get(item.chunk_id),
                    )
                    for item in hybrid_candidates
                ),
            )
        )

    if embedder.query_call_count != len(dataset.cases):
        raise RetrievalExperimentError("Vector 必须对每道题恰好调用一次查询 Embedding")
    if embedder.document_batch_count != 0:
        raise RetrievalExperimentError("Day 14 不得重新调用文档 Embedding")
    if bm25.retrieval_call_count != len(dataset.cases):
        raise RetrievalExperimentError("BM25 必须对每道题恰好检索一次")

    runs = (
        _score_precomputed(dataset, "vector", vector_by_case),
        _score_precomputed(dataset, "bm25", bm25_by_case),
        _score_precomputed(dataset, "hybrid_rrf", hybrid_by_case),
    )
    method_results = _method_results(runs, diagnostic_case_ids=_diagnostic_ids())
    vector_drift = _vector_drift(method_results[0], day12)
    reviews = _case_reviews(method_results)
    recommendation = _recommend(method_results, vector_drift=vector_drift)

    after_hashes = _input_hashes(paths)
    if after_hashes != before_hashes:
        raise RetrievalExperimentError("正式实验前后受保护输入哈希发生变化，拒绝保存")

    timestamp = datetime.now(UTC)
    identity = vector.corpus_identity
    result = RetrievalExperimentResult(
        schema_version=1,
        experiment_id=(
            f"{dataset.dataset_id}-retrieval-day14-"
            f"{timestamp.strftime('%Y%m%dt%H%M%Sz').lower()}"
        ),
        created_at_utc=timestamp,
        dataset_id=dataset.dataset_id,
        dataset_cases_sha256=dataset.cases_sha256,
        episode_id=dataset.episode_id,
        k_values=DEFAULT_K_VALUES,
        corpus_identity=Day14CorpusIdentity(
            episode_id=identity.episode_id,
            source_file=identity.source_file,
            chunking_algorithm=identity.chunking_algorithm,
            chunk_size=identity.chunk_size,
            overlap=identity.overlap,
            input_chunk_sha256=identity.input_chunk_sha256,
            chunk_count=len(identity.chunk_id_order),
            chunk_id_order_sha256=_ordered_ids_sha256(identity.chunk_id_order),
        ),
        input_artifact_hashes=before_hashes,
        bm25_configuration=_bm25_configuration(),
        rrf_configuration=_rrf_configuration(),
        selection_policy=SELECTION_POLICY,
        diagnostic_subset=DIAGNOSTIC_SUBSET,
        methods=method_results,
        hybrid_fusion_audit=tuple(fusion_audit),
        focus_and_regression_reviews=reviews,
        vector_drift=vector_drift,
        call_counts=Day14CallCounts(
            vector_retrieval_calls=20,
            vector_query_embedding_calls=embedder.query_call_count,
            bm25_retrieval_calls=bm25.retrieval_call_count,
            rrf_fusion_calls=20,
            document_embedding_calls=embedder.document_batch_count,
            answer_model_calls=0,
        ),
        latency_and_size=Day14LatencyAndSize(
            bm25_build_seconds=bm25.build_elapsed_seconds,
            vector_query_seconds=sum(vector_times),
            vector_query_median_seconds=statistics.median(vector_times),
            bm25_query_seconds=sum(bm25_times),
            bm25_query_median_seconds=statistics.median(bm25_times),
            rrf_fusion_seconds=sum(fusion_times),
            rrf_fusion_median_seconds=statistics.median(fusion_times),
            total_experiment_seconds=perf_counter() - started,
            bm25_document_token_count=bm25.document_token_count,
            bm25_vocabulary_size=bm25.vocabulary_size,
            bm25_posting_count=bm25.posting_count,
            bm25_python_memory_bytes_estimate=bm25.approximate_memory_bytes,
            faiss_file_bytes=paths["faiss"].stat().st_size,
            latency_evidence_limit=(
                "Vector includes one live provider run and network variability; BM25 and "
                "RRF are local per-query samples from this run. These timings show scale, "
                "not stable provider speed superiority."
            ),
        ),
        recommendation=recommendation,
    )
    save_retrieval_experiment(result, paths["output"], overwrite=overwrite)
    return result


def _score_precomputed(
    dataset,
    retriever_name: str,
    candidates_by_case_id: dict[str, list[RetrievedCandidate]],
) -> RetrievalEvaluationRun:
    return run_precomputed_retrieval_evaluation(
        dataset,
        retriever_name=retriever_name,
        candidates_by_case_id=candidates_by_case_id,
        candidate_depth=CHILD_CANDIDATE_DEPTH,
    )


def _method_results(
    runs: tuple[RetrievalEvaluationRun, ...],
    *,
    diagnostic_case_ids: set[str],
) -> tuple[RetrievalMethodResult, ...]:
    vector = runs[0]
    results: list[RetrievalMethodResult] = []
    semantics = {
        "vector": "cosine_similarity",
        "bm25": "bm25_okapi",
        "hybrid_rrf": "reciprocal_rank_fusion",
    }
    for run in runs:
        subset = [case for case in run.cases if case.case_id in diagnostic_case_ids]
        results.append(
            RetrievalMethodResult(
                retriever_name=run.retriever_name,
                score_semantics=semantics[run.retriever_name],
                retrieval_calls=len(run.cases),
                aggregate_metrics=run.aggregate_metrics,
                category_metrics=run.category_metrics,
                diagnostic_subset_metrics=aggregate_retrieval_cases(subset),
                delta_vs_vector=_metric_delta(run.aggregate_metrics, vector.aggregate_metrics),
                cases=run.cases,
            )
        )
    return tuple(results)  # type: ignore[return-value]


def _metric_delta(current, baseline) -> dict[int, MethodMetricDeltaAtK]:
    return {
        k: MethodMetricDeltaAtK(
            hit_count=current[k].hit_count - baseline[k].hit_count,
            hit_at_k=(current[k].hit_at_k or 0) - (baseline[k].hit_at_k or 0),
            macro_recall_sum=current[k].macro_recall_sum - baseline[k].macro_recall_sum,
            macro_recall_at_k=(current[k].macro_recall_at_k or 0)
            - (baseline[k].macro_recall_at_k or 0),
        )
        for k in DEFAULT_K_VALUES
    }


def _vector_drift(
    current: RetrievalMethodResult,
    day12,
) -> VectorDriftComparison:
    baseline_by_case = {case.case_id: case for case in day12.cases}
    ranking_drift = tuple(
        case.case_id
        for case in current.cases
        if tuple(item.chunk_id for item in case.retrieved_candidates[:8])
        != tuple(
            item.chunk_id
            for item in baseline_by_case[case.case_id].retrieved_candidates[:8]
        )
    )
    delta = _metric_delta(current.aggregate_metrics, day12.aggregate_metrics)
    metrics_match = all(
        item.hit_count == 0 and abs(item.macro_recall_sum) < 1e-12
        for item in delta.values()
    )
    return VectorDriftComparison(
        day12_evaluation_id=day12.evaluation_id,
        metrics_match_day12=metrics_match,
        top8_exact_order_match_count=20 - len(ranking_drift),
        top8_case_count=20,
        ranking_drift_case_ids=ranking_drift,
        metric_delta_vs_day12=delta,
        interpretation=(
            "Same-run Vector metrics match Day 12; ranking differences, if any, are "
            "recorded as provider/index drift and are not attributed to BM25 or RRF."
            if metrics_match
            else "Same-run Vector metrics differ from Day 12, so the change is recorded "
            "as provider/index drift rather than attributed to BM25 or RRF."
        ),
    )


def _case_reviews(
    methods: tuple[RetrievalMethodResult, ...],
) -> tuple[Day14CaseReview, ...]:
    by_method = {
        method.retriever_name: {case.case_id: case for case in method.cases}
        for method in methods
    }
    case_ids = list(FOCUS_CASE_IDS)
    vector_cases = by_method["vector"]
    for method_name in ("bm25", "hybrid_rrf"):
        for case_id, case in by_method[method_name].items():
            if any(
                vector_cases[case_id].metrics_by_k[k].hit_at_k is True
                and case.metrics_by_k[k].hit_at_k is False
                for k in (1, 3)
            ) and case_id not in case_ids:
                case_ids.append(case_id)
    return tuple(
        Day14CaseReview(
            case_id=case_id,
            review_reason=(
                "preselected_focus" if case_id in FOCUS_CASE_IDS else "new_hit_regression"
            ),
            methods=tuple(
                _method_case_review(by_method[name][case_id], name)
                for name in ("vector", "bm25", "hybrid_rrf")
            ),
        )
        for case_id in case_ids
    )


def _method_case_review(
    case: EvaluationCaseResult,
    retriever_name: str,
) -> MethodCaseReview:
    correct_ranks = {
        segment_id: tuple(
            candidate.rank
            for candidate in case.retrieved_candidates
            if segment_id in candidate.source_segment_ids
        )
        for segment_id in case.expected_source_segment_ids
    }
    retrieved = tuple(key for key, ranks in correct_ranks.items() if ranks)
    missing = tuple(key for key, ranks in correct_ranks.items() if not ranks)
    return MethodCaseReview(
        retriever_name=retriever_name,
        correct_candidate_ranks=correct_ranks,
        retrieved_expected_segment_ids=retrieved,
        missing_expected_segment_ids=missing,
        hit_at_1=case.metrics_by_k[1].hit_at_k,
        hit_at_3=case.metrics_by_k[3].hit_at_k,
        recall_at_5=case.metrics_by_k[5].recall_at_k,
        recall_at_8=case.metrics_by_k[8].recall_at_k,
    )


def _recommend(
    methods: tuple[RetrievalMethodResult, ...],
    *,
    vector_drift: VectorDriftComparison,
) -> Day14Recommendation:
    vector, _, hybrid = methods
    vector_cases = {case.case_id: case for case in vector.cases}
    hybrid_cases = {case.case_id: case for case in hybrid.cases}
    regressions_at_1 = tuple(
        case_id
        for case_id, case in hybrid_cases.items()
        if vector_cases[case_id].metrics_by_k[1].hit_at_k is True
        and case.metrics_by_k[1].hit_at_k is False
    )
    regressions_at_3 = tuple(
        case_id
        for case_id, case in hybrid_cases.items()
        if vector_cases[case_id].metrics_by_k[3].hit_at_k is True
        and case.metrics_by_k[3].hit_at_k is False
    )
    eligible = hybrid.aggregate_metrics[3].hit_count >= vector.aggregate_metrics[3].hit_count
    recall8_improved = (
        hybrid.aggregate_metrics[8].macro_recall_sum
        > vector.aggregate_metrics[8].macro_recall_sum
    )
    multi_not_lower = (
        hybrid.category_metrics["multi_segment"][8].macro_recall_sum
        >= vector.category_metrics["multi_segment"][8].macro_recall_sum
    )
    qualifies = (
        eligible
        and recall8_improved
        and multi_not_lower
        and not regressions_at_1
        and not regressions_at_3
    )
    if qualifies and vector_drift.metrics_match_day12:
        value = "hybrid_rrf_candidate_default"
        rationale = (
            "Hybrid-RRF satisfies the predeclared Hit@3 gate, strictly improves overall "
            "Macro Recall@8, does not lower multi-segment Recall@8, and introduces no new "
            "Hit@1/Hit@3 regression. It is only a candidate default; production remains unchanged."
        )
    elif qualifies:
        value = "insufficient_evidence_no_switch"
        rationale = (
            "Hybrid meets the same-run quality rule, but the current Vector metrics drifted "
            "from Day 12. The provider/index drift prevents a confident default change."
        )
    else:
        value = "retain_vector"
        rationale = (
            "Hybrid-RRF does not satisfy every predeclared requirement for a stable quality "
            "gain over the same-run Vector ranking, so the current Vector default is retained."
        )
    return Day14Recommendation(
        value=value,
        hybrid_eligible_by_hit_at_3=eligible,
        new_hybrid_hit_at_1_regressions=regressions_at_1,
        new_hybrid_hit_at_3_regressions=regressions_at_3,
        rationale=rationale,
        uncertainty=(
            "The evidence is one English episode, one embedding model, 16 scored cases, "
            "and one live provider run. Category and latency comparisons are diagnostic, "
            "not claims of broad or stable superiority."
        ),
        production_default_changed=False,
    )


def _resolve_paths(**values: str | Path) -> dict[str, Path]:
    index_directory = Path(values.pop("index_directory"))
    paths = {key.removesuffix("_path"): Path(value) for key, value in values.items()}
    paths["day12"] = paths.pop("day12_baseline")
    paths["day13"] = paths.pop("day13_experiment")
    paths["index_directory"] = index_directory
    paths["manifest"] = index_directory / MANIFEST_FILENAME
    paths["faiss"] = index_directory / INDEX_FILENAME
    return paths


def _preflight_output(paths: dict[str, Path], *, overwrite: bool) -> None:
    output = paths["output"]
    protected = {
        path.resolve()
        for key, path in paths.items()
        if key not in {"output", "index_directory"}
    }
    if output.resolve() in protected:
        raise RetrievalExperimentPersistenceError(
            "Day 14 输出不能覆盖任何冻结输入产物"
        )
    if output.exists() and not overwrite:
        raise RetrievalExperimentPersistenceError(
            f"目标 Day 14 结果已存在：{output}；不会静默覆盖"
        )


def _input_hashes(paths: dict[str, Path]) -> Day14InputArtifactHashes:
    return Day14InputArtifactHashes(
        dataset_file_sha256=file_sha256(paths["dataset"]),
        transcript_file_sha256=file_sha256(paths["transcript"]),
        chunk_catalog_file_sha256=file_sha256(paths["chunk_catalog"]),
        manifest_file_sha256=file_sha256(paths["manifest"]),
        faiss_file_sha256=file_sha256(paths["faiss"]),
        day12_baseline_sha256=file_sha256(paths["day12"]),
        day13_experiment_sha256=file_sha256(paths["day13"]),
    )


def _load_chunk_catalog(path: Path) -> ChunkedTranscript:
    try:
        return ChunkedTranscript.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as error:
        raise RetrievalExperimentError(
            "1000/200 Chunk catalog 无法严格加载："
            f"{safe_error_message(error)}"
        ) from error


def _load_day13_experiment(path: Path) -> ChunkParameterExperimentResult:
    try:
        return ChunkParameterExperimentResult.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, ValidationError) as error:
        raise RetrievalExperimentError(
            f"Day 13 正式结果无法严格加载：{safe_error_message(error)}"
        ) from error


def _validate_chunk_catalog(chunked, manifest) -> None:
    actual = (
        chunked.episode_id,
        chunked.source_file,
        chunked.algorithm,
        chunked.chunk_size,
        chunked.overlap,
        tuple(chunk.chunk_id for chunk in chunked.chunks),
        chunked_transcript_fingerprint(chunked),
    )
    expected = (
        manifest.episode_id,
        manifest.source_file,
        manifest.chunking_algorithm,
        manifest.chunk_size,
        manifest.overlap,
        tuple(manifest.chunk_id_order),
        manifest.input_chunk_sha256,
    )
    if actual != expected:
        raise RetrievalExperimentError(
            "Chunk catalog 与 Vector manifest 的 episode、顺序、配置或内容指纹不一致"
        )
    if (manifest.chunk_size, manifest.overlap) != (1000, 200):
        raise RetrievalExperimentError("Day 14 必须固定使用 1000/200 Chunk")


def _validate_formal_inputs(*, dataset, manifest, day12, day13, settings, hashes) -> None:
    if (
        day12.dataset_cases_sha256 != dataset.cases_sha256
        or day12.episode_id != dataset.episode_id
        or day12.input_artifact_hashes.dataset_file_sha256
        != hashes.dataset_file_sha256
        or day12.input_artifact_hashes.manifest_file_sha256
        != hashes.manifest_file_sha256
        or day12.input_artifact_hashes.faiss_file_sha256 != hashes.faiss_file_sha256
    ):
        raise RetrievalExperimentError("Day 12 正式基线与当前冻结输入不一致")
    if (
        day13.dataset_cases_sha256 != dataset.cases_sha256
        or day13.episode_id != dataset.episode_id
        or day13.selected_default.chunk_size != 1000
        or day13.selected_default.overlap != 200
        or not day13.selected_default.retained_day12_baseline
    ):
        raise RetrievalExperimentError("Day 13 正式结论没有保留当前 1000/200 基线")
    expected_settings = (
        manifest.embedding_model,
        manifest.vector_dimension,
        manifest.embedding_document_task,
        manifest.embedding_query_task,
        manifest.embedding_type,
        manifest.provider_normalized,
        manifest.provider_truncate,
    )
    actual_settings = (
        settings.model,
        settings.expected_dimension,
        settings.document_task,
        settings.query_task,
        settings.embedding_type,
        settings.normalized,
        settings.truncate,
    )
    if actual_settings != expected_settings:
        raise RetrievalExperimentError("Embedding 配置与当前 Vector manifest 不一致")


def _validate_predeclared_case_ids(dataset) -> None:
    available = {case.case_id for case in dataset.cases}
    declared = _diagnostic_ids() | set(FOCUS_CASE_IDS)
    if not declared.issubset(available):
        raise RetrievalExperimentError("预先声明的诊断或重点 case ID 不在冻结评测集")
    if any(
        not case.answerable
        for case in dataset.cases
        if case.case_id in _diagnostic_ids()
    ):
        raise RetrievalExperimentError("专有名词诊断子集必须全部为可计分案例")


def _diagnostic_ids() -> set[str]:
    return {item.case_id for item in DIAGNOSTIC_SUBSET}


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


def _bm25_configuration() -> BM25Configuration:
    return BM25Configuration(
        implementation="auditable_bm25_okapi_v1",
        score_semantics="bm25_okapi",
        idf_formula="ln(1+(N-df+0.5)/(df+0.5))",
        k1=BM25_K1,
        b=BM25_B,
        query_term_duplicates="deduplicated_in_first_occurrence_order",
        equal_score_tiebreaker="chunk_catalog_position_ascending",
        tokenizer=BM25TokenizerConfiguration(
            version=TOKENIZER_VERSION,
            unicode_normalization="NFKC",
            casing="casefold",
            token_characters="unicode_letters_and_numbers",
            hyphen_handling="preserve_internal_and_normalize_variants",
            apostrophe_handling="preserve_internal_and_normalize_variants",
            stemming=False,
            stopword_removal=False,
        ),
    )


def _rrf_configuration() -> RRFConfiguration:
    return RRFConfiguration(
        implementation="reciprocal_rank_fusion_v1",
        score_semantics="reciprocal_rank_fusion",
        formula="sum(1/(rrf_k+rank_i))",
        rank_origin=1,
        rrf_k=RRF_K,
        child_candidate_depth=CHILD_CANDIDATE_DEPTH,
        final_k_values=DEFAULT_K_VALUES,
        merge_key="chunk_id",
        equal_score_tiebreakers=(
            "best_child_rank_ascending",
            "chunk_catalog_position_ascending",
            "chunk_id_ascending",
        ),
    )


def _ordered_ids_sha256(chunk_ids: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(chunk_ids).encode("utf-8")).hexdigest()


def save_retrieval_experiment(
    result: RetrievalExperimentResult,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise RetrievalExperimentPersistenceError(
            f"目标 Day 14 结果已存在：{destination}；不会静默覆盖"
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
        raise RetrievalExperimentPersistenceError("Day 14 结果写入失败") from error
    return destination
