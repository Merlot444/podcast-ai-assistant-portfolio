"""Day 15 frozen-ranking rerank experiment, evaluation and safe persistence."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from pydantic import ValidationError

from app.core.errors import safe_error_message
from app.chunking.pipeline import load_processed_transcript
from app.evaluation.baseline import file_sha256
from app.evaluation.core import (
    DEFAULT_EVALUATION_DATASET,
    DEFAULT_EVALUATION_INDEX_DIRECTORY,
    DEFAULT_EVALUATION_TRANSCRIPT,
    DEFAULT_RETRIEVAL_BASELINE,
    load_evaluation_dataset,
    validate_evaluation_sources,
)
from app.evaluation.retrieval_experiment import (
    DEFAULT_CHUNK_CATALOG,
    DEFAULT_DAY13_EXPERIMENT,
    DEFAULT_RETRIEVAL_EXPERIMENT_OUTPUT,
    DIAGNOSTIC_SUBSET,
)
from app.evaluation.runner import (
    DEFAULT_K_VALUES,
    aggregate_retrieval_cases,
    run_precomputed_retrieval_evaluation,
)
from app.indexing import load_vector_index
from app.indexing.index import chunked_transcript_fingerprint
from app.retrieval.rerank import (
    JINA_RERANK_ENDPOINT,
    JINA_RERANK_MODEL,
    RERANK_CANDIDATE_DEPTH,
    RERANK_EVALUATION_K_VALUES,
    RERANK_MAX_DOCUMENT_CHARACTERS,
    RERANK_MAX_INPUT_TOKENS,
    RERANK_MAX_RETRIES,
    RERANK_PRODUCT_FOCUS_K,
    RERANK_RETURNED_DEPTH,
    RERANK_TIMEOUT_SECONDS,
    JinaAPIReranker,
    RerankInput,
    RerankedCandidate,
    Reranker,
)
from app.schemas.chunk import ChunkedTranscript
from app.schemas.chunk_experiment import ChunkParameterExperimentResult
from app.schemas.evaluation_result import (
    EvaluationCaseResult,
    MetricSummaryAtK,
    RetrievalBaselineResult,
)
from app.schemas.rerank_experiment import (
    CandidatePoolCaseLimit,
    CandidatePoolLimit,
    Day15CaseReview,
    Day15InputArtifactHashes,
    Day15MethodCaseReview,
    Day15MethodResult,
    Day15Recommendation,
    Day15SelectionPolicy,
    LatencySummary,
    RegressionSummary,
    RerankCallCounts,
    RerankCandidateAudit,
    RerankCaseAudit,
    RerankExperimentResult,
    RerankerConfiguration,
    RerankLatencyAndCost,
    RerankRunAudit,
)
from app.schemas.retrieval_experiment import (
    MethodMetricDeltaAtK,
    RetrievalExperimentResult,
    RetrievalMethodResult,
)
from app.schemas.retriever import RetrievedCandidate


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RERANK_EXPERIMENT_OUTPUT = (
    _PROJECT_ROOT
    / "data"
    / "evaluation"
    / "eval_dataset_v1-rerank-experiment-day15.json"
)
EXPECTED_DAY14_EXPERIMENT_SHA256 = (
    "bbbeae51724c13032e5efc41394558988c81ec431302bcafab135aa456491de5"
)
FOCUS_CASE_IDS = (
    "gemini_scaling_limits",
    "agi_timeline_uncertainty",
    "alignment_safeguards_pause_trigger",
    "neuroscience_provenance",
    "agi_architecture_original_bets",
)
HYBRID_DAY14_HIT1_REGRESSIONS = (
    "agi_timeline_uncertainty",
    "alignment_safeguards_pause_trigger",
    "neuroscience_provenance",
)

SELECTION_POLICY = Day15SelectionPolicy(
    fixed_before_results=True,
    vector_rerank_requirements=(
        "Hit@3 remains 16/16",
        "no new Hit@1 regression relative to Vector",
        "no new Hit@3 regression relative to Vector",
        "Macro Recall@1 is strictly higher than Vector",
        "Macro Recall@3/5/8 are not lower than Vector",
        "multi-segment Macro Recall@1/3/5/8 are not lower than Vector",
        "Recall@1 improvement occurs in at least two distinct cases",
        "request P95 is at most 5 seconds with zero failed formal calls",
    ),
    hybrid_rerank_additional_requirements=(
        "all three Day 14 Hybrid Hit@1 regressions are repaired",
        "Macro Recall@1 is strictly higher than pure Vector",
        "the result exceeds pure Vector rather than merely restoring it at added cost",
    ),
    minimum_distinct_recall_at_1_improvement_cases=2,
    maximum_acceptable_p95_request_seconds=5.0,
    one_case_percentage_points=6.25,
    production_change_deferred_to_day17=True,
    small_sample_warning=(
        "Only 16 scored questions from one English episode are available; one case "
        "equals 6.25 percentage points and no result establishes broad generality."
    ),
)


class RerankExperimentError(RuntimeError):
    """Raised when a complete trustworthy Day 15 result cannot be produced."""


class RerankExperimentPersistenceError(RuntimeError):
    """Raised when the formal Day 15 artifact cannot be written safely."""


def run_rerank_experiment(
    *,
    api_key: str | None = None,
    reranker: Reranker | None = None,
    dataset_path: str | Path = DEFAULT_EVALUATION_DATASET,
    transcript_path: str | Path = DEFAULT_EVALUATION_TRANSCRIPT,
    chunk_catalog_path: str | Path = DEFAULT_CHUNK_CATALOG,
    index_directory: str | Path = DEFAULT_EVALUATION_INDEX_DIRECTORY,
    day12_baseline_path: str | Path = DEFAULT_RETRIEVAL_BASELINE,
    day13_experiment_path: str | Path = DEFAULT_DAY13_EXPERIMENT,
    day14_experiment_path: str | Path = DEFAULT_RETRIEVAL_EXPERIMENT_OUTPUT,
    output_path: str | Path = DEFAULT_RERANK_EXPERIMENT_OUTPUT,
    prior_failed_rerank_attempts: int = 0,
    prior_successful_synthetic_diagnostic_requests: int = 0,
    overwrite: bool = False,
) -> RerankExperimentResult:
    """Load two frozen Top-20 sources, rerank independently, score and save once."""
    if (
        isinstance(prior_failed_rerank_attempts, bool)
        or not isinstance(prior_failed_rerank_attempts, int)
        or prior_failed_rerank_attempts < 0
    ):
        raise RerankExperimentError("正式运行前失败的 Rerank 尝试次数必须是非负整数")
    if (
        isinstance(prior_successful_synthetic_diagnostic_requests, bool)
        or not isinstance(prior_successful_synthetic_diagnostic_requests, int)
        or prior_successful_synthetic_diagnostic_requests < 0
    ):
        raise RerankExperimentError("正式运行前成功的合成诊断请求次数必须是非负整数")
    started = perf_counter()
    paths = _resolve_paths(
        dataset_path=dataset_path,
        transcript_path=transcript_path,
        chunk_catalog_path=chunk_catalog_path,
        index_directory=index_directory,
        day12_baseline_path=day12_baseline_path,
        day13_experiment_path=day13_experiment_path,
        day14_experiment_path=day14_experiment_path,
        output_path=output_path,
    )
    _preflight_output(paths, overwrite=overwrite)
    before_hashes = _input_hashes(paths)
    if before_hashes.day14_experiment_sha256 != EXPECTED_DAY14_EXPERIMENT_SHA256:
        raise RerankExperimentError("Day 14 正式 artifact SHA-256 与冻结值不一致")

    dataset = load_evaluation_dataset(paths["dataset"])
    transcript = load_processed_transcript(paths["transcript"])
    vector_index = load_vector_index(paths["index_directory"])
    validate_evaluation_sources(
        dataset,
        transcript=transcript,
        manifest=vector_index.manifest,
    )
    chunked = _load_model(paths["chunk_catalog"], ChunkedTranscript, "Chunk catalog")
    day12 = _load_model(paths["day12"], RetrievalBaselineResult, "Day 12 baseline")
    day13 = _load_model(paths["day13"], ChunkParameterExperimentResult, "Day 13 result")
    day14 = _load_model(paths["day14"], RetrievalExperimentResult, "Day 14 result")
    _validate_formal_inputs(
        dataset=dataset,
        chunked=chunked,
        manifest=vector_index.manifest,
        day12=day12,
        day13=day13,
        day14=day14,
        hashes=before_hashes,
    )

    actual_reranker = reranker or JinaAPIReranker(api_key=api_key or "")
    if not isinstance(actual_reranker, JinaAPIReranker):
        raise RerankExperimentError("正式编排只接受严格的 JinaAPIReranker 实现")

    vector_method = _day14_method(day14, "vector")
    hybrid_method = _day14_method(day14, "hybrid_rrf")
    catalog_position = {
        chunk_id: position
        for position, chunk_id in enumerate(vector_index.manifest.chunk_id_order)
    }
    chunk_by_id = {chunk.chunk_id: chunk for chunk in vector_index.manifest.chunks}
    candidate_pool_limits = (
        _candidate_pool_limit(vector_method),
        _candidate_pool_limit(hybrid_method),
    )

    vector_outputs, vector_audit, vector_times = _run_one_source(
        dataset=dataset,
        method=vector_method,
        output_method="vector_rerank",
        reranker=actual_reranker,
        chunk_by_id=chunk_by_id,
        catalog_position=catalog_position,
    )
    hybrid_outputs, hybrid_audit, hybrid_times = _run_one_source(
        dataset=dataset,
        method=hybrid_method,
        output_method="hybrid_rrf_rerank",
        reranker=actual_reranker,
        chunk_by_id=chunk_by_id,
        catalog_position=catalog_position,
    )
    if actual_reranker.request_attempt_count != 40 or len(actual_reranker.telemetry) != 40:
        raise RerankExperimentError("Day 15 必须恰好完成 40 次成功 Rerank 请求")

    vector_rerank_run = run_precomputed_retrieval_evaluation(
        dataset,
        retriever_name="vector_rerank",
        candidates_by_case_id=vector_outputs,
        candidate_depth=20,
    )
    hybrid_rerank_run = run_precomputed_retrieval_evaluation(
        dataset,
        retriever_name="hybrid_rrf_rerank",
        candidates_by_case_id=hybrid_outputs,
        candidate_depth=20,
    )
    methods = _method_results(
        vector_method=vector_method,
        vector_rerank_run=vector_rerank_run,
        hybrid_method=hybrid_method,
        hybrid_rerank_run=hybrid_rerank_run,
    )
    regressions = _regression_summaries(methods)
    reviews = _case_reviews(methods)
    all_times = vector_times + hybrid_times
    latency_acceptable = _p95(all_times) <= 5.0
    recommendation = _recommend(
        methods,
        regressions=regressions,
        latency_acceptable=latency_acceptable,
    )

    after_hashes = _input_hashes(paths)
    if after_hashes != before_hashes:
        raise RerankExperimentError("正式实验前后冻结输入哈希发生变化，拒绝保存")

    timestamp = datetime.now(UTC)
    telemetry = actual_reranker.telemetry
    token_values = [item.provider_total_tokens for item in telemetry]
    provider_tokens = (
        sum(value for value in token_values if value is not None)
        if all(value is not None for value in token_values)
        else None
    )
    result = RerankExperimentResult(
        schema_version=1,
        experiment_id=(
            f"{dataset.dataset_id}-rerank-day15-"
            f"{timestamp.strftime('%Y%m%dt%H%M%Sz').lower()}"
        ),
        created_at_utc=timestamp,
        dataset_id=dataset.dataset_id,
        dataset_cases_sha256=dataset.cases_sha256,
        episode_id=dataset.episode_id,
        k_values=DEFAULT_K_VALUES,
        corpus_identity=day14.corpus_identity,
        day14_experiment_id=day14.experiment_id,
        input_artifact_hashes=before_hashes,
        reranker_configuration=_reranker_configuration(),
        selection_policy=SELECTION_POLICY,
        diagnostic_subset=DIAGNOSTIC_SUBSET,
        candidate_pool_limits=candidate_pool_limits,
        methods=methods,
        rerank_audits=(vector_audit, hybrid_audit),
        regression_summaries=regressions,
        focus_and_regression_reviews=reviews,
        call_counts=RerankCallCounts(
            vector_query_embedding_calls=0,
            vector_retrieval_calls=0,
            bm25_retrieval_calls=0,
            rrf_fusion_calls=0,
            rerank_logical_calls=40,
            rerank_http_attempts=actual_reranker.request_attempt_count,
            successful_provider_responses=len(actual_reranker.telemetry),
            pre_formal_failed_rerank_attempts=prior_failed_rerank_attempts,
            pre_formal_successful_synthetic_diagnostic_requests=(
                prior_successful_synthetic_diagnostic_requests
            ),
            query_document_pairs=sum(item.document_count for item in telemetry),
            document_embedding_calls=0,
            answer_model_calls=0,
        ),
        latency_and_cost=RerankLatencyAndCost(
            model_load_seconds=None,
            cold_start_note="remote_api_not_measured_separately",
            vector_rerank=_latency_summary(vector_times),
            hybrid_rrf_rerank=_latency_summary(hybrid_times),
            all_requests=_latency_summary(all_times),
            total_experiment_seconds=perf_counter() - started,
            query_characters=sum(item.query_characters for item in telemetry),
            document_characters=sum(item.document_characters for item in telemetry),
            provider_reported_total_tokens=provider_tokens,
            provider_billing_unit="input_tokens",
            confirmed_monetary_cost_usd=None,
            monetary_cost_status="unknown_provider_did_not_return_currency_cost",
            local_peak_memory_bytes=None,
            local_model_disk_bytes=None,
            remote_model_parameter_count=597_000_000,
            latency_evidence_limit=(
                "One sequential remote run includes provider and network variability. "
                "It establishes request-scale evidence, not stable production latency."
            ),
        ),
        recommendation=recommendation,
        limitations=(
            "One English episode and 16 scored cases cannot establish broad superiority.",
            "Rerank can only reorder sources already present in each frozen Top-20 pool.",
            "Provider model revision was not returned, so only the model name is pinned.",
            "The provider returned token usage but no currency-denominated actual charge.",
            "Remote cold start and provider-side memory or disk footprint are not observable.",
            "No answer quality, citation correctness or unanswerable refusal was evaluated.",
        ),
    )
    save_rerank_experiment(result, paths["output"], overwrite=overwrite)
    return result


def _run_one_source(
    *,
    dataset,
    method: RetrievalMethodResult,
    output_method: str,
    reranker: JinaAPIReranker,
    chunk_by_id: dict[str, object],
    catalog_position: dict[str, int],
) -> tuple[dict[str, list[RetrievedCandidate]], RerankRunAudit, list[float]]:
    cases_by_id = {case.case_id: case for case in method.cases}
    outputs: dict[str, list[RetrievedCandidate]] = {}
    audits: list[RerankCaseAudit] = []
    times: list[float] = []
    for case in dataset.cases:
        original_case = cases_by_id[case.case_id]
        inputs: list[RerankInput] = []
        for candidate in original_case.retrieved_candidates:
            chunk = chunk_by_id.get(candidate.chunk_id)
            if chunk is None:
                raise RerankExperimentError(
                    f"案例 {case.case_id} 的候选 Chunk 无法映射到 manifest 正文"
                )
            source_segment_ids = tuple(getattr(chunk, "source_segment_ids"))
            text = getattr(chunk, "text")
            if source_segment_ids != candidate.source_segment_ids:
                raise RerankExperimentError(
                    f"案例 {case.case_id} 的候选来源与 manifest 不一致"
                )
            inputs.append(
                RerankInput(
                    chunk_id=candidate.chunk_id,
                    text=text,
                    source_segment_ids=candidate.source_segment_ids,
                    original_retriever=method.retriever_name,
                    original_rank=candidate.rank,
                    original_score=candidate.score,
                    original_score_semantics=candidate.score_semantics,
                    catalog_position=catalog_position[candidate.chunk_id],
                )
            )
        if len(inputs) != 20:
            raise RerankExperimentError(
                f"案例 {case.case_id} 的 {method.retriever_name} 候选不是完整 Top-20"
            )
        try:
            started = perf_counter()
            reranked = reranker.rerank(case.question, inputs, top_n=20)
            times.append(perf_counter() - started)
        except Exception as error:
            raise RerankExperimentError(
                f"案例 {case.case_id} 的 {method.retriever_name} Rerank 失败，"
                f"未生成完整 Day 15 结果：{type(error).__name__}"
            ) from error
        outputs[case.case_id] = [
            RetrievedCandidate(
                rank=item.rerank_rank,
                chunk_id=item.chunk_id,
                source_segment_ids=item.source_segment_ids,
                score=item.rerank_score,
                score_semantics=item.rerank_score_semantics,
                retriever_name=output_method,
            )
            for item in reranked
        ]
        audits.append(
            RerankCaseAudit(
                case_id=case.case_id,
                candidates=tuple(_audit_candidate(item) for item in reranked),
            )
        )
    return (
        outputs,
        RerankRunAudit(
            source_method=method.retriever_name,
            output_method=output_method,
            cases=tuple(audits),
        ),
        times,
    )


def _audit_candidate(item: RerankedCandidate) -> RerankCandidateAudit:
    return RerankCandidateAudit(
        chunk_id=item.chunk_id,
        source_segment_ids=item.source_segment_ids,
        original_retriever=item.original_retriever,
        original_rank=item.original_rank,
        original_score=item.original_score,
        original_score_semantics=item.original_score_semantics,
        rerank_rank=item.rerank_rank,
        rerank_score=item.rerank_score,
        rerank_score_semantics=item.rerank_score_semantics,
    )


def _method_results(
    *,
    vector_method: RetrievalMethodResult,
    vector_rerank_run,
    hybrid_method: RetrievalMethodResult,
    hybrid_rerank_run,
) -> tuple[Day15MethodResult, ...]:
    vector_metrics = vector_method.aggregate_metrics
    diagnostic_ids = {item.case_id for item in DIAGNOSTIC_SUBSET}

    def build(
        *,
        method_name: str,
        score_semantics: str,
        model_calls: int,
        aggregate,
        categories,
        cases: Sequence[EvaluationCaseResult],
        diagnostic=None,
    ) -> Day15MethodResult:
        subset = [case for case in cases if case.case_id in diagnostic_ids]
        return Day15MethodResult(
            method_name=method_name,
            score_semantics=score_semantics,
            day15_model_calls=model_calls,
            aggregate_metrics=aggregate,
            category_metrics=categories,
            diagnostic_subset_metrics=(
                diagnostic or aggregate_retrieval_cases(subset)
            ),
            delta_vs_vector=_metric_delta(aggregate, vector_metrics),
            cases=tuple(cases),
        )

    return (
        build(
            method_name="vector",
            score_semantics="cosine_similarity",
            model_calls=0,
            aggregate=vector_method.aggregate_metrics,
            categories=vector_method.category_metrics,
            diagnostic=vector_method.diagnostic_subset_metrics,
            cases=vector_method.cases,
        ),
        build(
            method_name="vector_rerank",
            score_semantics="jina_relevance_score_higher_is_better",
            model_calls=20,
            aggregate=vector_rerank_run.aggregate_metrics,
            categories=vector_rerank_run.category_metrics,
            cases=vector_rerank_run.cases,
        ),
        build(
            method_name="hybrid_rrf",
            score_semantics="reciprocal_rank_fusion",
            model_calls=0,
            aggregate=hybrid_method.aggregate_metrics,
            categories=hybrid_method.category_metrics,
            diagnostic=hybrid_method.diagnostic_subset_metrics,
            cases=hybrid_method.cases,
        ),
        build(
            method_name="hybrid_rrf_rerank",
            score_semantics="jina_relevance_score_higher_is_better",
            model_calls=20,
            aggregate=hybrid_rerank_run.aggregate_metrics,
            categories=hybrid_rerank_run.category_metrics,
            cases=hybrid_rerank_run.cases,
        ),
    )


def _candidate_pool_limit(method: RetrievalMethodResult) -> CandidatePoolLimit:
    case_limits: list[CandidatePoolCaseLimit] = []
    scored: list[tuple[bool, float]] = []
    multi: list[tuple[bool, float]] = []
    for case in method.cases:
        retrieved = {
            segment_id
            for candidate in case.retrieved_candidates
            for segment_id in candidate.source_segment_ids
        }
        present = tuple(
            segment_id
            for segment_id in case.expected_source_segment_ids
            if segment_id in retrieved
        )
        missing = tuple(
            segment_id
            for segment_id in case.expected_source_segment_ids
            if segment_id not in retrieved
        )
        hit = bool(present) if case.answerable else None
        recall = (
            len(present) / len(case.expected_source_segment_ids)
            if case.answerable
            else None
        )
        case_limits.append(
            CandidatePoolCaseLimit(
                case_id=case.case_id,
                answerable=case.answerable,
                expected_source_segment_ids=case.expected_source_segment_ids,
                present_expected_source_segment_ids=present,
                missing_expected_source_segment_ids=missing,
                hit_at_20=hit,
                recall_at_20=recall,
            )
        )
        if case.answerable:
            scored.append((bool(hit), float(recall)))
            if case.category == "multi_segment":
                multi.append((bool(hit), float(recall)))
    return CandidatePoolLimit(
        source_method=method.retriever_name,
        candidate_depth=20,
        aggregate_at_20=_summary(scored),
        multi_segment_at_20=_summary(multi),
        cases=tuple(case_limits),
    )


def _summary(values: Sequence[tuple[bool, float]]) -> MetricSummaryAtK:
    denominator = len(values)
    hit_count = sum(hit for hit, _ in values)
    recall_sum = sum(recall for _, recall in values)
    return MetricSummaryAtK(
        hit_count=hit_count,
        hit_denominator=denominator,
        hit_at_k=hit_count / denominator if denominator else None,
        macro_recall_sum=recall_sum,
        macro_recall_denominator=denominator,
        macro_recall_at_k=recall_sum / denominator if denominator else None,
    )


def _regression_summaries(
    methods: tuple[Day15MethodResult, ...],
) -> tuple[RegressionSummary, ...]:
    by_name = {method.method_name: method for method in methods}
    vector = {case.case_id: case for case in by_name["vector"].cases}
    summaries: list[RegressionSummary] = []
    source_by_method = {
        "vector": "vector",
        "vector_rerank": "vector",
        "hybrid_rrf": "hybrid_rrf",
        "hybrid_rrf_rerank": "hybrid_rrf",
    }
    for method in methods:
        current = {case.case_id: case for case in method.cases}
        source = {
            case.case_id: case
            for case in by_name[source_by_method[method.method_name]].cases
        }
        summaries.append(
            RegressionSummary(
                method_name=method.method_name,
                new_hit_at_1_regressions_vs_vector=_hit_changes(vector, current, 1, True),
                fixed_hit_at_1_failures_vs_vector=_hit_changes(vector, current, 1, False),
                new_hit_at_3_regressions_vs_vector=_hit_changes(vector, current, 3, True),
                fixed_hit_at_3_failures_vs_vector=_hit_changes(vector, current, 3, False),
                recall_at_1_improvement_case_ids_vs_vector=tuple(
                    case_id
                    for case_id, case in current.items()
                    if _recall(case, 1) > _recall(vector[case_id], 1)
                ),
                source_method_hit_at_1_repairs=_hit_changes(source, current, 1, False),
                source_method_hit_at_1_regressions=_hit_changes(source, current, 1, True),
            )
        )
    return tuple(summaries)


def _hit_changes(base, current, k: int, regressions: bool) -> tuple[str, ...]:
    return tuple(
        case_id
        for case_id, case in current.items()
        if (
            base[case_id].metrics_by_k[k].hit_at_k is True
            and case.metrics_by_k[k].hit_at_k is False
            if regressions
            else base[case_id].metrics_by_k[k].hit_at_k is False
            and case.metrics_by_k[k].hit_at_k is True
        )
    )


def _recall(case: EvaluationCaseResult, k: int) -> float:
    return case.metrics_by_k[k].recall_at_k or 0.0


def _case_reviews(methods: tuple[Day15MethodResult, ...]) -> tuple[Day15CaseReview, ...]:
    by_method = {
        method.method_name: {case.case_id: case for case in method.cases}
        for method in methods
    }
    case_ids = list(FOCUS_CASE_IDS)
    vector = by_method["vector"]
    for method in methods[1:]:
        for case_id, case in by_method[method.method_name].items():
            if any(
                vector[case_id].metrics_by_k[k].hit_at_k is True
                and case.metrics_by_k[k].hit_at_k is False
                for k in (1, 3)
            ) and case_id not in case_ids:
                case_ids.append(case_id)
    return tuple(
        Day15CaseReview(
            case_id=case_id,
            review_reason=(
                "predeclared_focus" if case_id in FOCUS_CASE_IDS else "new_hit_regression"
            ),
            methods=tuple(
                _method_case_review(method.method_name, by_method[method.method_name][case_id])
                for method in methods
            ),
        )
        for case_id in case_ids
    )


def _method_case_review(method_name: str, case: EvaluationCaseResult) -> Day15MethodCaseReview:
    return Day15MethodCaseReview(
        method_name=method_name,
        correct_candidate_ranks={
            segment_id: tuple(
                candidate.rank
                for candidate in case.retrieved_candidates
                if segment_id in candidate.source_segment_ids
            )
            for segment_id in case.expected_source_segment_ids
        },
        hit_at_1=case.metrics_by_k[1].hit_at_k,
        hit_at_3=case.metrics_by_k[3].hit_at_k,
        recall_at_5=case.metrics_by_k[5].recall_at_k,
        recall_at_8=case.metrics_by_k[8].recall_at_k,
    )


def _recommend(
    methods: tuple[Day15MethodResult, ...],
    *,
    regressions: tuple[RegressionSummary, ...],
    latency_acceptable: bool,
) -> Day15Recommendation:
    by_name = {method.method_name: method for method in methods}
    regression_by_name = {item.method_name: item for item in regressions}
    vector = by_name["vector"]
    vector_rerank = by_name["vector_rerank"]
    hybrid_rerank = by_name["hybrid_rrf_rerank"]

    def base_quality(method: Day15MethodResult) -> bool:
        summary = regression_by_name[method.method_name]
        return (
            method.aggregate_metrics[3].hit_count == 16
            and not summary.new_hit_at_1_regressions_vs_vector
            and not summary.new_hit_at_3_regressions_vs_vector
            and method.aggregate_metrics[1].macro_recall_sum
            > vector.aggregate_metrics[1].macro_recall_sum
            and all(
                method.aggregate_metrics[k].macro_recall_sum
                >= vector.aggregate_metrics[k].macro_recall_sum
                for k in (3, 5, 8)
            )
            and all(
                method.category_metrics["multi_segment"][k].macro_recall_sum
                >= vector.category_metrics["multi_segment"][k].macro_recall_sum
                for k in (1, 3, 5, 8)
            )
            and len(summary.recall_at_1_improvement_case_ids_vs_vector) >= 2
        )

    vector_quality = base_quality(vector_rerank)
    hybrid_summary = regression_by_name["hybrid_rrf_rerank"]
    hybrid_quality = (
        base_quality(hybrid_rerank)
        and set(HYBRID_DAY14_HIT1_REGRESSIONS).issubset(
            hybrid_summary.source_method_hit_at_1_repairs
        )
        and hybrid_rerank.aggregate_metrics[1].macro_recall_sum
        > vector.aggregate_metrics[1].macro_recall_sum
    )
    any_improvement = any(
        regression_by_name[name].recall_at_1_improvement_case_ids_vs_vector
        for name in ("vector_rerank", "hybrid_rrf_rerank")
    )
    if vector_quality or hybrid_quality:
        value = "insufficient_sample_no_switch"
        rationale = (
            "At least one reranked method satisfies the frozen quality rule, but the "
            "provider returned no currency-denominated charge and the evidence remains "
            "one episode with 16 scored cases. Keep Vector until Day 17 synthesis."
        )
    elif any_improvement:
        value = "rerank_query_specific"
        rationale = (
            "Rerank improves selected frozen cases but does not satisfy every predeclared "
            "default-candidate gate. Keep Vector and treat reranking as query-specific evidence."
        )
    else:
        value = "retain_vector"
        rationale = (
            "Neither reranked method satisfies the predeclared quality gates or shows a "
            "clear multi-case Recall@1 benefit over Vector."
        )
    return Day15Recommendation(
        value=value,
        vector_rerank_quality_qualified=vector_quality,
        hybrid_rerank_quality_qualified=hybrid_quality,
        latency_acceptable_by_predeclared_threshold=latency_acceptable,
        monetary_cost_confirmed=False,
        production_default_changed=False,
        rationale=rationale,
        uncertainty=(
            "One case equals 6.25 Hit percentage points. Provider/network latency and the "
            "unreported monetary charge may vary, and no answer-layer behavior was tested."
        ),
    )


def _reranker_configuration() -> RerankerConfiguration:
    return RerankerConfiguration(
        backend="jina_reranker_api",
        provider="Jina AI",
        endpoint=JINA_RERANK_ENDPOINT,
        model_name=JINA_RERANK_MODEL,
        model_revision=None,
        model_parameter_count=597_000_000,
        score_semantics="jina_relevance_score_higher_is_better",
        candidate_depth=RERANK_CANDIDATE_DEPTH,
        returned_depth=RERANK_RETURNED_DEPTH,
        product_focus_k=RERANK_PRODUCT_FOCUS_K,
        evaluation_k_values=RERANK_EVALUATION_K_VALUES,
        maximum_input_tokens=RERANK_MAX_INPUT_TOKENS,
        maximum_documents_per_model_pass=64,
        application_document_character_limit=RERANK_MAX_DOCUMENT_CHARACTERS,
        truncation_rule="first_1000_unicode_code_points",
        provider_auto_truncation=True,
        timeout_seconds=RERANK_TIMEOUT_SECONDS,
        maximum_retries=RERANK_MAX_RETRIES,
        batching="one_query_with_20_documents_per_request",
        concurrency="sequential_single_request",
        return_documents=False,
        input_order="original_rank_ascending",
        equal_score_tiebreakers=(
            "original_rank_ascending",
            "chunk_catalog_position_ascending",
            "chunk_id_ascending",
        ),
        official_model_url="https://jina.ai/models/jina-reranker-v3/",
    )


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


def _latency_summary(values: Sequence[float]) -> LatencySummary:
    if not values:
        return LatencySummary(
            request_count=0,
            total_seconds=0.0,
            median_seconds=0.0,
            p95_seconds=0.0,
            maximum_seconds=0.0,
        )
    return LatencySummary(
        request_count=len(values),
        total_seconds=sum(values),
        median_seconds=statistics.median(values),
        p95_seconds=_p95(values),
        maximum_seconds=max(values),
    )


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return ordered[index]


def _day14_method(day14: RetrievalExperimentResult, name: str) -> RetrievalMethodResult:
    return next(method for method in day14.methods if method.retriever_name == name)


def _resolve_paths(**values: str | Path) -> dict[str, Path]:
    index_directory = Path(values.pop("index_directory"))
    paths = {key.removesuffix("_path"): Path(value) for key, value in values.items()}
    paths["day12"] = paths.pop("day12_baseline")
    paths["day13"] = paths.pop("day13_experiment")
    paths["day14"] = paths.pop("day14_experiment")
    paths["index_directory"] = index_directory
    paths["manifest"] = index_directory / "manifest.json"
    paths["faiss"] = index_directory / "index.faiss"
    return paths


def _preflight_output(paths: dict[str, Path], *, overwrite: bool) -> None:
    protected = {
        path.resolve()
        for key, path in paths.items()
        if key not in {"output", "index_directory"}
    }
    if paths["output"].resolve() in protected:
        raise RerankExperimentPersistenceError("Day 15 输出不能覆盖任何冻结输入")
    if paths["output"].exists() and not overwrite:
        raise RerankExperimentPersistenceError(
            f"目标 Day 15 结果已存在：{paths['output']}；不会静默覆盖"
        )


def _input_hashes(paths: dict[str, Path]) -> Day15InputArtifactHashes:
    return Day15InputArtifactHashes(
        dataset_file_sha256=file_sha256(paths["dataset"]),
        transcript_file_sha256=file_sha256(paths["transcript"]),
        chunk_catalog_file_sha256=file_sha256(paths["chunk_catalog"]),
        manifest_file_sha256=file_sha256(paths["manifest"]),
        faiss_file_sha256=file_sha256(paths["faiss"]),
        day12_baseline_sha256=file_sha256(paths["day12"]),
        day13_experiment_sha256=file_sha256(paths["day13"]),
        day14_experiment_sha256=file_sha256(paths["day14"]),
    )


def _load_model(path: Path, model, label: str):
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as error:
        raise RerankExperimentError(
            f"{label} 无法严格加载：{safe_error_message(error)}"
        ) from error


def _validate_formal_inputs(*, dataset, chunked, manifest, day12, day13, day14, hashes) -> None:
    actual_catalog = (
        chunked.episode_id,
        chunked.source_file,
        chunked.algorithm,
        chunked.chunk_size,
        chunked.overlap,
        tuple(chunk.chunk_id for chunk in chunked.chunks),
        chunked_transcript_fingerprint(chunked),
    )
    expected_catalog = (
        manifest.episode_id,
        manifest.source_file,
        manifest.chunking_algorithm,
        manifest.chunk_size,
        manifest.overlap,
        tuple(manifest.chunk_id_order),
        manifest.input_chunk_sha256,
    )
    if actual_catalog != expected_catalog or (manifest.chunk_size, manifest.overlap) != (1000, 200):
        raise RerankExperimentError("1000/200 Chunk catalog 与 manifest 不一致")
    if (
        day12.dataset_cases_sha256 != dataset.cases_sha256
        or day13.dataset_cases_sha256 != dataset.cases_sha256
        or day14.dataset_cases_sha256 != dataset.cases_sha256
        or day14.episode_id != dataset.episode_id
    ):
        raise RerankExperimentError("Day 11～14 的评测集身份或 episode 不一致")
    embedded = day14.input_artifact_hashes
    if (
        embedded.dataset_file_sha256 != hashes.dataset_file_sha256
        or embedded.transcript_file_sha256 != hashes.transcript_file_sha256
        or embedded.chunk_catalog_file_sha256 != hashes.chunk_catalog_file_sha256
        or embedded.manifest_file_sha256 != hashes.manifest_file_sha256
        or embedded.faiss_file_sha256 != hashes.faiss_file_sha256
        or embedded.day12_baseline_sha256 != hashes.day12_baseline_sha256
        or embedded.day13_experiment_sha256 != hashes.day13_experiment_sha256
    ):
        raise RerankExperimentError("Day 14 内嵌输入哈希与当前冻结产物不一致")
    ordered_hash = hashlib.sha256(
        "\n".join(manifest.chunk_id_order).encode("utf-8")
    ).hexdigest()
    if (
        day14.corpus_identity.input_chunk_sha256 != manifest.input_chunk_sha256
        or day14.corpus_identity.chunk_id_order_sha256 != ordered_hash
        or day14.corpus_identity.chunk_count != 76
    ):
        raise RerankExperimentError("Day 14 corpus identity 与 manifest 不一致")
    dataset_case_ids = tuple(case.case_id for case in dataset.cases)
    chunk_by_id = {chunk.chunk_id: chunk for chunk in manifest.chunks}
    for method_name in ("vector", "hybrid_rrf"):
        method = _day14_method(day14, method_name)
        if tuple(case.case_id for case in method.cases) != dataset_case_ids:
            raise RerankExperimentError("Day 14 候选 case 顺序与冻结评测集不一致")
        for case in method.cases:
            if len(case.retrieved_candidates) != 20:
                raise RerankExperimentError("Day 14 每道题必须保存完整 Top-20")
            if [candidate.rank for candidate in case.retrieved_candidates] != list(range(1, 21)):
                raise RerankExperimentError("Day 14 候选 rank 必须连续从 1 开始")
            ids = [candidate.chunk_id for candidate in case.retrieved_candidates]
            if len(ids) != len(set(ids)):
                raise RerankExperimentError("Day 14 单题候选不能重复 Chunk")
            for candidate in case.retrieved_candidates:
                chunk = chunk_by_id.get(candidate.chunk_id)
                if chunk is None or tuple(chunk.source_segment_ids) != candidate.source_segment_ids:
                    raise RerankExperimentError("Day 14 候选身份或来源元数据与 manifest 不一致")


def save_rerank_experiment(
    result: RerankExperimentResult,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise RerankExperimentPersistenceError(
            f"目标 Day 15 结果已存在：{destination}；不会静默覆盖"
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
        raise RerankExperimentPersistenceError("Day 15 结果写入失败") from error
    return destination
