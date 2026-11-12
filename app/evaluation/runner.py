"""Injected, provider-independent Day 12 retrieval evaluation runner."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter

from app.evaluation.metrics import compute_case_metrics_at_k
from app.retrieval.base import EvaluationRetriever
from app.schemas.evaluation import EvaluationCase, EvaluationDataset
from app.schemas.evaluation_result import (
    EvaluationCaseResult,
    MetricSummaryAtK,
    RetrievalEvaluationRun,
    SavedCandidate,
)
from app.schemas.retriever import RetrievedCandidate


DEFAULT_K_VALUES = (1, 3, 5, 8)


class RetrievalEvaluationError(RuntimeError):
    """Raised when a complete trustworthy baseline cannot be produced."""


def run_retrieval_evaluation(
    dataset: EvaluationDataset,
    *,
    retriever: EvaluationRetriever,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> RetrievalEvaluationRun:
    """Retrieve each question exactly once at max K, then score its prefixes."""
    validated_k = _validate_k_values(k_values)
    max_k = max(validated_k)
    started = perf_counter()
    case_results: list[EvaluationCaseResult] = []

    for case in dataset.cases:
        case_started = perf_counter()
        try:
            candidates = retriever.retrieve(case.question, max_k)
            _validate_candidate_contract(
                candidates,
                retriever_name=retriever.name,
                top_k=max_k,
            )
        except Exception as error:
            raise RetrievalEvaluationError(
                f"案例 {case.case_id} 检索失败，未生成完整基线："
                f"{type(error).__name__}"
            ) from error
        case_results.append(
            _build_case_result(
                case,
                candidates=candidates,
                k_values=validated_k,
                elapsed_seconds=perf_counter() - case_started,
            )
        )

    return RetrievalEvaluationRun(
        retriever_name=retriever.name,
        k_values=validated_k,
        query_count=len(case_results),
        elapsed_seconds=perf_counter() - started,
        aggregate_metrics=_aggregate(case_results, validated_k),
        category_metrics={
            category: _aggregate(
                [item for item in case_results if item.category == category],
                validated_k,
            )
            for category in sorted({item.category for item in case_results})
        },
        cases=tuple(case_results),
    )


def run_precomputed_retrieval_evaluation(
    dataset: EvaluationDataset,
    *,
    retriever_name: str,
    candidates_by_case_id: dict[str, list[RetrievedCandidate]],
    candidate_depth: int,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> RetrievalEvaluationRun:
    """Score fixed rankings while preserving a depth larger than final metric K."""
    return run_precomputed_case_evaluation(
        dataset.cases,
        retriever_name=retriever_name,
        candidates_by_case_id=candidates_by_case_id,
        candidate_depth=candidate_depth,
        k_values=k_values,
    )


def run_precomputed_case_evaluation(
    cases: Sequence[EvaluationCase],
    *,
    retriever_name: str,
    candidates_by_case_id: dict[str, list[RetrievedCandidate]],
    candidate_depth: int,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> RetrievalEvaluationRun:
    """Score fixed rankings for any validated ordered collection of cases."""
    validated_k = _validate_k_values(k_values)
    if isinstance(candidate_depth, bool) or not isinstance(candidate_depth, int):
        raise ValueError("candidate_depth 必须是正整数")
    if candidate_depth <= 0 or candidate_depth < max(validated_k):
        raise ValueError("candidate_depth 不能小于最终最大 K")
    ordered_cases = tuple(cases)
    if not ordered_cases:
        raise RetrievalEvaluationError("预计算评测案例不能为空")
    expected_case_ids = tuple(case.case_id for case in ordered_cases)
    if len(expected_case_ids) != len(set(expected_case_ids)):
        raise RetrievalEvaluationError("预计算评测案例 ID 不能重复")
    if set(candidates_by_case_id) != set(expected_case_ids):
        raise RetrievalEvaluationError("预计算排名必须恰好覆盖冻结评测集的全部案例")

    started = perf_counter()
    case_results: list[EvaluationCaseResult] = []
    for case in ordered_cases:
        candidates = candidates_by_case_id[case.case_id]
        _validate_candidate_contract(
            candidates,
            retriever_name=retriever_name,
            top_k=candidate_depth,
        )
        case_results.append(
            _build_case_result(
                case,
                candidates=candidates,
                k_values=validated_k,
                elapsed_seconds=0,
            )
        )

    return RetrievalEvaluationRun(
        retriever_name=retriever_name,
        k_values=validated_k,
        query_count=len(case_results),
        elapsed_seconds=perf_counter() - started,
        aggregate_metrics=_aggregate(case_results, validated_k),
        category_metrics={
            category: _aggregate(
                [item for item in case_results if item.category == category],
                validated_k,
            )
            for category in sorted({item.category for item in case_results})
        },
        cases=tuple(case_results),
    )


def aggregate_retrieval_cases(
    cases: Sequence[EvaluationCaseResult],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> dict[int, MetricSummaryAtK]:
    """Aggregate any declared case subset with the same Day 12 metric contract."""
    validated_k = _validate_k_values(k_values)
    return _aggregate(cases, validated_k)


def _validate_k_values(k_values: Sequence[int]) -> tuple[int, ...]:
    values = tuple(k_values)
    if not values:
        raise ValueError("k_values 不能为空")
    if any(isinstance(k, bool) or not isinstance(k, int) or k <= 0 for k in values):
        raise ValueError("k_values 必须只包含正整数")
    if tuple(sorted(set(values))) != values:
        raise ValueError("k_values 必须严格递增且不能重复")
    return values


def _validate_candidate_contract(
    candidates: Sequence[RetrievedCandidate],
    *,
    retriever_name: str,
    top_k: int,
) -> None:
    if len(candidates) > top_k:
        raise RetrievalEvaluationError("检索器返回的候选数量超过 top_k")
    expected_ranks = list(range(1, len(candidates) + 1))
    if [candidate.rank for candidate in candidates] != expected_ranks:
        raise RetrievalEvaluationError("候选 rank 必须从 1 连续并与列表顺序一致")
    if any(candidate.retriever_name != retriever_name for candidate in candidates):
        raise RetrievalEvaluationError("候选 retriever_name 与注入检索器不一致")


def _build_case_result(
    case: EvaluationCase,
    *,
    candidates: Sequence[RetrievedCandidate],
    k_values: tuple[int, ...],
    elapsed_seconds: float,
) -> EvaluationCaseResult:
    return EvaluationCaseResult(
        case_id=case.case_id,
        category=case.category,
        answerable=case.answerable,
        scored=case.answerable,
        expected_source_segment_ids=case.expected_source_segment_ids,
        retrieved_candidates=tuple(
            SavedCandidate(
                rank=candidate.rank,
                chunk_id=candidate.chunk_id,
                source_segment_ids=candidate.source_segment_ids,
                score=candidate.score,
                score_semantics=candidate.score_semantics,
            )
            for candidate in candidates
        ),
        metrics_by_k={
            k: compute_case_metrics_at_k(
                expected_source_segment_ids=case.expected_source_segment_ids,
                candidates=candidates,
                k=k,
                scored=case.answerable,
            )
            for k in k_values
        },
        elapsed_seconds=elapsed_seconds,
    )


def _aggregate(
    cases: Sequence[EvaluationCaseResult],
    k_values: tuple[int, ...],
) -> dict[int, MetricSummaryAtK]:
    scored_cases = [case for case in cases if case.scored]
    denominator = len(scored_cases)
    summaries: dict[int, MetricSummaryAtK] = {}
    for k in k_values:
        metrics = [case.metrics_by_k[k] for case in scored_cases]
        hit_count = sum(metric.hit_at_k is True for metric in metrics)
        recall_sum = sum(
            metric.recall_at_k for metric in metrics if metric.recall_at_k is not None
        )
        summaries[k] = MetricSummaryAtK(
            hit_count=hit_count,
            hit_denominator=denominator,
            hit_at_k=(hit_count / denominator if denominator else None),
            macro_recall_sum=recall_sum,
            macro_recall_denominator=denominator,
            macro_recall_at_k=(recall_sum / denominator if denominator else None),
        )
    return summaries
