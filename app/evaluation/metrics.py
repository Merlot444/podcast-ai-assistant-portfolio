"""Pure score-independent retrieval metrics over stable source segments."""

from __future__ import annotations

from collections.abc import Sequence

from app.schemas.evaluation_result import CaseMetricsAtK
from app.schemas.retriever import RetrievedCandidate


def compute_case_metrics_at_k(
    *,
    expected_source_segment_ids: Sequence[str],
    candidates: Sequence[RetrievedCandidate],
    k: int,
    scored: bool = True,
) -> CaseMetricsAtK:
    """Compute Hit@K and segment Recall@K without reading candidate scores."""
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k 必须是大于 0 的整数")
    expected = set(expected_source_segment_ids)
    if not scored:
        return CaseMetricsAtK(
            scored=False,
            hit_at_k=None,
            recall_at_k=None,
            correct_segment_count=None,
            expected_segment_count=len(expected),
        )
    if not expected:
        raise ValueError("计分案例必须至少包含一个稳定来源 segment")

    retrieved = {
        segment_id
        for candidate in candidates[:k]
        for segment_id in candidate.source_segment_ids
    }
    correct_count = len(retrieved.intersection(expected))
    return CaseMetricsAtK(
        scored=True,
        hit_at_k=correct_count > 0,
        recall_at_k=correct_count / len(expected),
        correct_segment_count=correct_count,
        expected_segment_count=len(expected),
    )
