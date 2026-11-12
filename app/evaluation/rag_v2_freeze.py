"""Offline Day 17 failure analysis and RAG 2.0 decision freeze."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError

from app.evaluation.baseline import file_sha256
from app.evaluation.payload_policy import find_unsafe_payload_path
from app.schemas.answer_quality import AnswerQualityCaseResult, AnswerQualityExperimentResult
from app.schemas.chunk_experiment import ChunkParameterExperimentResult
from app.schemas.evaluation import EvaluationDataset
from app.schemas.evaluation_result import EvaluationCaseResult, RetrievalBaselineResult
from app.schemas.evaluation_v2 import EvaluationDatasetV2Pilot
from app.schemas.evaluation_v2_retrieval import EvaluationV2RetrievalExperiment
from app.schemas.rag_v2_freeze import (
    AnswerConfiguration,
    AnswerMetricFreeze,
    CandidateRetriever,
    ChunkConfiguration,
    Day16ScoreSummary,
    Day17CaseReview,
    Day17InputArtifactHashes,
    Day17ProviderCallCounts,
    EvidenceSummary,
    FailureTaxonomyEntry,
    MethodCaseFinding,
    RagV2FreezeResult,
    RagVersionComparison,
    RankEvidence,
    RerankDecision,
    RetrievalDefault,
    RetrievalMetricFreeze,
)
from app.schemas.rerank_experiment import RerankExperimentResult
from app.schemas.retrieval_experiment import RetrievalExperimentResult, RetrievalMethodResult


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAG_V2_FREEZE_OUTPUT = (
    _PROJECT_ROOT / "data" / "evaluation" / "rag-v2-freeze-day17.json"
)
DEFAULT_DAY17_INPUTS = {
    "eval_dataset_v1": _PROJECT_ROOT / "data" / "evaluation" / "eval_dataset_v1.json",
    "day12_vector_baseline": _PROJECT_ROOT
    / "data"
    / "evaluation"
    / "eval_dataset_v1-vector-baseline.json",
    "day13_chunk_experiment": _PROJECT_ROOT
    / "data"
    / "evaluation"
    / "eval_dataset_v1-chunk-experiment-day13.json",
    "day14_retrieval_experiment": _PROJECT_ROOT
    / "data"
    / "evaluation"
    / "eval_dataset_v1-retrieval-experiment-day14.json",
    "day15_rerank_experiment": _PROJECT_ROOT
    / "data"
    / "evaluation"
    / "eval_dataset_v1-rerank-experiment-day15.json",
    "day16_answer_quality": _PROJECT_ROOT
    / "data"
    / "evaluation"
    / "eval_dataset_v1-answer-quality-experiment-day16.json",
    "eval_dataset_v2_pilot": _PROJECT_ROOT
    / "data"
    / "evaluation"
    / "eval_dataset_v2-pilot.json",
    "eval_dataset_v2_retrieval_experiment": _PROJECT_ROOT
    / "data"
    / "evaluation"
    / "eval_dataset_v2-pilot-retrieval-experiment.json",
}


class RagV2FreezeError(RuntimeError):
    """Raised when frozen inputs or Day 17 evidence are inconsistent."""


class RagV2FreezePersistenceError(RuntimeError):
    """Raised when the formal Day 17 artifact cannot be written safely."""


class _LoadedInputs(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    dataset_v1: EvaluationDataset
    day12: RetrievalBaselineResult
    day13: ChunkParameterExperimentResult
    day14: RetrievalExperimentResult
    day15: RerankExperimentResult
    day16: AnswerQualityExperimentResult
    dataset_v2: EvaluationDatasetV2Pilot
    v2_retrieval: EvaluationV2RetrievalExperiment


ModelT = TypeVar("ModelT", bound=BaseModel)


def freeze_rag_v2(
    *,
    output_path: str | Path = DEFAULT_RAG_V2_FREEZE_OUTPUT,
    input_paths: dict[str, str | Path] | None = None,
    overwrite: bool = False,
    now: Callable[[], datetime] | None = None,
) -> RagV2FreezeResult:
    """Build and atomically persist the fully offline Day 17 decision artifact."""
    paths = _resolve_input_paths(input_paths)
    destination = Path(output_path)
    _preflight(paths, destination=destination, overwrite=overwrite)
    before_hashes = _input_hashes(paths)
    loaded = _load_inputs(paths)
    _validate_input_chain(loaded, before_hashes=before_hashes)
    created_at = (now or (lambda: datetime.now(UTC)))()
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise RagV2FreezeError("created_at_utc 必须包含时区")
    created_at = created_at.astimezone(UTC)
    result = _build_result(loaded, hashes=before_hashes, created_at=created_at)
    _assert_safe_payload(result.model_dump(mode="json"))
    after_hashes = _input_hashes(paths)
    if after_hashes != before_hashes:
        raise RagV2FreezeError("Day 17 运行前后输入摘要发生漂移，拒绝保存正式结果")
    save_rag_v2_freeze(result, destination, overwrite=overwrite)
    return result


def load_rag_v2_freeze(path: str | Path) -> RagV2FreezeResult:
    """Strictly reload one metadata-only Day 17 artifact."""
    return _load_model(Path(path), RagV2FreezeResult, "Day 17 冻结结果")


def save_rag_v2_freeze(
    result: RagV2FreezeResult,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically save a complete freeze and refuse silent replacement."""
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise RagV2FreezePersistenceError(
            f"目标 Day 17 冻结结果已存在：{destination}；不会静默覆盖"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        result.model_dump(mode="json"), ensure_ascii=False, indent=2
    ) + "\n"
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
        raise RagV2FreezePersistenceError("Day 17 冻结结果写入失败") from error
    return destination


def _build_result(
    inputs: _LoadedInputs,
    *,
    hashes: Day17InputArtifactHashes,
    created_at: datetime,
) -> RagV2FreezeResult:
    return RagV2FreezeResult(
        schema_version=1,
        freeze_id=f"rag-v2-day17-{created_at.strftime('%Y%m%dt%H%M%Sz')}",
        created_at_utc=created_at,
        input_artifact_hashes=hashes,
        retrieval_default=RetrievalDefault(
            name="vector",
            embedding_model="jina-embeddings-v5-text-small",
            score_semantics="cosine_similarity",
            evidence_depth=5,
            rationale=(
                "Vector has the only frozen Top-5 answer-quality evidence. Hybrid-RRF "
                "improves expanded Hit@5 by one case but lowers Hit@3, regresses one "
                "multi-source Recall@5 case, and has no corresponding answer evaluation."
            ),
        ),
        candidate_retrievers=_candidate_retrievers(),
        rerank_enabled=False,
        rerank_decision=RerankDecision(
            status="query_specific_candidate_not_enabled",
            evaluated_methods=("vector_rerank", "hybrid_rrf_rerank"),
            stable_quality_gain=False,
            latency_p95_seconds=inputs.day15.latency_and_cost.all_requests.p95_seconds,
            monetary_cost_confirmed=False,
            answer_quality_evaluated=False,
            rationale=(
                "Day 15 repaired selected Top-1 failures but introduced "
                "neuroscience_provenance as a new Top-1 failure. Neither reranked path "
                "passed the predeclared quality gate; monetary cost and answer-layer "
                "behavior also remain unverified."
            ),
        ),
        chunk_configuration=_chunk_configuration(inputs),
        answer_configuration=_answer_configuration(inputs),
        failure_taxonomy=_failure_taxonomy(inputs),
        case_reviews=_case_reviews(inputs),
        rag_1_vs_rag_2_comparison=_rag_version_comparison(),
        evidence_summary=_evidence_summary(inputs),
        decision_rationale=(
            "Production changes require both retrieval and answer-quality evidence.",
            "Hybrid-RRF adds only one expanded Top-5 hit, while its Top-3 Hit is lower than Vector.",
            "The Hybrid gain is concentrated in one Mark chapter-time case and includes an Ilya multi-source regression.",
            "Day 16 validates only frozen Vector Top-5, so it does not justify changing retriever, model, prompt, or evidence depth.",
            "Day 15 does not show stable rerank benefit after quality, regression, latency, cost, and answer-layer limits are considered.",
            "RAG 2.0 is frozen as an evidence and governance upgrade with the online default intentionally unchanged.",
        ),
        known_limitations=(
            "The expanded set covers only three English podcast episodes and 64 scored retrieval cases.",
            "The v2 pilot has one-agent semantic review and still requires a second independent human review.",
            "The paired bootstrap interval for Hybrid minus Vector Top-5 Recall crosses zero.",
            "Hybrid Top-5 has no answer, citation, or refusal quality evaluation.",
            "Day 16 covers one model, one run per question, and one named reviewer.",
            "Answer citations are answer-level rather than sentence- or claim-level.",
            "Source segment end times are unavailable for the original episode, and Mark timestamps are chapter-start only.",
            "Provider revisions and currency-denominated actual costs were not returned.",
        ),
        future_evaluation_requirements=(
            "Complete an independent second human review of v2, prioritizing unanswerable, multi-source, and Mark chapter-time cases.",
            "Add two or three independently frozen episodes across speakers, topics, formats, and languages.",
            "Run matched Vector and Hybrid Top-5 answer-quality evaluations before any production retriever switch.",
            "Increase non-adjacent multi-source, hard-unanswerable, temporal-provenance, entity, acronym, number, and version questions.",
            "Add claim-to-citation annotation, repeated answer runs, and reviewer-agreement measurement.",
            "Use separate development and held-out test partitions before tuning fusion, routing, Top-K, prompts, or models.",
        ),
        provider_call_counts=Day17ProviderCallCounts(
            embedding_calls=0,
            vector_query_calls=0,
            bm25_calls=0,
            rrf_calls=0,
            rerank_calls=0,
            answer_model_calls=0,
            judge_model_calls=0,
        ),
        production_default_changed=False,
    )


def _candidate_retrievers() -> tuple[CandidateRetriever, ...]:
    return (
        CandidateRetriever(
            name="vector",
            role="production_default",
            production_enabled=True,
            answer_quality_evaluated=True,
            retrieval_evidence_scope="20-case v1 and 80-case expanded retrieval; Day 16 Vector Top-5 answers",
            improvement_summary="Expanded Vector reaches 63/64 Hit@5 and 96.09% Macro Recall@5.",
            regression_summary="It misses the Mark chapter-start case at rank 9 and remains incomplete on selected multi-source cases.",
            next_gate="Recompare against Hybrid on more episodes and matched answer-quality runs.",
        ),
        CandidateRetriever(
            name="bm25",
            role="hybrid_component",
            production_enabled=False,
            answer_quality_evaluated=False,
            retrieval_evidence_scope="20-case v1 and 80-case expanded retrieval only",
            improvement_summary="It supplies complementary exact-term evidence, notably the Mark chapter-start source at rank 2.",
            regression_summary="Expanded aggregate Hit@5 and Recall@5 are both below Vector; semantic and multi-source cases regress.",
            next_gate="Keep as a deterministic Hybrid child or future routed candidate, not a standalone default.",
        ),
        CandidateRetriever(
            name="hybrid_rrf",
            role="candidate_experiment",
            production_enabled=False,
            answer_quality_evaluated=False,
            retrieval_evidence_scope="20-case v1 and 80-case expanded retrieval; no answer-quality run",
            improvement_summary="Expanded Hit@5 is 64/64 and Macro Recall@5 is 97.66%, one hit above Vector.",
            regression_summary="Hit@3 is one case below Vector and Ilya SSI multi-source Recall@5 drops from 1.0 to 0.5.",
            next_gate="Require more episodes, independent v2 review, and matched Top-5 answer/citation/refusal evaluation.",
        ),
    )


def _chunk_configuration(inputs: _LoadedInputs) -> ChunkConfiguration:
    episode_hashes = {
        identity.episode_slug: identity.input_chunk_sha256
        for identity in inputs.v2_retrieval.episode_identities
    }
    return ChunkConfiguration(
        algorithm="fixed_character_windows_v1",
        chunk_size=1000,
        overlap=200,
        selection_source="day13_controlled_comparison",
        episode_input_chunk_sha256=episode_hashes,
        globally_optimal_claim=False,
    )


def _answer_configuration(inputs: _LoadedInputs) -> AnswerConfiguration:
    configuration = inputs.day16.answer_model_configuration
    return AnswerConfiguration(
        model_name=configuration.model_name,
        temperature=configuration.temperature,
        thinking=configuration.thinking,
        maximum_retries=configuration.maximum_retries,
        evidence_depth=configuration.evidence_depth,
        grounded_system_prompt_sha256=(
            inputs.day16.input_artifact_hashes.grounded_system_prompt_sha256
        ),
        output_schema_sha256=configuration.output_schema_sha256,
        configuration_changed=False,
        rationale=(
            "Day 16 establishes real Vector Top-5 answer behavior but contains no "
            "controlled alternative model, prompt, or Top-K comparison. The frozen "
            "configuration is retained rather than changed without comparative evidence."
        ),
    )


def _failure_taxonomy(inputs: _LoadedInputs) -> tuple[FailureTaxonomyEntry, ...]:
    day14_vector = _retrieval_method(inputs.day14, "vector")
    day14_hybrid = _retrieval_method(inputs.day14, "hybrid_rrf")
    v2_vector = _expanded_method(inputs.v2_retrieval, "vector")
    v2_hybrid = _expanded_method(inputs.v2_retrieval, "hybrid_rrf")
    return (
        _unobserved_failure(
            "retrieval_no_recall",
            "No necessary source in the reviewed failures was absent from the frozen Top-20 pools.",
            "Candidate-pool depth is sufficient for the reviewed cases, though not proven universally.",
            "Continue recording Top-20 pool misses separately from Top-5 ranking misses.",
        ),
        FailureTaxonomyEntry(
            category="ranking_below_top_k",
            observed=True,
            case_ids=(
                "alignment_safeguards_pause_trigger",
                "mark25_b1_chapter_intelligence_explosion_start",
                "agi_architecture_original_bets",
            ),
            methods=("vector",),
            evidence_ranks=(
                _rank_evidence(
                    day14_vector,
                    "alignment_safeguards_pause_trigger",
                    method="vector",
                    depth=20,
                    finding="The pause-trigger source is at Vector rank 19, outside answer Top-5.",
                ),
                _rank_evidence(
                    v2_vector,
                    "mark25_b1_chapter_intelligence_explosion_start",
                    method="vector",
                    depth=20,
                    finding="The chapter-start truth first appears at Vector rank 9.",
                ),
                _rank_evidence(
                    day14_vector,
                    "agi_architecture_original_bets",
                    method="vector",
                    depth=20,
                    finding="The second historical-bets source appears at Vector ranks 7/8/15.",
                ),
            ),
            answer_impact="Top-5 answers omit required details even though the larger candidate pool contains them.",
            root_cause="Relevant evidence is recalled but ranked below the fixed evidence budget.",
            current_handling="Keep Top-5 frozen and disclose missing context rather than inventing it.",
            next_action="Evaluate matched Vector/Hybrid answers and multi-source-aware ranking on held-out episodes.",
            blocks_rag_v2_freeze=False,
        ),
        FailureTaxonomyEntry(
            category="partial_multi_source_recall",
            observed=True,
            case_ids=("alignment_safeguards_pause_trigger", "ilya25_b2_ssi_compute"),
            methods=("hybrid_rrf",),
            evidence_ranks=(
                _rank_evidence(
                    day14_hybrid,
                    "alignment_safeguards_pause_trigger",
                    method="hybrid_rrf",
                    depth=20,
                    finding="Hybrid keeps the second required source at rank 9, so Top-8 recall stays 0.5.",
                ),
                _rank_evidence(
                    v2_hybrid,
                    "ilya25_b2_ssi_compute",
                    method="hybrid_rrf",
                    depth=20,
                    finding="Hybrid pushes the second required turn to rank 6; Recall@5 falls from Vector 1.0 to 0.5.",
                ),
            ),
            answer_impact="A question can be counted as a Hit while a required second source is absent from the answer context.",
            root_cause="Fusion or ranking favors one subtopic/source and fails to keep all independent evidence in the fixed prefix.",
            current_handling="Use Macro Recall and per-source ranks alongside Hit@K; do not equate Hit with completeness.",
            next_action="Expand non-adjacent multi-source cases and test answer completeness for each retrieval method.",
            blocks_rag_v2_freeze=False,
        ),
        FailureTaxonomyEntry(
            category="chunk_boundary_or_duplication",
            observed=True,
            case_ids=("gato_exchange_provenance",),
            methods=("answer_generation",),
            evidence_ranks=(
                _answer_rank_evidence(
                    day14_vector,
                    "gato_exchange_provenance",
                    finding="Vector rank 1 contains the correct turns but begins at an earlier source segment.",
                ),
            ),
            answer_impact="The answer reported a broad Chunk time range instead of the two exact source-turn start times.",
            root_cause="Chunk-level coverage spans earlier context; the generator treated that broad boundary as claim-level provenance. Duplicate displacement itself was not observed.",
            current_handling="Preserve source-segment IDs and incomplete-time semantics; document that Chunk ranges are not exact claim times.",
            next_action="Add claim-to-segment citation formatting and more boundary-sensitive temporal cases.",
            blocks_rag_v2_freeze=False,
        ),
        _unobserved_failure(
            "question_ambiguity",
            "The Day 16 reviewer did not attribute any observed failure to ambiguous wording.",
            "Current cases have reviewable intent, but the sample is small.",
            "Add deliberately ambiguous and clarification-required cases in a later dataset version.",
        ),
        FailureTaxonomyEntry(
            category="insufficient_top_k_context",
            observed=True,
            case_ids=("alignment_safeguards_pause_trigger", "agi_architecture_original_bets"),
            methods=("vector",),
            evidence_ranks=(
                _rank_evidence(
                    day14_vector,
                    "alignment_safeguards_pause_trigger",
                    method="vector",
                    depth=20,
                    finding="The second truth source is rank 19 and cannot enter frozen Top-5 context.",
                ),
                _rank_evidence(
                    day14_vector,
                    "agi_architecture_original_bets",
                    method="vector",
                    depth=20,
                    finding="The second truth source starts at rank 7, outside frozen Top-5 context.",
                ),
            ),
            answer_impact="Answers are necessarily incomplete and may form weak connections from nearby evidence.",
            root_cause="The five-evidence ceiling excludes required sources already present deeper in the ranking.",
            current_handling="Retain Top-5 because no answer-quality comparison supports raising it; record the ceiling explicitly.",
            next_action="Run controlled Top-5 versus alternative-depth answer evaluations on a separate development/test design.",
            blocks_rag_v2_freeze=False,
        ),
        FailureTaxonomyEntry(
            category="generation_incorrect",
            observed=True,
            case_ids=("gato_exchange_provenance", "neuroscience_provenance"),
            methods=("answer_generation",),
            evidence_ranks=(
                _answer_rank_evidence(
                    day14_vector,
                    "gato_exchange_provenance",
                    finding="Correct provenance is present at Vector rank 1, but generation reports the wrong start range.",
                ),
                _answer_rank_evidence(
                    day14_vector,
                    "neuroscience_provenance",
                    finding="The stable source is present in Top-5, but generation confuses Chunk coverage with segment time.",
                ),
            ),
            answer_impact="Factual or temporal details are wrong despite sufficient retrieved evidence.",
            root_cause="The model misinterprets available evidence metadata or omits a supported detail.",
            current_handling="Keep strict application-owned citation mapping and expose the single-run quality limit.",
            next_action="Add claim-level provenance instructions and repeat-run evaluation before changing the frozen prompt.",
            blocks_rag_v2_freeze=False,
        ),
        FailureTaxonomyEntry(
            category="generation_incomplete",
            observed=True,
            case_ids=("robotics_data_poor_research", "tree_search_rl_synthetic_reward"),
            methods=("answer_generation",),
            evidence_ranks=(
                _answer_rank_evidence(
                    day14_vector,
                    "robotics_data_poor_research",
                    finding="The model omits a control-problem point already present in Top-5 evidence.",
                ),
                _answer_rank_evidence(
                    day14_vector,
                    "tree_search_rl_synthetic_reward",
                    finding="All required sources are in Top-5 but completeness receives 1/2.",
                ),
            ),
            answer_impact="The answer is grounded but misses requested aspects, reducing completeness.",
            root_cause="Generation does not reliably synthesize every supported sub-answer even when context is sufficient.",
            current_handling="Score completeness separately from correctness and citation legality.",
            next_action="Increase multi-part synthesis cases and compare repeated runs before prompt modification.",
            blocks_rag_v2_freeze=False,
        ),
        FailureTaxonomyEntry(
            category="citation_wrong_or_incomplete",
            observed=True,
            case_ids=("gato_exchange_provenance", "agi_architecture_original_bets"),
            methods=("answer_generation",),
            evidence_ranks=(
                _answer_rank_evidence(
                    day14_vector,
                    "gato_exchange_provenance",
                    finding="The legal E1 citation is semantically too broad for the reported time claim.",
                ),
                _answer_rank_evidence(
                    day14_vector,
                    "agi_architecture_original_bets",
                    finding="The required historical source is outside Top-5, leaving the historical connection incompletely cited.",
                ),
            ),
            answer_impact="Application mapping remains legal, but semantic support or source coverage is incomplete.",
            root_cause="Answer-level citations do not explicitly bind each claim to the precise stable segment.",
            current_handling="Distinguish legal mapping from semantic citation correctness in evaluation.",
            next_action="Introduce sentence- or claim-level citation annotations and temporal provenance checks.",
            blocks_rag_v2_freeze=False,
        ),
        _unobserved_failure(
            "incorrect_refusal",
            "All 16 answerable Day 16 cases were answered; no false refusal was observed.",
            "This is evidence for the frozen 20-case run only.",
            "Retain answerable false-refusal checks in expanded answer-quality evaluation.",
        ),
        _unobserved_failure(
            "unsafe_answer_on_unanswerable",
            "All four Day 16 unanswerable cases used the fixed refusal with empty citations and no invented fact.",
            "The additional 12 v2 unanswerable cases have retrieval results but no answer-quality run.",
            "Prioritize all 16 unanswerable cases in matched Vector/Hybrid answer evaluation.",
        ),
    )


def _case_reviews(inputs: _LoadedInputs) -> tuple[Day17CaseReview, ...]:
    return (
        _day16_case_review(
            inputs,
            "alignment_safeguards_pause_trigger",
            roles=("required_focus", "multi_source_incomplete"),
            root_cause="The second required pause-trigger source is Vector rank 19, outside Top-5.",
            answer_impact="The safeguards are grounded, but the requested trigger behavior is missing.",
            disposition="Keep as a multi-source regression sentinel; it does not justify a default switch because Hybrid still ranks the second source at 9.",
        ),
        _v2_case_review(
            inputs,
            "mark25_b1_chapter_intelligence_explosion_start",
            root_cause="Exact chapter wording favors BM25; Vector places the chapter-start source at rank 9.",
            answer_impact="Vector Top-5 lacks the source, while BM25 rank 2 and Hybrid rank 3 restore the retrieval hit.",
            disposition="Count as the principal Hybrid Top-5 improvement, but not as cross-program stability evidence.",
        ),
        _v2_case_review(
            inputs,
            "ilya25_b2_ssi_compute",
            root_cause="Weak BM25 ranks dilute a strong Vector multi-source ordering during equal-weight RRF.",
            answer_impact="Hybrid moves the second required turn to rank 6 and lowers Recall@5 from 1.0 to 0.5.",
            disposition="Keep as a mandatory Hybrid regression case for future answer-quality evaluation.",
        ),
        _v2_case_review(
            inputs,
            "ilya25_b1_timestamp_second_student",
            root_cause="Fusion promotes lexical distractors and pushes the correct timestamp source below the Top-3 prefix.",
            answer_impact="This case contributes to Hybrid's lower expanded Hit@3 despite its higher Top-5 result.",
            disposition="Track both Top-3 and Top-5; do not describe Hybrid as uniformly better.",
        ),
        _v2_case_review(
            inputs,
            "mark25_b3_productivity_and_customer_support",
            root_cause="The two requested Mark topics benefit differently from dense and lexical rankings.",
            answer_impact="Hybrid improves source coverage within Top-5 relative to the weaker individual ordering.",
            disposition="Retain as a positive multi-source fusion example that still lacks answer-quality evidence.",
        ),
        _day16_case_review(
            inputs,
            "bioweapon_mitigation_provenance",
            roles=("complete_answer_and_citations",),
            root_cause="No failure observed; adjacent Top-2 evidence completely supports content and provenance.",
            answer_impact="The answer receives 2/2 on all four dimensions with complete citations.",
            disposition="Use as the frozen positive control for grounded, complete, temporally honest answers.",
        ),
        _day16_case_review(
            inputs,
            "agi_architecture_original_bets",
            roles=("retrieval_success_answer_not_full_score",),
            root_cause="The second stable source is at Vector ranks 7/8, and generation forms a partially supported historical link.",
            answer_impact="All four Day 16 dimensions score 1/2; citation coverage is incomplete.",
            disposition="Keep as the clearest retrieval-context plus generation/citation compound failure.",
        ),
        _day16_case_review(
            inputs,
            "sourdough_refusal",
            roles=("correct_unanswerable_refusal",),
            root_cause="No answer exists in the episode despite lexical similarity in retrieved candidates.",
            answer_impact="The fixed refusal is correct, uses no citation, and invents no fact.",
            disposition="Preserve as a refusal regression sentinel and extend the same check to v2 unanswerable cases.",
        ),
        _day16_case_review(
            inputs,
            "gato_exchange_provenance",
            roles=("highest_risk_failure",),
            root_cause="Generation mistakes a broad Chunk start for the exact source-turn time even though the correct turns are present.",
            answer_impact="The answer gives incorrect temporal provenance and incomplete content, with all four dimensions at 1/2.",
            disposition="Treat temporal provenance misuse as the highest-risk observed failure; require claim-level source-time validation.",
        ),
    )


def _day16_case_review(
    inputs: _LoadedInputs,
    case_id: str,
    *,
    roles: tuple[str, ...],
    root_cause: str,
    answer_impact: str,
    disposition: str,
) -> Day17CaseReview:
    answer_case = _answer_case(inputs.day16, case_id)
    vector_case = _method_case(_retrieval_method(inputs.day14, "vector"), case_id)
    review = answer_case.answerable_review
    refusal = answer_case.unanswerable_review
    if review is not None:
        scores = Day16ScoreSummary(
            correctness=review.correctness.score,
            evidence_alignment=review.evidence_alignment.score,
            citation_correctness=review.citation_correctness.score,
            completeness=review.completeness.score,
            correct_refusal=None,
        )
        finding = (
            f"Day 16 scores {review.correctness.score}/{review.evidence_alignment.score}/"
            f"{review.citation_correctness.score}/{review.completeness.score}; "
            f"failure categories: {', '.join(review.failure_categories) or 'none'}."
        )
    else:
        if refusal is None:
            raise RagV2FreezeError(f"Day 16 案例缺少复核：{case_id}")
        scores = Day16ScoreSummary(correct_refusal=refusal.correct_refusal)
        finding = "Day 16 records a correct fixed refusal with empty citations and no invented fact."
    metric = vector_case.metrics_by_k[5]
    return Day17CaseReview(
        case_id=case_id,
        evidence_source="day16_answer_quality",
        review_roles=roles,
        method_findings=(
            MethodCaseFinding(
                method="vector",
                expected_source_ranks=_source_rank_map(vector_case),
                hit_at_5=metric.hit_at_k,
                recall_at_5=metric.recall_at_k,
                answer_quality_available=True,
                finding=finding,
            ),
        ),
        day16_scores=scores,
        root_cause=root_cause,
        answer_impact=answer_impact,
        disposition=disposition,
    )


def _v2_case_review(
    inputs: _LoadedInputs,
    case_id: str,
    *,
    root_cause: str,
    answer_impact: str,
    disposition: str,
) -> Day17CaseReview:
    findings: list[MethodCaseFinding] = []
    for method_name in ("vector", "bm25", "hybrid_rrf"):
        method = _expanded_method(inputs.v2_retrieval, method_name)
        case = _method_case(method, case_id)
        metric = case.metrics_by_k[5]
        findings.append(
            MethodCaseFinding(
                method=method_name,
                expected_source_ranks=_source_rank_map(case),
                hit_at_5=metric.hit_at_k,
                recall_at_5=metric.recall_at_k,
                answer_quality_available=False,
                finding=(
                    f"Top-5 hit={metric.hit_at_k}; recall={metric.recall_at_k}; "
                    f"expected-source ranks={_flat_expected_ranks(case)}."
                ),
            )
        )
    return Day17CaseReview(
        case_id=case_id,
        evidence_source="v2_retrieval_comparison",
        review_roles=("required_focus",),
        method_findings=tuple(findings),
        day16_scores=None,
        root_cause=root_cause,
        answer_impact=answer_impact,
        disposition=disposition,
    )


def _evidence_summary(inputs: _LoadedInputs) -> EvidenceSummary:
    metrics: list[RetrievalMetricFreeze] = []
    for method_name in ("vector", "bm25", "hybrid_rrf"):
        method = _expanded_method(inputs.v2_retrieval, method_name)
        top5 = method.aggregate_metrics[5]
        metrics.append(
            RetrievalMetricFreeze(
                method=method_name,
                hit_at_5_count=top5.hit_count,
                hit_at_5_denominator=top5.hit_denominator,
                hit_at_5=top5.hit_at_k,
                macro_recall_at_5=top5.macro_recall_at_k,
            )
        )
    answer_metrics = inputs.day16.metrics
    return EvidenceSummary(
        expanded_retrieval_case_count=80,
        expanded_answerable_case_count=64,
        expanded_unanswerable_case_count=16,
        retrieval_top5=tuple(metrics),
        day16_answer_quality=AnswerMetricFreeze(
            answerable_case_count=16,
            unanswerable_case_count=4,
            correctness_mean=answer_metrics.answerable_scores.correctness.mean_score,
            evidence_alignment_mean=(
                answer_metrics.answerable_scores.evidence_alignment.mean_score
            ),
            citation_correctness_mean=(
                answer_metrics.answerable_scores.citation_correctness.mean_score
            ),
            completeness_mean=answer_metrics.answerable_scores.completeness.mean_score,
            correct_refusal_count=answer_metrics.answerability.correct_refusal_count,
            core_claim_support_rate=answer_metrics.citations.core_claim_support_rate or 0.0,
        ),
        hybrid_top3_hit_lower_than_vector=True,
        hybrid_top5_answer_quality_evaluated=False,
        v2_independent_human_review_complete=False,
    )


def _rag_version_comparison() -> tuple[RagVersionComparison, ...]:
    return (
        RagVersionComparison(
            dimension="online_pipeline",
            rag_1_0="Vector retrieval, five evidence items, grounded generation, application-owned citation mapping, and deterministic refusal.",
            rag_2_0="The same evidence-selected online path remains production default; candidate retrieval and rerank code stay outside the default.",
            production_behavior_changed=False,
        ),
        RagVersionComparison(
            dimension="evaluation_scope",
            rag_1_0="A five-question smoke regression and one-episode operational evidence.",
            rag_2_0="Frozen 20-case answer quality plus 80-case, three-episode retrieval comparison with stable source truth.",
            production_behavior_changed=False,
        ),
        RagVersionComparison(
            dimension="retrieval_comparison",
            rag_1_0="Dense Vector retrieval only.",
            rag_2_0="Versioned Vector, BM25, Hybrid-RRF, and Rerank comparisons with improvements, regressions, provenance, calls, and latency.",
            production_behavior_changed=False,
        ),
        RagVersionComparison(
            dimension="answer_and_refusal_evidence",
            rag_1_0="Two illustrative real answers and smoke checks without a formal quality score.",
            rag_2_0="Twenty frozen Vector Top-5 outputs scored for correctness, alignment, citations, completeness, and refusal safety.",
            production_behavior_changed=False,
        ),
        RagVersionComparison(
            dimension="failure_analysis",
            rag_1_0="Known limits were documented per stage.",
            rag_2_0="A fixed taxonomy records observed and unobserved failures, real cases, ranks, impacts, causes, handling, and next gates.",
            production_behavior_changed=False,
        ),
        RagVersionComparison(
            dimension="configuration_freeze",
            rag_1_0="Operational defaults existed across separate implementation stages.",
            rag_2_0="One strict, checksummed, no-overwrite artifact freezes default and candidate status, Chunk, Top-K, model, prompt digest, failures, and evidence limits.",
            production_behavior_changed=False,
        ),
    )


def _validate_input_chain(
    inputs: _LoadedInputs,
    *,
    before_hashes: Day17InputArtifactHashes,
) -> None:
    cases_sha = inputs.dataset_v1.cases_sha256
    if any(
        value != cases_sha
        for value in (
            inputs.day12.dataset_cases_sha256,
            inputs.day13.dataset_cases_sha256,
            inputs.day14.dataset_cases_sha256,
            inputs.day15.dataset_cases_sha256,
            inputs.day16.dataset_cases_sha256,
            inputs.v2_retrieval.dataset.base_cases_sha256,
        )
    ):
        raise RagV2FreezeError("Day 12-16 或 v2 的 dataset cases 摘要不一致")
    if inputs.day12.input_artifact_hashes.dataset_file_sha256 != before_hashes.eval_dataset_v1_sha256:
        raise RagV2FreezeError("Day 12 绑定的 v1 文件摘要与当前输入不一致")
    if inputs.day13.input_artifact_hashes.baseline_result_sha256 != before_hashes.day12_vector_baseline_sha256:
        raise RagV2FreezeError("Day 13 绑定的 Day 12 摘要不一致")
    if inputs.day14.input_artifact_hashes.day13_experiment_sha256 != before_hashes.day13_chunk_experiment_sha256:
        raise RagV2FreezeError("Day 14 绑定的 Day 13 摘要不一致")
    if inputs.day15.input_artifact_hashes.day14_experiment_sha256 != before_hashes.day14_retrieval_experiment_sha256:
        raise RagV2FreezeError("Day 15 绑定的 Day 14 摘要不一致")
    if inputs.day16.input_artifact_hashes.day15_experiment_sha256 != before_hashes.day15_rerank_experiment_sha256:
        raise RagV2FreezeError("Day 16 绑定的 Day 15 摘要不一致")
    if (
        inputs.dataset_v2.parent.artifact_sha256 != before_hashes.eval_dataset_v1_sha256
        or inputs.dataset_v2.parent.cases_sha256 != cases_sha
    ):
        raise RagV2FreezeError("v2 pilot 的 v1 parent identity 不一致")
    v2_hashes = inputs.v2_retrieval.input_hashes
    if (
        v2_hashes.get("data/evaluation/eval_dataset_v1.json")
        != before_hashes.eval_dataset_v1_sha256
        or v2_hashes.get("data/evaluation/eval_dataset_v2-pilot.json")
        != before_hashes.eval_dataset_v2_pilot_sha256
    ):
        raise RagV2FreezeError("80 题检索结果绑定的数据集摘要不一致")
    if (
        inputs.day13.selected_default.chunk_size,
        inputs.day13.selected_default.overlap,
        inputs.day13.selected_default.retained_day12_baseline,
    ) != (1000, 200, True):
        raise RagV2FreezeError("Day 13 不支持冻结 1000/200")
    if inputs.day14.recommendation.value != "retain_vector":
        raise RagV2FreezeError("Day 14 默认检索结论不是 retain_vector")
    if (
        inputs.day15.recommendation.value != "rerank_query_specific"
        or inputs.day15.recommendation.vector_rerank_quality_qualified
        or inputs.day15.recommendation.hybrid_rerank_quality_qualified
    ):
        raise RagV2FreezeError("Day 15 结论不支持当前 Rerank 禁用状态")
    if inputs.dataset_v2.independent_human_review_required is not True:
        raise RagV2FreezeError("v2 pilot 必须保留第二位独立人工复核边界")
    _validate_expanded_metrics(inputs.v2_retrieval)
    if inputs.day16.answer_model_configuration.evidence_depth != 5:
        raise RagV2FreezeError("Day 16 不支持冻结 Top-5")


def _validate_expanded_metrics(result: EvaluationV2RetrievalExperiment) -> None:
    expected = {
        "vector": (63, 0.9609375),
        "bm25": (61, 0.9296875),
        "hybrid_rrf": (64, 0.9765625),
    }
    for method_name, (hit_count, recall) in expected.items():
        summary = _expanded_method(result, method_name).aggregate_metrics[5]
        if summary.hit_count != hit_count or abs(summary.macro_recall_at_k - recall) > 1e-12:
            raise RagV2FreezeError(f"80 题 {method_name} Top-5 指标与冻结证据不一致")
    vector_top3 = _expanded_method(result, "vector").aggregate_metrics[3].hit_count
    hybrid_top3 = _expanded_method(result, "hybrid_rrf").aggregate_metrics[3].hit_count
    if not hybrid_top3 < vector_top3:
        raise RagV2FreezeError("Hybrid Top-3 退化证据发生变化，必须重新决策")


def _input_hashes(paths: dict[str, Path]) -> Day17InputArtifactHashes:
    return Day17InputArtifactHashes(
        eval_dataset_v1_sha256=file_sha256(paths["eval_dataset_v1"]),
        day12_vector_baseline_sha256=file_sha256(paths["day12_vector_baseline"]),
        day13_chunk_experiment_sha256=file_sha256(paths["day13_chunk_experiment"]),
        day14_retrieval_experiment_sha256=file_sha256(
            paths["day14_retrieval_experiment"]
        ),
        day15_rerank_experiment_sha256=file_sha256(paths["day15_rerank_experiment"]),
        day16_answer_quality_sha256=file_sha256(paths["day16_answer_quality"]),
        eval_dataset_v2_pilot_sha256=file_sha256(paths["eval_dataset_v2_pilot"]),
        eval_dataset_v2_retrieval_experiment_sha256=file_sha256(
            paths["eval_dataset_v2_retrieval_experiment"]
        ),
    )


def _load_inputs(paths: dict[str, Path]) -> _LoadedInputs:
    return _LoadedInputs(
        dataset_v1=_load_model(paths["eval_dataset_v1"], EvaluationDataset, "v1 评测集"),
        day12=_load_model(paths["day12_vector_baseline"], RetrievalBaselineResult, "Day 12"),
        day13=_load_model(paths["day13_chunk_experiment"], ChunkParameterExperimentResult, "Day 13"),
        day14=_load_model(paths["day14_retrieval_experiment"], RetrievalExperimentResult, "Day 14"),
        day15=_load_model(paths["day15_rerank_experiment"], RerankExperimentResult, "Day 15"),
        day16=_load_model(paths["day16_answer_quality"], AnswerQualityExperimentResult, "Day 16"),
        dataset_v2=_load_model(paths["eval_dataset_v2_pilot"], EvaluationDatasetV2Pilot, "v2 pilot"),
        v2_retrieval=_load_model(
            paths["eval_dataset_v2_retrieval_experiment"],
            EvaluationV2RetrievalExperiment,
            "80 题检索结果",
        ),
    )


def _load_model(path: Path, model: type[ModelT], label: str) -> ModelT:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return model.model_validate(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as error:
        raise RagV2FreezeError(f"无法严格加载{label}：{path}") from error


def _resolve_input_paths(values: dict[str, str | Path] | None) -> dict[str, Path]:
    if values is None:
        return dict(DEFAULT_DAY17_INPUTS)
    if set(values) != set(DEFAULT_DAY17_INPUTS):
        missing = sorted(set(DEFAULT_DAY17_INPUTS) - set(values))
        extra = sorted(set(values) - set(DEFAULT_DAY17_INPUTS))
        raise RagV2FreezeError(f"Day 17 输入集合不完整；missing={missing}, extra={extra}")
    return {key: Path(value) for key, value in values.items()}


def _preflight(paths: dict[str, Path], *, destination: Path, overwrite: bool) -> None:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RagV2FreezeError(f"Day 17 缺少必要正式输入：{missing}")
    if destination.resolve() in {path.resolve() for path in paths.values()}:
        raise RagV2FreezePersistenceError("Day 17 输出不能覆盖任何正式输入")
    if destination.exists() and not overwrite:
        raise RagV2FreezePersistenceError(
            f"目标 Day 17 冻结结果已存在：{destination}；不会静默覆盖"
        )


def _retrieval_method(
    result: RetrievalExperimentResult, name: str
) -> RetrievalMethodResult:
    try:
        return next(item for item in result.methods if item.retriever_name == name)
    except StopIteration as error:
        raise RagV2FreezeError(f"Day 14 缺少方法：{name}") from error


def _expanded_method(result: EvaluationV2RetrievalExperiment, name: str):
    try:
        return next(item for item in result.methods if item.retriever_name == name)
    except StopIteration as error:
        raise RagV2FreezeError(f"80 题结果缺少方法：{name}") from error


def _method_case(method, case_id: str) -> EvaluationCaseResult:
    try:
        return next(item for item in method.cases if item.case_id == case_id)
    except StopIteration as error:
        raise RagV2FreezeError(f"方法 {method.retriever_name} 缺少案例：{case_id}") from error


def _answer_case(result: AnswerQualityExperimentResult, case_id: str) -> AnswerQualityCaseResult:
    try:
        return next(item for item in result.cases if item.case_id == case_id)
    except StopIteration as error:
        raise RagV2FreezeError(f"Day 16 缺少案例：{case_id}") from error


def _source_rank_map(case: EvaluationCaseResult) -> dict[str, tuple[int, ...]]:
    return {
        source_id: tuple(
            candidate.rank
            for candidate in case.retrieved_candidates
            if source_id in candidate.source_segment_ids
        )
        for source_id in case.expected_source_segment_ids
    }


def _flat_expected_ranks(case: EvaluationCaseResult) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                rank
                for ranks in _source_rank_map(case).values()
                for rank in ranks
            }
        )
    )


def _rank_evidence(
    method_result,
    case_id: str,
    *,
    method: str,
    depth: int,
    finding: str,
) -> RankEvidence:
    case = _method_case(method_result, case_id)
    return RankEvidence(
        case_id=case_id,
        method=method,
        expected_source_ranks=tuple(
            rank for rank in _flat_expected_ranks(case) if rank <= depth
        ),
        evaluated_depth=depth,
        finding=finding,
    )


def _answer_rank_evidence(
    vector_method: RetrievalMethodResult,
    case_id: str,
    *,
    finding: str,
) -> RankEvidence:
    case = _method_case(vector_method, case_id)
    return RankEvidence(
        case_id=case_id,
        method="answer_generation",
        expected_source_ranks=tuple(
            rank for rank in _flat_expected_ranks(case) if rank <= 5
        ),
        evaluated_depth=5,
        finding=finding,
    )


def _unobserved_failure(
    category: str,
    answer_impact: str,
    root_cause: str,
    next_action: str,
) -> FailureTaxonomyEntry:
    return FailureTaxonomyEntry(
        category=category,
        observed=False,
        case_ids=(),
        methods=(),
        evidence_ranks=(),
        answer_impact=answer_impact,
        root_cause=root_cause,
        current_handling="本轮未观察到；不编造案例。",
        next_action=next_action,
        blocks_rag_v2_freeze=False,
    )


def _assert_safe_payload(value: object, *, path: str = "$") -> None:
    unsafe_path = find_unsafe_payload_path(value, path=path)
    if unsafe_path is not None:
        raise RagV2FreezeError(
            f"Day 17 正式结果包含禁存字段或凭据：{unsafe_path}"
        )
