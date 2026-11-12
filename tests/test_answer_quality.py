from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import app.evaluation.answer_quality as day16
from app.core.config import Settings
from app.rag.core import REFUSAL_ANSWER
from app.schemas.answer_quality import (
    AnswerableManualReview,
    CitationSemanticReview,
    Day16CallCounts,
    Day16GitState,
    Day16InputArtifactHashes,
    Day16ManualReviewArtifact,
    ManualDimensionScore,
    UnanswerableManualReview,
    VectorTop5Identity,
)
from app.schemas.chunk import ChunkSourceSpan
from app.schemas.retrieval import RetrievalResult
from app.schemas.retrieval_experiment import Day14CorpusIdentity


REVIEWER = "主执行代理单审阅者语义复核"


class _FakeCompletions:
    def __init__(self, *, payloads: list[object] | None = None, error: Exception | None = None):
        self.payloads = payloads
        self.error = error
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        index = len(self.requests) - 1
        if self.payloads is None:
            if index < 16:
                payload: object = {
                    "answer": f"Synthetic supported answer {index + 1}.",
                    "citations": [{"evidence_id": "E1"}],
                    "has_sufficient_evidence": True,
                }
            else:
                payload = {
                    "answer": "The supplied synthetic evidence is insufficient.",
                    "citations": [],
                    "has_sufficient_evidence": False,
                }
        else:
            payload = self.payloads[min(index, len(self.payloads) - 1)]
        content = payload if isinstance(payload, str) else json.dumps(payload)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
            ),
            model="deepseek-v4-pro",
        )


class _FakeClient:
    def __init__(self, **kwargs: object):
        self.completions = _FakeCompletions(**kwargs)
        self.chat = SimpleNamespace(completions=self.completions)


def _settings() -> Settings:
    return Settings(
        api_key="synthetic-key",
        base_url="https://synthetic.invalid/v1",
        model_fast="synthetic-fast",
        model_strong="deepseek-v4-pro",
        timeout_seconds=30.0,
    )


def _result(case_index: int, rank: int) -> RetrievalResult:
    segment_index = case_index * 10 + rank
    segment_id = f"ep_{'1' * 16}:{segment_index:06d}"
    return RetrievalResult(
        rank=rank,
        score=1.0 - rank / 10,
        row_index=case_index * 5 + rank - 1,
        chunk_id=f"chunk_{case_index:08x}{rank:08x}",
        episode_id=f"ep_{'1' * 16}",
        source_file="synthetic.txt",
        text=f"Synthetic evidence for case {case_index}, rank {rank}.",
        source_segment_ids=[segment_id],
        source_segment_start_index=segment_index,
        source_segment_end_index=segment_index,
        source_spans=[
            ChunkSourceSpan(
                segment_id=segment_id,
                segment_index=segment_index,
                start_char=0,
                end_char=20,
            )
        ],
        speakers=["Synthetic Guest"],
        section_titles=["Synthetic Section"],
        start_time=float(segment_index),
        end_time=None,
        time_range_complete=False,
    )


def _loaded_inputs() -> day16._LoadedInputs:
    case_ids = (
        "gemini_scaling_limits",
        "agi_timeline_uncertainty",
        "robotics_data_poor_research",
        "multimodal_interaction_beyond_chat",
        "tree_search_rl_synthetic_reward",
        "scaling_invention_compute",
        "alignment_safeguards_pause_trigger",
        "agi_architecture_original_bets",
        "neuroscience_provenance",
        "responsible_scaling_publication_provenance",
        "bioweapon_mitigation_provenance",
        "gato_exchange_provenance",
        "superhuman_ai_consensus_ranking",
        "memory_vs_context_distractor",
        "language_grounding_distractor",
        "weight_security_layers_distractor",
        "sourdough_refusal",
        "gemini_size_refusal",
        "rt2_score_refusal",
        "un_meeting_refusal",
    )
    categories = (
        ["fact"] * 4
        + ["multi_segment"] * 4
        + ["timestamp"] * 4
        + ["similar_term_distractor"] * 4
        + ["unanswerable"] * 4
    )
    cases = tuple(
        SimpleNamespace(
            case_id=case_ids[index],
            question=f"Synthetic question {index}?",
            category=category,
            answerable=index < 16,
            reference_answer="REFERENCE_ANSWER_MUST_NOT_LEAK",
            expected_source_segment_ids=("TRUTH_OUTSIDE_TOP5_MUST_NOT_LEAK",),
        )
        for index, category in enumerate(categories)
    )
    dataset = SimpleNamespace(
        dataset_id="eval_dataset_v1",
        cases_sha256="2" * 64,
        episode_id=f"ep_{'1' * 16}",
        cases=cases,
    )
    hashes = Day16InputArtifactHashes(
        dataset_file_sha256="a" * 64,
        transcript_file_sha256="b" * 64,
        chunk_catalog_file_sha256="c" * 64,
        input_chunk_sha256="d" * 64,
        manifest_file_sha256="e" * 64,
        faiss_file_sha256="f" * 64,
        day12_baseline_sha256="1" * 64,
        day13_experiment_sha256="3" * 64,
        day14_experiment_sha256="4" * 64,
        day15_experiment_sha256="5" * 64,
        grounded_prompt_file_sha256="6" * 64,
        grounded_system_prompt_sha256="7" * 64,
    )
    generation_inputs = tuple(
        day16._GenerationInput(
            case_id=case.case_id,
            query=case.question,
            evidence=tuple(_result(index, rank) for rank in range(1, 6)),
        )
        for index, case in enumerate(cases)
    )
    corpus = Day14CorpusIdentity(
        episode_id=dataset.episode_id,
        source_file="synthetic.txt",
        chunking_algorithm="fixed_character_windows_v1",
        chunk_size=1000,
        overlap=200,
        input_chunk_sha256=hashes.input_chunk_sha256,
        chunk_count=100,
        chunk_id_order_sha256="8" * 64,
    )
    top5 = VectorTop5Identity(
        source_artifact=(
            "data/evaluation/eval_dataset_v1-retrieval-experiment-day14.json"
        ),
        source_experiment_id="synthetic-day14",
        source_retriever="vector",
        source_candidate_depth=20,
        evidence_depth=5,
        case_count=20,
        ordered_top5_sha256="9" * 64,
    )
    return day16._LoadedInputs(
        dataset=dataset,
        vector_index=SimpleNamespace(
            manifest=SimpleNamespace(input_chunk_sha256=hashes.input_chunk_sha256)
        ),
        day14=SimpleNamespace(corpus_identity=corpus),
        hashes=hashes,
        generation_inputs=generation_inputs,
        top5_identity=top5,
    )


@pytest.fixture
def synthetic_run(monkeypatch: pytest.MonkeyPatch) -> day16._LoadedInputs:
    loaded = _loaded_inputs()
    monkeypatch.setattr(day16, "_load_and_validate_inputs", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(day16, "_input_hashes", lambda *args, **kwargs: loaded.hashes)
    monkeypatch.setattr(
        day16,
        "_git_state",
        lambda *args: Day16GitState(
            head_commit="a" * 40,
            branch="synthetic/day16",
            working_tree_dirty=False,
            protected_untracked_files=(),
        ),
    )
    return loaded


def _generate(tmp_path: Path, client: _FakeClient | None = None):
    actual_client = client or _FakeClient()
    path = tmp_path / "generation.json"
    artifact = day16.run_answer_generation(
        settings=_settings(),
        answer_client=actual_client,
        generation_output_path=path,
    )
    return artifact, path, actual_client


def _review_artifact(generation) -> Day16ManualReviewArtifact:
    answerable = []
    unanswerable = []
    for item in generation.cases[:16]:
        answerable.append(
            AnswerableManualReview(
                case_id=item.case_id,
                reviewer_method=REVIEWER,
                correctness=ManualDimensionScore(score=2, reason="关键事实正确。"),
                evidence_alignment=ManualDimensionScore(
                    score=2, reason="核心结论由冻结 Top-5 直接支持。"
                ),
                citation_correctness=ManualDimensionScore(
                    score=2, reason="E1 直接支持核心回答。"
                ),
                completeness=ManualDimensionScore(
                    score=2, reason="覆盖了合成参考中的全部关键要点。"
                ),
                core_claim_count=1,
                supported_core_claim_count=1,
                all_core_claims_supported=True,
                citation_reviews=(
                    CitationSemanticReview(
                        evidence_id="E1",
                        chunk_id=item.citations[0].chunk_id,
                        supports_core_claim=True,
                        truth_segment_alignment="match",
                        reason="该 Chunk 直接承载合成核心结论。",
                    ),
                ),
                citation_segments_match_truth=True,
                duplicate_chunk_displaced_distinct_source=False,
                top5_context_sufficient=True,
                primary_failure_category=None,
                failure_categories=(),
                overall_reason="回答、证据与引用一致。",
            )
        )
    for item in generation.cases[16:]:
        unanswerable.append(
            UnanswerableManualReview(
                case_id=item.case_id,
                reviewer_method=REVIEWER,
                correct_refusal=True,
                incorrect_answer=False,
                used_fixed_refusal=True,
                citations_empty=True,
                invented_fact=False,
                primary_failure_category=None,
                failure_categories=(),
                reason="使用固定拒答，引用为空，未添加事实。",
            )
        )
    return Day16ManualReviewArtifact(
        schema_version=1,
        artifact_type="day16_single_reviewer_manual_review",
        generation_freeze_sha256=generation.generation_freeze_sha256,
        rubric_sha256=day16.RUBRIC_SHA256,
        reviewer_method=REVIEWER,
        answerable_reviews=tuple(answerable),
        unanswerable_reviews=tuple(unanswerable),
    )


def test_generation_uses_only_vector_top5_and_never_leaks_truth(
    synthetic_run, tmp_path: Path
) -> None:
    artifact, path, client = _generate(tmp_path)
    assert path.is_file()
    assert len(client.completions.requests) == 20
    assert artifact.call_counts == Day16CallCounts(
        answer_model_calls=20,
        successful_answer_model_calls=20,
        failed_answer_model_calls=0,
        vector_query_embedding_calls=0,
        document_embedding_calls=0,
        vector_retrieval_calls=0,
        bm25_retrieval_calls=0,
        rrf_fusion_calls=0,
        rerank_calls=0,
        judge_model_calls=0,
    )
    for request, case in zip(client.completions.requests, artifact.cases, strict=True):
        rendered = json.dumps(request["messages"], ensure_ascii=False)
        assert "REFERENCE_ANSWER_MUST_NOT_LEAK" not in rendered
        assert "TRUTH_OUTSIDE_TOP5_MUST_NOT_LEAK" not in rendered
        assert "E6" not in rendered
        assert [item.rank for item in case.evidence] == [1, 2, 3, 4, 5]
        assert len(case.evidence) == 5
    raw = path.read_text(encoding="utf-8")
    assert '"question"' not in raw
    assert '"reference_answer"' not in raw
    assert '"text"' not in raw
    assert "synthetic-key" not in raw


def test_valid_answers_and_refusals_are_frozen_with_provenance(
    synthetic_run, tmp_path: Path
) -> None:
    artifact, _, _ = _generate(tmp_path)
    assert all(item.has_sufficient_evidence for item in artifact.cases[:16])
    assert all(item.citations[0].evidence_id == "E1" for item in artifact.cases[:16])
    assert all(not item.has_sufficient_evidence for item in artifact.cases[16:])
    assert all(item.answer == REFUSAL_ANSWER for item in artifact.cases[16:])
    assert all(not item.citations for item in artifact.cases[16:])
    citation = artifact.cases[0].citations[0]
    assert citation.chunk_id == artifact.cases[0].evidence[0].chunk_id
    assert citation.time_range_complete is False
    assert citation.source_spans[0].start_char == 0


@pytest.mark.parametrize(
    "bad_payload",
    [
        {"answer": "x", "citations": [{"evidence_id": "E6"}], "has_sufficient_evidence": True},
        {"answer": "x", "citations": [], "has_sufficient_evidence": True},
        {"answer": "x", "citations": [{"evidence_id": "E1"}], "has_sufficient_evidence": False},
        {"answer": "", "citations": [], "has_sufficient_evidence": False},
        "not-json",
        {"answer": "x", "citations": [], "has_sufficient_evidence": False, "extra": 1},
    ],
)
def test_contract_failures_leave_no_generation_artifact(
    synthetic_run, tmp_path: Path, bad_payload: object
) -> None:
    client = _FakeClient(payloads=[bad_payload])
    output = tmp_path / "generation.json"
    with pytest.raises(day16.AnswerQualityExperimentError):
        day16.run_answer_generation(
            settings=_settings(),
            answer_client=client,
            generation_output_path=output,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("timeout"),
        RuntimeError("authentication"),
        RuntimeError("rate limit"),
        RuntimeError("service unavailable"),
    ],
)
def test_provider_failures_leave_no_generation_artifact(
    synthetic_run, tmp_path: Path, error: Exception
) -> None:
    output = tmp_path / "generation.json"
    with pytest.raises(day16.AnswerQualityExperimentError):
        day16.run_answer_generation(
            settings=_settings(),
            answer_client=_FakeClient(error=error),
            generation_output_path=output,
        )
    assert not output.exists()


def test_duplicate_model_citation_is_deduplicated_by_existing_boundary(
    synthetic_run, tmp_path: Path
) -> None:
    duplicate = {
        "answer": "Synthetic answer.",
        "citations": [{"evidence_id": "E1"}, {"evidence_id": "E1"}],
        "has_sufficient_evidence": True,
    }
    artifact, _, _ = _generate(tmp_path, _FakeClient(payloads=[duplicate]))
    assert len(artifact.cases[0].citations) == 1


def test_generation_refuses_no_overwrite(synthetic_run, tmp_path: Path) -> None:
    _, output, _ = _generate(tmp_path)
    before = output.read_bytes()
    with pytest.raises(day16.AnswerQualityPersistenceError):
        day16.run_answer_generation(
            settings=_settings(),
            answer_client=_FakeClient(),
            generation_output_path=output,
        )
    assert output.read_bytes() == before


@pytest.mark.parametrize("score", [-1, 3, True])
def test_manual_score_range_is_strict(score: object) -> None:
    with pytest.raises(ValidationError):
        ManualDimensionScore(score=score, reason="Non-empty.")


@pytest.mark.parametrize("reason", ["", "   ", " padded "])
def test_manual_score_requires_auditable_reason(reason: str) -> None:
    with pytest.raises(ValidationError):
        ManualDimensionScore(score=1, reason=reason)


def test_day16_payload_policy_discloses_only_answer_key_exception() -> None:
    day16._assert_safe_payload({"cases": [{"answer": "reviewable output"}]})

    for forbidden in ("question", "text", "prompt", "secret", "raw_response"):
        with pytest.raises(day16.AnswerQualityExperimentError, match=forbidden):
            day16._assert_safe_payload({forbidden: "PRIVATE-CONTENT"})


def test_finalize_computes_scores_refusals_citations_and_failure_counts(
    synthetic_run, tmp_path: Path
) -> None:
    generation, generation_path, _ = _generate(tmp_path)
    review = _review_artifact(generation)
    review_path = tmp_path / "review.json"
    day16.save_answer_quality_artifact(review, review_path)
    output = tmp_path / "formal.json"
    result = day16.finalize_answer_quality_experiment(
        generation_path=generation_path,
        review_path=review_path,
        output_path=output,
    )
    assert result.metrics.answerable_scores.correctness.score_sum == 32
    assert result.metrics.answerable_scores.correctness.maximum_score == 32
    assert result.metrics.answerable_scores.correctness.mean_score == 2
    assert result.metrics.multi_segment_scores.correctness.case_count == 4
    assert result.metrics.proper_noun_subset_scores.correctness.case_count == 6
    assert result.metrics.answerability.correct_refusal_count == 4
    assert result.metrics.answerability.incorrect_answer_count == 0
    assert result.metrics.answerability.false_refusal_count == 0
    assert result.metrics.answerability.successful_answer_count == 16
    assert result.metrics.citations.mapping_legal_rate == 1
    assert result.metrics.citations.semantic_support_rate == 1
    assert result.metrics.citations.core_claim_support_rate == 1
    assert result.metrics.citations.fully_supported_answer_count == 16
    assert all(value == 0 for value in result.metrics.failure_category_counts.values())
    assert day16.load_answer_quality_result(output) == result


def test_finalize_rejects_review_bound_to_another_generation(
    synthetic_run, tmp_path: Path
) -> None:
    generation, generation_path, _ = _generate(tmp_path)
    payload = _review_artifact(generation).model_dump(mode="json")
    payload["generation_freeze_sha256"] = "0" * 64
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "formal.json"
    with pytest.raises(day16.AnswerQualityExperimentError, match="未绑定"):
        day16.finalize_answer_quality_experiment(
            generation_path=generation_path,
            review_path=review_path,
            output_path=output,
        )
    assert not output.exists()


def test_finalize_rejects_missing_score_and_does_not_save(
    synthetic_run, tmp_path: Path
) -> None:
    generation, generation_path, _ = _generate(tmp_path)
    payload = _review_artifact(generation).model_dump(mode="json")
    del payload["answerable_reviews"][0]["correctness"]
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "formal.json"
    with pytest.raises(day16.AnswerQualityExperimentError, match="无法严格加载"):
        day16.finalize_answer_quality_experiment(
            generation_path=generation_path,
            review_path=review_path,
            output_path=output,
        )
    assert not output.exists()


def test_formal_json_excludes_questions_text_prompts_truth_and_secrets(
    synthetic_run, tmp_path: Path
) -> None:
    generation, generation_path, _ = _generate(tmp_path)
    review_path = tmp_path / "review.json"
    day16.save_answer_quality_artifact(_review_artifact(generation), review_path)
    output = tmp_path / "formal.json"
    day16.finalize_answer_quality_experiment(
        generation_path=generation_path,
        review_path=review_path,
        output_path=output,
    )
    raw = output.read_text(encoding="utf-8")
    for forbidden in (
        '"question"',
        '"reference_answer"',
        '"expected_source_segment_ids"',
        '"text"',
        '"prompt"',
        '"raw_response"',
        "synthetic-key",
        "REFERENCE_ANSWER_MUST_NOT_LEAK",
        "TRUTH_OUTSIDE_TOP5_MUST_NOT_LEAK",
    ):
        assert forbidden not in raw


def test_token_counts_and_top5_counts_are_self_consistent(
    synthetic_run, tmp_path: Path
) -> None:
    artifact, _, _ = _generate(tmp_path)
    assert artifact.latency_and_cost.input_tokens == 2000
    assert artifact.latency_and_cost.output_tokens == 400
    assert artifact.latency_and_cost.total_tokens == 2400
    assert sum(item.telemetry.token_usage.total_tokens for item in artifact.cases) == 2400
    assert sum(len(item.evidence) for item in artifact.cases) == 100


def test_wrong_model_or_timeout_is_rejected_before_any_call(
    synthetic_run, tmp_path: Path
) -> None:
    client = _FakeClient()
    bad = Settings(
        api_key="x",
        base_url="https://synthetic.invalid",
        model_fast="x",
        model_strong="another-model",
        timeout_seconds=31,
    )
    with pytest.raises(day16.AnswerQualityExperimentError):
        day16.run_answer_generation(
            settings=bad,
            answer_client=client,
            generation_output_path=tmp_path / "generation.json",
        )
    assert not client.completions.requests
