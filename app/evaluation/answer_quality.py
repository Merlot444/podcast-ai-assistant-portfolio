"""Day 16 frozen-Vector answer generation and single-reviewer evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Sequence

from pydantic import BaseModel, ValidationError

from app.core.errors import safe_error_message
from app.chunking.pipeline import load_processed_transcript
from app.core.config import Settings
from app.evaluation.baseline import file_sha256
from app.evaluation.payload_policy import (
    DAY16_DISCLOSED_CONTENT_KEY_EXCEPTIONS,
    find_unsafe_payload_path,
)
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
)
from app.evaluation.rerank_experiment import DEFAULT_RERANK_EXPERIMENT_OUTPUT
from app.indexing import load_vector_index
from app.indexing.index import chunked_transcript_fingerprint
from app.rag.core import REFUSAL_ANSWER, generate_grounded_answer
from app.rag.prompts import GROUNDED_ANSWER_SYSTEM_PROMPT
from app.schemas.answer_quality import (
    FAILURE_CATEGORIES,
    AnswerCallTelemetry,
    AnswerEvidenceMetadata,
    AnswerGenerationArtifact,
    AnswerModelConfiguration,
    AnswerQualityCaseResult,
    AnswerQualityExperimentResult,
    AnswerTokenUsage,
    AnswerabilityMetrics,
    CitationMetrics,
    Day16CallCounts,
    Day16GitState,
    Day16InputArtifactHashes,
    Day16LatencyAndCost,
    Day16ManualReviewArtifact,
    Day16Metrics,
    DimensionAggregate,
    GeneratedAnswerCase,
    ScoreAggregate,
    VectorTop5Identity,
)
from app.schemas.chunk import ChunkedTranscript, TranscriptChunk
from app.schemas.chunk_experiment import ChunkParameterExperimentResult
from app.schemas.evaluation_result import RetrievalBaselineResult
from app.schemas.rag import GroundedAnswer, ModelGroundedAnswer
from app.schemas.rerank_experiment import RerankExperimentResult
from app.schemas.retrieval import RetrievalResult
from app.schemas.retrieval_experiment import (
    RetrievalExperimentResult,
    RetrievalMethodResult,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANSWER_QUALITY_OUTPUT = (
    _PROJECT_ROOT
    / "data"
    / "evaluation"
    / "eval_dataset_v1-answer-quality-experiment-day16.json"
)
DEFAULT_GROUNDED_PROMPT_FILE = _PROJECT_ROOT / "app" / "rag" / "prompts.py"

EXPECTED_FROZEN_HASHES = {
    "dataset_file_sha256": "73703862b0976cb7bed78fd2171beff594b4817047d5ca4ab211cf26fd2995ab",
    "transcript_file_sha256": "566013e83578476e31df839ea98065f8a9be5df38882ba63894c10bf49f98af1",
    "chunk_catalog_file_sha256": "338da2e92a5be3a8862d4f4c1885e54c3d6ccd45655a823eba4231105cb2916c",
    "input_chunk_sha256": "89324c632d8b25975e1891041608761c771e8baabad8a020af70324a023f805b",
    "manifest_file_sha256": "ea4361c600cb1448a254127ea17aee03d55e84b813ac53402d2bdf1a4a79d231",
    "faiss_file_sha256": "2d7df97a78354cece4c233b75a73d0649ecbd979ac39edef9a05f4851b45a7b3",
    "day12_baseline_sha256": "bffc430f2abfeb68e7f8d414946fac307d23d03ec9d46a623e745b4939752a82",
    "day13_experiment_sha256": "b5c989e1077274029f158600ebba72fac022408b357d8160c820279ab6397c78",
    "day14_experiment_sha256": "bbbeae51724c13032e5efc41394558988c81ec431302bcafab135aa456491de5",
    "day15_experiment_sha256": "e2191394c2b6016599472b18070c5df11ba11e4b70effc1709252bf79c41a7dd",
    "grounded_prompt_file_sha256": "47c30c4e193f651e3ffe10b42b663a8431fc85d263e8613ea5045a50154981fa",
    "grounded_system_prompt_sha256": "ac152ad2a4044645663d50f71a163cb4f6fa5a345cbf99bf709f438c32de8f57",
}

PROPER_NOUN_SUBSET_CASE_IDS = (
    "gemini_scaling_limits",
    "agi_timeline_uncertainty",
    "robotics_data_poor_research",
    "tree_search_rl_synthetic_reward",
    "agi_architecture_original_bets",
    "gato_exchange_provenance",
)

SCORING_RUBRIC = {
    "correctness": {
        "2": "关键事实正确",
        "1": "核心方向正确，但存在遗漏、模糊或轻微错误",
        "0": "核心答案错误、矛盾或没有回答问题",
    },
    "evidence_alignment": {
        "2": "所有实质性结论均由 Top-5 支持",
        "1": "主要结论有支持，但存在次要无依据内容",
        "0": "核心结论无依据或与证据矛盾",
    },
    "citation_correctness": {
        "2": "所选引用直接支持核心回答",
        "1": "引用相关但支持不完整、过宽或部分错配",
        "0": "引用无法支持核心回答",
    },
    "completeness": {
        "2": "覆盖参考答案和稳定真值中的关键要点",
        "1": "回答核心问题，但遗漏重要要点或第二来源",
        "0": "严重不完整",
    },
}
RUBRIC_SHA256 = hashlib.sha256(
    json.dumps(
        SCORING_RUBRIC,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

class AnswerQualityExperimentError(RuntimeError):
    """Raised when a complete and trustworthy Day 16 run cannot be produced."""


class AnswerQualityPersistenceError(RuntimeError):
    """Raised when a Day 16 artifact cannot be written without ambiguity."""


@dataclass(frozen=True)
class _GenerationInput:
    case_id: str
    query: str
    evidence: tuple[RetrievalResult, ...]


@dataclass(frozen=True)
class _LoadedInputs:
    dataset: Any
    vector_index: Any
    day14: RetrievalExperimentResult
    hashes: Day16InputArtifactHashes
    generation_inputs: tuple[_GenerationInput, ...]
    top5_identity: VectorTop5Identity


@dataclass(frozen=True)
class _ProviderRecord:
    usage: AnswerTokenUsage
    response_model_name: str | None


class _RecordingAnswerClient:
    """Count the exact provider calls without retaining raw provider responses."""

    def __init__(self, delegate: object) -> None:
        self.delegate = delegate
        self.attempt_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.records: list[_ProviderRecord] = []
        self.chat = _ChatProxy(delegate, self)


class _ChatProxy:
    def __init__(self, delegate: object, owner: _RecordingAnswerClient) -> None:
        self.completions = _CompletionsProxy(delegate, owner)


class _CompletionsProxy:
    def __init__(self, delegate: object, owner: _RecordingAnswerClient) -> None:
        self.delegate = delegate
        self.owner = owner

    def create(self, **kwargs: object) -> object:
        self.owner.attempt_count += 1
        try:
            response = self.delegate.chat.completions.create(**kwargs)
        except Exception:
            self.owner.failure_count += 1
            raise
        self.owner.success_count += 1
        usage = getattr(response, "usage", None)
        if usage is None:
            token_usage = AnswerTokenUsage()
        else:
            token_usage = AnswerTokenUsage(
                input_tokens=getattr(usage, "prompt_tokens", None),
                output_tokens=getattr(usage, "completion_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
            )
        response_model = getattr(response, "model", None)
        self.owner.records.append(
            _ProviderRecord(
                usage=token_usage,
                response_model_name=(
                    response_model.strip()
                    if isinstance(response_model, str) and response_model.strip()
                    else None
                ),
            )
        )
        return response


def run_answer_generation(
    *,
    settings: Settings,
    answer_client: object,
    generation_output_path: str | Path,
    dataset_path: str | Path = DEFAULT_EVALUATION_DATASET,
    transcript_path: str | Path = DEFAULT_EVALUATION_TRANSCRIPT,
    chunk_catalog_path: str | Path = DEFAULT_CHUNK_CATALOG,
    index_directory: str | Path = DEFAULT_EVALUATION_INDEX_DIRECTORY,
    day12_baseline_path: str | Path = DEFAULT_RETRIEVAL_BASELINE,
    day13_experiment_path: str | Path = DEFAULT_DAY13_EXPERIMENT,
    day14_experiment_path: str | Path = DEFAULT_RETRIEVAL_EXPERIMENT_OUTPUT,
    day15_experiment_path: str | Path = DEFAULT_RERANK_EXPERIMENT_OUTPUT,
    overwrite: bool = False,
    enforce_frozen_hashes: bool = True,
    clock: Callable[[], float] = perf_counter,
) -> AnswerGenerationArtifact:
    """Generate and freeze all 20 answers before any review artifact is read."""
    _validate_answer_settings(settings)
    paths = _resolve_paths(
        dataset_path=dataset_path,
        transcript_path=transcript_path,
        chunk_catalog_path=chunk_catalog_path,
        index_directory=index_directory,
        day12_baseline_path=day12_baseline_path,
        day13_experiment_path=day13_experiment_path,
        day14_experiment_path=day14_experiment_path,
        day15_experiment_path=day15_experiment_path,
        output_path=generation_output_path,
    )
    _preflight_output(paths, overwrite=overwrite, label="生成冻结")
    loaded = _load_and_validate_inputs(
        paths,
        enforce_frozen_hashes=enforce_frozen_hashes,
    )
    recording_client = _RecordingAnswerClient(answer_client)
    generated_cases: list[GeneratedAnswerCase] = []
    response_models: list[str] = []

    for item in loaded.generation_inputs:
        before_attempts = recording_client.attempt_count
        started = clock()
        try:
            answer = generate_grounded_answer(
                recording_client,
                model=settings.model_strong,
                query=item.query,
                evidence=list(item.evidence),
                max_tokens=1200,
            )
        except Exception as error:
            raise AnswerQualityExperimentError(
                f"案例 {item.case_id} 回答生成失败，未保存完整生成冻结："
                f"{type(error).__name__}"
            ) from error
        elapsed = clock() - started
        if recording_client.attempt_count != before_attempts + 1:
            raise AnswerQualityExperimentError("每道题必须恰好调用一次回答模型")
        provider_record = recording_client.records[-1]
        if provider_record.response_model_name is not None:
            response_models.append(provider_record.response_model_name)
        generated_cases.append(
            _generated_case(
                case_id=item.case_id,
                evidence=item.evidence,
                answer=answer,
                elapsed_seconds=elapsed,
                question_characters=len(item.query),
                token_usage=provider_record.usage,
            )
        )

    if (
        recording_client.attempt_count != 20
        or recording_client.success_count != 20
        or recording_client.failure_count != 0
    ):
        raise AnswerQualityExperimentError("正式生成必须完成 20 次成功回答调用")
    after_hashes = _input_hashes(paths, loaded.vector_index.manifest.input_chunk_sha256)
    if after_hashes != loaded.hashes:
        raise AnswerQualityExperimentError("生成前后冻结输入哈希发生变化，拒绝保存")

    response_model = _single_response_model(response_models)
    model_configuration = _model_configuration(response_model)
    freeze_sha256 = generation_cases_sha256(generated_cases)
    timestamp = datetime.now(UTC)
    artifact = AnswerGenerationArtifact(
        schema_version=1,
        artifact_type="day16_answer_generation_freeze",
        generation_id=(
            f"eval_dataset_v1-answer-generation-day16-"
            f"{timestamp.strftime('%Y%m%dt%H%M%Sz').lower()}"
        ),
        created_at_utc=timestamp,
        dataset_id=loaded.dataset.dataset_id,
        dataset_cases_sha256=loaded.dataset.cases_sha256,
        episode_id=loaded.dataset.episode_id,
        git_state=_git_state(_PROJECT_ROOT),
        corpus_identity=loaded.day14.corpus_identity,
        input_artifact_hashes=loaded.hashes,
        vector_top5_identity=loaded.top5_identity,
        answer_model_configuration=model_configuration,
        generation_freeze_sha256=freeze_sha256,
        cases=tuple(generated_cases),
        call_counts=Day16CallCounts(
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
        ),
        latency_and_cost=_latency_and_cost(generated_cases),
    )
    _assert_safe_payload(artifact.model_dump(mode="json"))
    save_answer_quality_artifact(artifact, paths["output"], overwrite=overwrite)
    return artifact


def finalize_answer_quality_experiment(
    *,
    generation_path: str | Path,
    review_path: str | Path,
    output_path: str | Path = DEFAULT_ANSWER_QUALITY_OUTPUT,
    dataset_path: str | Path = DEFAULT_EVALUATION_DATASET,
    transcript_path: str | Path = DEFAULT_EVALUATION_TRANSCRIPT,
    chunk_catalog_path: str | Path = DEFAULT_CHUNK_CATALOG,
    index_directory: str | Path = DEFAULT_EVALUATION_INDEX_DIRECTORY,
    day12_baseline_path: str | Path = DEFAULT_RETRIEVAL_BASELINE,
    day13_experiment_path: str | Path = DEFAULT_DAY13_EXPERIMENT,
    day14_experiment_path: str | Path = DEFAULT_RETRIEVAL_EXPERIMENT_OUTPUT,
    day15_experiment_path: str | Path = DEFAULT_RERANK_EXPERIMENT_OUTPUT,
    overwrite: bool = False,
    enforce_frozen_hashes: bool = True,
) -> AnswerQualityExperimentResult:
    """Load the frozen answers, then apply the separately entered manual review."""
    paths = _resolve_paths(
        dataset_path=dataset_path,
        transcript_path=transcript_path,
        chunk_catalog_path=chunk_catalog_path,
        index_directory=index_directory,
        day12_baseline_path=day12_baseline_path,
        day13_experiment_path=day13_experiment_path,
        day14_experiment_path=day14_experiment_path,
        day15_experiment_path=day15_experiment_path,
        output_path=output_path,
    )
    paths["generation"] = Path(generation_path)
    paths["review"] = Path(review_path)
    _preflight_output(paths, overwrite=overwrite, label="正式 Day 16 结果")
    loaded = _load_and_validate_inputs(
        paths,
        enforce_frozen_hashes=enforce_frozen_hashes,
    )
    generation = load_answer_generation(paths["generation"])
    review = load_manual_review(paths["review"])
    _validate_generation_against_inputs(generation, loaded)
    _validate_manual_review(generation, review, loaded.dataset)

    generated_by_id = {item.case_id: item for item in generation.cases}
    answerable_by_id = {item.case_id: item for item in review.answerable_reviews}
    unanswerable_by_id = {
        item.case_id: item for item in review.unanswerable_reviews
    }
    cases: list[AnswerQualityCaseResult] = []
    for case in loaded.dataset.cases:
        generated = generated_by_id[case.case_id]
        cases.append(
            AnswerQualityCaseResult(
                case_id=case.case_id,
                category=case.category,
                answerable=case.answerable,
                evidence=generated.evidence,
                answer=generated.answer,
                citations=generated.citations,
                has_sufficient_evidence=generated.has_sufficient_evidence,
                telemetry=generated.telemetry,
                answerable_review=answerable_by_id.get(case.case_id),
                unanswerable_review=unanswerable_by_id.get(case.case_id),
            )
        )

    after_hashes = _input_hashes(paths, loaded.vector_index.manifest.input_chunk_sha256)
    if after_hashes != loaded.hashes:
        raise AnswerQualityExperimentError("评分前后冻结输入哈希发生变化，拒绝保存")
    timestamp = datetime.now(UTC)
    result = AnswerQualityExperimentResult(
        schema_version=1,
        experiment_id=(
            f"eval_dataset_v1-answer-quality-day16-"
            f"{timestamp.strftime('%Y%m%dt%H%M%Sz').lower()}"
        ),
        created_at_utc=timestamp,
        dataset_id=loaded.dataset.dataset_id,
        dataset_cases_sha256=loaded.dataset.cases_sha256,
        episode_id=loaded.dataset.episode_id,
        reviewer_method="主执行代理单审阅者语义复核",
        rubric_sha256=RUBRIC_SHA256,
        generation_freeze_sha256=generation.generation_freeze_sha256,
        git_state=generation.git_state,
        corpus_identity=generation.corpus_identity,
        input_artifact_hashes=generation.input_artifact_hashes,
        vector_top5_identity=generation.vector_top5_identity,
        answer_model_configuration=generation.answer_model_configuration,
        cases=tuple(cases),
        metrics=_build_metrics(cases),
        call_counts=generation.call_counts,
        latency_and_cost=generation.latency_and_cost,
        limitations=(
            "The evaluation covers one English podcast episode and only 20 frozen cases.",
            "Each answer was generated once with one provider model; run-to-run variance was not measured.",
            "Semantic scoring is a main-agent single-reviewer review, not multi-annotator consensus.",
            "Citations are answer-level selections and are not aligned to individual sentences or clauses.",
            "Vector Top-5 is a fixed context ceiling; missing second sources cannot be repaired by generation.",
            "Provider model revision and currency-denominated actual cost were not returned.",
        ),
        future_evaluation_expansion_priorities=(
            "Add independently frozen episodes, speakers, and podcast genres without modifying v1.",
            "Increase multi-source, ambiguous, entity-heavy, temporal, and hard-unanswerable cases.",
            "Use a development/test split before tuning retrieval, fusion, prompts, or thresholds.",
            "Add sentence-level claim-to-citation annotation and repeated-run variance checks.",
        ),
        day17_required_inputs=(
            "Frozen Day 12-16 artifacts and their verified hashes.",
            "Per-case Day 16 scores, refusal outcomes, citation audits, and failure categories.",
            "Day 14 default-retention decision and Day 15 query-specific rerank evidence.",
            "Documented latency, token, unknown-cost, and small-sample limitations.",
        ),
    )
    _assert_safe_payload(result.model_dump(mode="json"))
    save_answer_quality_artifact(result, paths["output"], overwrite=overwrite)
    return result


def load_answer_generation(path: str | Path) -> AnswerGenerationArtifact:
    return _load_model(Path(path), AnswerGenerationArtifact, "Day 16 生成冻结")


def load_manual_review(path: str | Path) -> Day16ManualReviewArtifact:
    return _load_model(Path(path), Day16ManualReviewArtifact, "Day 16 人工式复核")


def load_answer_quality_result(path: str | Path) -> AnswerQualityExperimentResult:
    return _load_model(Path(path), AnswerQualityExperimentResult, "Day 16 正式结果")


def save_answer_quality_artifact(
    result: BaseModel,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically save one complete strict artifact and refuse silent replacement."""
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise AnswerQualityPersistenceError(
            f"目标已存在：{destination}；不会静默覆盖"
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
        raise AnswerQualityPersistenceError("Day 16 artifact 写入失败") from error
    return destination


def generation_cases_sha256(cases: Sequence[GeneratedAnswerCase]) -> str:
    canonical = json.dumps(
        [item.model_dump(mode="json") for item in cases],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def ordered_top5_sha256(method: RetrievalMethodResult) -> str:
    payload = [
        {
            "case_id": case.case_id,
            "candidates": [
                item.model_dump(mode="json")
                for item in case.retrieved_candidates[:5]
            ],
        }
        for case in method.cases
    ]
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_and_validate_inputs(
    paths: dict[str, Path],
    *,
    enforce_frozen_hashes: bool,
) -> _LoadedInputs:
    vector_index = load_vector_index(paths["index_directory"])
    before_hashes = _input_hashes(paths, vector_index.manifest.input_chunk_sha256)
    if enforce_frozen_hashes and before_hashes.model_dump() != EXPECTED_FROZEN_HASHES:
        differences = {
            key: value
            for key, value in before_hashes.model_dump().items()
            if EXPECTED_FROZEN_HASHES.get(key) != value
        }
        raise AnswerQualityExperimentError(
            f"Day 16 冻结输入 SHA-256 漂移：{sorted(differences)}"
        )
    dataset = load_evaluation_dataset(paths["dataset"])
    transcript = load_processed_transcript(paths["transcript"])
    validate_evaluation_sources(
        dataset,
        transcript=transcript,
        manifest=vector_index.manifest,
    )
    chunked = _load_model(paths["chunk_catalog"], ChunkedTranscript, "Chunk catalog")
    day12 = _load_model(paths["day12"], RetrievalBaselineResult, "Day 12 baseline")
    day13 = _load_model(paths["day13"], ChunkParameterExperimentResult, "Day 13 result")
    day14 = _load_model(paths["day14"], RetrievalExperimentResult, "Day 14 result")
    day15 = _load_model(paths["day15"], RerankExperimentResult, "Day 15 result")
    _validate_formal_inputs(
        dataset=dataset,
        chunked=chunked,
        manifest=vector_index.manifest,
        day12=day12,
        day13=day13,
        day14=day14,
        day15=day15,
        hashes=before_hashes,
    )
    vector_method = _day14_vector_method(day14)
    generation_inputs = _build_generation_inputs(
        dataset=dataset,
        vector_method=vector_method,
        manifest=vector_index.manifest,
    )
    top5_identity = VectorTop5Identity(
        source_artifact=(
            "data/evaluation/eval_dataset_v1-retrieval-experiment-day14.json"
        ),
        source_experiment_id=day14.experiment_id,
        source_retriever="vector",
        source_candidate_depth=20,
        evidence_depth=5,
        case_count=20,
        ordered_top5_sha256=ordered_top5_sha256(vector_method),
    )
    return _LoadedInputs(
        dataset=dataset,
        vector_index=vector_index,
        day14=day14,
        hashes=before_hashes,
        generation_inputs=generation_inputs,
        top5_identity=top5_identity,
    )


def _validate_formal_inputs(
    *,
    dataset: Any,
    chunked: ChunkedTranscript,
    manifest: Any,
    day12: RetrievalBaselineResult,
    day13: ChunkParameterExperimentResult,
    day14: RetrievalExperimentResult,
    day15: RerankExperimentResult,
    hashes: Day16InputArtifactHashes,
) -> None:
    if len(dataset.cases) != 20:
        raise AnswerQualityExperimentError("Day 16 必须使用冻结的 20 道题")
    answerable_count = sum(case.answerable for case in dataset.cases)
    if answerable_count != 16 or len(dataset.cases) - answerable_count != 4:
        raise AnswerQualityExperimentError("Day 16 必须保持 16 可答 / 4 无答案")
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
    if actual_catalog != expected_catalog or (manifest.chunk_size, manifest.overlap) != (
        1000,
        200,
    ):
        raise AnswerQualityExperimentError("1000/200 Chunk catalog 与 manifest 不一致")
    if manifest.input_chunk_sha256 != hashes.input_chunk_sha256:
        raise AnswerQualityExperimentError("input_chunk_sha256 与 manifest 不一致")
    identities = (
        day12.dataset_cases_sha256,
        day13.dataset_cases_sha256,
        day14.dataset_cases_sha256,
        day15.dataset_cases_sha256,
    )
    if any(value != dataset.cases_sha256 for value in identities):
        raise AnswerQualityExperimentError("Day 11～15 评测集身份不一致")
    if any(
        episode != dataset.episode_id
        for episode in (day14.episode_id, day15.episode_id, manifest.episode_id)
    ):
        raise AnswerQualityExperimentError("Day 11～15 episode 身份不一致")
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
        raise AnswerQualityExperimentError("Day 14 内嵌哈希与当前冻结输入不一致")
    if day15.input_artifact_hashes.day14_experiment_sha256 != hashes.day14_experiment_sha256:
        raise AnswerQualityExperimentError("Day 15 未绑定当前冻结 Day 14 artifact")
    if day15.recommendation.production_default_changed is not False:
        raise AnswerQualityExperimentError("Day 15 默认链路决策发生漂移")
    vector_method = _day14_vector_method(day14)
    expected_case_ids = tuple(case.case_id for case in dataset.cases)
    if tuple(case.case_id for case in vector_method.cases) != expected_case_ids:
        raise AnswerQualityExperimentError("Day 14 Vector case 顺序与 Day 11 不一致")
    chunk_by_id = {chunk.chunk_id: chunk for chunk in manifest.chunks}
    for case in vector_method.cases:
        candidates = case.retrieved_candidates
        if len(candidates) != 20 or [item.rank for item in candidates] != list(
            range(1, 21)
        ):
            raise AnswerQualityExperimentError("Day 14 Vector 必须保存连续完整 Top-20")
        if len({item.chunk_id for item in candidates}) != 20:
            raise AnswerQualityExperimentError("Day 14 Vector Top-20 不能重复 Chunk")
        for candidate in candidates:
            chunk = chunk_by_id.get(candidate.chunk_id)
            if chunk is None or tuple(chunk.source_segment_ids) != candidate.source_segment_ids:
                raise AnswerQualityExperimentError(
                    "Day 14 Vector 候选无法唯一映射到 manifest 正文与来源"
                )


def _build_generation_inputs(
    *,
    dataset: Any,
    vector_method: RetrievalMethodResult,
    manifest: Any,
) -> tuple[_GenerationInput, ...]:
    cases_by_id = {case.case_id: case for case in vector_method.cases}
    chunk_by_id: dict[str, TranscriptChunk] = {
        chunk.chunk_id: chunk for chunk in manifest.chunks
    }
    inputs: list[_GenerationInput] = []
    for case in dataset.cases:
        saved_case = cases_by_id[case.case_id]
        evidence: list[RetrievalResult] = []
        for candidate in saved_case.retrieved_candidates[:5]:
            chunk = chunk_by_id[candidate.chunk_id]
            evidence.append(
                RetrievalResult(
                    rank=candidate.rank,
                    score=candidate.score,
                    row_index=chunk.chunk_index,
                    chunk_id=chunk.chunk_id,
                    episode_id=chunk.episode_id,
                    source_file=chunk.source_file,
                    text=chunk.text,
                    source_segment_ids=list(chunk.source_segment_ids),
                    source_segment_start_index=chunk.source_segment_start_index,
                    source_segment_end_index=chunk.source_segment_end_index,
                    source_spans=[span.model_copy(deep=True) for span in chunk.source_spans],
                    speakers=list(chunk.speakers),
                    section_titles=list(chunk.section_titles),
                    start_time=chunk.start_time,
                    end_time=chunk.end_time,
                    time_range_complete=chunk.time_range_complete,
                )
            )
        if len(evidence) != 5 or [item.rank for item in evidence] != [1, 2, 3, 4, 5]:
            raise AnswerQualityExperimentError("每题必须只使用原始 Vector Top-5")
        inputs.append(
            _GenerationInput(
                case_id=case.case_id,
                query=case.question,
                evidence=tuple(evidence),
            )
        )
    return tuple(inputs)


def _generated_case(
    *,
    case_id: str,
    evidence: Sequence[RetrievalResult],
    answer: GroundedAnswer,
    elapsed_seconds: float,
    question_characters: int,
    token_usage: AnswerTokenUsage,
) -> GeneratedAnswerCase:
    evidence_metadata = tuple(
        AnswerEvidenceMetadata(
            evidence_id=f"E{index}",
            rank=item.rank,
            score=item.score,
            row_index=item.row_index,
            chunk_id=item.chunk_id,
            episode_id=item.episode_id,
            source_file=item.source_file,
            speakers=tuple(item.speakers),
            section_titles=tuple(item.section_titles),
            start_time=item.start_time,
            end_time=item.end_time,
            time_range_complete=item.time_range_complete,
            source_segment_ids=tuple(item.source_segment_ids),
            source_segment_start_index=item.source_segment_start_index,
            source_segment_end_index=item.source_segment_end_index,
            source_spans=tuple(span.model_copy(deep=True) for span in item.source_spans),
        )
        for index, item in enumerate(evidence, start=1)
    )
    return GeneratedAnswerCase(
        case_id=case_id,
        evidence=evidence_metadata,
        answer=answer.answer,
        citations=tuple(item.model_copy(deep=True) for item in answer.citations),
        has_sufficient_evidence=answer.has_sufficient_evidence,
        telemetry=AnswerCallTelemetry(
            elapsed_seconds=elapsed_seconds,
            question_characters=question_characters,
            evidence_characters=sum(len(item.text) for item in evidence),
            token_usage=token_usage,
        ),
    )


def _validate_generation_against_inputs(
    generation: AnswerGenerationArtifact,
    loaded: _LoadedInputs,
) -> None:
    if generation.input_artifact_hashes != loaded.hashes:
        raise AnswerQualityExperimentError("生成冻结未绑定当前 Day 16 输入哈希")
    if generation.vector_top5_identity != loaded.top5_identity:
        raise AnswerQualityExperimentError("生成冻结未使用当前原始 Vector Top-5")
    if generation.generation_freeze_sha256 != generation_cases_sha256(generation.cases):
        raise AnswerQualityExperimentError("生成冻结 SHA-256 与答案内容不一致")
    if tuple(item.case_id for item in generation.cases) != tuple(
        case.case_id for case in loaded.dataset.cases
    ):
        raise AnswerQualityExperimentError("生成冻结 case 顺序与 Day 11 不一致")
    expected_by_id = {item.case_id: item for item in loaded.generation_inputs}
    for generated in generation.cases:
        expected = expected_by_id[generated.case_id]
        expected_metadata = _generated_case(
            case_id=generated.case_id,
            evidence=expected.evidence,
            answer=GroundedAnswer(
                answer=generated.answer,
                citations=list(generated.citations),
                has_sufficient_evidence=generated.has_sufficient_evidence,
            ),
            elapsed_seconds=generated.telemetry.elapsed_seconds,
            question_characters=generated.telemetry.question_characters,
            token_usage=generated.telemetry.token_usage,
        )
        if generated.evidence != expected_metadata.evidence:
            raise AnswerQualityExperimentError("生成冻结 evidence 与原始 Top-5 不一致")


def _validate_manual_review(
    generation: AnswerGenerationArtifact,
    review: Day16ManualReviewArtifact,
    dataset: Any,
) -> None:
    if review.generation_freeze_sha256 != generation.generation_freeze_sha256:
        raise AnswerQualityExperimentError("人工式复核未绑定当前生成冻结")
    if review.rubric_sha256 != RUBRIC_SHA256:
        raise AnswerQualityExperimentError("人工式复核量表 SHA-256 发生漂移")
    expected_answerable = tuple(
        case.case_id for case in dataset.cases if case.answerable
    )
    expected_unanswerable = tuple(
        case.case_id for case in dataset.cases if not case.answerable
    )
    if tuple(item.case_id for item in review.answerable_reviews) != expected_answerable:
        raise AnswerQualityExperimentError("四项评分必须按顺序覆盖 16 道可回答题")
    if (
        tuple(item.case_id for item in review.unanswerable_reviews)
        != expected_unanswerable
    ):
        raise AnswerQualityExperimentError("拒答复核必须按顺序覆盖 4 道无答案题")
    generated_by_id = {item.case_id: item for item in generation.cases}
    for item in review.answerable_reviews:
        generated = generated_by_id[item.case_id]
        expected_citations = tuple(
            (citation.evidence_id, citation.chunk_id)
            for citation in generated.citations
        )
        reviewed_citations = tuple(
            (citation.evidence_id, citation.chunk_id)
            for citation in item.citation_reviews
        )
        if reviewed_citations != expected_citations:
            raise AnswerQualityExperimentError(
                f"案例 {item.case_id} 的语义引用复核未逐条覆盖实际引用"
            )
        if not generated.has_sufficient_evidence and "false_refusal" not in item.failure_categories:
            raise AnswerQualityExperimentError("可回答题拒答必须归类为 false_refusal")
    for item in review.unanswerable_reviews:
        generated = generated_by_id[item.case_id]
        actual_fixed = (
            not generated.has_sufficient_evidence
            and generated.answer == REFUSAL_ANSWER
            and not generated.citations
        )
        if (
            item.correct_refusal != actual_fixed
            or item.incorrect_answer != generated.has_sufficient_evidence
            or item.used_fixed_refusal != (generated.answer == REFUSAL_ANSWER)
            or item.citations_empty != (not generated.citations)
        ):
            raise AnswerQualityExperimentError(
                f"案例 {item.case_id} 的拒答复核与冻结输出不一致"
            )


def _build_metrics(cases: Sequence[AnswerQualityCaseResult]) -> Day16Metrics:
    answerable = [item for item in cases if item.answerable]
    reviews = [item.answerable_review for item in answerable]
    if any(item is None for item in reviews):
        raise AnswerQualityExperimentError("16 道可回答题评分不完整")
    typed_reviews = [item for item in reviews if item is not None]
    category_scores = {
        category: _score_aggregate(
            [
                item.answerable_review
                for item in answerable
                if item.category == category and item.answerable_review is not None
            ]
        )
        for category in (
            "fact",
            "multi_segment",
            "timestamp",
            "similar_term_distractor",
        )
    }
    proper_noun_reviews = [
        item.answerable_review
        for item in answerable
        if item.case_id in PROPER_NOUN_SUBSET_CASE_IDS
        and item.answerable_review is not None
    ]
    unanswerable_reviews = [
        item.unanswerable_review
        for item in cases
        if not item.answerable and item.unanswerable_review is not None
    ]
    citations = [citation for item in answerable for citation in item.citations]
    citation_reviews = [
        audit for review in typed_reviews for audit in review.citation_reviews
    ]
    supported_citations = sum(item.supports_core_claim for item in citation_reviews)
    truth_aligned = sum(item.truth_segment_alignment == "match" for item in citation_reviews)
    supported_claims = sum(item.supported_core_claim_count for item in typed_reviews)
    total_claims = sum(item.core_claim_count for item in typed_reviews)
    failure_counts = {
        category: sum(
            category in review.failure_categories
            for review in typed_reviews + unanswerable_reviews
        )
        for category in FAILURE_CATEGORIES
    }
    primary_counts = {
        category: sum(
            review.primary_failure_category == category
            for review in typed_reviews + unanswerable_reviews
        )
        for category in FAILURE_CATEGORIES
    }
    return Day16Metrics(
        answerable_scores=_score_aggregate(typed_reviews),
        category_scores=category_scores,
        multi_segment_scores=category_scores["multi_segment"],
        proper_noun_subset_case_ids=PROPER_NOUN_SUBSET_CASE_IDS,
        proper_noun_subset_scores=_score_aggregate(proper_noun_reviews),
        answerability=AnswerabilityMetrics(
            correct_refusal_count=sum(item.correct_refusal for item in unanswerable_reviews),
            unanswerable_denominator=4,
            incorrect_answer_count=sum(item.incorrect_answer for item in unanswerable_reviews),
            false_refusal_count=sum(not item.has_sufficient_evidence for item in answerable),
            answerable_denominator=16,
            successful_answer_count=sum(item.has_sufficient_evidence for item in answerable),
        ),
        citations=CitationMetrics(
            mapping_legal_citation_count=len(citations),
            total_citation_count=len(citations),
            mapping_legal_rate=(1.0 if citations else None),
            semantically_supported_citation_count=supported_citations,
            semantic_support_rate=_rate(supported_citations, len(citation_reviews)),
            truth_aligned_citation_count=truth_aligned,
            supported_core_claim_count=supported_claims,
            total_core_claim_count=total_claims,
            core_claim_support_rate=_rate(supported_claims, total_claims),
            fully_supported_answer_count=sum(
                item.all_core_claims_supported for item in typed_reviews
            ),
            answerable_denominator=16,
            duplicate_displacement_answer_count=sum(
                item.duplicate_chunk_displaced_distinct_source for item in typed_reviews
            ),
            citation_granularity_limit=(
                "answer_level_citations_without_sentence_level_alignment"
            ),
        ),
        failure_category_counts=failure_counts,
        primary_failure_category_counts=primary_counts,
    )


def _score_aggregate(reviews: Sequence[Any]) -> ScoreAggregate:
    return ScoreAggregate(
        correctness=_dimension_aggregate(reviews, "correctness"),
        evidence_alignment=_dimension_aggregate(reviews, "evidence_alignment"),
        citation_correctness=_dimension_aggregate(reviews, "citation_correctness"),
        completeness=_dimension_aggregate(reviews, "completeness"),
    )


def _dimension_aggregate(reviews: Sequence[Any], field: str) -> DimensionAggregate:
    values = [getattr(item, field).score for item in reviews]
    count = len(values)
    total = sum(values)
    return DimensionAggregate(
        score_sum=total,
        maximum_score=2 * count,
        case_count=count,
        mean_score=total / count if count else 0.0,
    )


def _latency_and_cost(cases: Sequence[GeneratedAnswerCase]) -> Day16LatencyAndCost:
    latencies = [item.telemetry.elapsed_seconds for item in cases]
    usages = [item.telemetry.token_usage for item in cases]
    tokens_known = all(item.total_tokens is not None for item in usages)
    return Day16LatencyAndCost(
        request_count=len(cases),
        total_seconds=sum(latencies),
        median_seconds=statistics.median(latencies),
        p95_seconds=_p95(latencies),
        maximum_seconds=max(latencies),
        input_tokens=(sum(item.input_tokens or 0 for item in usages) if tokens_known else None),
        output_tokens=(sum(item.output_tokens or 0 for item in usages) if tokens_known else None),
        total_tokens=(sum(item.total_tokens or 0 for item in usages) if tokens_known else None),
        question_characters=sum(item.telemetry.question_characters for item in cases),
        evidence_characters=sum(item.telemetry.evidence_characters for item in cases),
        provider_billing_unit="input_and_output_tokens",
        confirmed_monetary_cost_usd=None,
        monetary_cost_status="unknown_provider_did_not_return_currency_cost",
    )


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    index = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return ordered[index]


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _model_configuration(response_model_name: str | None) -> AnswerModelConfiguration:
    schema_json = json.dumps(
        ModelGroundedAnswer.model_json_schema(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return AnswerModelConfiguration(
        provider="OpenAI-compatible DeepSeek service",
        model_name="deepseek-v4-pro",
        model_revision=None,
        response_model_name=response_model_name,
        response_format="json_object",
        output_schema="answer,citations.evidence_id,has_sufficient_evidence",
        output_schema_sha256=hashlib.sha256(schema_json.encode("utf-8")).hexdigest(),
        temperature=0.2,
        thinking="disabled",
        max_tokens=1200,
        timeout_seconds=30.0,
        maximum_retries=0,
        evidence_depth=5,
        batching="one_question_per_request",
        concurrency="sequential",
    )


def _single_response_model(values: Sequence[str]) -> str | None:
    unique = tuple(dict.fromkeys(values))
    if len(unique) > 1:
        raise AnswerQualityExperimentError("供应商响应报告了多个回答模型名称")
    return unique[0] if unique else None


def _validate_answer_settings(settings: Settings) -> None:
    if settings.model_strong != "deepseek-v4-pro":
        raise AnswerQualityExperimentError("Day 16 正式回答模型必须固定为 deepseek-v4-pro")
    if settings.timeout_seconds != 30.0:
        raise AnswerQualityExperimentError("Day 16 正式回答 timeout 必须固定为 30 秒")


def _day14_vector_method(day14: RetrievalExperimentResult) -> RetrievalMethodResult:
    matches = [item for item in day14.methods if item.retriever_name == "vector"]
    if len(matches) != 1:
        raise AnswerQualityExperimentError("Day 14 必须且只能有一份 Vector 排名")
    return matches[0]


def _input_hashes(paths: dict[str, Path], input_chunk_sha256: str) -> Day16InputArtifactHashes:
    return Day16InputArtifactHashes(
        dataset_file_sha256=file_sha256(paths["dataset"]),
        transcript_file_sha256=file_sha256(paths["transcript"]),
        chunk_catalog_file_sha256=file_sha256(paths["chunk_catalog"]),
        input_chunk_sha256=input_chunk_sha256,
        manifest_file_sha256=file_sha256(paths["manifest"]),
        faiss_file_sha256=file_sha256(paths["faiss"]),
        day12_baseline_sha256=file_sha256(paths["day12"]),
        day13_experiment_sha256=file_sha256(paths["day13"]),
        day14_experiment_sha256=file_sha256(paths["day14"]),
        day15_experiment_sha256=file_sha256(paths["day15"]),
        grounded_prompt_file_sha256=file_sha256(DEFAULT_GROUNDED_PROMPT_FILE),
        grounded_system_prompt_sha256=hashlib.sha256(
            GROUNDED_ANSWER_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
    )


def _resolve_paths(**values: str | Path) -> dict[str, Path]:
    index_directory = Path(values.pop("index_directory"))
    paths = {key.removesuffix("_path"): Path(value) for key, value in values.items()}
    paths["day12"] = paths.pop("day12_baseline")
    paths["day13"] = paths.pop("day13_experiment")
    paths["day14"] = paths.pop("day14_experiment")
    paths["day15"] = paths.pop("day15_experiment")
    paths["index_directory"] = index_directory
    paths["manifest"] = index_directory / "manifest.json"
    paths["faiss"] = index_directory / "index.faiss"
    return paths


def _preflight_output(
    paths: dict[str, Path],
    *,
    overwrite: bool,
    label: str,
) -> None:
    protected = {
        path.resolve()
        for key, path in paths.items()
        if key not in {"output", "index_directory"}
    }
    if paths["output"].resolve() in protected:
        raise AnswerQualityPersistenceError(f"{label}不能覆盖任何输入 artifact")
    if paths["output"].exists() and not overwrite:
        raise AnswerQualityPersistenceError(
            f"目标{label}已存在：{paths['output']}；不会静默覆盖"
        )


def _load_model(path: Path, model: type[BaseModel], label: str) -> Any:
    try:
        raw = path.read_text(encoding="utf-8")
        return model.model_validate_json(raw)
    except (OSError, UnicodeDecodeError, ValidationError) as error:
        raise AnswerQualityExperimentError(
            f"{label} 无法严格加载：{safe_error_message(error)}"
        ) from error


def _git_state(project_root: Path) -> Day16GitState:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        raise AnswerQualityExperimentError("无法记录 Day 16 Git 身份") from error
    protected = tuple(
        item
        for item in ("AGENTS.md", "docs/day12-retrieval-boundary-draft.md")
        if any(line[3:] == item for line in status if len(line) >= 4)
    )
    return Day16GitState(
        head_commit=head,
        branch=branch,
        working_tree_dirty=bool(status),
        protected_untracked_files=protected,
    )


def _assert_safe_payload(value: object, *, path: str = "$") -> None:
    unsafe_path = find_unsafe_payload_path(
        value,
        allowed_keys=DAY16_DISCLOSED_CONTENT_KEY_EXCEPTIONS,
        path=path,
    )
    if unsafe_path is not None:
        raise AnswerQualityExperimentError(
            f"Day 16 持久化 payload 包含禁存字段或凭据：{unsafe_path}"
        )
