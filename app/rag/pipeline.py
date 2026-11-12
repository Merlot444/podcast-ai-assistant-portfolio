"""Day 10 orchestration over the existing ingestion-to-answer boundaries."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, TypeVar

from openai import OpenAI

from app.chunking.core import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    chunk_transcript,
)
from app.chunking.pipeline import load_processed_transcript
from app.indexing.embedding import Embedder
from app.indexing.index import (
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    IndexPersistenceError,
    PersistentVectorIndex,
    build_vector_index,
    load_vector_index,
    save_vector_index,
)
from app.ingestion.transcripts import load_transcript
from app.rag.core import generate_grounded_answer
from app.rag.events import NullEventSink, RagEventSink
from app.retrieval.core import Retriever
from app.schemas.chunk import ChunkedTranscript
from app.schemas.rag import GroundedAnswer
from app.schemas.rag_pipeline import (
    RagRunMetrics,
    RagRunResult,
    RagTokenUsage,
)
from app.schemas.transcript import ProcessedTranscript


InputKind = Literal["raw", "processed"]
IndexMode = Literal["reuse", "build"]
Clock = Callable[[], float]
_ResultT = TypeVar("_ResultT")
MAX_QUERY_CHARACTERS = 2_000
MAX_ANSWER_TOKENS = 8_000


@dataclass(frozen=True)
class RagPipelineRequest:
    """All user-controlled choices for one RAG 1.0 run."""

    input_path: str | Path
    input_kind: InputKind
    query: str
    top_k: int
    index_directory: str | Path
    index_mode: IndexMode
    answer_model: str
    chunk_size: int = DEFAULT_CHUNK_SIZE
    overlap: int = DEFAULT_OVERLAP
    overwrite_index: bool = False
    answer_max_tokens: int = 1_200


def run_rag_pipeline(
    request: RagPipelineRequest,
    *,
    embedder: Embedder,
    answer_client: OpenAI,
    event_sink: RagEventSink | None = None,
    clock: Clock = perf_counter,
) -> RagRunResult:
    """Compose Day 3, 6, 7, 8, and 9 without reimplementing their algorithms."""
    sink = event_sink or NullEventSink()
    counted_embedder = _CountingEmbedder(embedder)
    counted_answer_client = _CountingAnswerClient(answer_client)
    observer = _RunObserver(
        request=request,
        embedder=counted_embedder,
        answer_client=counted_answer_client,
        event_sink=sink,
        clock=clock,
    )
    pipeline_started = clock()
    observer.emit(stage="pipeline", status="started", elapsed_seconds=0.0)

    try:
        _validate_request(request)
        transcript = observer.run_stage(
            "ingestion",
            lambda: _load_input(request),
            on_success=observer.observe_transcript,
        )
        chunked = observer.run_stage(
            "chunking",
            lambda: chunk_transcript(
                transcript,
                chunk_size=request.chunk_size,
                overlap=request.overlap,
            ),
            on_success=observer.observe_chunks,
        )
        vector_index = observer.run_stage(
            "index",
            lambda: _prepare_index(
                request,
                chunked=chunked,
                embedder=counted_embedder,
            ),
            on_success=observer.observe_index,
        )
        retrieval_results = observer.run_stage(
            "retrieval",
            lambda: Retriever(
                vector_index,
                counted_embedder,
            ).retrieve(request.query, request.top_k),
            on_success=observer.observe_retrieval,
        )
        answer = observer.run_stage(
            "answer",
            lambda: generate_grounded_answer(
                counted_answer_client,  # type: ignore[arg-type]
                model=request.answer_model,
                query=request.query,
                evidence=retrieval_results,
                max_tokens=request.answer_max_tokens,
                transcript=transcript,
            ),
            on_success=observer.observe_answer,
        )

        total_elapsed = _elapsed(clock(), pipeline_started)
        result = RagRunResult(
            episode_id=transcript.episode_id,
            source_file=transcript.source_file,
            input_file=str(Path(request.input_path)),
            input_kind=request.input_kind,
            input_segment_count=len(transcript.segments),
            chunk_count=len(chunked.chunks),
            index_directory=str(Path(request.index_directory)),
            index_action=(
                "reused" if request.index_mode == "reuse" else "built"
            ),
            embedding_model=vector_index.manifest.embedding_model,
            answer_model=request.answer_model,
            question=request.query,
            vector_count=vector_index.manifest.vector_count,
            vector_dimension=vector_index.manifest.vector_dimension,
            top_k=request.top_k,
            retrieved_count=len(retrieval_results),
            retrieved_chunk_ids=[
                result.chunk_id for result in retrieval_results
            ],
            answer=answer,
            metrics=RagRunMetrics(
                stage_elapsed_seconds=dict(observer.stage_timings),
                total_elapsed_seconds=total_elapsed,
                query_embedding_calls=counted_embedder.query_call_count,
                document_embedding_calls=counted_embedder.document_call_count,
                document_embedding_batches=(
                    counted_embedder.document_batch_count
                ),
                answer_model_calls=counted_answer_client.call_count,
                answer_token_usage=counted_answer_client.token_usage,
            ),
        )
        observer.emit(
            stage="pipeline",
            status="succeeded",
            elapsed_seconds=total_elapsed,
        )
        return result
    except Exception as error:
        observer.emit(
            stage="pipeline",
            status="failed",
            elapsed_seconds=_elapsed(clock(), pipeline_started),
            error_type=type(error).__name__,
        )
        raise


def _validate_request(request: RagPipelineRequest) -> None:
    if request.input_kind not in {"raw", "processed"}:
        raise ValueError("input_kind 必须是 raw 或 processed")
    if request.index_mode not in {"reuse", "build"}:
        raise ValueError("index_mode 必须是 reuse 或 build")
    if not isinstance(request.query, str) or not request.query.strip():
        raise ValueError("问题不能为空")
    if len(request.query) > MAX_QUERY_CHARACTERS:
        raise ValueError(
            f"问题不能超过 {MAX_QUERY_CHARACTERS} 个字符"
        )
    if (
        isinstance(request.top_k, bool)
        or not isinstance(request.top_k, int)
        or request.top_k <= 0
    ):
        raise ValueError("top_k 必须是大于 0 的整数")
    if not isinstance(request.answer_model, str) or not request.answer_model.strip():
        raise ValueError("回答模型名称不能为空")
    if (
        isinstance(request.answer_max_tokens, bool)
        or not isinstance(request.answer_max_tokens, int)
        or request.answer_max_tokens <= 0
    ):
        raise ValueError("answer_max_tokens 必须是大于 0 的整数")
    if request.answer_max_tokens > MAX_ANSWER_TOKENS:
        raise ValueError(
            f"answer_max_tokens 不能超过 {MAX_ANSWER_TOKENS}"
        )
    if request.overwrite_index and request.index_mode != "build":
        raise ValueError("overwrite_index 只能与 index_mode=build 一起使用")


def _load_input(request: RagPipelineRequest) -> ProcessedTranscript:
    if request.input_kind == "raw":
        return load_transcript(request.input_path)
    return load_processed_transcript(request.input_path)


def _prepare_index(
    request: RagPipelineRequest,
    *,
    chunked: ChunkedTranscript,
    embedder: _CountingEmbedder,
) -> PersistentVectorIndex:
    try:
        return _prepare_index_files(
            request,
            chunked=chunked,
            embedder=embedder,
        )
    except OSError as error:
        raise IndexPersistenceError("索引文件系统操作失败") from error


def _prepare_index_files(
    request: RagPipelineRequest,
    *,
    chunked: ChunkedTranscript,
    embedder: _CountingEmbedder,
) -> PersistentVectorIndex:
    directory = Path(request.index_directory)
    if request.index_mode == "reuse":
        vector_index = _load_matching_index(
            directory,
            chunked=chunked,
            embedder=embedder,
        )
        embedder.set_index_dimension(vector_index.manifest.vector_dimension)
        return vector_index

    _preflight_build_destination(
        directory,
        overwrite=request.overwrite_index,
    )
    source_stem = Path(request.input_path).stem
    vector_index = build_vector_index(
        chunked,
        embedder=embedder,
        input_chunk_file=f"{source_stem}-chunks.json",
    )
    save_vector_index(
        vector_index,
        directory,
        overwrite=request.overwrite_index,
    )
    reloaded = _load_matching_index(
        directory,
        chunked=chunked,
        embedder=embedder,
    )
    embedder.set_index_dimension(reloaded.manifest.vector_dimension)
    return reloaded


def _preflight_build_destination(
    directory: Path,
    *,
    overwrite: bool,
) -> None:
    existing = [
        path
        for path in (
            directory / INDEX_FILENAME,
            directory / MANIFEST_FILENAME,
        )
        if path.exists()
    ]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise IndexPersistenceError(
            f"目标已有索引产物（{names}）；"
            "构建前已停止，不会计算文档 Embedding 或静默覆盖"
        )


def _load_matching_index(
    directory: Path,
    *,
    chunked: ChunkedTranscript,
    embedder: _CountingEmbedder,
) -> PersistentVectorIndex:
    return load_vector_index(
        directory,
        expected_chunks=chunked,
        expected_embedding_model=embedder.model_name,
        expected_dimension=_configured_dimension(embedder.delegate),
        expected_document_task=getattr(
            embedder.delegate,
            "document_task",
            None,
        ),
        expected_query_task=getattr(
            embedder.delegate,
            "query_task",
            None,
        ),
        expected_embedding_type=getattr(
            embedder.delegate,
            "embedding_type",
            None,
        ),
        expected_provider_normalized=getattr(
            embedder.delegate,
            "normalized",
            None,
        ),
        expected_provider_truncate=getattr(
            embedder.delegate,
            "truncate",
            None,
        ),
    )


def _configured_dimension(embedder: Embedder) -> int | None:
    expected = getattr(embedder, "expected_dimension", None)
    if expected is not None:
        return expected
    dimension = getattr(embedder, "dimension", None)
    if isinstance(dimension, int) and not isinstance(dimension, bool):
        return dimension
    return None


class _CountingEmbedder:
    """Count this run's calls while preserving the configured Embedder surface."""

    def __init__(self, delegate: Embedder) -> None:
        self.delegate = delegate
        self.document_call_count = 0
        self.query_call_count = 0
        self._index_dimension: int | None = None
        self._initial_document_batch_count = _optional_nonnegative_counter(
            delegate,
            "document_batch_count",
        )

    @property
    def model_name(self) -> str:
        return self.delegate.model_name

    @property
    def expected_dimension(self) -> int | None:
        configured = getattr(self.delegate, "expected_dimension", None)
        if configured is not None:
            return configured
        return self._index_dimension

    @property
    def document_batch_count(self) -> int | None:
        initial = self._initial_document_batch_count
        current = _optional_nonnegative_counter(
            self.delegate,
            "document_batch_count",
        )
        if initial is None or current is None:
            return None
        return max(0, current - initial)

    def set_index_dimension(self, dimension: int) -> None:
        self._index_dimension = dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_call_count += 1
        return self.delegate.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        self.query_call_count += 1
        return self.delegate.embed_query(text)

    def __getattr__(self, name: str) -> object:
        return getattr(self.delegate, name)


class _CountingAnswerClient:
    """Proxy the one SDK method used by Day 9 and capture real usage counts."""

    def __init__(self, delegate: OpenAI) -> None:
        self.delegate = delegate
        self.call_count = 0
        self.token_usage: RagTokenUsage | None = None
        self.chat = _CountingChat(delegate.chat, self)

    def observe_response(self, response: object) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        prompt_tokens = _usage_integer(usage, "prompt_tokens")
        completion_tokens = _usage_integer(usage, "completion_tokens")
        total_tokens = _usage_integer(usage, "total_tokens")
        if None in (prompt_tokens, completion_tokens, total_tokens):
            return
        current = self.token_usage
        if current is None:
            self.token_usage = RagTokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
            return
        self.token_usage = RagTokenUsage(
            prompt_tokens=current.prompt_tokens + prompt_tokens,
            completion_tokens=current.completion_tokens + completion_tokens,
            total_tokens=current.total_tokens + total_tokens,
        )

    def __getattr__(self, name: str) -> object:
        return getattr(self.delegate, name)


class _CountingChat:
    def __init__(self, delegate: object, owner: _CountingAnswerClient) -> None:
        self._delegate = delegate
        self.completions = _CountingCompletions(
            getattr(delegate, "completions"),
            owner,
        )

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)


class _CountingCompletions:
    def __init__(self, delegate: object, owner: _CountingAnswerClient) -> None:
        self._delegate = delegate
        self._owner = owner

    def create(self, *args: object, **kwargs: object) -> object:
        self._owner.call_count += 1
        response = getattr(self._delegate, "create")(*args, **kwargs)
        self._owner.observe_response(response)
        return response

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)


class _RunObserver:
    """Emit redacted events and retain exact timings for the final result."""

    def __init__(
        self,
        *,
        request: RagPipelineRequest,
        embedder: _CountingEmbedder,
        answer_client: _CountingAnswerClient,
        event_sink: RagEventSink,
        clock: Clock,
    ) -> None:
        self.request = request
        self.embedder = embedder
        self.answer_client = answer_client
        self.event_sink = event_sink
        self.clock = clock
        self.stage_timings: dict[str, float] = {}
        self.episode_id: str | None = None
        self.input_segment_count: int | None = None
        self.chunk_count: int | None = None
        self.index_action: str | None = None
        self.vector_count: int | None = None
        self.vector_dimension: int | None = None
        self.retrieved_count: int | None = None
        self.has_sufficient_evidence: bool | None = None
        self.citation_count: int | None = None

    def run_stage(
        self,
        stage: str,
        action: Callable[[], _ResultT],
        *,
        on_success: Callable[[_ResultT], None],
    ) -> _ResultT:
        started = self.clock()
        self.emit(stage=stage, status="started", elapsed_seconds=0.0)
        try:
            result = action()
        except Exception as error:
            elapsed = _elapsed(self.clock(), started)
            self.stage_timings[stage] = elapsed
            self.emit(
                stage=stage,
                status="failed",
                elapsed_seconds=elapsed,
                error_type=type(error).__name__,
            )
            raise
        on_success(result)
        elapsed = _elapsed(self.clock(), started)
        self.stage_timings[stage] = elapsed
        self.emit(
            stage=stage,
            status="succeeded",
            elapsed_seconds=elapsed,
        )
        return result

    def observe_transcript(self, transcript: ProcessedTranscript) -> None:
        self.episode_id = transcript.episode_id
        self.input_segment_count = len(transcript.segments)

    def observe_chunks(self, chunked: ChunkedTranscript) -> None:
        self.chunk_count = len(chunked.chunks)

    def observe_index(self, vector_index: PersistentVectorIndex) -> None:
        self.index_action = (
            "reused" if self.request.index_mode == "reuse" else "built"
        )
        self.vector_count = vector_index.manifest.vector_count
        self.vector_dimension = vector_index.manifest.vector_dimension

    def observe_retrieval(self, results: Sequence[object]) -> None:
        self.retrieved_count = len(results)

    def observe_answer(self, answer: GroundedAnswer) -> None:
        self.has_sufficient_evidence = answer.has_sufficient_evidence
        self.citation_count = len(answer.citations)

    def emit(
        self,
        *,
        stage: str,
        status: str,
        elapsed_seconds: float,
        error_type: str | None = None,
    ) -> None:
        event: dict[str, object] = {
            "event": "rag_pipeline_stage",
            "stage": stage,
            "status": status,
            "elapsed_seconds": elapsed_seconds,
            "episode_id": self.episode_id,
            "input_file": str(Path(self.request.input_path)),
            "input_kind": self.request.input_kind,
            "input_segment_count": self.input_segment_count,
            "chunk_count": self.chunk_count,
            "index_directory": str(Path(self.request.index_directory)),
            "index_action": self.index_action,
            "top_k": self.request.top_k,
            "retrieved_count": self.retrieved_count,
            "vector_count": self.vector_count,
            "vector_dimension": self.vector_dimension,
            "embedding_model": self.embedder.model_name,
            "answer_model": self.request.answer_model,
            "query_embedding_calls": self.embedder.query_call_count,
            "document_embedding_calls": self.embedder.document_call_count,
            "document_embedding_batches": self.embedder.document_batch_count,
            "answer_model_calls": self.answer_client.call_count,
            "has_sufficient_evidence": self.has_sufficient_evidence,
            "citation_count": self.citation_count,
            "error_type": error_type,
        }
        self.event_sink.emit(event)


def _elapsed(current: float, started: float) -> float:
    return max(0.0, float(current) - float(started))


def _optional_nonnegative_counter(
    value: object,
    name: str,
) -> int | None:
    counter = getattr(value, name, None)
    if isinstance(counter, bool) or not isinstance(counter, int) or counter < 0:
        return None
    return counter


def _usage_integer(usage: object, name: str) -> int | None:
    if isinstance(usage, Mapping):
        value: Any = usage.get(name)
    else:
        value = getattr(usage, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value
