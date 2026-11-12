"""External composition root for the explicitly enabled Day 27 Ask service."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from app.api.runtime import (
    RUNTIME_CONFIGURATION_ERROR_MESSAGE,
    AgentAskRuntimeConfig,
    EmbeddingRuntimeConfig,
    LanguageModelRuntimeConfig,
)
from app.api.services import AskService, RetrieveService
from app.core.config import EmbeddingSettings, Settings


DEFAULT_AGENT_MAX_STEPS = 4
PRODUCTION_CHUNK_SIZE = 1_000
PRODUCTION_CHUNK_OVERLAP = 200


class AskServiceBootstrapError(RuntimeError):
    """A fixed startup failure that never reflects config or artifact details."""

    def __init__(self) -> None:
        super().__init__(RUNTIME_CONFIGURATION_ERROR_MESSAGE)


class ProviderEventSink(Protocol):
    def __call__(self, event: object) -> None:
        ...


@dataclass(frozen=True)
class AgentBootstrapFactories:
    """Injectable construction surface used by offline assembly tests."""

    transcript_loader: Callable[..., object]
    chunker: Callable[..., object]
    index_loader: Callable[..., object]
    embedding_client_factory: Callable[..., object]
    embedder_factory: Callable[..., object]
    vector_retriever_factory: Callable[..., object]
    tool_dependencies_factory: Callable[..., object]
    tool_handlers_factory: Callable[..., object]
    dispatcher_factory: Callable[..., object]
    provider_client_factory: Callable[..., object]
    provider_model_factory: Callable[..., object]
    workflow_factory: Callable[..., object]
    ask_service_factory: Callable[..., AskService]


@dataclass(frozen=True)
class AgentApiServices:
    """The Ask and retrieval services sharing one validated runtime."""

    ask_service: AskService
    retrieve_service: RetrieveService


@dataclass(frozen=True)
class _AgentAssembly:
    ask_service: AskService
    vector_index: object
    vector_retriever: object


def build_agent_ask_service(
    config: AgentAskRuntimeConfig,
    *,
    factories: AgentBootstrapFactories | None = None,
    provider_event_sink: ProviderEventSink | None = None,
) -> AskService:
    """Construct one fully validated Agent Ask service without making a request."""
    return _build_agent_assembly(
        config,
        factories=factories,
        provider_event_sink=provider_event_sink,
    ).ask_service


def build_agent_api_services(
    config: AgentAskRuntimeConfig,
    *,
    factories: AgentBootstrapFactories | None = None,
    provider_event_sink: ProviderEventSink | None = None,
) -> AgentApiServices:
    """Construct Ask and retrieval services over one loaded index/retriever."""
    assembly = _build_agent_assembly(
        config,
        factories=factories,
        provider_event_sink=provider_event_sink,
    )
    manifest = getattr(assembly.vector_index, "manifest", None)
    chunks = getattr(manifest, "chunks", None)
    input_chunk_sha256 = getattr(manifest, "input_chunk_sha256", None)
    _require_artifact(isinstance(chunks, (list, tuple)) and bool(chunks))
    _require_artifact(isinstance(input_chunk_sha256, str))

    from app.api.adapters import EvaluationRetrieveService

    try:
        retrieve_service = EvaluationRetrieveService(
            retriever=assembly.vector_retriever,  # type: ignore[arg-type]
            chunks=tuple(chunks),
            episode_id=config.episode_id,
            input_chunk_sha256=input_chunk_sha256,
        )
    except ValueError:
        raise AskServiceBootstrapError() from None
    return AgentApiServices(
        ask_service=assembly.ask_service,
        retrieve_service=retrieve_service,
    )


def _build_agent_assembly(
    config: AgentAskRuntimeConfig,
    *,
    factories: AgentBootstrapFactories | None = None,
    provider_event_sink: ProviderEventSink | None = None,
) -> _AgentAssembly:
    """Build and retain the shared validated components for API services.

    All local artifacts are loaded and cross-checked before either provider
    client is constructed. OpenAI client construction is local-only; the first
    network operation remains a later ``ask`` call.
    """
    resolved = factories if factories is not None else _default_factories()
    try:
        transcript = resolved.transcript_loader(config.transcript_path)
    except Exception as error:
        if not _is_expected_transcript_error(error):
            raise
        raise AskServiceBootstrapError() from None
    _require_artifact(getattr(transcript, "episode_id", None) == config.episode_id)
    transcript_source = getattr(transcript, "source_file", None)
    _require_artifact(
        isinstance(transcript_source, str) and bool(transcript_source)
    )

    try:
        chunked = resolved.chunker(
            transcript,
            chunk_size=PRODUCTION_CHUNK_SIZE,
            overlap=PRODUCTION_CHUNK_OVERLAP,
        )
    except Exception as error:
        if not _is_expected_chunk_error(error):
            raise
        raise AskServiceBootstrapError() from None
    _require_artifact(getattr(chunked, "episode_id", None) == config.episode_id)
    _require_artifact(getattr(chunked, "source_file", None) == transcript_source)
    _require_artifact(
        getattr(chunked, "chunk_size", None) == PRODUCTION_CHUNK_SIZE
    )
    _require_artifact(getattr(chunked, "overlap", None) == PRODUCTION_CHUNK_OVERLAP)

    embedding = config.embedding
    try:
        vector_index = resolved.index_loader(
            config.index_directory,
            expected_chunks=chunked,
            expected_embedding_model=embedding.model,
            expected_dimension=embedding.expected_dimension,
            expected_document_task=embedding.document_task,
            expected_query_task=embedding.query_task,
            expected_embedding_type=embedding.embedding_type,
            expected_provider_normalized=embedding.normalized,
            expected_provider_truncate=embedding.truncate,
        )
    except Exception as error:
        if not _is_expected_index_error(error):
            raise
        raise AskServiceBootstrapError() from None
    manifest = getattr(vector_index, "manifest", None)
    _require_artifact(getattr(manifest, "episode_id", None) == config.episode_id)
    _require_artifact(getattr(manifest, "source_file", None) == transcript_source)
    _require_artifact(
        getattr(manifest, "chunk_size", None) == PRODUCTION_CHUNK_SIZE
    )
    _require_artifact(getattr(manifest, "overlap", None) == PRODUCTION_CHUNK_OVERLAP)

    embedding_settings = _embedding_settings(embedding)
    embedding_client = resolved.embedding_client_factory(embedding_settings)
    embedder = resolved.embedder_factory(
        embedding_client,
        config=embedding,
    )
    vector_retriever = resolved.vector_retriever_factory(
        vector_index,
        embedder,
    )

    tool_dependencies = resolved.tool_dependencies_factory(
        transcripts={config.episode_id: transcript},
        retrievers={config.episode_id: vector_retriever},
    )
    handlers = resolved.tool_handlers_factory(tool_dependencies)
    dispatcher = resolved.dispatcher_factory(handlers)

    language_model_settings = _language_model_settings(config.language_model)
    provider_client = resolved.provider_client_factory(language_model_settings)
    provider_model = resolved.provider_model_factory(
        provider_client,
        config=config.language_model,
        event_sink=provider_event_sink,
    )
    workflow = resolved.workflow_factory(
        model=provider_model,
        dispatcher=dispatcher,
        max_steps=DEFAULT_AGENT_MAX_STEPS,
        require_tool_grounding=True,
    )
    ask_service = resolved.ask_service_factory(
        workflow=workflow,
        episode_id=config.episode_id,
    )
    return _AgentAssembly(
        ask_service=ask_service,
        vector_index=vector_index,
        vector_retriever=vector_retriever,
    )


def _require_artifact(condition: bool) -> None:
    if not condition:
        raise AskServiceBootstrapError()


def _is_expected_transcript_error(error: Exception) -> bool:
    from app.chunking.core import ChunkingError

    return isinstance(error, (ChunkingError, OSError))


def _is_expected_chunk_error(error: Exception) -> bool:
    from app.chunking.core import ChunkingError

    return isinstance(error, ChunkingError)


def _is_expected_index_error(error: Exception) -> bool:
    from app.indexing.index import IndexPersistenceError, VectorIndexError

    return isinstance(error, (IndexPersistenceError, VectorIndexError, OSError))


def _language_model_settings(config: LanguageModelRuntimeConfig) -> Settings:
    return Settings(
        api_key=config.api_key,
        base_url=config.base_url,
        model_fast=config.model_fast,
        model_strong=config.model_strong,
        timeout_seconds=config.timeout_seconds,
    )


def _embedding_settings(config: EmbeddingRuntimeConfig) -> EmbeddingSettings:
    return EmbeddingSettings(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        expected_dimension=config.expected_dimension,
        batch_size=config.batch_size,
        timeout_seconds=config.timeout_seconds,
        document_task=config.document_task,
        query_task=config.query_task,
        embedding_type=config.embedding_type,
        normalized=config.normalized,
        truncate=config.truncate,
    )


def _default_factories() -> AgentBootstrapFactories:
    """Import infrastructure only after ``agent`` mode was explicitly selected."""
    from app.agent.openai_compatible import create_openai_compatible_client
    from app.api.adapters import AgentWorkflowAskService
    from app.chunking.core import chunk_transcript
    from app.chunking.pipeline import load_processed_transcript
    from app.indexing.embedding import create_embedding_client
    from app.indexing.index import load_vector_index
    from app.tooling.handlers import ToolDependencies, ToolHandlers
    from app.tooling.dispatcher import ToolDispatcher

    return AgentBootstrapFactories(
        transcript_loader=load_processed_transcript,
        chunker=chunk_transcript,
        index_loader=load_vector_index,
        embedding_client_factory=create_embedding_client,
        embedder_factory=_create_embedder,
        vector_retriever_factory=_create_vector_retriever,
        tool_dependencies_factory=ToolDependencies,
        tool_handlers_factory=ToolHandlers,
        dispatcher_factory=ToolDispatcher,
        provider_client_factory=create_openai_compatible_client,
        provider_model_factory=_create_provider_model,
        workflow_factory=_create_workflow,
        ask_service_factory=AgentWorkflowAskService,
    )


def _create_embedder(
    client: object,
    *,
    config: EmbeddingRuntimeConfig,
) -> object:
    from app.indexing.embedding import OpenAICompatibleEmbedder

    return OpenAICompatibleEmbedder(
        client,  # type: ignore[arg-type]
        model_name=config.model,
        batch_size=config.batch_size,
        expected_dimension=config.expected_dimension,
        document_task=config.document_task,
        query_task=config.query_task,
        embedding_type=config.embedding_type,
        normalized=config.normalized,
        truncate=config.truncate,
    )


def _create_vector_retriever(vector_index: object, embedder: object) -> object:
    from app.retrieval.core import Retriever
    from app.retrieval.vector import VectorRetrieverAdapter

    return VectorRetrieverAdapter(
        Retriever(
            vector_index=vector_index,  # type: ignore[arg-type]
            embedder=embedder,  # type: ignore[arg-type]
        )
    )


def _create_provider_model(
    client: object,
    *,
    config: LanguageModelRuntimeConfig,
    event_sink: ProviderEventSink | None,
) -> object:
    from app.agent.openai_compatible import (
        OpenAICompatibleAgentModel,
        OpenAICompatibleAgentModelConfig,
    )

    provider_config = OpenAICompatibleAgentModelConfig(
        model=config.model_strong,
        timeout_seconds=config.timeout_seconds,
    )
    return OpenAICompatibleAgentModel(
        client,  # type: ignore[arg-type]
        config=provider_config,
        event_sink=event_sink,  # type: ignore[arg-type]
    )


def _create_workflow(
    *,
    model: object,
    dispatcher: object,
    max_steps: int,
    require_tool_grounding: bool,
) -> object:
    from app.agent.schemas import AgentConfig
    from app.agent.workflow import AgentWorkflow

    return AgentWorkflow(
        model=model,  # type: ignore[arg-type]
        dispatcher=dispatcher,  # type: ignore[arg-type]
        config=AgentConfig(
            max_steps=max_steps,
            require_tool_grounding=require_tool_grounding,
        ),
    )
