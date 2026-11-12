"""Day 27 explicit Ask mode, fail-closed startup, and composition tests."""

from __future__ import annotations

import io
from collections.abc import Mapping
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from app.api import __main__ as api_main
from app.api.runtime import (
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSION,
    EMBEDDING_DOCUMENT_TASK,
    EMBEDDING_MODEL,
    EMBEDDING_NORMALIZED,
    EMBEDDING_QUERY_TASK,
    EMBEDDING_TIMEOUT_SECONDS,
    EMBEDDING_TRUNCATE,
    EMBEDDING_TYPE,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL_FAST,
    LLM_MODEL_STRONG,
    LLM_TIMEOUT_SECONDS,
    PODCAST_API_BEARER_TOKEN,
    PODCAST_ASK_EPISODE_ID,
    PODCAST_ASK_INDEX_DIRECTORY,
    PODCAST_ASK_MODE,
    PODCAST_ASK_TRANSCRIPT_PATH,
    RUNTIME_CONFIGURATION_ERROR_MESSAGE,
    AgentAskRuntimeConfig,
    RuntimeConfigurationError,
    load_runtime_config,
)
from app.bootstrap import (
    PRODUCTION_CHUNK_OVERLAP,
    PRODUCTION_CHUNK_SIZE,
    AgentBootstrapFactories,
    AskServiceBootstrapError,
    build_agent_ask_service,
)
from app.core.config import EmbeddingSettings, Settings


EPISODE_ID = "ep_1111111111111111"
SYNTHETIC_LLM_KEY = "synthetic-llm-key-never-log"
SYNTHETIC_EMBEDDING_KEY = "synthetic-embedding-key-never-log"
SYNTHETIC_BEARER_TOKEN = "synthetic-bearer-token-never-log-0001"
PRIVATE_TRANSCRIPT = "/private/transcripts/episode.json"
PRIVATE_INDEX = "/private/indexes/episode"


def _agent_environment() -> dict[str, object]:
    return {
        PODCAST_ASK_MODE: "agent",
        PODCAST_API_BEARER_TOKEN: SYNTHETIC_BEARER_TOKEN,
        PODCAST_ASK_EPISODE_ID: EPISODE_ID,
        PODCAST_ASK_TRANSCRIPT_PATH: PRIVATE_TRANSCRIPT,
        PODCAST_ASK_INDEX_DIRECTORY: PRIVATE_INDEX,
        LLM_API_KEY: SYNTHETIC_LLM_KEY,
        LLM_BASE_URL: "https://models.example.test/v1",
        LLM_MODEL_FAST: "model-fast",
        LLM_MODEL_STRONG: "model-strong",
        LLM_TIMEOUT_SECONDS: "30",
        EMBEDDING_API_KEY: SYNTHETIC_EMBEDDING_KEY,
        EMBEDDING_BASE_URL: "https://embeddings.example.test/v1",
        EMBEDDING_MODEL: "embedding-model",
        EMBEDDING_DIMENSION: "1024",
        EMBEDDING_DOCUMENT_TASK: "retrieval.passage",
        EMBEDDING_QUERY_TASK: "retrieval.query",
        EMBEDDING_TYPE: "float",
        EMBEDDING_NORMALIZED: "true",
        EMBEDDING_TRUNCATE: "false",
        EMBEDDING_BATCH_SIZE: "16",
        EMBEDDING_TIMEOUT_SECONDS: "20.5",
    }


class GuardedUnavailableEnvironment(Mapping[str, object]):
    """Fail if the unavailable branch even looks up an Agent-only key."""

    def __init__(self) -> None:
        self.lookups: list[str] = []

    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0

    def get(self, key: str, default: object = None) -> object:
        self.lookups.append(key)
        if key.startswith("LLM_") or key.startswith("EMBEDDING_"):
            raise AssertionError("unavailable mode inspected provider config")
        if key in {
            PODCAST_ASK_EPISODE_ID,
            PODCAST_ASK_TRANSCRIPT_PATH,
            PODCAST_ASK_INDEX_DIRECTORY,
        }:
            raise AssertionError("unavailable mode inspected artifact config")
        return default


def test_unavailable_default_never_reads_agent_only_configuration() -> None:
    environment = GuardedUnavailableEnvironment()

    config = load_runtime_config(environment)

    assert config.ask_mode == "unavailable"
    assert config.agent_ask is None
    assert environment.lookups == [
        "PODCAST_API_HOST",
        "PODCAST_API_PORT",
        "PODCAST_MAX_REQUEST_BODY_BYTES",
        PODCAST_ASK_MODE,
    ]


def test_agent_mode_strictly_parses_complete_injected_mapping_and_hides_keys() -> None:
    config = load_runtime_config(_agent_environment())

    assert config.ask_mode == "agent"
    assert config.agent_ask is not None
    assert config.agent_ask.episode_id == EPISODE_ID
    assert config.agent_ask.transcript_path == PRIVATE_TRANSCRIPT
    assert config.agent_ask.index_directory == PRIVATE_INDEX
    assert config.agent_ask.language_model.model_strong == "model-strong"
    assert config.agent_ask.language_model.timeout_seconds == 30.0
    assert config.agent_ask.embedding.expected_dimension == 1024
    assert config.agent_ask.embedding.normalized is True
    assert config.agent_ask.embedding.truncate is False
    assert config.agent_ask.embedding.timeout_seconds == 20.5
    rendered = repr(config) + repr(config.agent_ask)
    rendered += repr(config.agent_ask.language_model)
    rendered += repr(config.agent_ask.embedding)
    assert SYNTHETIC_LLM_KEY not in rendered
    assert SYNTHETIC_EMBEDDING_KEY not in rendered
    assert SYNTHETIC_BEARER_TOKEN not in rendered
    serialized = repr(config.model_dump())
    assert "agent_ask" not in serialized
    assert SYNTHETIC_LLM_KEY not in serialized
    assert SYNTHETIC_EMBEDDING_KEY not in serialized
    assert SYNTHETIC_BEARER_TOKEN not in serialized


@pytest.mark.parametrize(
    "missing_name",
    sorted(_agent_environment().keys() - {PODCAST_ASK_MODE}),
)
def test_agent_mode_requires_every_provider_and_artifact_value(
    missing_name: str,
) -> None:
    environment = _agent_environment()
    del environment[missing_name]

    with pytest.raises(RuntimeConfigurationError) as caught:
        load_runtime_config(environment)

    assert str(caught.value) == RUNTIME_CONFIGURATION_ERROR_MESSAGE
    for forbidden in (
        SYNTHETIC_LLM_KEY,
        SYNTHETIC_EMBEDDING_KEY,
        PRIVATE_TRANSCRIPT,
        PRIVATE_INDEX,
    ):
        assert forbidden not in str(caught.value)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        (PODCAST_ASK_MODE, "fake"),
        (PODCAST_ASK_MODE, " agent"),
        (PODCAST_API_BEARER_TOKEN, "short"),
        (PODCAST_API_BEARER_TOKEN, "x" * 257),
        (PODCAST_ASK_EPISODE_ID, "episode-invalid"),
        (PODCAST_ASK_TRANSCRIPT_PATH, " "),
        (PODCAST_ASK_TRANSCRIPT_PATH, "relative/transcript.json"),
        (PODCAST_ASK_INDEX_DIRECTORY, "/private/index\x00suffix"),
        (LLM_BASE_URL, "https://user:password@example.test/v1"),
        (LLM_MODEL_STRONG, "model with spaces"),
        (LLM_TIMEOUT_SECONDS, "0"),
        (LLM_TIMEOUT_SECONDS, "nan"),
        (EMBEDDING_DIMENSION, "1024.0"),
        (EMBEDDING_NORMALIZED, "TRUE"),
        (EMBEDDING_TRUNCATE, False),
        (EMBEDDING_TYPE, "base64"),
        (EMBEDDING_BATCH_SIZE, "0"),
        (EMBEDDING_TIMEOUT_SECONDS, "301"),
    ],
)
def test_invalid_agent_values_have_one_fixed_redacted_failure(
    name: str,
    value: object,
) -> None:
    environment = _agent_environment()
    environment[name] = value

    with pytest.raises(RuntimeConfigurationError) as caught:
        load_runtime_config(environment)

    assert str(caught.value) == RUNTIME_CONFIGURATION_ERROR_MESSAGE
    serialized = str(caught.value)
    for forbidden in (
        str(value),
        SYNTHETIC_LLM_KEY,
        SYNTHETIC_EMBEDDING_KEY,
        SYNTHETIC_BEARER_TOKEN,
        PRIVATE_TRANSCRIPT,
        PRIVATE_INDEX,
        "traceback",
    ):
        assert forbidden not in serialized


def test_runtime_mapping_programming_failure_is_not_mislabeled_as_config() -> None:
    class BrokenMapping(dict[str, object]):
        def get(self, key: str, default: object = None) -> object:
            del key, default
            raise RuntimeError("unexpected mapping implementation bug")

    with pytest.raises(RuntimeError, match="unexpected mapping implementation bug"):
        load_runtime_config(BrokenMapping())


def test_main_unavailable_never_resolves_agent_builder_and_preserves_default_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel_app = object()
    app_calls: list[dict[str, object]] = []
    server_calls: list[object] = []

    def fake_create_app(**kwargs: object) -> object:
        app_calls.append(kwargs)
        return sentinel_app

    def forbidden_builder(_config: AgentAskRuntimeConfig) -> object:
        raise AssertionError("unavailable mode must not construct Agent dependencies")

    monkeypatch.setattr(api_main, "create_app", fake_create_app)

    result = api_main.main(
        environ={},
        ask_service_builder=forbidden_builder,  # type: ignore[arg-type]
        run_server=lambda app, **_kwargs: server_calls.append(app),
    )

    assert result == 0
    assert app_calls == [{"max_request_body_bytes": 16 * 1024}]
    assert server_calls == [sentinel_app]


def test_main_agent_builds_service_before_app_and_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel_service = object()
    sentinel_app = object()
    order: list[str] = []
    app_kwargs: list[dict[str, object]] = []

    def fake_builder(config: AgentAskRuntimeConfig) -> object:
        order.append("service")
        assert config.episode_id == EPISODE_ID
        return sentinel_service

    def fake_create_app(**kwargs: object) -> object:
        order.append("app")
        app_kwargs.append(kwargs)
        return sentinel_app

    def fake_runner(app: object, **_kwargs: object) -> None:
        order.append("server")
        assert app is sentinel_app

    monkeypatch.setattr(api_main, "create_app", fake_create_app)

    result = api_main.main(
        environ=_agent_environment(),
        ask_service_builder=fake_builder,  # type: ignore[arg-type]
        run_server=fake_runner,
    )

    assert result == 0
    assert order == ["service", "app", "server"]
    assert app_kwargs == [
        {
            "ask_service": sentinel_service,
            "api_bearer_token": SYNTHETIC_BEARER_TOKEN,
            "max_request_body_bytes": 16 * 1024,
        }
    ]


@pytest.mark.parametrize("failure_stage", ["config", "builder"])
def test_agent_startup_failure_is_fixed_and_happens_before_server(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    stream = io.StringIO()
    server_calls: list[object] = []
    environment = _agent_environment()
    if failure_stage == "config":
        environment[LLM_API_KEY] = ""

    def fake_builder(_config: AgentAskRuntimeConfig) -> object:
        if failure_stage == "builder":
            raise AskServiceBootstrapError()
        return object()

    def fake_create_app(**_kwargs: object) -> object:
        return object()

    monkeypatch.setattr(api_main, "create_app", fake_create_app)
    result = api_main.main(
        environ=environment,
        ask_service_builder=fake_builder,  # type: ignore[arg-type]
        run_server=lambda app, **_kwargs: server_calls.append(app),
        stderr=stream,
    )

    assert result == 2
    assert server_calls == []
    assert stream.getvalue() == RUNTIME_CONFIGURATION_ERROR_MESSAGE + "\n"
    for forbidden in (
        SYNTHETIC_LLM_KEY,
        SYNTHETIC_EMBEDDING_KEY,
        SYNTHETIC_BEARER_TOKEN,
        PRIVATE_TRANSCRIPT,
        PRIVATE_INDEX,
        "traceback",
        "RuntimeError",
    ):
        assert forbidden not in stream.getvalue()


def test_unexpected_app_bug_is_not_mislabeled_as_configuration_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = io.StringIO()
    sentinel = "unexpected application construction bug"

    def broken_create_app(**_kwargs: object) -> object:
        raise RuntimeError(sentinel)

    monkeypatch.setattr(api_main, "create_app", broken_create_app)

    with pytest.raises(RuntimeError, match=sentinel):
        api_main.main(
            environ=_agent_environment(),
            ask_service_builder=lambda _config: object(),  # type: ignore[arg-type]
            run_server=lambda *_args, **_kwargs: None,
            stderr=stream,
        )

    assert stream.getvalue() == ""


def test_unexpected_bootstrap_factory_bug_is_not_mislabeled_as_configuration() -> None:
    config = load_runtime_config(_agent_environment()).agent_ask
    assert config is not None
    unused = lambda *_args, **_kwargs: object()

    def broken_transcript_loader(_path: str) -> object:
        raise RuntimeError("unexpected transcript factory bug")

    factories = AgentBootstrapFactories(
        transcript_loader=broken_transcript_loader,
        chunker=unused,
        index_loader=unused,
        embedding_client_factory=unused,
        embedder_factory=unused,
        vector_retriever_factory=unused,
        tool_dependencies_factory=unused,
        tool_handlers_factory=unused,
        dispatcher_factory=unused,
        provider_client_factory=unused,
        provider_model_factory=unused,
        workflow_factory=unused,
        ask_service_factory=unused,  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="unexpected transcript factory bug"):
        build_agent_ask_service(config, factories=factories)


def test_bootstrap_assembles_existing_boundaries_in_order_without_network() -> None:
    config = load_runtime_config(_agent_environment()).agent_ask
    assert config is not None
    order: list[str] = []
    transcript = SimpleNamespace(
        episode_id=EPISODE_ID,
        source_file="episode.json",
    )
    chunked = SimpleNamespace(
        episode_id=EPISODE_ID,
        source_file="episode.json",
        chunk_size=PRODUCTION_CHUNK_SIZE,
        overlap=PRODUCTION_CHUNK_OVERLAP,
    )
    vector_index = SimpleNamespace(
        manifest=SimpleNamespace(
            episode_id=EPISODE_ID,
            source_file="episode.json",
            chunk_size=PRODUCTION_CHUNK_SIZE,
            overlap=PRODUCTION_CHUNK_OVERLAP,
        )
    )
    sentinels = {
        name: object()
        for name in (
            "embedding_client",
            "embedder",
            "retriever",
            "dependencies",
            "handlers",
            "dispatcher",
            "provider_client",
            "provider_model",
            "workflow",
            "service",
        )
    }
    provider_events: list[object] = []

    def transcript_loader(path: str) -> object:
        order.append("transcript")
        assert path == PRIVATE_TRANSCRIPT
        return transcript

    def chunker(value: object, **kwargs: object) -> object:
        order.append("chunks")
        assert value is transcript
        assert kwargs == {
            "chunk_size": PRODUCTION_CHUNK_SIZE,
            "overlap": PRODUCTION_CHUNK_OVERLAP,
        }
        return chunked

    def index_loader(path: str, **kwargs: object) -> object:
        order.append("index")
        assert path == PRIVATE_INDEX
        assert kwargs == {
            "expected_chunks": chunked,
            "expected_embedding_model": "embedding-model",
            "expected_dimension": 1024,
            "expected_document_task": "retrieval.passage",
            "expected_query_task": "retrieval.query",
            "expected_embedding_type": "float",
            "expected_provider_normalized": True,
            "expected_provider_truncate": False,
        }
        return vector_index

    def embedding_client_factory(settings: EmbeddingSettings) -> object:
        order.append("embedding_client")
        assert settings.api_key == SYNTHETIC_EMBEDDING_KEY
        assert settings.expected_dimension == 1024
        return sentinels["embedding_client"]

    def embedder_factory(client: object, **kwargs: object) -> object:
        order.append("embedder")
        assert client is sentinels["embedding_client"]
        assert kwargs == {"config": config.embedding}
        return sentinels["embedder"]

    def vector_retriever_factory(index: object, embedder: object) -> object:
        order.append("retriever")
        assert index is vector_index
        assert embedder is sentinels["embedder"]
        return sentinels["retriever"]

    def tool_dependencies_factory(**kwargs: object) -> object:
        order.append("dependencies")
        assert kwargs == {
            "transcripts": {EPISODE_ID: transcript},
            "retrievers": {EPISODE_ID: sentinels["retriever"]},
        }
        return sentinels["dependencies"]

    def tool_handlers_factory(dependencies: object) -> object:
        order.append("handlers")
        assert dependencies is sentinels["dependencies"]
        return sentinels["handlers"]

    def dispatcher_factory(handlers: object) -> object:
        order.append("dispatcher")
        assert handlers is sentinels["handlers"]
        return sentinels["dispatcher"]

    def provider_client_factory(settings: Settings) -> object:
        order.append("provider_client")
        assert settings.api_key == SYNTHETIC_LLM_KEY
        assert settings.model_strong == "model-strong"
        return sentinels["provider_client"]

    def provider_model_factory(client: object, **kwargs: object) -> object:
        order.append("provider_model")
        assert client is sentinels["provider_client"]
        assert kwargs == {
            "config": config.language_model,
            "event_sink": provider_events.append,
        }
        return sentinels["provider_model"]

    def workflow_factory(**kwargs: object) -> object:
        order.append("workflow")
        assert kwargs == {
            "model": sentinels["provider_model"],
            "dispatcher": sentinels["dispatcher"],
            "max_steps": 4,
            "require_tool_grounding": True,
        }
        return sentinels["workflow"]

    def ask_service_factory(**kwargs: object) -> object:
        order.append("service")
        assert kwargs == {
            "workflow": sentinels["workflow"],
            "episode_id": EPISODE_ID,
        }
        return sentinels["service"]

    factories = AgentBootstrapFactories(
        transcript_loader=transcript_loader,
        chunker=chunker,
        index_loader=index_loader,
        embedding_client_factory=embedding_client_factory,
        embedder_factory=embedder_factory,
        vector_retriever_factory=vector_retriever_factory,
        tool_dependencies_factory=tool_dependencies_factory,
        tool_handlers_factory=tool_handlers_factory,
        dispatcher_factory=dispatcher_factory,
        provider_client_factory=provider_client_factory,
        provider_model_factory=provider_model_factory,
        workflow_factory=workflow_factory,
        ask_service_factory=ask_service_factory,  # type: ignore[arg-type]
    )

    service = build_agent_ask_service(
        config,
        factories=factories,
        provider_event_sink=provider_events.append,
    )

    assert service is sentinels["service"]
    assert order == [
        "transcript",
        "chunks",
        "index",
        "embedding_client",
        "embedder",
        "retriever",
        "dependencies",
        "handlers",
        "dispatcher",
        "provider_client",
        "provider_model",
        "workflow",
        "service",
    ]
    assert provider_events == []


@pytest.mark.parametrize(
    "mismatch",
    [
        "transcript_episode",
        "chunk_episode",
        "chunk_source",
        "chunk_size",
        "chunk_overlap",
        "index_episode",
        "index_source",
        "index_size",
        "index_overlap",
    ],
)
def test_bootstrap_artifact_mismatch_is_fixed_before_any_client(
    mismatch: str,
) -> None:
    config = load_runtime_config(_agent_environment()).agent_ask
    assert config is not None
    client_calls: list[str] = []
    wrong_episode = "ep_2222222222222222"
    transcript = SimpleNamespace(
        episode_id=(
            wrong_episode if mismatch == "transcript_episode" else EPISODE_ID
        ),
        source_file="episode.json",
    )
    chunked = SimpleNamespace(
        episode_id=wrong_episode if mismatch == "chunk_episode" else EPISODE_ID,
        source_file=(
            "other.json" if mismatch == "chunk_source" else "episode.json"
        ),
        chunk_size=(
            600 if mismatch == "chunk_size" else PRODUCTION_CHUNK_SIZE
        ),
        overlap=(
            120 if mismatch == "chunk_overlap" else PRODUCTION_CHUNK_OVERLAP
        ),
    )
    vector_index = SimpleNamespace(
        manifest=SimpleNamespace(
            episode_id=(
                wrong_episode if mismatch == "index_episode" else EPISODE_ID
            ),
            source_file=(
                "other.json" if mismatch == "index_source" else "episode.json"
            ),
            chunk_size=(
                600 if mismatch == "index_size" else PRODUCTION_CHUNK_SIZE
            ),
            overlap=(
                120
                if mismatch == "index_overlap"
                else PRODUCTION_CHUNK_OVERLAP
            ),
        )
    )
    forbidden = lambda *_args, **_kwargs: client_calls.append("called")
    factories = AgentBootstrapFactories(
        transcript_loader=lambda _path: transcript,
        chunker=lambda *_args, **_kwargs: chunked,
        index_loader=lambda *_args, **_kwargs: vector_index,
        embedding_client_factory=forbidden,
        embedder_factory=forbidden,
        vector_retriever_factory=forbidden,
        tool_dependencies_factory=forbidden,
        tool_handlers_factory=forbidden,
        dispatcher_factory=forbidden,
        provider_client_factory=forbidden,
        provider_model_factory=forbidden,
        workflow_factory=forbidden,
        ask_service_factory=forbidden,  # type: ignore[arg-type]
    )

    with pytest.raises(AskServiceBootstrapError) as caught:
        build_agent_ask_service(config, factories=factories)

    assert str(caught.value) == RUNTIME_CONFIGURATION_ERROR_MESSAGE
    assert client_calls == []
    for forbidden_value in (
        SYNTHETIC_LLM_KEY,
        SYNTHETIC_EMBEDDING_KEY,
        SYNTHETIC_BEARER_TOKEN,
        PRIVATE_TRANSCRIPT,
        PRIVATE_INDEX,
    ):
        assert forbidden_value not in str(caught.value)
