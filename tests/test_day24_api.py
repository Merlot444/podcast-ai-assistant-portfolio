"""Offline Day 24 FastAPI factory, routes, schemas, and safety tests."""

from __future__ import annotations

import ast
import importlib.metadata
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from app.agent import AgentCitation, AgentFinalAnswer
from app.api import create_app
from app.api.adapters import ConversationSessionReader
from app.api.schemas import (
    ApiErrorResponse,
    AskRequest,
    AskResponse,
    HealthResponse,
    SessionMetadata,
    SessionResponse,
)
from app.api.services import AskServiceSuccess, ServiceFailure
from app.storage.schemas import SessionRead


PROJECT_ROOT = Path(__file__).parents[1]
EPISODE_ID = "ep_1111111111111111"
SESSION_ID = "session_day24"
QUESTION = "  What does the synthetic episode explain?  "
QUESTION_CLEAN = QUESTION.strip()
PRIVATE_PATH = "/private/data/day24.sqlite3"
DATABASE_URL = "postgresql://admin:password@localhost/private"
SECRET = "DAY24_SECRET_KEY"
FAILURE_SENTINEL = f"{QUESTION_CLEAN} {PRIVATE_PATH} {DATABASE_URL} {SECRET} traceback"


class FakeAskService:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[AskRequest] = []

    def ask(self, request: AskRequest) -> object:
        self.calls.append(request)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class FakeSessionReader:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[str] = []

    def get_session(self, session_id: str) -> object:
        self.calls.append(session_id)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def _answer_success() -> AskServiceSuccess:
    return AskServiceSuccess(
        answer=AgentFinalAnswer(
            answer="A bounded synthetic answer.",
            citations=[
                AgentCitation(
                    kind="chunk",
                    episode_id=EPISODE_ID,
                    reference_id="chunk_1111111111111111",
                )
            ],
        )
    )


def _session() -> SessionMetadata:
    return SessionMetadata(
        session_id=SESSION_ID,
        title="Synthetic session",
        status="active",
        created_at=datetime(2026, 8, 20, 8, 0, tzinfo=UTC),
        updated_at=datetime(2026, 8, 20, 8, 1, tzinfo=UTC),
    )


def _assert_error(response: object, *, status: int, code: str) -> ApiErrorResponse:
    assert response.status_code == status
    parsed = ApiErrorResponse.model_validate_json(response.text)
    assert parsed.ok is False
    assert parsed.error.code == code
    return parsed


def test_application_factory_starts_with_no_real_services() -> None:
    app = create_app()

    with TestClient(app) as client:
        response = client.get("/health")

    assert HealthResponse.model_validate_json(response.text) == HealthResponse()


def test_health_returns_the_fixed_minimal_response() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True, "status": "ok"}
    serialized = response.text
    for forbidden in ("DATABASE_URL", DATABASE_URL, PRIVATE_PATH, SECRET, "provider"):
        assert forbidden not in serialized


def test_ask_valid_input_calls_fake_once_and_returns_structured_answer() -> None:
    service = FakeAskService(_answer_success())
    app = create_app(ask_service=service)

    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={
                "question": QUESTION,
                "episode_id": EPISODE_ID,
                "top_k": 3,
                "session_id": SESSION_ID,
            },
        )

    parsed = AskResponse.model_validate_json(response.text)
    assert response.status_code == 200
    assert parsed.ok is True
    assert parsed.answer == "A bounded synthetic answer."
    assert parsed.citations[0].reference_id == "chunk_1111111111111111"
    assert len(service.calls) == 1
    assert service.calls[0].question == QUESTION_CLEAN
    assert service.calls[0].episode_id == EPISODE_ID
    assert service.calls[0].top_k == 3
    assert service.calls[0].session_id == SESSION_ID


def test_falsey_injected_services_are_not_replaced_by_defaults() -> None:
    class FalseyAskService(FakeAskService):
        def __bool__(self) -> bool:
            return False

    class FalseySessionReader(FakeSessionReader):
        def __bool__(self) -> bool:
            return False

    ask = FalseyAskService(_answer_success())
    session = FalseySessionReader(_session())

    with TestClient(create_app(ask_service=ask, session_reader=session)) as client:
        ask_response = client.post(
            "/ask",
            json={"question": "valid", "episode_id": EPISODE_ID, "top_k": 1},
        )
        session_response = client.get(f"/sessions/{SESSION_ID}")

    assert AskResponse.model_validate_json(ask_response.text).ok is True
    assert SessionResponse.model_validate_json(session_response.text).ok is True
    assert len(ask.calls) == 1
    assert session.calls == [SESSION_ID]


@pytest.mark.parametrize(
    "payload",
    [
        {"question": " ", "episode_id": EPISODE_ID, "top_k": 3},
        {"question": "x" * 2_001, "episode_id": EPISODE_ID, "top_k": 3},
        {"question": "valid", "episode_id": "episode-invalid", "top_k": 3},
        {"question": "valid", "episode_id": EPISODE_ID, "top_k": 0},
        {"question": "valid", "episode_id": EPISODE_ID, "top_k": 9},
        {"question": "valid", "episode_id": EPISODE_ID, "top_k": True},
        {"question": "valid", "episode_id": EPISODE_ID, "top_k": "3"},
        {
            "question": "valid",
            "episode_id": EPISODE_ID,
            "top_k": 3,
            "session_id": "../../private",
        },
        {
            "question": "valid",
            "episode_id": EPISODE_ID,
            "top_k": 3,
            "path": PRIVATE_PATH,
        },
    ],
)
def test_invalid_ask_input_is_redacted_and_never_calls_service(
    payload: dict[str, object],
) -> None:
    service = FakeAskService(_answer_success())
    app = create_app(ask_service=service)

    with TestClient(app) as client:
        response = client.post("/ask", json=payload)

    _assert_error(response, status=422, code="invalid_request")
    assert service.calls == []
    serialized = response.text
    for forbidden in (
        QUESTION_CLEAN,
        "x" * 100,
        PRIVATE_PATH,
        "../../private",
        "episode-invalid",
    ):
        assert forbidden not in serialized


def test_malformed_non_object_and_missing_ask_bodies_are_safely_rejected() -> None:
    service = FakeAskService(_answer_success())

    with TestClient(create_app(ask_service=service)) as client:
        responses = [
            client.post("/ask"),
            client.post("/ask", json=[QUESTION_CLEAN, PRIVATE_PATH]),
            client.post(
                "/ask",
                content=f'{{"question":"{SECRET}',
                headers={"content-type": "application/json"},
            ),
        ]

    for response in responses:
        _assert_error(response, status=422, code="invalid_request")
        for forbidden in (QUESTION_CLEAN, PRIVATE_PATH, SECRET):
            assert forbidden not in response.text
    assert service.calls == []


def test_default_ask_service_returns_stable_unavailable_instead_of_fake_answer() -> None:
    with TestClient(create_app()) as client:
        response = client.post(
            "/ask",
            json={"question": "valid", "episode_id": EPISODE_ID, "top_k": 3},
        )

    error = _assert_error(response, status=503, code="service_unavailable")
    assert error.error.message == "问答服务暂时不可用。"
    assert "answer" not in response.json()


@pytest.mark.parametrize(
    ("failure", "status", "expected_code"),
    [
        (ServiceFailure(code="invalid_request"), 422, "invalid_request"),
        (ServiceFailure(code="not_found"), 404, "not_found"),
        (ServiceFailure(code="service_unavailable"), 503, "service_unavailable"),
        (ServiceFailure(code="service_failed"), 503, "service_failed"),
    ],
)
def test_ask_service_safe_failures_use_fixed_api_messages(
    failure: ServiceFailure,
    status: int,
    expected_code: str,
) -> None:
    service = FakeAskService(failure)

    with TestClient(create_app(ask_service=service)) as client:
        response = client.post(
            "/ask",
            json={"question": "valid", "episode_id": EPISODE_ID, "top_k": 3},
        )

    _assert_error(response, status=status, code=expected_code)
    assert len(service.calls) == 1
    assert FAILURE_SENTINEL not in response.text


@pytest.mark.parametrize(
    "result",
    [
        RuntimeError(FAILURE_SENTINEL),
        {"ok": True, "answer": {"answer": FAILURE_SENTINEL}},
    ],
)
def test_ask_service_exception_or_invalid_result_is_stable_and_redacted(
    result: object,
) -> None:
    service = FakeAskService(result)

    with TestClient(create_app(ask_service=service)) as client:
        response = client.post(
            "/ask",
            json={"question": QUESTION_CLEAN, "episode_id": EPISODE_ID, "top_k": 3},
        )

    error = _assert_error(response, status=503, code="service_failed")
    assert error.error.message == "问答服务无法完成请求。"
    assert len(service.calls) == 1
    for forbidden in (
        QUESTION_CLEAN,
        PRIVATE_PATH,
        DATABASE_URL,
        SECRET,
        "traceback",
    ):
        assert forbidden not in response.text


def test_session_metadata_success_calls_reader_once() -> None:
    reader = FakeSessionReader(_session())

    with TestClient(create_app(session_reader=reader)) as client:
        response = client.get(f"/sessions/{SESSION_ID}")

    parsed = SessionResponse.model_validate_json(response.text)
    assert response.status_code == 200
    assert parsed.session.session_id == SESSION_ID
    assert parsed.session.status == "active"
    assert reader.calls == [SESSION_ID]


def test_missing_session_returns_stable_404() -> None:
    reader = FakeSessionReader(None)

    with TestClient(create_app(session_reader=reader)) as client:
        response = client.get(f"/sessions/{SESSION_ID}")

    error = _assert_error(response, status=404, code="not_found")
    assert error.error.message == "请求的会话不存在。"
    assert reader.calls == [SESSION_ID]


@pytest.mark.parametrize(
    ("failure", "status", "expected_code"),
    [
        (ServiceFailure(code="invalid_request"), 422, "invalid_request"),
        (ServiceFailure(code="not_found"), 404, "not_found"),
        (ServiceFailure(code="service_unavailable"), 503, "service_unavailable"),
        (ServiceFailure(code="service_failed"), 503, "service_failed"),
    ],
)
def test_session_reader_safe_failures_use_fixed_api_messages(
    failure: ServiceFailure,
    status: int,
    expected_code: str,
) -> None:
    reader = FakeSessionReader(failure)

    with TestClient(create_app(session_reader=reader)) as client:
        response = client.get(f"/sessions/{SESSION_ID}")

    _assert_error(response, status=status, code=expected_code)
    assert reader.calls == [SESSION_ID]


def test_malformed_session_result_is_redacted() -> None:
    reader = FakeSessionReader(
        {"session_id": SESSION_ID, "title": FAILURE_SENTINEL}
    )

    with TestClient(create_app(session_reader=reader)) as client:
        response = client.get(f"/sessions/{SESSION_ID}")

    _assert_error(response, status=503, code="service_failed")
    assert reader.calls == [SESSION_ID]
    for forbidden in (QUESTION_CLEAN, PRIVATE_PATH, DATABASE_URL, SECRET, "traceback"):
        assert forbidden not in response.text


@pytest.mark.parametrize(
    "session_id",
    ["bad/id", ".starts-with-dot", "x" * 129],
)
def test_invalid_session_id_returns_uniform_validation_error_without_reader_call(
    session_id: str,
) -> None:
    reader = FakeSessionReader(_session())

    with TestClient(create_app(session_reader=reader)) as client:
        response = client.get(f"/sessions/{session_id}")

    _assert_error(response, status=422, code="invalid_request")
    assert reader.calls == []
    assert session_id not in response.text


def test_session_database_exception_is_redacted() -> None:
    reader = FakeSessionReader(RuntimeError(FAILURE_SENTINEL))

    with TestClient(create_app(session_reader=reader)) as client:
        response = client.get(f"/sessions/{SESSION_ID}")

    error = _assert_error(response, status=503, code="service_failed")
    assert error.error.message == "会话服务无法完成请求。"
    assert reader.calls == [SESSION_ID]
    for forbidden in (PRIVATE_PATH, DATABASE_URL, SECRET, "traceback"):
        assert forbidden not in response.text


def test_default_session_reader_is_unavailable_without_creating_a_database() -> None:
    with TestClient(create_app()) as client:
        response = client.get(f"/sessions/{SESSION_ID}")

    _assert_error(response, status=503, code="service_unavailable")


def test_conversation_adapter_is_a_thin_single_read_projection() -> None:
    stored = SessionRead(
        session_id=SESSION_ID,
        title="Stored synthetic session",
        status="closed",
        created_at=datetime(2026, 8, 20, 8, 0, tzinfo=UTC),
        updated_at=datetime(2026, 8, 20, 8, 2, tzinfo=UTC),
    )

    class ConversationServiceSpy:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def get(self, session_id: str) -> SessionRead | None:
            self.calls.append(session_id)
            return stored

    service = ConversationServiceSpy()
    reader = ConversationSessionReader(service)  # type: ignore[arg-type]

    result = reader.get_session(SESSION_ID)

    assert result == SessionMetadata(
        session_id=SESSION_ID,
        title="Stored synthetic session",
        status="closed",
        created_at=stored.created_at,
        updated_at=stored.updated_at,
    )
    assert service.calls == [SESSION_ID]


def test_all_documented_endpoint_results_validate_with_pydantic_schemas() -> None:
    ask = FakeAskService(_answer_success())
    session = FakeSessionReader(_session())
    app = create_app(ask_service=ask, session_reader=session)

    with TestClient(app) as client:
        health_response = client.get("/health")
        ask_response = client.post(
            "/ask",
            json={"question": "valid", "episode_id": EPISODE_ID, "top_k": 1},
        )
        session_response = client.get(f"/sessions/{SESSION_ID}")
        error_response = client.post(
            "/ask",
            json={"question": QUESTION_CLEAN, "episode_id": "bad", "top_k": 1},
        )

    HealthResponse.model_validate_json(health_response.text)
    AskResponse.model_validate_json(ask_response.text)
    SessionResponse.model_validate_json(session_response.text)
    ApiErrorResponse.model_validate_json(error_response.text)


def test_unknown_route_and_method_also_use_the_unified_error_shape() -> None:
    with TestClient(create_app()) as client:
        missing = client.get("/private/path/that-does-not-exist")
        wrong_method = client.post("/health", json={"secret": SECRET})

    _assert_error(missing, status=404, code="not_found")
    _assert_error(wrong_method, status=405, code="invalid_request")
    assert SECRET not in wrong_method.text


def test_unexpected_route_exception_uses_fixed_redacted_internal_error() -> None:
    app = create_app()

    @app.get("/unexpected-test-only")
    def unexpected() -> None:
        raise RuntimeError(FAILURE_SENTINEL)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/unexpected-test-only")

    error = _assert_error(response, status=500, code="internal_error")
    assert error.error.message == "服务暂时无法完成请求。"
    for forbidden in (QUESTION_CLEAN, PRIVATE_PATH, DATABASE_URL, SECRET, "traceback"):
        assert forbidden not in response.text


def test_api_modules_do_not_construct_business_or_infrastructure_dependencies() -> None:
    api_root = PROJECT_ROOT / "app" / "api"
    forbidden_import_fragments = {
        "app.agent.workflow",
        "app.retrieval",
        "app.tooling.dispatcher",
        "app.storage.database",
        "experiments.framework_compare",
        "langgraph",
        "openai",
    }
    forbidden_constructor_names = {
        "AgentWorkflow",
        "ConversationPersistenceService",
        "Database",
        "Retriever",
        "ToolDispatcher",
        "create_database",
    }
    for path in sorted(api_root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert all(fragment not in module for fragment in forbidden_import_fragments)
                assert all(
                    alias.name not in forbidden_constructor_names
                    for alias in node.names
                )
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert all(
                        fragment not in alias.name
                        for fragment in forbidden_import_fragments
                    )
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_constructor_names
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_constructor_names


def test_api_dependencies_are_optional_and_minimal() -> None:
    configuration = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    optional = configuration["project"]["optional-dependencies"]
    direct = configuration["project"]["dependencies"]

    assert optional["api"] == [
        "fastapi>=0.141,<0.142",
        "uvicorn>=0.52,<0.53",
    ]
    assert all("fastapi" not in item.lower() and "uvicorn" not in item.lower() for item in direct)
    assert importlib.metadata.version("fastapi").startswith("0.141.")
    assert importlib.metadata.version("uvicorn").startswith("0.52.")
    assert "app*" in configuration["tool"]["setuptools"]["packages"]["find"]["include"]
