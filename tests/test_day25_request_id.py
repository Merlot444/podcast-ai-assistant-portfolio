"""Day 25 server-owned request ID propagation tests."""

from __future__ import annotations

import re
from datetime import UTC, datetime

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from app.agent import AgentCitation, AgentFinalAnswer
from app.api import REQUEST_ID_PATTERN, create_app
from app.api.schemas import AskRequest, SessionMetadata
from app.api.services import AskServiceSuccess


EPISODE_ID = "ep_1111111111111111"
SESSION_ID = "session_day25"
FIXED_REQUEST_ID = "req_11111111111111111111111111111111"
CLIENT_REQUEST_ID = "req_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


class CountingRequestIdFactory:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        return f"req_{self.calls:032x}"


class SuccessfulAskService:
    def __init__(self) -> None:
        self.calls: list[AskRequest] = []

    def ask(self, request: AskRequest) -> AskServiceSuccess:
        self.calls.append(request)
        return AskServiceSuccess(
            answer=AgentFinalAnswer(
                answer="Synthetic answer.",
                citations=[
                    AgentCitation(
                        kind="chunk",
                        episode_id=EPISODE_ID,
                        reference_id="chunk_1111111111111111",
                    )
                ],
            )
        )


class SuccessfulSessionReader:
    def get_session(self, session_id: str) -> SessionMetadata:
        return SessionMetadata(
            session_id=session_id,
            title="Synthetic",
            status="active",
            created_at=datetime(2026, 8, 21, tzinfo=UTC),
            updated_at=datetime(2026, 8, 21, tzinfo=UTC),
        )


def test_page_assets_health_ask_and_session_have_one_server_id_each() -> None:
    factory = CountingRequestIdFactory()
    ask = SuccessfulAskService()
    app = create_app(
        ask_service=ask,
        session_reader=SuccessfulSessionReader(),
        request_id_factory=factory,
        event_sink=lambda _event: None,
    )

    with TestClient(app) as client:
        responses = [
            client.get("/"),
            client.get("/static/day25.css"),
            client.get("/static/day25.js"),
            client.get("/health"),
            client.post(
                "/ask",
                json={"question": "valid", "episode_id": EPISODE_ID, "top_k": 3},
            ),
            client.get(f"/sessions/{SESSION_ID}"),
        ]

    assert factory.calls == len(responses)
    assert [response.headers["x-request-id"] for response in responses] == [
        f"req_{index:032x}" for index in range(1, len(responses) + 1)
    ]
    assert len(ask.calls) == 1
    for response in responses:
        assert "x-request-id" in response.headers


def test_client_request_id_is_ignored_for_response_and_log() -> None:
    events = []
    with TestClient(
        create_app(
            request_id_factory=lambda: FIXED_REQUEST_ID,
            event_sink=events.append,
        )
    ) as client:
        response = client.get(
            "/health",
            headers={"X-Request-ID": CLIENT_REQUEST_ID + "malicious"},
        )

    assert response.headers["x-request-id"] == FIXED_REQUEST_ID
    assert events[0].request_id == FIXED_REQUEST_ID
    assert CLIENT_REQUEST_ID not in response.text
    assert CLIENT_REQUEST_ID not in events[0].model_dump_json()


@pytest.mark.parametrize("bad_result", [None, "", "req_bad", CLIENT_REQUEST_ID + "x"])
def test_malformed_factory_result_gets_a_safe_valid_fallback(bad_result: object) -> None:
    calls = 0

    def bad_factory() -> object:
        nonlocal calls
        calls += 1
        return bad_result

    with TestClient(
        create_app(request_id_factory=bad_factory, event_sink=lambda _event: None)
    ) as client:
        response = client.get("/health")

    assert calls == 1
    assert re.fullmatch(REQUEST_ID_PATTERN, response.headers["x-request-id"])


def test_factory_exception_gets_a_safe_valid_fallback() -> None:
    calls = 0

    def broken_factory() -> str:
        nonlocal calls
        calls += 1
        raise RuntimeError("request id failure must stay private")

    with TestClient(
        create_app(request_id_factory=broken_factory, event_sink=lambda _event: None)
    ) as client:
        response = client.get("/health")

    assert calls == 1
    assert re.fullmatch(REQUEST_ID_PATTERN, response.headers["x-request-id"])
    assert "failure" not in response.text


def test_404_405_413_422_and_500_all_include_request_id() -> None:
    app = create_app(
        request_id_factory=lambda: FIXED_REQUEST_ID,
        event_sink=lambda _event: None,
        max_request_body_bytes=128,
    )

    @app.get("/test-only-unexpected")
    def unexpected() -> None:
        raise RuntimeError("private failure")

    with TestClient(app, raise_server_exceptions=False) as client:
        responses = [
            client.get("/missing"),
            client.post("/health"),
            client.post(
                "/ask",
                content=b"x" * 129,
                headers={"content-type": "application/json"},
            ),
            client.post("/ask", json={"question": "bad"}),
            client.get("/test-only-unexpected"),
        ]

    assert [response.status_code for response in responses] == [404, 405, 413, 422, 500]
    assert all(
        response.headers["x-request-id"] == FIXED_REQUEST_ID
        for response in responses
    )


def test_request_id_stays_in_header_not_day24_json_bodies() -> None:
    with TestClient(
        create_app(
            ask_service=SuccessfulAskService(),
            request_id_factory=lambda: FIXED_REQUEST_ID,
            event_sink=lambda _event: None,
        )
    ) as client:
        health = client.get("/health")
        ask = client.post(
            "/ask",
            json={"question": "valid", "episode_id": EPISODE_ID, "top_k": 3},
        )

    assert "request_id" not in health.json()
    assert "request_id" not in ask.json()
    assert FIXED_REQUEST_ID not in health.text
    assert FIXED_REQUEST_ID not in ask.text
