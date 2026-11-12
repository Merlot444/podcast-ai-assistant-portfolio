"""Day 25 fixed, redacted, one-event-per-request logging tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from app.agent import AgentFinalAnswer
from app.api import HttpRequestCompletedEvent, create_app
from app.api.schemas import AskRequest
from app.api.services import AskServiceSuccess


EPISODE_ID = "ep_1111111111111111"
SESSION_ID = "session_private_value"
REQUEST_ID = "req_22222222222222222222222222222222"
QUESTION = "private question body"
ANSWER = "private answer body"
SECRET = "postgresql://admin:password@localhost/private"


class SequenceClock:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


class AnswerService:
    def ask(self, request: AskRequest) -> AskServiceSuccess:
        return AskServiceSuccess(
            answer=AgentFinalAnswer(answer=ANSWER, citations=[])
        )


def test_one_strict_completion_event_matches_header_and_fake_clocks() -> None:
    events: list[HttpRequestCompletedEvent] = []
    app = create_app(
        request_id_factory=lambda: REQUEST_ID,
        event_sink=events.append,
        clock=lambda: datetime(2026, 8, 20, 8, 0, tzinfo=UTC),
        monotonic_clock=SequenceClock([10.0, 10.001234]),
    )

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert len(events) == 1
    event = events[0]
    assert event == HttpRequestCompletedEvent(
        occurred_at=datetime(2026, 8, 20, 8, 0, tzinfo=UTC),
        request_id=REQUEST_ID,
        method="GET",
        route="/health",
        status_code=200,
        duration_ms=1.234,
    )
    assert response.headers["x-request-id"] == event.request_id
    assert set(event.model_dump()) == {
        "event",
        "occurred_at",
        "request_id",
        "method",
        "route",
        "status_code",
        "duration_ms",
    }


def test_session_uses_route_template_and_unknown_route_is_unmatched() -> None:
    events: list[HttpRequestCompletedEvent] = []
    app = create_app(
        event_sink=events.append,
        request_id_factory=lambda: REQUEST_ID,
    )

    with TestClient(app) as client:
        client.get(f"/sessions/{SESSION_ID}")
        client.get("/sessions/")
        client.get(f"/unknown/{SESSION_ID}")

    assert [event.route for event in events] == [
        "/sessions/{session_id}",
        "/sessions/{session_id}",
        "unmatched",
    ]
    serialized = "\n".join(event.model_dump_json() for event in events)
    assert SESSION_ID not in serialized
    assert f"/unknown/{SESSION_ID}" not in serialized


def test_page_and_static_requests_each_emit_one_event() -> None:
    events: list[HttpRequestCompletedEvent] = []
    with TestClient(create_app(event_sink=events.append)) as client:
        responses = [
            client.get("/"),
            client.get("/static/day25.css"),
            client.get("/static/day25.js"),
        ]

    assert all(response.status_code == 200 for response in responses)
    assert [event.route for event in events] == [
        "/",
        "/static/day25.css",
        "/static/day25.js",
    ]
    assert len(events) == len(responses)


def test_event_never_contains_request_answer_session_headers_or_exception() -> None:
    events: list[HttpRequestCompletedEvent] = []
    app = create_app(
        ask_service=AnswerService(),
        event_sink=events.append,
        request_id_factory=lambda: REQUEST_ID,
    )

    @app.get("/safe-test-error")
    def fail() -> None:
        raise RuntimeError(f"{QUESTION} {ANSWER} {SESSION_ID} {SECRET} traceback")

    with TestClient(app, raise_server_exceptions=False) as client:
        success = client.post(
            "/ask",
            json={
                "question": QUESTION,
                "episode_id": EPISODE_ID,
                "top_k": 3,
                "session_id": SESSION_ID,
            },
            headers={"User-Agent": "private-agent", "X-Secret": SECRET},
        )
        failure = client.get("/safe-test-error")

    assert success.status_code == 200
    assert failure.status_code == 500
    assert len(events) == 2
    serialized = "\n".join(event.model_dump_json() for event in events)
    for forbidden in (
        QUESTION,
        ANSWER,
        SESSION_ID,
        SECRET,
        "private-agent",
        "X-Secret",
        "traceback",
    ):
        assert forbidden not in serialized


def test_sink_failure_does_not_change_the_http_response_or_retry() -> None:
    calls = 0

    def broken_sink(_event: HttpRequestCompletedEvent) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("logging failed")

    with TestClient(
        create_app(event_sink=broken_sink, request_id_factory=lambda: REQUEST_ID)
    ) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True, "status": "ok"}
    assert response.headers["x-request-id"] == REQUEST_ID
    assert calls == 1


def test_falsey_sink_is_preserved_instead_of_replaced() -> None:
    class FalseySink:
        def __init__(self) -> None:
            self.events: list[HttpRequestCompletedEvent] = []

        def __bool__(self) -> bool:
            return False

        def __call__(self, event: HttpRequestCompletedEvent) -> None:
            self.events.append(event)

    sink = FalseySink()
    with TestClient(create_app(event_sink=sink)) as client:
        client.get("/health")
    assert len(sink.events) == 1


def test_default_request_sink_emits_one_completion_json_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with TestClient(
        create_app(request_id_factory=lambda: REQUEST_ID)
    ) as client:
        response = client.get("/health")

    captured = capsys.readouterr()
    lines = captured.err.splitlines()
    completion_lines = [
        line
        for line in lines
        if json.loads(line).get("event") == "http_request_completed"
    ]
    assert len(completion_lines) == 1
    payload = json.loads(completion_lines[0])
    parsed = HttpRequestCompletedEvent.model_validate_json(completion_lines[0])
    assert payload["event"] == "http_request_completed"
    assert payload["request_id"] == response.headers["x-request-id"]
    assert parsed.occurred_at.utcoffset().total_seconds() == 0
    assert parsed.duration_ms >= 0
