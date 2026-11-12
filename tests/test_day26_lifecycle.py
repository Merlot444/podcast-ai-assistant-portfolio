"""Day 26 strict, fail-open service lifecycle event tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from app.api import (
    HttpRequestCompletedEvent,
    SERVICE_NAME,
    SERVICE_VERSION,
    ServiceLifecycleEvent,
    create_app,
)


EPISODE_ID = "ep_1111111111111111"
REQUEST_ID = "req_44444444444444444444444444444444"
PRIVATE_VALUE = "postgresql://admin:password@localhost/private"
STARTED_AT = datetime(2026, 8, 24, 8, 0, tzinfo=UTC)
STOPPED_AT = datetime(2026, 8, 24, 8, 1, tzinfo=UTC)


class SequenceClock:
    def __init__(self, values: list[datetime]) -> None:
        self._values = iter(values)

    def __call__(self) -> datetime:
        return next(self._values)


def test_lifecycle_schema_is_strict_utc_and_has_only_low_sensitivity_fields() -> None:
    event = ServiceLifecycleEvent(
        event="service_started",
        occurred_at=STARTED_AT,
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
    )

    assert set(event.model_dump()) == {"event", "occurred_at", "service", "version"}
    assert event.model_dump_json() == (
        '{"event":"service_started",'
        '"occurred_at":"2026-08-24T08:00:00.000Z",'
        f'"service":"{SERVICE_NAME}","version":"{SERVICE_VERSION}"}}'
    )

    invalid_payloads = [
        {
            "event": "service_started",
            "occurred_at": datetime(2026, 8, 24, 8, 0),
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
        },
        {
            "event": "started",
            "occurred_at": STARTED_AT,
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
        },
        {
            "event": "service_started",
            "occurred_at": STARTED_AT,
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "host": "0.0.0.0",
        },
        {
            "event": "service_started",
            "occurred_at": "2026-08-24T08:00:00Z",
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
        },
        {
            "event": "service_started",
            "occurred_at": STARTED_AT,
            "service": "another-service",
            "version": SERVICE_VERSION,
        },
        {
            "event": "service_started",
            "occurred_at": STARTED_AT,
            "service": SERVICE_NAME,
            "version": "private-build",
        },
    ]
    for payload in invalid_payloads:
        with pytest.raises(ValidationError):
            ServiceLifecycleEvent.model_validate(payload)


def test_one_lifespan_emits_started_request_completed_stopped_in_order() -> None:
    events: list[object] = []
    app = create_app(
        request_id_factory=lambda: REQUEST_ID,
        event_sink=events.append,
        lifecycle_event_sink=events.append,
        lifecycle_clock=SequenceClock([STARTED_AT, STOPPED_AT]),
    )

    with TestClient(app) as client:
        assert len(events) == 1
        response = client.get("/health")
        assert len(events) == 2

    assert response.status_code == 200
    assert response.headers["x-request-id"] == REQUEST_ID
    assert [getattr(event, "event") for event in events] == [
        "service_started",
        "http_request_completed",
        "service_stopped",
    ]
    assert isinstance(events[0], ServiceLifecycleEvent)
    assert isinstance(events[1], HttpRequestCompletedEvent)
    assert isinstance(events[2], ServiceLifecycleEvent)
    assert events[0].occurred_at == STARTED_AT  # type: ignore[union-attr]
    assert events[2].occurred_at == STOPPED_AT  # type: ignore[union-attr]


def test_each_repeated_lifespan_has_at_most_one_started_and_stopped_event() -> None:
    events: list[ServiceLifecycleEvent] = []
    app = create_app(
        event_sink=lambda _event: None,
        lifecycle_event_sink=events.append,
        lifecycle_clock=lambda: STARTED_AT,
    )

    with TestClient(app):
        pass
    with TestClient(app):
        pass

    assert [event.event for event in events] == [
        "service_started",
        "service_stopped",
        "service_started",
        "service_stopped",
    ]


def test_lifecycle_sink_failure_cannot_block_start_response_or_shutdown() -> None:
    lifecycle_calls = 0
    request_events: list[HttpRequestCompletedEvent] = []

    def broken_sink(_event: ServiceLifecycleEvent) -> None:
        nonlocal lifecycle_calls
        lifecycle_calls += 1
        raise RuntimeError(f"{PRIVATE_VALUE} traceback")

    with TestClient(
        create_app(
            request_id_factory=lambda: REQUEST_ID,
            event_sink=request_events.append,
            lifecycle_event_sink=broken_sink,
        )
    ) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True, "status": "ok"}
    assert response.headers["x-request-id"] == REQUEST_ID
    assert lifecycle_calls == 2
    assert len(request_events) == 1


def test_falsey_lifecycle_sink_is_preserved() -> None:
    class FalseySink:
        def __init__(self) -> None:
            self.events: list[ServiceLifecycleEvent] = []

        def __bool__(self) -> bool:
            return False

        def __call__(self, event: ServiceLifecycleEvent) -> None:
            self.events.append(event)

    sink = FalseySink()
    with TestClient(
        create_app(
            event_sink=lambda _event: None,
            lifecycle_event_sink=sink,
        )
    ):
        pass

    assert [event.event for event in sink.events] == [
        "service_started",
        "service_stopped",
    ]


def test_non_utc_aware_clock_is_normalized_and_naive_clock_falls_back_to_utc() -> None:
    plus_eight = datetime(
        2026,
        8,
        24,
        16,
        0,
        tzinfo=timezone(timedelta(hours=8)),
    )
    values = iter([plus_eight, datetime(2026, 8, 24, 8, 1)])
    events: list[ServiceLifecycleEvent] = []

    with TestClient(
        create_app(
            event_sink=lambda _event: None,
            lifecycle_event_sink=events.append,
            lifecycle_clock=lambda: next(values),
        )
    ):
        pass

    assert events[0].occurred_at == STARTED_AT
    assert events[1].occurred_at.tzinfo is not None
    assert events[1].occurred_at.utcoffset() == timedelta(0)


def test_default_lifecycle_sink_writes_two_independent_json_lines_to_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with TestClient(
        create_app(
            event_sink=lambda _event: None,
            lifecycle_clock=SequenceClock([STARTED_AT, STOPPED_AT]),
        )
    ):
        pass

    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 2
    payloads = [json.loads(line) for line in lines]
    parsed = [ServiceLifecycleEvent.model_validate_json(line) for line in lines]
    assert [payload["event"] for payload in payloads] == [
        "service_started",
        "service_stopped",
    ]
    assert all(item.occurred_at.utcoffset() == timedelta(0) for item in parsed)


def test_lifecycle_events_exclude_runtime_request_and_exception_details() -> None:
    events: list[ServiceLifecycleEvent] = []

    with TestClient(
        create_app(
            event_sink=lambda _event: None,
            lifecycle_event_sink=events.append,
        )
    ) as client:
        client.get(
            "/health?private=yes",
            headers={"X-Secret": PRIVATE_VALUE},
        )

    serialized = "\n".join(event.model_dump_json() for event in events)
    assert len(events) == 2
    assert all(
        set(event.model_dump()) == {"event", "occurred_at", "service", "version"}
        for event in events
    )
    for forbidden in (
        PRIVATE_VALUE,
        "0.0.0.0",
        "127.0.0.1",
        "PODCAST_API_HOST",
        "pid",
        "/health",
        "private=yes",
        "X-Secret",
        "traceback",
    ):
        assert forbidden not in serialized


def test_day25_request_boundaries_and_default_ask_remain_compatible() -> None:
    request_events: list[HttpRequestCompletedEvent] = []
    lifecycle_events: list[ServiceLifecycleEvent] = []
    app = create_app(
        request_id_factory=lambda: REQUEST_ID,
        event_sink=request_events.append,
        lifecycle_event_sink=lifecycle_events.append,
        max_request_body_bytes=128,
    )

    @app.get("/test-only-unexpected")
    def unexpected() -> None:
        raise RuntimeError(f"{PRIVATE_VALUE} traceback")

    with TestClient(app, raise_server_exceptions=False) as client:
        health = client.get("/health")
        ask = client.post(
            "/ask",
            json={"question": "valid", "episode_id": EPISODE_ID, "top_k": 3},
        )
        oversized = client.post(
            "/ask",
            content=b"x" * 129,
            headers={"content-type": "application/json"},
        )
        failure = client.get("/test-only-unexpected")

    assert [response.status_code for response in (health, ask, oversized, failure)] == [
        200,
        503,
        413,
        500,
    ]
    assert ask.json() == {
        "ok": False,
        "error": {
            "code": "service_unavailable",
            "message": "问答服务暂时不可用。",
        },
    }
    assert all(
        response.headers["x-request-id"] == REQUEST_ID
        for response in (health, ask, oversized, failure)
    )
    assert len(request_events) == 4
    assert [event.status_code for event in request_events] == [200, 503, 413, 500]
    assert all(
        set(event.model_dump())
        == {
            "event",
            "occurred_at",
            "request_id",
            "method",
            "route",
            "status_code",
            "duration_ms",
        }
        for event in request_events
    )
    assert [event.event for event in lifecycle_events] == [
        "service_started",
        "service_stopped",
    ]
    serialized = "\n".join(event.model_dump_json() for event in request_events)
    assert PRIVATE_VALUE not in serialized
    assert "traceback" not in serialized
