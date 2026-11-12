"""Day 25 pre-Pydantic ASGI request-body byte limit tests."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from starlette.types import Message

from app.agent import AgentFinalAnswer
from app.api import HttpRequestCompletedEvent, create_app
from app.api.request_boundaries import MAX_CONFIGURABLE_REQUEST_BODY_BYTES, _replay_receive
from app.api.schemas import ApiErrorResponse, AskRequest
from app.api.services import AskServiceSuccess


EPISODE_ID = "ep_1111111111111111"
REQUEST_ID = "req_33333333333333333333333333333333"


def test_buffered_body_waits_for_the_real_disconnect() -> None:
    async def check() -> None:
        disconnected = asyncio.Event()

        async def receive() -> Message:
            await disconnected.wait()
            return {"type": "http.disconnect"}

        replay = _replay_receive(b"bounded", disconnected=False, receive=receive)
        assert await replay() == {
            "type": "http.request", "body": b"bounded", "more_body": False,
        }
        pending = asyncio.create_task(replay())
        await asyncio.sleep(0)
        assert not pending.done()
        disconnected.set()
        assert await pending == {"type": "http.disconnect"}

    asyncio.run(check())


class CountingAskService:
    def __init__(self) -> None:
        self.calls: list[AskRequest] = []

    def ask(self, request: AskRequest) -> AskServiceSuccess:
        self.calls.append(request)
        return AskServiceSuccess(
            answer=AgentFinalAnswer(answer="Synthetic answer.", citations=[])
        )


def _ask_body(question: str = "valid") -> bytes:
    return json.dumps(
        {"question": question, "episode_id": EPISODE_ID, "top_k": 3},
        separators=(",", ":"),
    ).encode()


def _direct_asgi_post(
    app: object,
    *,
    chunks: Iterable[bytes],
    headers: list[tuple[bytes, bytes]],
    disconnect_after_chunks: bool = False,
) -> tuple[int, dict[bytes, bytes], bytes, int]:
    queued = list(chunks)
    sent: list[Message] = []
    receive_calls = 0

    async def run() -> None:
        nonlocal receive_calls
        messages: list[Message] = [
            {
                "type": "http.request",
                "body": chunk,
                "more_body": (
                    index < len(queued) - 1 or disconnect_after_chunks
                ),
            }
            for index, chunk in enumerate(queued)
        ]
        if not messages:
            messages.append(
                {"type": "http.request", "body": b"", "more_body": False}
            )

        async def receive() -> Message:
            nonlocal receive_calls
            receive_calls += 1
            if messages:
                return messages.pop(0)
            return {"type": "http.disconnect"}

        async def send(message: Message) -> None:
            sent.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/ask",
            "raw_path": b"/ask",
            "root_path": "",
            "query_string": b"",
            "headers": headers,
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
        }
        await app(scope, receive, send)  # type: ignore[misc]

    asyncio.run(run())
    start = next(message for message in sent if message["type"] == "http.response.start")
    body = b"".join(
        message.get("body", b"")
        for message in sent
        if message["type"] == "http.response.body"
    )
    return start["status"], dict(start["headers"]), body, receive_calls


@pytest.mark.parametrize(
    "bad_limit",
    [
        True,
        False,
        0,
        -1,
        1.0,
        float("inf"),
        "16384",
        MAX_CONFIGURABLE_REQUEST_BODY_BYTES + 1,
    ],
)
def test_body_limit_configuration_is_a_bounded_strict_integer(bad_limit: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        create_app(max_request_body_bytes=bad_limit)  # type: ignore[arg-type]


def test_content_length_over_limit_rejects_before_reading_or_service() -> None:
    service = CountingAskService()
    events: list[HttpRequestCompletedEvent] = []
    app = create_app(
        ask_service=service,
        max_request_body_bytes=64,
        request_id_factory=lambda: REQUEST_ID,
        event_sink=events.append,
    )

    status, headers, body, receive_calls = _direct_asgi_post(
        app,
        chunks=[b"this body must never be read"],
        headers=[
            (b"content-type", b"application/json"),
            (b"content-length", b"65"),
        ],
    )

    assert status == 413
    assert dict(headers)[b"x-request-id"] == REQUEST_ID.encode()
    assert receive_calls == 0
    assert service.calls == []
    error = ApiErrorResponse.model_validate_json(body)
    assert error.error.code == "invalid_request"
    assert error.error.message == "请求体过大。"
    assert len(events) == 1 and events[0].status_code == 413


def test_missing_content_length_actual_bytes_are_counted_across_chunks() -> None:
    service = CountingAskService()
    events: list[HttpRequestCompletedEvent] = []
    app = create_app(
        ask_service=service,
        max_request_body_bytes=16,
        request_id_factory=lambda: REQUEST_ID,
        event_sink=events.append,
    )

    status, _headers, body, receive_calls = _direct_asgi_post(
        app,
        chunks=[b"12345678", b"abcdefgh", b"X", b"unread-private-data"],
        headers=[(b"content-type", b"application/json")],
    )

    assert status == 413
    assert receive_calls == 3
    assert service.calls == []
    assert ApiErrorResponse.model_validate_json(body).error.message == "请求体过大。"
    assert len(events) == 1


def test_forged_smaller_content_length_does_not_bypass_actual_byte_count() -> None:
    service = CountingAskService()
    app = create_app(
        ask_service=service,
        max_request_body_bytes=32,
        request_id_factory=lambda: REQUEST_ID,
        event_sink=lambda _event: None,
    )
    oversized = b"{" + b"x" * 64

    status, _headers, body, _receive_calls = _direct_asgi_post(
        app,
        chunks=[oversized],
        headers=[
            (b"content-type", b"application/json"),
            (b"content-length", b"1"),
        ],
    )

    assert status == 413
    assert service.calls == []
    assert ApiErrorResponse.model_validate_json(body).error.code == "invalid_request"


def test_disconnect_during_body_returns_fixed_500_without_partial_body_leak() -> None:
    service = CountingAskService()
    events: list[HttpRequestCompletedEvent] = []
    app = create_app(
        ask_service=service,
        max_request_body_bytes=1024,
        request_id_factory=lambda: REQUEST_ID,
        event_sink=events.append,
    )
    partial_private_body = b'{"question":"/private/path postgres://password'

    status, headers, body, receive_calls = _direct_asgi_post(
        app,
        chunks=[partial_private_body],
        headers=[(b"content-type", b"application/json")],
        disconnect_after_chunks=True,
    )

    assert status == 500
    assert headers[b"x-request-id"] == REQUEST_ID.encode()
    assert receive_calls == 2
    assert service.calls == []
    error = ApiErrorResponse.model_validate_json(body)
    assert error.error.code == "internal_error"
    assert error.error.message == "服务暂时无法完成请求。"
    assert len(events) == 1
    assert events[0].request_id == REQUEST_ID
    assert events[0].route == "/ask"
    assert events[0].status_code == 500
    serialized = body.decode() + events[0].model_dump_json()
    for forbidden in ("/private/path", "postgres://password"):
        assert forbidden not in serialized


def test_body_exactly_at_limit_is_replayed_and_calls_service_once() -> None:
    service = CountingAskService()
    body = _ask_body()
    app = create_app(
        ask_service=service,
        max_request_body_bytes=len(body),
        request_id_factory=lambda: REQUEST_ID,
        event_sink=lambda _event: None,
    )

    status, headers, response_body, receive_calls = _direct_asgi_post(
        app,
        chunks=[body[:15], body[15:]],
        headers=[
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
    )

    assert status == 200
    assert receive_calls == 2
    assert headers[b"x-request-id"] == REQUEST_ID.encode()
    assert json.loads(response_body)["answer"] == "Synthetic answer."
    assert len(service.calls) == 1


def test_one_byte_over_exact_limit_is_413_and_contains_no_sensitive_input() -> None:
    service = CountingAskService()
    private = b"/private/path postgres://password traceback"
    limit = len(private) - 1
    events: list[HttpRequestCompletedEvent] = []
    app = create_app(
        ask_service=service,
        max_request_body_bytes=limit,
        request_id_factory=lambda: REQUEST_ID,
        event_sink=events.append,
    )

    with TestClient(app) as client:
        response = client.post(
            "/ask?query=private",
            content=private,
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 413
    assert response.headers["x-request-id"] == REQUEST_ID
    assert service.calls == []
    serialized = response.text + events[0].model_dump_json()
    for forbidden in ("/private/path", "postgres://password", "traceback", "query=private"):
        assert forbidden not in serialized
    assert len(events) == 1
