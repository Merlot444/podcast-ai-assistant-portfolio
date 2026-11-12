"""Day 25 HTTP request identity, bounded-body, and completion logging."""

from __future__ import annotations

import math
import re
import secrets
import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send


DEFAULT_MAX_REQUEST_BODY_BYTES = 16 * 1024
MAX_CONFIGURABLE_REQUEST_BODY_BYTES = 1024 * 1024
REQUEST_ID_PATTERN = r"^req_[0-9a-f]{32}$"
_REQUEST_ID_RE = re.compile(REQUEST_ID_PATTERN)

RequestIdFactory = Callable[[], object]
UtcClock = Callable[[], object]
MonotonicClock = Callable[[], object]
BoundaryErrorKind = Literal["body_too_large", "internal_error"]


class BoundaryErrorResponseFactory(Protocol):
    def __call__(self, kind: BoundaryErrorKind) -> Response:
        """Return one application-owned fixed error response."""


class HttpRequestCompletedEvent(BaseModel):
    """The only structured event emitted for one completed HTTP request."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
    )

    event: Literal["http_request_completed"] = "http_request_completed"
    occurred_at: datetime
    request_id: str = Field(pattern=REQUEST_ID_PATTERN)
    method: str = Field(min_length=1, max_length=32, pattern=r"^[A-Z]+$")
    route: str = Field(min_length=1, max_length=160)
    status_code: int = Field(ge=100, le=599)
    duration_ms: float = Field(ge=0, allow_inf_nan=False)

    @field_validator("occurred_at")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("occurred_at 必须是 UTC 时间")
        return value.astimezone(UTC)

    @field_serializer("occurred_at")
    def serialize_utc(self, value: datetime) -> str:
        return value.astimezone(UTC).isoformat(timespec="milliseconds").replace(
            "+00:00",
            "Z",
        )


RequestEventSink = Callable[[HttpRequestCompletedEvent], None]


def generate_request_id() -> str:
    """Return one server-owned request identifier using OS randomness."""
    return f"req_{secrets.token_hex(16)}"


def write_request_event(event: HttpRequestCompletedEvent) -> None:
    """Write exactly one compact UTF-8 JSON object to stderr."""
    sys.stderr.write(event.model_dump_json() + "\n")
    sys.stderr.flush()


def validate_max_request_body_bytes(value: object) -> int:
    """Validate the small trusted configuration surface without coercion."""
    if type(value) is not int:
        raise TypeError("max_request_body_bytes 必须是严格整数")
    if value < 1 or value > MAX_CONFIGURABLE_REQUEST_BODY_BYTES:
        raise ValueError(
            "max_request_body_bytes 必须在 1 到 "
            f"{MAX_CONFIGURABLE_REQUEST_BODY_BYTES} 之间"
        )
    return value


class HttpRequestBoundaryMiddleware:
    """Own request IDs, pre-parse body limits, and one completion event."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        request_id_factory: RequestIdFactory,
        event_sink: RequestEventSink,
        clock: UtcClock,
        monotonic_clock: MonotonicClock,
        max_request_body_bytes: int,
        error_response_factory: BoundaryErrorResponseFactory,
    ) -> None:
        self.app = app
        self.request_id_factory = request_id_factory
        self.event_sink = event_sink
        self.clock = clock
        self.monotonic_clock = monotonic_clock
        self.max_request_body_bytes = validate_max_request_body_bytes(
            max_request_body_bytes
        )
        self.error_response_factory = error_response_factory

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _safe_request_id(self.request_id_factory)
        started_at = _read_monotonic(self.monotonic_clock)
        status_code = 500
        response_started = False

        async def send_with_request_id(message: Message) -> None:
            nonlocal response_started, status_code
            if message["type"] == "http.response.start":
                response_started = True
                status_code = int(message["status"])
                headers = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() != b"x-request-id"
                ]
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            declared_length = _content_length(scope)
            if (
                declared_length is not None
                and declared_length > self.max_request_body_bytes
            ):
                await self._send_error(
                    scope,
                    send_with_request_id,
                    "body_too_large",
                )
                return

            body, disconnected, exceeded = await _read_bounded_body(
                receive,
                self.max_request_body_bytes,
            )
            if exceeded:
                await self._send_error(
                    scope,
                    send_with_request_id,
                    "body_too_large",
                )
                return

            replay_receive = _replay_receive(
                body, disconnected=disconnected, receive=receive
            )
            await self.app(scope, replay_receive, send_with_request_id)
        except Exception:
            if response_started:
                raise
            await self._send_error(
                scope,
                send_with_request_id,
                "internal_error",
            )
        finally:
            finished_at = _read_monotonic(self.monotonic_clock)
            duration_ms = _duration_ms(started_at, finished_at)
            occurred_at = _read_utc(self.clock)
            route = _route_template(scope)
            method = _safe_method(scope)
            try:
                event = HttpRequestCompletedEvent(
                    occurred_at=occurred_at,
                    request_id=request_id,
                    method=method,
                    route=route,
                    status_code=status_code,
                    duration_ms=duration_ms,
                )
                self.event_sink(event)
            except Exception:
                # Observability is deliberately fail-open for the HTTP response.
                pass

    async def _send_error(
        self,
        scope: Scope,
        send: Send,
        kind: BoundaryErrorKind,
    ) -> None:
        response = self.error_response_factory(kind)

        async def empty_receive() -> Message:
            return {"type": "http.disconnect"}

        await response(scope, empty_receive, send)


def _safe_request_id(factory: RequestIdFactory) -> str:
    try:
        candidate = factory()
    except Exception:
        candidate = None
    if isinstance(candidate, str) and _REQUEST_ID_RE.fullmatch(candidate):
        return candidate
    return generate_request_id()


def _content_length(scope: Scope) -> int | None:
    values = [
        value
        for name, value in scope.get("headers", [])
        if name.lower() == b"content-length"
    ]
    if len(values) != 1:
        return None
    try:
        text = values[0].decode("ascii")
    except UnicodeDecodeError:
        return None
    if not text or not text.isdigit():
        return None
    return int(text)


async def _read_bounded_body(
    receive: Receive,
    limit: int,
) -> tuple[bytes, bool, bool]:
    """Read no more than ``limit + 1`` bytes before routing can parse JSON."""
    buffered = bytearray()
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return bytes(buffered), True, False
        if message["type"] != "http.request":
            continue

        chunk = message.get("body", b"")
        remaining = limit + 1 - len(buffered)
        if remaining > 0:
            buffered.extend(chunk[:remaining])
        if len(chunk) > remaining or len(buffered) > limit:
            return bytes(buffered), False, True
        if not message.get("more_body", False):
            return bytes(buffered), False, False


def _replay_receive(body: bytes, *, disconnected: bool, receive: Receive) -> Receive:
    sent_body = False

    async def replay() -> Message:
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {
                "type": "http.request",
                "body": body,
                "more_body": disconnected,
            }
        if disconnected:
            return {"type": "http.disconnect"}
        # Body exhaustion is not a client disconnect. Preserve the real receive
        # channel so response streaming can observe an actual disconnect.
        return await receive()

    return replay


def _read_monotonic(clock: MonotonicClock) -> float | None:
    try:
        value = clock()
    except Exception:
        return None
    if type(value) not in {int, float}:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def _duration_ms(started_at: float | None, finished_at: float | None) -> float:
    if started_at is None or finished_at is None or finished_at < started_at:
        return 0.0
    return round((finished_at - started_at) * 1000, 3)


def _read_utc(clock: UtcClock) -> datetime:
    try:
        value = clock()
    except Exception:
        value = None
    if isinstance(value, datetime) and value.tzinfo is not None:
        return value.astimezone(UTC)
    return datetime.now(UTC)


def _safe_method(scope: Scope) -> str:
    value = scope.get("method")
    if isinstance(value, str):
        candidate = value.upper()
        if 1 <= len(candidate) <= 32 and candidate.isascii() and candidate.isalpha():
            return candidate
    return "UNKNOWN"


def _route_template(scope: Scope) -> str:
    route = scope.get("route")
    path = getattr(route, "path", None)
    if isinstance(path, str) and 1 <= len(path) <= 160:
        if path.startswith("/sessions/{"):
            return "/sessions/{session_id}"
        return path

    raw_path = scope.get("path")
    if raw_path in {"/", "/health", "/ask", "/static/day25.css", "/static/day25.js"}:
        return str(raw_path)
    if isinstance(raw_path, str) and raw_path.startswith("/sessions/"):
        return "/sessions/{session_id}"
    return "unmatched"
