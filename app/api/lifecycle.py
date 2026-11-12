"""Minimal, application-owned Day 26 service lifecycle JSONL events."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator


SERVICE_NAME = "podcast-ai-assistant-api"
SERVICE_VERSION = "0.1.0"

LifecycleEventName = Literal["service_started", "service_stopped"]
LifecycleClock = Callable[[], object]


class ServiceLifecycleEvent(BaseModel):
    """One low-sensitivity start or stop fact for a single app lifespan."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
    )

    event: LifecycleEventName
    occurred_at: datetime
    service: Literal["podcast-ai-assistant-api"] = SERVICE_NAME
    version: Literal["0.1.0"] = SERVICE_VERSION

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


LifecycleEventSink = Callable[[ServiceLifecycleEvent], None]


def write_lifecycle_event(event: ServiceLifecycleEvent) -> None:
    """Write exactly one compact UTF-8 JSON object to stderr."""
    sys.stderr.write(event.model_dump_json() + "\n")
    sys.stderr.flush()


@asynccontextmanager
async def service_lifespan(
    _app: object,
    *,
    event_sink: LifecycleEventSink,
    clock: LifecycleClock,
) -> AsyncIterator[None]:
    """Emit at most one started and stopped event for this lifespan."""
    _emit_lifecycle_event("service_started", event_sink=event_sink, clock=clock)
    try:
        yield
    finally:
        _emit_lifecycle_event("service_stopped", event_sink=event_sink, clock=clock)


def _emit_lifecycle_event(
    name: LifecycleEventName,
    *,
    event_sink: LifecycleEventSink,
    clock: LifecycleClock,
) -> None:
    try:
        event_sink(
            ServiceLifecycleEvent(
                event=name,
                occurred_at=_read_utc(clock),
                service=SERVICE_NAME,
                version=SERVICE_VERSION,
            )
        )
    except Exception:
        # Observability is deliberately fail-open for startup and shutdown.
        pass


def _read_utc(clock: LifecycleClock) -> datetime:
    try:
        value = clock()
    except Exception:
        value = None
    if isinstance(value, datetime) and value.tzinfo is not None:
        return value.astimezone(UTC)
    return datetime.now(UTC)
