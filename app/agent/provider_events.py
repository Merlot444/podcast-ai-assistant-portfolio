"""Strict, low-sensitivity events for Day 27 model-provider calls."""

from __future__ import annotations

import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Literal, Self, TextIO

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from pydantic import model_validator


PROVIDER_IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$"

ProviderFailureCategory = Literal[
    "authentication_permission",
    "invalid_request",
    "timeout",
    "rate_limit",
    "connection",
    "provider_5xx",
    "malformed_response",
    "unknown_provider_failure",
]
ProviderCallOutcome = Literal["success", "failure"]
ProviderUtcClock = Callable[[], object]


class ProviderCallEvent(BaseModel):
    """One completed provider attempt without prompt or response content."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
    )

    event: Literal["agent_provider_call_completed"] = (
        "agent_provider_call_completed"
    )
    occurred_at: datetime
    provider: str = Field(
        min_length=1,
        max_length=128,
        pattern=PROVIDER_IDENTIFIER_PATTERN,
    )
    model: str = Field(
        min_length=1,
        max_length=128,
        pattern=PROVIDER_IDENTIFIER_PATTERN,
    )
    stage: Literal["agent_decision"] = "agent_decision"
    latency_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    outcome: ProviderCallOutcome
    failure_category: ProviderFailureCategory | None = None
    attempt_number: Literal[1] = 1
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    retry_count: Literal[0] = 0
    cost: Literal["unknown"] = "unknown"

    @field_validator("occurred_at")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("occurred_at must be UTC")
        return value.astimezone(UTC)

    @field_serializer("occurred_at")
    def serialize_utc(self, value: datetime) -> str:
        return value.astimezone(UTC).isoformat(timespec="milliseconds").replace(
            "+00:00",
            "Z",
        )

    @model_validator(mode="after")
    def require_consistent_outcome_and_usage(self) -> Self:
        if self.outcome == "success" and self.failure_category is not None:
            raise ValueError("successful provider events cannot have a failure category")
        if self.outcome == "failure" and self.failure_category is None:
            raise ValueError("failed provider events require a failure category")

        usage = (self.input_tokens, self.output_tokens, self.total_tokens)
        if any(value is None for value in usage) and any(
            value is not None for value in usage
        ):
            raise ValueError("provider usage must be complete or absent")
        return self


ProviderEventSink = Callable[[ProviderCallEvent], None]


def write_provider_call_event(
    event: ProviderCallEvent,
    *,
    stream: TextIO | None = None,
) -> None:
    """Write exactly one compact UTF-8 JSON event to stderr by default."""
    output = sys.stderr if stream is None else stream
    output.write(event.model_dump_json() + "\n")
    output.flush()


def emit_provider_call_event(
    *,
    event_sink: ProviderEventSink,
    utc_clock: ProviderUtcClock,
    provider: str,
    model: str,
    latency_ms: float | None,
    outcome: ProviderCallOutcome,
    failure_category: ProviderFailureCategory | None,
    input_tokens: int | None,
    output_tokens: int | None,
    total_tokens: int | None,
) -> None:
    """Emit one event fail-open so observability cannot alter call semantics."""
    try:
        event_sink(
            ProviderCallEvent(
                occurred_at=_read_utc(utc_clock),
                provider=provider,
                model=model,
                latency_ms=latency_ms,
                outcome=outcome,
                failure_category=failure_category,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )
        )
    except Exception:
        pass


def _read_utc(clock: ProviderUtcClock) -> datetime:
    try:
        value = clock()
    except Exception:
        value = None
    if isinstance(value, datetime) and value.tzinfo is not None:
        return value.astimezone(UTC)
    return datetime.now(UTC)
