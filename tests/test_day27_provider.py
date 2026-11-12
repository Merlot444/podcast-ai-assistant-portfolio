"""Offline tests for the bounded Day 27 OpenAI-compatible Agent provider."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
)
from openai import OpenAI as OpenAISdkClient
from pydantic import ValidationError

from app.agent.model import AgentModelRequest
from app.agent.openai_compatible import (
    MAX_PROVIDER_INPUT_CHARACTERS,
    MAX_PROVIDER_OUTPUT_TOKENS,
    PROVIDER_MAX_RETRIES,
    OpenAICompatibleAgentModel,
    OpenAICompatibleAgentModelConfig,
    ProviderAdapterError,
    create_openai_compatible_client,
)
from app.agent.provider_events import (
    ProviderCallEvent,
    ProviderFailureCategory,
    write_provider_call_event,
)
from app.core.config import Settings
from app.tooling.definitions import export_openai_tool_definitions
from app.tooling.schemas import EpisodeMetadataData, ToolSuccess


SECRET = "fixture-secret-never-valid"
QUESTION = "private synthetic question"
MODEL = "offline-model"
OCCURRED_AT = datetime(2026, 8, 25, 8, 0, tzinfo=UTC)


class SequenceClock:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def _request(*, tool_definitions: tuple[dict[str, object], ...] | None = None) -> AgentModelRequest:
    metadata = ToolSuccess(
        tool_name="get_episode_metadata",
        call_id="call_11111111111111111111111111111111",
        data=EpisodeMetadataData(
            episode_id="ep_1111111111111111",
            source_format="json",
            segment_count=3,
            timed_segment_count=3,
            speaker_count=1,
            section_count=1,
            section_names=("Synthetic",),
        ),
    )
    definitions = export_openai_tool_definitions()
    return AgentModelRequest(
        question=QUESTION,
        step_number=2,
        max_steps=4,
        tool_definitions=(
            tuple(definitions)
            if tool_definitions is None
            else tool_definitions
        ),
        tool_results=(metadata,),
        episode_id="ep_1111111111111111",
        top_k=3,
    )


def _completion(
    raw_content: str,
    *,
    finish_reason: str = "stop",
    usage: object | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content=raw_content,
                    tool_calls=None,
                    refusal=None,
                ),
            )
        ],
        usage=usage,
        model=MODEL,
    )


def _json_completion(payload: object, *, usage: object | None = None) -> SimpleNamespace:
    return _completion(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        usage=usage,
    )


def _provider(
    client: Mock,
    *,
    events: list[ProviderCallEvent] | None = None,
    sink: object | None = None,
) -> OpenAICompatibleAgentModel:
    resolved_sink = events.append if events is not None else sink
    return OpenAICompatibleAgentModel(
        client,
        config=OpenAICompatibleAgentModelConfig(
            model=MODEL,
            timeout_seconds=12.5,
        ),
        clock=SequenceClock([10.0, 10.125]),
        utc_clock=lambda: OCCURRED_AT,
        event_sink=resolved_sink,  # type: ignore[arg-type]
    )


def _http_request() -> httpx.Request:
    return httpx.Request(
        "POST",
        "https://example.test/chat/completions",
        headers={"Authorization": f"Bearer {SECRET}"},
    )


def _http_response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, request=_http_request())


def test_client_factory_sets_transport_timeout_and_disables_sdk_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_openai(**kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr("app.agent.openai_compatible.OpenAI", fake_openai)
    settings = Settings(
        api_key=SECRET,
        base_url="https://example.test/v1",
        model_fast=MODEL,
        model_strong="offline-strong-model",
        timeout_seconds=12.5,
    )

    client = create_openai_compatible_client(settings)

    assert client is sentinel
    assert captured == {
        "api_key": SECRET,
        "base_url": "https://example.test/v1",
        "timeout": 12.5,
        "max_retries": 0,
    }
    assert captured["max_retries"] == PROVIDER_MAX_RETRIES


def test_real_sdk_encodes_one_bounded_wire_request_with_transport_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire_requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        wire_requests.append(request)
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "chatcmpl_offline",
                "object": "chat.completion",
                "created": 1,
                "model": MODEL,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": '{"decision_type":"invented"}',
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
        monkeypatch.setattr(
            "app.agent.openai_compatible.OpenAI",
            lambda **kwargs: OpenAISdkClient(http_client=http_client, **kwargs),
        )
        client = create_openai_compatible_client(
            Settings(
                api_key=SECRET,
                base_url="https://example.test/v1",
                model_fast=MODEL,
                model_strong=MODEL,
                timeout_seconds=7.5,
            )
        )
        events: list[ProviderCallEvent] = []
        provider = OpenAICompatibleAgentModel(
            client,
            config=OpenAICompatibleAgentModelConfig(
                model=MODEL,
                timeout_seconds=7.5,
            ),
            clock=SequenceClock([1.0, 1.01]),
            utc_clock=lambda: OCCURRED_AT,
            event_sink=events.append,
        )

        assert provider.decide(_request()) == {"decision_type": "invented"}

    assert len(wire_requests) == 1
    wire_request = wire_requests[0]
    assert wire_request.url.path == "/v1/chat/completions"
    timeout = wire_request.extensions["timeout"]
    assert timeout == {
        "connect": 7.5,
        "read": 7.5,
        "write": 7.5,
        "pool": 7.5,
    }
    body = json.loads(wire_request.content)
    assert body["model"] == MODEL
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_tokens"] == MAX_PROVIDER_OUTPUT_TOKENS
    assert body["stream"] is False
    assert body["n"] == 1
    assert len(events) == 1
    assert events[0].total_tokens == 12


def test_real_sdk_503_is_one_wire_attempt_when_retries_are_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire_attempts = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal wire_attempts
        wire_attempts += 1
        return httpx.Response(
            503,
            request=request,
            json={"error": {"message": f"synthetic upstream {SECRET}"}},
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
        monkeypatch.setattr(
            "app.agent.openai_compatible.OpenAI",
            lambda **kwargs: OpenAISdkClient(http_client=http_client, **kwargs),
        )
        client = create_openai_compatible_client(
            Settings(
                api_key=SECRET,
                base_url="https://example.test/v1",
                model_fast=MODEL,
                model_strong=MODEL,
                timeout_seconds=7.5,
            )
        )
        events: list[ProviderCallEvent] = []
        provider = OpenAICompatibleAgentModel(
            client,
            config=OpenAICompatibleAgentModelConfig(
                model=MODEL,
                timeout_seconds=7.5,
            ),
            clock=SequenceClock([1.0, 1.01]),
            utc_clock=lambda: OCCURRED_AT,
            event_sink=events.append,
        )

        with pytest.raises(ProviderAdapterError) as caught:
            provider.decide(_request())

    assert wire_attempts == 1
    assert caught.value.category == "provider_5xx"
    assert SECRET not in str(caught.value)
    assert len(events) == 1
    assert events[0].failure_category == "provider_5xx"
    assert events[0].retry_count == 0


def test_config_fixes_input_output_retry_bounds_and_hides_extra_values() -> None:
    config = OpenAICompatibleAgentModelConfig(
        model=MODEL,
        timeout_seconds=12.5,
    )

    assert config.max_input_characters == MAX_PROVIDER_INPUT_CHARACTERS
    assert config.max_output_tokens == MAX_PROVIDER_OUTPUT_TOKENS
    assert config.max_retries == 0

    with pytest.raises(ValidationError):
        OpenAICompatibleAgentModelConfig(
            model=MODEL,
            timeout_seconds=12.5,
            max_retries=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError):
        OpenAICompatibleAgentModelConfig(
            model=MODEL,
            timeout_seconds=12.5,
            extra_value=SECRET,  # type: ignore[call-arg]
        )


def test_one_request_has_deterministic_messages_and_all_explicit_bounds() -> None:
    decision = {
        "decision_type": "final_answer",
        "final_answer": {"answer": "Synthetic answer.", "citations": []},
    }
    client = Mock()
    client.chat.completions.create.return_value = _json_completion(decision)
    events: list[ProviderCallEvent] = []
    provider = _provider(client, events=events)

    assert provider.decide(_request()) == decision

    client.chat.completions.create.assert_called_once()
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == MODEL
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["temperature"] == 0
    assert kwargs["max_tokens"] == MAX_PROVIDER_OUTPUT_TOKENS
    assert kwargs["n"] == 1
    assert kwargs["stream"] is False
    assert kwargs["timeout"] == 12.5

    messages = kwargs["messages"]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "request_scope" in messages[0]["content"]
    assert "search_transcript must use its top_k" in messages[0]["content"]
    payload = json.loads(messages[1]["content"])
    assert payload["question"] == QUESTION
    assert payload["step_number"] == 2
    assert payload["max_steps"] == 4
    assert payload["request_scope"] == {
        "episode_id": "ep_1111111111111111",
        "top_k": 3,
    }
    assert [
        item["function"]["name"] for item in payload["tool_definitions"]
    ] == [
        "search_transcript",
        "get_episode_metadata",
        "get_segment_context",
    ]
    assert payload["tool_results"][0]["tool_name"] == "get_episode_metadata"
    assert len(events) == 1


def test_request_scope_is_omitted_unless_episode_and_top_k_are_both_present() -> None:
    client = Mock()
    client.chat.completions.create.return_value = _json_completion(
        {"decision_type": "invented"}
    )
    request = replace(_request(), top_k=None)

    _provider(client, events=[]).decide(request)

    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert "request_scope" not in json.loads(messages[1]["content"])


def test_message_json_is_byte_deterministic_across_mapping_insertion_order() -> None:
    first_client = Mock()
    second_client = Mock()
    decision = {"decision_type": "invented"}
    first_client.chat.completions.create.return_value = _json_completion(decision)
    second_client.chat.completions.create.return_value = _json_completion(decision)
    first = _provider(first_client, events=[])
    second = _provider(second_client, events=[])
    first_definition = ({"z": 1, "a": {"y": 2, "b": 3}},)
    second_definition = ({"a": {"b": 3, "y": 2}, "z": 1},)

    first.decide(_request(tool_definitions=first_definition))
    second.decide(_request(tool_definitions=second_definition))

    first_messages = first_client.chat.completions.create.call_args.kwargs["messages"]
    second_messages = second_client.chat.completions.create.call_args.kwargs["messages"]
    assert first_messages == second_messages


@pytest.mark.parametrize(
    "decision",
    [
        {
            "decision_type": "tool_call",
            "tool_name": "search_transcript",
            "arguments": {
                "query": "synthetic",
                "episode_id": "ep_1111111111111111",
                "top_k": 3,
            },
        },
        {
            "decision_type": "final_answer",
            "final_answer": {"answer": "Synthetic answer.", "citations": []},
        },
        {
            "decision_type": "tool_call",
            "tool_name": "unregistered_tool",
            "arguments": {},
        },
        {"decision_type": "final_answer", "final_answer": {"answer": "missing"}},
        ["valid JSON", "invalid Day 22 schema"],
    ],
    ids=[
        "tool_call",
        "final_answer",
        "unknown_tool",
        "invalid_schema",
        "non_object_json",
    ],
)
def test_every_valid_json_value_is_returned_unmodified_for_day22_validation(
    decision: object,
) -> None:
    client = Mock()
    client.chat.completions.create.return_value = _json_completion(decision)

    result = _provider(client, events=[]).decide(_request())

    assert result == decision
    client.chat.completions.create.assert_called_once()


@pytest.mark.parametrize(
    "response",
    [
        _completion("not JSON"),
        _completion('{"decision_type":"tool_call","decision_type":"final_answer"}'),
        _completion('{"score":NaN}'),
        _completion("{}", finish_reason="length"),
        SimpleNamespace(choices=[], usage=None),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="{}",
                        tool_calls=[{"unsafe": True}],
                        refusal=None,
                    ),
                )
            ],
            usage=None,
        ),
    ],
    ids=[
        "invalid_json",
        "duplicate_key",
        "nonstandard_number",
        "truncated",
        "missing_choice",
        "unexpected_native_tool_call",
    ],
)
def test_malformed_provider_responses_have_one_fixed_safe_failure(
    response: object,
) -> None:
    client = Mock()
    client.chat.completions.create.return_value = response
    events: list[ProviderCallEvent] = []

    with pytest.raises(ProviderAdapterError) as captured:
        _provider(client, events=events).decide(_request())

    assert captured.value.category == "malformed_response"
    assert str(captured.value) == "模型供应商返回了无效响应。"
    client.chat.completions.create.assert_called_once()
    assert len(events) == 1
    assert events[0].outcome == "failure"
    assert events[0].failure_category == "malformed_response"


@pytest.mark.parametrize(
    ("sdk_error", "expected"),
    [
        (
            AuthenticationError(
                f"bad key {SECRET}",
                response=_http_response(401),
                body=None,
            ),
            "authentication_permission",
        ),
        (
            PermissionDeniedError(
                f"forbidden key {SECRET}",
                response=_http_response(403),
                body=None,
            ),
            "authentication_permission",
        ),
        (
            BadRequestError(
                f"bad prompt {QUESTION} {SECRET}",
                response=_http_response(400),
                body=None,
            ),
            "invalid_request",
        ),
        (APITimeoutError(_http_request()), "timeout"),
        (
            RateLimitError(
                f"limited {SECRET}",
                response=_http_response(429),
                body=None,
            ),
            "rate_limit",
        ),
        (
            APIConnectionError(
                message=f"connection {SECRET}",
                request=_http_request(),
            ),
            "connection",
        ),
        (
            APIStatusError(
                f"upstream {SECRET}",
                response=_http_response(503),
                body=None,
            ),
            "provider_5xx",
        ),
        (OpenAIError(f"unknown sdk failure {SECRET}"), "unknown_provider_failure"),
        (RuntimeError(f"foreign failure {SECRET}"), "unknown_provider_failure"),
    ],
)
def test_provider_exceptions_are_classified_redacted_and_never_retried(
    sdk_error: Exception,
    expected: ProviderFailureCategory,
) -> None:
    client = Mock()
    client.chat.completions.create.side_effect = sdk_error
    events: list[ProviderCallEvent] = []

    with pytest.raises(ProviderAdapterError) as captured:
        _provider(client, events=events).decide(_request())

    client.chat.completions.create.assert_called_once()
    assert captured.value.category == expected
    assert SECRET not in str(captured.value)
    assert QUESTION not in str(captured.value)
    assert len(events) == 1
    event = events[0]
    assert event.failure_category == expected
    assert event.attempt_number == 1
    assert event.retry_count == 0
    serialized = event.model_dump_json()
    assert SECRET not in serialized
    assert QUESTION not in serialized


def test_success_event_records_only_real_usage_and_cost_remains_unknown() -> None:
    client = Mock()
    client.chat.completions.create.return_value = _json_completion(
        {"decision_type": "invented"},
        usage=SimpleNamespace(
            prompt_tokens=101,
            completion_tokens=17,
            total_tokens=118,
        ),
    )
    events: list[ProviderCallEvent] = []

    _provider(client, events=events).decide(_request())

    assert events == [
        ProviderCallEvent(
            occurred_at=OCCURRED_AT,
            provider="openai_compatible",
            model=MODEL,
            latency_ms=125.0,
            outcome="success",
            input_tokens=101,
            output_tokens=17,
            total_tokens=118,
        )
    ]
    event = events[0]
    assert event.cost == "unknown"
    assert set(event.model_dump()) == {
        "event",
        "occurred_at",
        "provider",
        "model",
        "stage",
        "latency_ms",
        "outcome",
        "failure_category",
        "attempt_number",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "retry_count",
        "cost",
    }


def test_missing_or_partial_usage_is_not_estimated() -> None:
    client = Mock()
    client.chat.completions.create.return_value = _json_completion(
        {"decision_type": "invented"},
        usage=SimpleNamespace(
            prompt_tokens=101,
            completion_tokens=None,
            total_tokens=118,
        ),
    )
    events: list[ProviderCallEvent] = []

    _provider(client, events=events).decide(_request())

    assert events[0].input_tokens is None
    assert events[0].output_tokens is None
    assert events[0].total_tokens is None


def test_unmeasurable_clock_records_unknown_latency_instead_of_zero() -> None:
    client = Mock()
    client.chat.completions.create.return_value = _json_completion(
        {"decision_type": "invented"}
    )
    events: list[ProviderCallEvent] = []
    provider = OpenAICompatibleAgentModel(
        client,
        config=OpenAICompatibleAgentModelConfig(
            model=MODEL,
            timeout_seconds=12.5,
        ),
        clock=lambda: float("nan"),
        utc_clock=lambda: OCCURRED_AT,
        event_sink=events.append,
    )

    provider.decide(_request())

    assert events[0].latency_ms is None
    assert '"latency_ms":null' in events[0].model_dump_json()


def test_metrics_sink_failure_cannot_change_success_or_add_provider_calls() -> None:
    decision = {"decision_type": "invented"}
    client = Mock()
    client.chat.completions.create.return_value = _json_completion(decision)
    sink_calls = 0

    def broken_sink(_event: ProviderCallEvent) -> None:
        nonlocal sink_calls
        sink_calls += 1
        raise RuntimeError(f"sink leaked {SECRET}")

    result = _provider(client, sink=broken_sink).decide(_request())

    assert result == decision
    assert sink_calls == 1
    client.chat.completions.create.assert_called_once()


def test_input_character_limit_fails_before_any_provider_attempt() -> None:
    client = Mock()
    oversized = ({"schema": "x" * MAX_PROVIDER_INPUT_CHARACTERS},)

    with pytest.raises(ProviderAdapterError) as captured:
        _provider(client, events=[]).decide(_request(tool_definitions=oversized))

    assert captured.value.category == "invalid_request"
    client.chat.completions.create.assert_not_called()


def test_event_schema_rejects_non_utc_extra_fields_and_partial_usage() -> None:
    valid = {
        "occurred_at": OCCURRED_AT,
        "provider": "openai_compatible",
        "model": MODEL,
        "latency_ms": 1.0,
        "outcome": "success",
    }
    invalid = [
        {**valid, "occurred_at": datetime(2026, 8, 25, 8, 0)},
        {**valid, "prompt": SECRET},
        {**valid, "input_tokens": 1},
        {**valid, "outcome": "failure", "failure_category": None},
    ]

    for payload in invalid:
        with pytest.raises(ValidationError):
            ProviderCallEvent.model_validate(payload)


def test_json_line_writer_serializes_one_strict_event_without_content() -> None:
    stream = StringIO()
    event = ProviderCallEvent(
        occurred_at=OCCURRED_AT,
        provider="openai_compatible",
        model=MODEL,
        latency_ms=1.25,
        outcome="failure",
        failure_category="rate_limit",
    )

    write_provider_call_event(event, stream=stream)

    lines = stream.getvalue().splitlines()
    assert len(lines) == 1
    assert ProviderCallEvent.model_validate_json(lines[0]) == event
    assert SECRET not in lines[0]
    assert QUESTION not in lines[0]
