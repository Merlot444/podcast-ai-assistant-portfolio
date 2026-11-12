"""Offline Day 22 bounded Agent workflow and security tests."""

from __future__ import annotations

import ast
import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.agent import (
    ALLOWED_STATE_TRANSITIONS,
    AgentConfig,
    AgentFailure,
    AgentModelRequest,
    AgentSuccess,
    AgentWorkflow,
)
from app.agent.schemas import MAX_AGENT_TRACE_ENTRIES
from app.schemas.retriever import RetrievedCandidate
from app.schemas.transcript import ProcessedTranscript, TranscriptSegment
from app.tooling import ToolDependencies, ToolDispatcher, ToolHandlers


EPISODE_ID = "ep_1111111111111111"
OTHER_EPISODE_ID = "ep_2222222222222222"
QUERY = "synthetic agent question"
PRIVATE_PATH = "/private/transcripts/day22-secret.json"
SEGMENT_TEXT = "Synthetic transcript evidence that must not enter the trace."


def _transcript() -> ProcessedTranscript:
    return ProcessedTranscript(
        episode_id=EPISODE_ID,
        source_file=PRIVATE_PATH,
        source_format="json",
        segments=[
            TranscriptSegment(
                segment_id=f"{EPISODE_ID}:{index:06d}",
                segment_index=index,
                text=f"{SEGMENT_TEXT} Item {index}.",
                source_file=PRIVATE_PATH,
                speaker="Guest",
                section_title="Synthetic section",
                start_time=float(index),
                end_time=float(index + 1),
            )
            for index in range(3)
        ],
    )


def _candidate(
    number: int = 1,
    *,
    segment_index: int = 0,
) -> RetrievedCandidate:
    return RetrievedCandidate(
        rank=number,
        chunk_id=f"chunk_{number:016x}",
        source_segment_ids=(f"{EPISODE_ID}:{segment_index:06d}",),
        score=1.0 / number,
        score_semantics="synthetic_score",
        retriever_name="fake",
    )


class FakeRetriever:
    name = "fake"

    def __init__(
        self,
        candidates: Sequence[RetrievedCandidate] = (),
        *,
        failure: Exception | None = None,
    ) -> None:
        self.candidates = list(candidates)
        self.failure = failure
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> list[RetrievedCandidate]:
        self.calls.append((query, top_k))
        if self.failure is not None:
            raise self.failure
        return self.candidates[:top_k]


class CallIdFactory:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self) -> str:
        self.count += 1
        return f"call_{self.count:032x}"


class RaisingCallIdFactory:
    def __call__(self) -> str:
        raise RuntimeError(f"{PRIVATE_PATH} SECRET call-id failure {QUERY}")


class InvalidCallIdFactory:
    def __call__(self) -> str:
        return f"invalid-call-id SECRET {PRIVATE_PATH}"


class RaisingRecorder:
    def record(self, record) -> None:  # noqa: ANN001
        del record
        raise RuntimeError(f"{PRIVATE_PATH} SECRET recorder failure {QUERY}")


class RaisingCanonicalDispatcher(ToolDispatcher):
    def canonical_call_sha256(
        self,
        tool_name: object,
        arguments: object,
    ) -> str | None:
        del tool_name, arguments
        raise RuntimeError(f"{PRIVATE_PATH} SECRET canonical failure {QUERY}")


class ScriptedModel:
    def __init__(self, decisions: Sequence[object]) -> None:
        self.decisions = list(decisions)
        self.requests: list[AgentModelRequest] = []

    def decide(self, request: AgentModelRequest) -> object:
        self.requests.append(request)
        if len(self.requests) > len(self.decisions):
            raise AssertionError("Fake Model 收到计划外调用")
        decision = self.decisions[len(self.requests) - 1]
        if isinstance(decision, Exception):
            raise decision
        return decision


def _tool_call(
    tool_name: str,
    arguments: dict[str, object],
) -> dict[str, object]:
    return {
        "decision_type": "tool_call",
        "tool_name": tool_name,
        "arguments": arguments,
    }


def _final(
    answer: str = "A safe synthetic answer.",
    *,
    citations: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    return {
        "decision_type": "final_answer",
        "final_answer": {
            "answer": answer,
            "citations": [] if citations is None else citations,
        },
    }


def _search_call(
    *,
    query: str = QUERY,
    top_k: object = 1,
) -> dict[str, object]:
    return _tool_call(
        "search_transcript",
        {"query": query, "episode_id": EPISODE_ID, "top_k": top_k},
    )


def _citation(
    kind: str,
    reference_id: str,
    *,
    episode_id: str = EPISODE_ID,
) -> dict[str, str]:
    return {
        "kind": kind,
        "episode_id": episode_id,
        "reference_id": reference_id,
    }


def _workflow(
    decisions: Sequence[object],
    *,
    candidates: Sequence[RetrievedCandidate] | None = None,
    retriever_failure: Exception | None = None,
    max_steps: int = 4,
) -> tuple[AgentWorkflow, ScriptedModel, FakeRetriever]:
    retriever = FakeRetriever(
        [_candidate()] if candidates is None else candidates,
        failure=retriever_failure,
    )
    dependencies = ToolDependencies(
        transcripts={EPISODE_ID: _transcript()},
        retrievers={EPISODE_ID: retriever},
    )
    dispatcher = ToolDispatcher(
        ToolHandlers(dependencies),
        call_id_factory=CallIdFactory(),
    )
    model = ScriptedModel(decisions)
    workflow = AgentWorkflow(
        model=model,
        dispatcher=dispatcher,
        config=AgentConfig(max_steps=max_steps),
    )
    return workflow, model, retriever


def _workflow_with_dispatcher(
    decisions: Sequence[object],
    dispatcher: ToolDispatcher,
) -> tuple[AgentWorkflow, ScriptedModel]:
    model = ScriptedModel(decisions)
    return (
        AgentWorkflow(model=model, dispatcher=dispatcher),
        model,
    )


def test_direct_answer_needs_no_tool_or_citation() -> None:
    workflow, model, retriever = _workflow([_final("Direct safe answer.")])

    result = workflow.run("What is a general concept?")

    assert isinstance(result, AgentSuccess)
    assert result.answer.answer == "Direct safe answer."
    assert result.answer.citations == []
    assert result.model_call_count == 1
    assert result.tool_call_count == 0
    assert retriever.calls == []
    assert len(model.requests) == 1
    assert [entry.state for entry in result.trace.entries] == [
        "received",
        "requesting_model",
        "validating_model_output",
        "validating_final_answer",
        "succeeded",
    ]


def test_optional_grounding_policy_rejects_direct_answer_without_dispatch() -> None:
    workflow, model, retriever = _workflow([_final("Unsupported direct answer.")])
    workflow = AgentWorkflow(
        model=model,
        dispatcher=workflow.dispatcher,
        config=AgentConfig(require_tool_grounding=True),
    )

    result = workflow.run(QUERY, episode_id=EPISODE_ID, top_k=1)

    assert isinstance(result, AgentFailure)
    assert result.termination_reason == "invalid_citation"
    assert result.tool_call_count == 0
    assert retriever.calls == []


def test_one_tool_call_then_answer_accepts_returned_chunk_citation() -> None:
    decisions = [
        _search_call(),
        _final(citations=[_citation("chunk", "chunk_0000000000000001")]),
    ]
    workflow, model, retriever = _workflow(decisions)

    result = workflow.run(QUERY)

    assert isinstance(result, AgentSuccess)
    assert result.model_call_count == 2
    assert result.tool_call_count == 1
    assert retriever.calls == [(QUERY, 1)]
    assert len(model.requests[1].tool_results) == 1
    assert (
        model.requests[1].tool_results[0].call_id
        == "call_00000000000000000000000000000001"
    )
    assert result.trace.entries[-2].citation_validation == "valid"
    assert result.trace.entries[-1].termination_reason == "completed"


def test_request_scope_is_forwarded_and_exact_search_scope_is_enforced() -> None:
    decisions = [
        _search_call(top_k=3),
        _final(citations=[_citation("chunk", "chunk_0000000000000001")]),
    ]
    workflow, model, retriever = _workflow(decisions)

    result = workflow.run(QUERY, episode_id=EPISODE_ID, top_k=3)

    assert isinstance(result, AgentSuccess)
    assert retriever.calls == [(QUERY, 3)]
    assert [(request.episode_id, request.top_k) for request in model.requests] == [
        (EPISODE_ID, 3),
        (EPISODE_ID, 3),
    ]


@pytest.mark.parametrize(
    "decision",
    [
        _search_call(top_k=2),
        _tool_call("get_episode_metadata", {"episode_id": OTHER_EPISODE_ID}),
    ],
    ids=["different_top_k", "different_episode"],
)
def test_request_scope_mismatch_stops_before_dispatch(
    decision: dict[str, object],
) -> None:
    workflow, model, retriever = _workflow([decision])

    result = workflow.run(QUERY, episode_id=EPISODE_ID, top_k=1)

    assert isinstance(result, AgentFailure)
    assert result.termination_reason == "invalid_arguments"
    assert result.tool_call_count == 0
    assert len(model.requests) == 1
    assert retriever.calls == []


@pytest.mark.parametrize(
    ("episode_id", "top_k"),
    [
        (None, 1),
        (EPISODE_ID, None),
        ("episode-invalid", 1),
        (EPISODE_ID, True),
        (EPISODE_ID, 9),
    ],
)
def test_invalid_request_scope_fails_before_model_or_dispatcher(
    episode_id: object,
    top_k: object,
) -> None:
    workflow, model, retriever = _workflow([_final()])

    result = workflow.run(QUERY, episode_id=episode_id, top_k=top_k)

    assert isinstance(result, AgentFailure)
    assert result.termination_reason == "invalid_request"
    assert result.model_call_count == result.tool_call_count == 0
    assert model.requests == []
    assert retriever.calls == []


def test_legal_multi_step_search_context_and_answer() -> None:
    segment_id = f"{EPISODE_ID}:000000"
    decisions = [
        _search_call(),
        _tool_call(
            "get_segment_context",
            {
                "episode_id": EPISODE_ID,
                "segment_id": segment_id,
                "context_before": 0,
                "context_after": 1,
            },
        ),
        _final(citations=[_citation("segment", segment_id)]),
    ]
    workflow, model, retriever = _workflow(decisions)

    result = workflow.run(QUERY)

    assert isinstance(result, AgentSuccess)
    assert result.model_call_count == 3
    assert result.tool_call_count == 2
    assert retriever.calls == [(QUERY, 1)]
    assert [len(request.tool_results) for request in model.requests] == [0, 1, 2]
    assert [
        entry.state
        for entry in result.trace.entries
        if entry.state in {"dispatching_tool", "saving_tool_result"}
    ] == [
        "dispatching_tool",
        "saving_tool_result",
        "dispatching_tool",
        "saving_tool_result",
    ]


def test_invalid_arguments_never_call_retriever() -> None:
    workflow, model, retriever = _workflow([_search_call(top_k=True)])

    result = workflow.run(QUERY)

    assert isinstance(result, AgentFailure)
    assert result.termination_reason == "invalid_arguments"
    assert result.error.message == "模型提供的工具参数无效。"
    assert result.model_call_count == 1
    assert result.tool_call_count == 1
    assert len(model.requests) == 1
    assert retriever.calls == []


def test_tool_failure_returns_safe_deterministic_fallback() -> None:
    secret = f"{PRIVATE_PATH} traceback SECRET {QUERY}"
    workflow, _, retriever = _workflow(
        [_search_call()],
        retriever_failure=RuntimeError(secret),
    )

    result = workflow.run(QUERY)

    assert isinstance(result, AgentFailure)
    assert result.termination_reason == "tool_failed"
    assert result.error.message == "工具暂时无法完成调用。"
    assert retriever.calls == [(QUERY, 1)]
    serialized = result.model_dump_json()
    for forbidden in (secret, PRIVATE_PATH, "traceback", "SECRET", QUERY):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    ("call_id_factory", "expected_retriever_calls"),
    [
        (RaisingCallIdFactory(), []),
        (InvalidCallIdFactory(), [(QUERY, 1)]),
    ],
    ids=["factory_raises", "factory_returns_invalid_id"],
)
def test_call_id_factory_faults_are_mapped_to_safe_tool_failure(
    call_id_factory: object,
    expected_retriever_calls: list[tuple[str, int]],
) -> None:
    retriever = FakeRetriever([_candidate()])
    dependencies = ToolDependencies(
        transcripts={EPISODE_ID: _transcript()},
        retrievers={EPISODE_ID: retriever},
    )
    dispatcher = ToolDispatcher(
        ToolHandlers(dependencies),
        call_id_factory=call_id_factory,  # type: ignore[arg-type]
    )
    workflow, model = _workflow_with_dispatcher([_search_call()], dispatcher)

    result = workflow.run(QUERY)

    assert isinstance(result, AgentFailure)
    assert result.termination_reason == "tool_failed"
    assert result.error.message == "工具暂时无法完成调用。"
    assert result.model_call_count == result.tool_call_count == 1
    assert len(model.requests) == 1
    assert retriever.calls == expected_retriever_calls
    serialized = result.model_dump_json()
    for forbidden in (PRIVATE_PATH, "SECRET", "RuntimeError", QUERY):
        assert forbidden not in serialized


def test_recorder_fault_is_mapped_to_safe_tool_failure() -> None:
    retriever = FakeRetriever([_candidate()])
    dependencies = ToolDependencies(
        transcripts={EPISODE_ID: _transcript()},
        retrievers={EPISODE_ID: retriever},
    )
    dispatcher = ToolDispatcher(
        ToolHandlers(dependencies),
        recorder=RaisingRecorder(),
        call_id_factory=CallIdFactory(),
    )
    workflow, model = _workflow_with_dispatcher([_search_call()], dispatcher)

    result = workflow.run(QUERY)

    assert isinstance(result, AgentFailure)
    assert result.termination_reason == "tool_failed"
    assert result.error.message == "工具暂时无法完成调用。"
    assert result.model_call_count == result.tool_call_count == 1
    assert len(model.requests) == 1
    assert retriever.calls == [(QUERY, 1)]
    serialized = result.model_dump_json()
    for forbidden in (PRIVATE_PATH, "SECRET", "RuntimeError", QUERY):
        assert forbidden not in serialized


def test_canonical_fingerprint_fault_is_mapped_before_dispatch() -> None:
    retriever = FakeRetriever([_candidate()])
    dependencies = ToolDependencies(
        transcripts={EPISODE_ID: _transcript()},
        retrievers={EPISODE_ID: retriever},
    )
    dispatcher = RaisingCanonicalDispatcher(
        ToolHandlers(dependencies),
        call_id_factory=CallIdFactory(),
    )
    workflow, model = _workflow_with_dispatcher([_search_call()], dispatcher)

    result = workflow.run(QUERY)

    assert isinstance(result, AgentFailure)
    assert result.termination_reason == "tool_failed"
    assert result.error.message == "工具暂时无法完成调用。"
    assert result.model_call_count == 1
    assert result.tool_call_count == 0
    assert len(model.requests) == 1
    assert retriever.calls == []
    serialized = result.model_dump_json()
    for forbidden in (PRIVATE_PATH, "SECRET", "RuntimeError", QUERY):
        assert forbidden not in serialized


def test_empty_search_result_stops_without_another_model_call() -> None:
    workflow, model, retriever = _workflow(
        [_search_call(), _final()],
        candidates=[],
    )

    result = workflow.run(QUERY)

    assert isinstance(result, AgentFailure)
    assert result.termination_reason == "empty_results"
    assert result.model_call_count == 1
    assert result.tool_call_count == 1
    assert len(model.requests) == 1
    assert retriever.calls == [(QUERY, 1)]
    assert any(entry.result_status == "empty" for entry in result.trace.entries)


def test_unknown_tool_is_rejected_by_dispatcher_without_reflection() -> None:
    unsafe_name = "os.system"
    workflow, model, retriever = _workflow([_tool_call(unsafe_name, {})])

    result = workflow.run(QUERY)

    assert isinstance(result, AgentFailure)
    assert result.termination_reason == "unknown_tool"
    assert result.error.message == "模型请求了不可用的工具。"
    assert result.tool_call_count == 1
    assert len(model.requests) == 1
    assert retriever.calls == []
    assert unsafe_name not in result.model_dump_json()
    assert {entry.tool_name for entry in result.trace.entries if entry.tool_name} == {
        "unknown"
    }


def test_not_found_stops_with_existing_safe_error_semantics() -> None:
    decisions = [
        _tool_call("get_episode_metadata", {"episode_id": OTHER_EPISODE_ID}),
        _final(),
    ]
    workflow, model, retriever = _workflow(decisions)

    result = workflow.run(QUERY)

    assert isinstance(result, AgentFailure)
    assert result.termination_reason == "not_found"
    assert result.error.message == "请求的节目或片段不存在。"
    assert result.model_call_count == 1
    assert result.tool_call_count == 1
    assert len(model.requests) == 1
    assert retriever.calls == []


@pytest.mark.parametrize(
    "invalid_output",
    [
        "not an object",
        {"decision_type": "final_answer", "final_answer": {"answer": "x"}},
        {
            "decision_type": "tool_call",
            "tool_name": "search_transcript",
            "arguments": {},
            "command": "rm -rf /",
        },
        {"decision_type": "invented", "raw": "unsafe"},
    ],
)
def test_model_output_must_pass_strict_schema(invalid_output: object) -> None:
    workflow, model, retriever = _workflow([invalid_output])

    result = workflow.run(QUERY)

    assert isinstance(result, AgentFailure)
    assert result.termination_reason == "model_output_invalid"
    assert result.model_call_count == 1
    assert result.tool_call_count == 0
    assert len(model.requests) == 1
    assert retriever.calls == []


def test_identical_schema_normalized_call_executes_only_once() -> None:
    decisions = [
        _search_call(query=f"  {QUERY}  "),
        _search_call(query=QUERY),
        _final(),
    ]
    workflow, model, retriever = _workflow(decisions)

    result = workflow.run(QUERY)

    assert isinstance(result, AgentFailure)
    assert result.termination_reason == "duplicate_call"
    assert result.model_call_count == 2
    assert result.tool_call_count == 1
    assert len(model.requests) == 2
    assert retriever.calls == [(QUERY, 1)]
    assert result.trace.entries[-1].result_status == "duplicate"


def test_distinct_calls_stop_exactly_at_max_steps() -> None:
    workflow, model, retriever = _workflow(
        [_search_call(query="first"), _search_call(query="second")],
        candidates=[_candidate()],
        max_steps=2,
    )

    result = workflow.run(QUERY)

    assert isinstance(result, AgentFailure)
    assert result.termination_reason == "max_steps_exceeded"
    assert result.model_call_count == 2
    assert result.tool_call_count == 2
    assert len(model.requests) == 2
    assert retriever.calls == [("first", 1), ("second", 1)]
    assert result.trace.entries[-1].termination_reason == "max_steps_exceeded"


@pytest.mark.parametrize(
    "citation",
    [
        _citation("chunk", "chunk_0000000000000099"),
        _citation(
            "segment",
            f"{OTHER_EPISODE_ID}:000000",
            episode_id=OTHER_EPISODE_ID,
        ),
        _citation("segment", f"{EPISODE_ID}:000002"),
    ],
    ids=["nonexistent", "cross_episode", "unseen_in_run"],
)
def test_unreturned_or_cross_episode_citation_refuses_success(
    citation: dict[str, str],
) -> None:
    workflow, model, retriever = _workflow(
        [_search_call(), _final(citations=[citation])]
    )

    result = workflow.run(QUERY)

    assert isinstance(result, AgentFailure)
    assert result.termination_reason == "invalid_citation"
    assert result.error.message == "回答引用未通过校验，未返回未经验证的内容。"
    assert result.model_call_count == 2
    assert result.tool_call_count == 1
    assert len(model.requests) == 2
    assert retriever.calls == [(QUERY, 1)]
    assert result.trace.entries[-2].citation_validation == "invalid"
    assert result.trace.entries[-1].citation_validation == "invalid"


def test_tool_grounded_answer_must_include_at_least_one_valid_citation() -> None:
    workflow, _, _ = _workflow([_search_call(), _final()])

    result = workflow.run(QUERY)

    assert isinstance(result, AgentFailure)
    assert result.termination_reason == "invalid_citation"


def test_direct_answer_cannot_invent_a_citation() -> None:
    workflow, _, _ = _workflow(
        [_final(citations=[_citation("episode", EPISODE_ID)])]
    )

    result = workflow.run(QUERY)

    assert isinstance(result, AgentFailure)
    assert result.termination_reason == "invalid_citation"
    assert result.tool_call_count == 0


def test_episode_metadata_success_allows_only_its_episode_reference() -> None:
    decisions = [
        _tool_call("get_episode_metadata", {"episode_id": EPISODE_ID}),
        _final(citations=[_citation("episode", EPISODE_ID)]),
    ]
    workflow, _, retriever = _workflow(decisions)

    result = workflow.run(QUERY)

    assert isinstance(result, AgentSuccess)
    assert result.tool_call_count == 1
    assert retriever.calls == []


def test_model_exception_is_redacted_and_stops_immediately() -> None:
    secret = f"provider failed at {PRIVATE_PATH} with SECRET and {QUERY}"
    workflow, model, retriever = _workflow([RuntimeError(secret), _final()])

    result = workflow.run(QUERY)

    assert isinstance(result, AgentFailure)
    assert result.termination_reason == "model_failed"
    assert result.error.message == "模型暂时无法完成决策。"
    assert result.model_call_count == 1
    assert result.tool_call_count == 0
    assert len(model.requests) == 1
    assert retriever.calls == []
    serialized = result.model_dump_json()
    for forbidden in (secret, PRIVATE_PATH, "SECRET", QUERY):
        assert forbidden not in serialized


def test_invalid_raw_model_output_and_trace_do_not_store_sensitive_content() -> None:
    sensitive_question = "private user wording 41b7d7"
    sensitive_raw = {
        "decision_type": "invalid",
        "query": sensitive_question,
        "path": PRIVATE_PATH,
        "exception": "traceback with SECRET",
    }
    workflow, _, _ = _workflow([sensitive_raw])

    result = workflow.run(sensitive_question)

    assert isinstance(result, AgentFailure)
    serialized = result.model_dump_json()
    for forbidden in (
        sensitive_question,
        PRIVATE_PATH,
        "traceback",
        "SECRET",
    ):
        assert forbidden not in serialized
    assert '"query":' not in serialized
    assert '"path":' not in serialized
    assert '"exception":' not in serialized
    assert result.trace.question_length == len(sensitive_question)
    assert len(result.trace.question_sha256) == 64
    assert result.trace.entries[1].decision_sha256 is not None


def test_context_text_is_available_to_model_but_absent_from_audit_trace() -> None:
    segment_id = f"{EPISODE_ID}:000000"
    context_call = _tool_call(
        "get_segment_context",
        {
            "episode_id": EPISODE_ID,
            "segment_id": segment_id,
            "context_before": 0,
            "context_after": 0,
        },
    )
    invalid_final = _final(
        answer=f"Unverified raw answer {PRIVATE_PATH}",
        citations=[_citation("segment", f"{EPISODE_ID}:000002")],
    )
    workflow, model, _ = _workflow([context_call, invalid_final])

    result = workflow.run(QUERY)

    assert isinstance(result, AgentFailure)
    assert SEGMENT_TEXT in model.requests[1].tool_results[0].model_dump_json()
    serialized = result.model_dump_json()
    assert SEGMENT_TEXT not in serialized
    assert PRIVATE_PATH not in serialized
    assert "Unverified raw answer" not in serialized
    assert QUERY not in serialized


def test_invalid_question_fails_before_model_or_tool_calls() -> None:
    workflow, model, retriever = _workflow([_final()])

    result = workflow.run("   ")

    assert isinstance(result, AgentFailure)
    assert result.termination_reason == "invalid_request"
    assert result.model_call_count == result.tool_call_count == 0
    assert model.requests == []
    assert retriever.calls == []


def test_config_is_strict_and_has_a_small_hard_ceiling() -> None:
    with pytest.raises(ValidationError):
        AgentConfig(max_steps=True)
    with pytest.raises(ValidationError):
        AgentConfig(max_steps=9)
    assert AgentConfig().max_steps == 4


def test_trace_is_bounded_even_at_hard_step_limit() -> None:
    decisions = [_search_call(query=f"query-{index}") for index in range(8)]
    workflow, model, retriever = _workflow(decisions, max_steps=8)

    result = workflow.run(QUERY)

    assert isinstance(result, AgentFailure)
    assert result.termination_reason == "max_steps_exceeded"
    assert len(model.requests) == len(retriever.calls) == 8
    assert len(result.trace.entries) <= MAX_AGENT_TRACE_ENTRIES
    assert result.trace.entries[-1].sequence == len(result.trace.entries)


def test_state_transition_table_has_only_terminal_dead_ends() -> None:
    assert ALLOWED_STATE_TRANSITIONS["succeeded"] == frozenset()
    assert ALLOWED_STATE_TRANSITIONS["failed"] == frozenset()
    assert "requesting_model" in ALLOWED_STATE_TRANSITIONS["received"]
    assert "dispatching_tool" in ALLOWED_STATE_TRANSITIONS[
        "validating_model_output"
    ]
    assert "validating_final_answer" in ALLOWED_STATE_TRANSITIONS[
        "validating_model_output"
    ]


def test_model_request_contains_schema_definitions_not_executable_handlers() -> None:
    workflow, model, _ = _workflow([_final()])

    result = workflow.run(QUERY)

    assert isinstance(result, AgentSuccess)
    request = model.requests[0]
    names = [
        definition["function"]["name"]
        for definition in request.tool_definitions
    ]
    assert names == [
        "search_transcript",
        "get_episode_metadata",
        "get_segment_context",
    ]
    assert all(
        definition["type"] == "function" for definition in request.tool_definitions
    )
    assert "handler" not in json.dumps(request.tool_definitions, sort_keys=True)


def test_agent_runtime_has_no_dynamic_execution_or_file_shell_capability() -> None:
    agent_root = Path(__file__).parents[1] / "app" / "agent"
    forbidden_names = {
        "open",
        "eval",
        "exec",
        "compile",
        "__import__",
        "getattr",
        "setattr",
    }
    forbidden_attributes = {
        "import_module",
        "open",
        "popen",
        "Popen",
        "read_bytes",
        "read_text",
        "run",
        "system",
        "write_bytes",
        "write_text",
    }
    for filename in ("model.py", "workflow.py"):
        tree = ast.parse((agent_root / filename).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_names
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_attributes
