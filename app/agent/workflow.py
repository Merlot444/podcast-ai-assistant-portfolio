"""Bounded Day 22 Agent state machine over the Day 21 Dispatcher."""

from __future__ import annotations

import re
from collections.abc import Callable
from time import perf_counter
from types import MappingProxyType
from typing import Literal, cast

from app.agent.citations import (
    ReferenceKey,
    citations_are_valid,
    references_from_tool_success,
)
from app.agent.model import AgentModel, AgentModelRequest
from app.agent.schemas import (
    MAX_AGENT_QUESTION_CHARACTERS,
    MAX_AGENT_TRACE_ENTRIES,
    MODEL_DECISION_ADAPTER,
    AgentCitationValidation,
    AgentConfig,
    AgentDecisionType,
    AgentError,
    AgentErrorCode,
    AgentFailure,
    AgentFinalAnswer,
    AgentResultStatus,
    AgentRunResult,
    AgentRunTrace,
    AgentState,
    AgentSuccess,
    AgentTerminationReason,
    AgentTraceEntry,
    ModelFinalAnswerDecision,
)
from app.tooling.definitions import export_openai_tool_definitions
from app.tooling.dispatcher import ToolDispatcher
from app.tooling.recording import stable_json_sha256, text_sha256
from app.tooling.schemas import (
    EPISODE_ID_PATTERN,
    RecordedToolName,
    SearchTranscriptData,
    SegmentContextData,
    ToolFailure,
    ToolSuccess,
)


Clock = Callable[[], float]
AGENT_ERROR_MESSAGES = MappingProxyType(
    {
        "invalid_request": "问题无效，无法开始处理。",
        "model_failed": "模型暂时无法完成决策。",
        "model_output_invalid": "模型返回的决策格式无效。",
        "unknown_tool": "模型请求了不可用的工具。",
        "invalid_arguments": "模型提供的工具参数无效。",
        "not_found": "请求的节目或片段不存在。",
        "tool_failed": "工具暂时无法完成调用。",
        "empty_results": "未找到可用于回答的相关内容。",
        "duplicate_call": "检测到重复工具调用，已安全停止。",
        "max_steps_exceeded": "已达到最大处理步骤，未能生成最终回答。",
        "invalid_citation": "回答引用未通过校验，未返回未经验证的内容。",
    }
)

ALLOWED_STATE_TRANSITIONS = MappingProxyType(
    {
        "received": frozenset({"requesting_model", "failed"}),
        "requesting_model": frozenset({"validating_model_output", "failed"}),
        "validating_model_output": frozenset(
            {"dispatching_tool", "validating_final_answer", "failed"}
        ),
        "dispatching_tool": frozenset({"saving_tool_result", "failed"}),
        "saving_tool_result": frozenset({"requesting_model", "failed"}),
        "validating_final_answer": frozenset({"succeeded", "failed"}),
        "succeeded": frozenset(),
        "failed": frozenset(),
    }
)


class AgentWorkflow:
    """Run one finite model/tool workflow with application-owned validation."""

    def __init__(
        self,
        *,
        model: AgentModel,
        dispatcher: ToolDispatcher,
        config: AgentConfig | None = None,
        clock: Clock = perf_counter,
    ) -> None:
        self.model = model
        self.dispatcher = dispatcher
        self.config = config or AgentConfig()
        self.clock = clock
        self._registered_tool_names = frozenset(dispatcher.registry)
        definitions = export_openai_tool_definitions()
        self._tool_definitions = tuple(
            cast(dict[str, object], definition) for definition in definitions
        )

    def run(
        self,
        question: object,
        *,
        episode_id: object | None = None,
        top_k: object | None = None,
    ) -> AgentRunResult:
        """Run until a verified answer or one deterministic safe failure."""
        question_length = len(question) if isinstance(question, str) else 0
        question_sha256 = (
            text_sha256(question)
            if isinstance(question, str)
            else stable_json_sha256(question)
        )
        trace = _TraceBuilder(
            question_length=question_length,
            question_sha256=question_sha256,
            max_steps=self.config.max_steps,
        )
        trace.record(model_step=0, duration_ms=0.0)

        cleaned_question = _clean_question(question)
        request_scope = _clean_request_scope(episode_id=episode_id, top_k=top_k)
        if cleaned_question is None or request_scope is False:
            return self._failure(
                trace=trace,
                reason="invalid_request",
                model_call_count=0,
                tool_call_count=0,
            )

        scoped_episode_id, scoped_top_k = (
            (None, None) if request_scope is None else request_scope
        )
        model_results: list[ToolSuccess] = []
        allowed_references: set[ReferenceKey] = set()
        seen_calls: set[str] = set()
        model_call_count = 0
        tool_call_count = 0

        for model_step in range(1, self.config.max_steps + 1):
            trace.transition("requesting_model")
            model_started = self.clock()
            model_call_count += 1
            try:
                raw_decision = self.model.decide(
                    AgentModelRequest(
                        question=cleaned_question,
                        step_number=model_step,
                        max_steps=self.config.max_steps,
                        tool_definitions=self._tool_definitions,
                        tool_results=tuple(model_results),
                        episode_id=scoped_episode_id,
                        top_k=scoped_top_k,
                    )
                )
            except Exception:
                trace.record(
                    model_step=model_step,
                    duration_ms=_elapsed_ms(self.clock(), model_started),
                )
                return self._failure(
                    trace=trace,
                    reason="model_failed",
                    model_call_count=model_call_count,
                    tool_call_count=tool_call_count,
                )

            decision_sha256 = stable_json_sha256(raw_decision)
            trace.record(
                model_step=model_step,
                decision_sha256=decision_sha256,
                duration_ms=_elapsed_ms(self.clock(), model_started),
            )
            trace.transition("validating_model_output")
            validation_started = self.clock()
            try:
                decision = MODEL_DECISION_ADAPTER.validate_python(raw_decision)
            except Exception:
                trace.record(
                    model_step=model_step,
                    decision_type="invalid",
                    decision_sha256=decision_sha256,
                    duration_ms=_elapsed_ms(self.clock(), validation_started),
                )
                return self._failure(
                    trace=trace,
                    reason="model_output_invalid",
                    model_call_count=model_call_count,
                    tool_call_count=tool_call_count,
                )

            if isinstance(decision, ModelFinalAnswerDecision):
                trace.record(
                    model_step=model_step,
                    decision_type="final_answer",
                    decision_sha256=decision_sha256,
                    duration_ms=_elapsed_ms(self.clock(), validation_started),
                )
                return self._finish_answer(
                    decision.final_answer,
                    trace=trace,
                    model_step=model_step,
                    model_call_count=model_call_count,
                    tool_call_count=tool_call_count,
                    has_tool_results=bool(model_results),
                    allowed_references=allowed_references,
                )

            argument_sha256 = stable_json_sha256(decision.arguments)
            recorded_tool_name = _recorded_tool_name(
                decision.tool_name,
                self._registered_tool_names,
            )
            trace.record(
                model_step=model_step,
                decision_type="tool_call",
                decision_sha256=decision_sha256,
                argument_sha256=argument_sha256,
                tool_name=recorded_tool_name,
                duration_ms=_elapsed_ms(self.clock(), validation_started),
            )
            if not _tool_call_is_within_request_scope(
                tool_name=decision.tool_name,
                arguments=decision.arguments,
                episode_id=scoped_episode_id,
                top_k=scoped_top_k,
                registered_tool_names=self._registered_tool_names,
            ):
                return self._failure(
                    trace=trace,
                    reason="invalid_arguments",
                    model_call_count=model_call_count,
                    tool_call_count=tool_call_count,
                    decision_type="tool_call",
                    argument_sha256=argument_sha256,
                    tool_name=recorded_tool_name,
                    result_status="failure",
                )
            try:
                fingerprint = self.dispatcher.canonical_call_sha256(
                    decision.tool_name,
                    decision.arguments,
                )
            except Exception:
                return self._failure(
                    trace=trace,
                    reason="tool_failed",
                    model_call_count=model_call_count,
                    tool_call_count=tool_call_count,
                    decision_type="tool_call",
                    argument_sha256=argument_sha256,
                    tool_name=recorded_tool_name,
                    result_status="failure",
                )
            if fingerprint is not None and fingerprint in seen_calls:
                return self._failure(
                    trace=trace,
                    reason="duplicate_call",
                    model_call_count=model_call_count,
                    tool_call_count=tool_call_count,
                    decision_type="tool_call",
                    argument_sha256=fingerprint,
                    tool_name=recorded_tool_name,
                    result_status="duplicate",
                )

            trace.transition("dispatching_tool")
            dispatch_started = self.clock()
            tool_call_count += 1
            try:
                result = self.dispatcher.dispatch(
                    decision.tool_name,
                    decision.arguments,
                )
            except Exception:
                trace.record(
                    model_step=model_step,
                    decision_type="tool_call",
                    argument_sha256=fingerprint or argument_sha256,
                    tool_name=recorded_tool_name,
                    result_status="failure",
                    result_count=0,
                    duration_ms=_elapsed_ms(self.clock(), dispatch_started),
                )
                return self._failure(
                    trace=trace,
                    reason="tool_failed",
                    model_call_count=model_call_count,
                    tool_call_count=tool_call_count,
                )
            if fingerprint is not None:
                seen_calls.add(fingerprint)
            result_count = _result_count(result)
            trace.record(
                model_step=model_step,
                decision_type="tool_call",
                argument_sha256=(
                    fingerprint
                    if fingerprint is not None
                    else stable_json_sha256(decision.arguments)
                ),
                call_id=result.call_id,
                tool_name=result.tool_name,
                result_status=(
                    "failure" if isinstance(result, ToolFailure) else "success"
                ),
                result_count=result_count,
                duration_ms=_elapsed_ms(self.clock(), dispatch_started),
            )

            trace.transition("saving_tool_result")
            save_started = self.clock()
            if isinstance(result, ToolFailure):
                trace.record(
                    model_step=model_step,
                    decision_type="tool_call",
                    argument_sha256=fingerprint,
                    call_id=result.call_id,
                    tool_name=result.tool_name,
                    result_status="failure",
                    result_count=0,
                    duration_ms=_elapsed_ms(self.clock(), save_started),
                )
                return self._failure(
                    trace=trace,
                    reason=cast(AgentErrorCode, result.error.code),
                    model_call_count=model_call_count,
                    tool_call_count=tool_call_count,
                )

            if (
                isinstance(result.data, SearchTranscriptData)
                and result.data.result_count == 0
            ):
                trace.record(
                    model_step=model_step,
                    decision_type="tool_call",
                    argument_sha256=fingerprint,
                    call_id=result.call_id,
                    tool_name=result.tool_name,
                    result_status="empty",
                    result_count=0,
                    duration_ms=_elapsed_ms(self.clock(), save_started),
                )
                return self._failure(
                    trace=trace,
                    reason="empty_results",
                    model_call_count=model_call_count,
                    tool_call_count=tool_call_count,
                )

            try:
                result_references = references_from_tool_success(result)
            except Exception:
                trace.record(
                    model_step=model_step,
                    decision_type="tool_call",
                    argument_sha256=fingerprint,
                    call_id=result.call_id,
                    tool_name=result.tool_name,
                    result_status="failure",
                    result_count=result_count,
                    duration_ms=_elapsed_ms(self.clock(), save_started),
                )
                return self._failure(
                    trace=trace,
                    reason="tool_failed",
                    model_call_count=model_call_count,
                    tool_call_count=tool_call_count,
                )
            model_results.append(result)
            allowed_references.update(result_references)
            trace.record(
                model_step=model_step,
                decision_type="tool_call",
                argument_sha256=fingerprint,
                call_id=result.call_id,
                tool_name=result.tool_name,
                result_status="success",
                result_count=result_count,
                duration_ms=_elapsed_ms(self.clock(), save_started),
            )
            if model_step == self.config.max_steps:
                return self._failure(
                    trace=trace,
                    reason="max_steps_exceeded",
                    model_call_count=model_call_count,
                    tool_call_count=tool_call_count,
                )

        raise RuntimeError("Agent 状态机到达不可达分支")

    def _finish_answer(
        self,
        answer: AgentFinalAnswer,
        *,
        trace: _TraceBuilder,
        model_step: int,
        model_call_count: int,
        tool_call_count: int,
        has_tool_results: bool,
        allowed_references: set[ReferenceKey],
    ) -> AgentRunResult:
        trace.transition("validating_final_answer")
        started = self.clock()
        citations_are_valid_result = citations_are_valid(
            answer.citations,
            has_tool_results=has_tool_results,
            allowed_references=allowed_references,
        )
        if self.config.require_tool_grounding and not has_tool_results:
            citations_are_valid_result = False
        trace.record(
            model_step=model_step,
            decision_type="final_answer",
            citation_validation=(
                "valid" if citations_are_valid_result else "invalid"
            ),
            duration_ms=_elapsed_ms(self.clock(), started),
        )
        if not citations_are_valid_result:
            return self._failure(
                trace=trace,
                reason="invalid_citation",
                model_call_count=model_call_count,
                tool_call_count=tool_call_count,
                decision_type="final_answer",
                citation_validation="invalid",
            )

        trace.transition("succeeded")
        trace.record(
            model_step=model_step,
            decision_type="final_answer",
            citation_validation="valid",
            termination_reason="completed",
            duration_ms=0.0,
        )
        return AgentSuccess(
            answer=answer,
            model_call_count=model_call_count,
            tool_call_count=tool_call_count,
            trace=trace.build(),
        )

    def _failure(
        self,
        *,
        trace: _TraceBuilder,
        reason: AgentErrorCode,
        model_call_count: int,
        tool_call_count: int,
        decision_type: AgentDecisionType | None = None,
        argument_sha256: str | None = None,
        tool_name: RecordedToolName | None = None,
        result_status: AgentResultStatus | None = None,
        citation_validation: AgentCitationValidation = "not_checked",
    ) -> AgentFailure:
        trace.transition("failed")
        trace.record(
            model_step=model_call_count,
            decision_type=decision_type,
            argument_sha256=argument_sha256,
            tool_name=tool_name,
            result_status=result_status,
            citation_validation=citation_validation,
            termination_reason=cast(AgentTerminationReason, reason),
            duration_ms=0.0,
        )
        return AgentFailure(
            error=AgentError(code=reason, message=AGENT_ERROR_MESSAGES[reason]),
            termination_reason=reason,
            model_call_count=model_call_count,
            tool_call_count=tool_call_count,
            trace=trace.build(),
        )


class _TraceBuilder:
    """Enforce transition legality and the fixed in-memory trace ceiling."""

    def __init__(
        self,
        *,
        question_length: int,
        question_sha256: str,
        max_steps: int,
    ) -> None:
        self.question_length = question_length
        self.question_sha256 = question_sha256
        self.max_steps = max_steps
        self.state: AgentState = "received"
        self.entries: list[AgentTraceEntry] = []

    def transition(self, next_state: AgentState) -> None:
        if next_state not in ALLOWED_STATE_TRANSITIONS[self.state]:
            raise RuntimeError(f"非法 Agent 状态转换：{self.state} -> {next_state}")
        self.state = next_state

    def record(
        self,
        *,
        model_step: int,
        duration_ms: float,
        decision_type: AgentDecisionType | None = None,
        decision_sha256: str | None = None,
        argument_sha256: str | None = None,
        call_id: str | None = None,
        tool_name: RecordedToolName | None = None,
        result_status: AgentResultStatus | None = None,
        result_count: int | None = None,
        citation_validation: AgentCitationValidation = "not_checked",
        termination_reason: AgentTerminationReason | None = None,
    ) -> None:
        if len(self.entries) >= MAX_AGENT_TRACE_ENTRIES:
            raise RuntimeError("Agent trace 超过固定内存边界")
        self.entries.append(
            AgentTraceEntry(
                sequence=len(self.entries) + 1,
                model_step=model_step,
                state=self.state,
                decision_type=decision_type,
                decision_sha256=decision_sha256,
                argument_sha256=argument_sha256,
                call_id=call_id,
                tool_name=tool_name,
                result_status=result_status,
                result_count=result_count,
                citation_validation=citation_validation,
                termination_reason=termination_reason,
                duration_ms=max(0.0, duration_ms),
            )
        )

    def build(self) -> AgentRunTrace:
        return AgentRunTrace(
            question_length=self.question_length,
            question_sha256=self.question_sha256,
            max_steps=self.max_steps,
            entries=list(self.entries),
        )


def _clean_question(question: object) -> str | None:
    if not isinstance(question, str):
        return None
    cleaned = question.strip()
    if not cleaned or len(cleaned) > MAX_AGENT_QUESTION_CHARACTERS:
        return None
    return cleaned


def _clean_request_scope(
    *,
    episode_id: object | None,
    top_k: object | None,
) -> tuple[str, int] | None | Literal[False]:
    """Validate an optional trusted HTTP scope without changing legacy runs."""
    if episode_id is None and top_k is None:
        return None
    if (
        type(episode_id) is not str
        or re.fullmatch(EPISODE_ID_PATTERN, episode_id) is None
        or type(top_k) is not int
        or not 1 <= top_k <= 8
    ):
        return False
    return episode_id, top_k


def _tool_call_is_within_request_scope(
    *,
    tool_name: str,
    arguments: dict[str, object],
    episode_id: str | None,
    top_k: int | None,
    registered_tool_names: frozenset[str],
) -> bool:
    """Keep model-selected arguments inside one trusted Ask request."""
    if episode_id is None or top_k is None or tool_name not in registered_tool_names:
        return True
    if arguments.get("episode_id") != episode_id:
        return False
    if tool_name == "search_transcript" and arguments.get("top_k") != top_k:
        return False
    return True


def _recorded_tool_name(
    tool_name: str,
    registered_tool_names: frozenset[str],
) -> RecordedToolName:
    if tool_name in registered_tool_names:
        return cast(RecordedToolName, tool_name)
    return "unknown"


def _result_count(result: ToolSuccess | ToolFailure) -> int:
    if isinstance(result, ToolFailure):
        return 0
    if isinstance(result.data, SearchTranscriptData | SegmentContextData):
        return result.data.result_count
    return 1


def _elapsed_ms(finished: float, started: float) -> float:
    return max(0.0, (finished - started) * 1_000)
