"""Strict model-facing, result, and audit schemas for the Day 22 Agent."""

from __future__ import annotations

import re
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from app.tooling.schemas import (
    CHUNK_ID_PATTERN,
    EPISODE_ID_PATTERN,
    SEGMENT_ID_PATTERN,
    SHA256_PATTERN,
    RecordedToolName,
)


MAX_AGENT_STEPS = 8
DEFAULT_MAX_AGENT_STEPS = 4
MAX_AGENT_TRACE_ENTRIES = 40
MAX_AGENT_ANSWER_CHARACTERS = 4_000
MAX_AGENT_CITATIONS = 16
MAX_AGENT_QUESTION_CHARACTERS = 2_000

AgentState = Literal[
    "received",
    "requesting_model",
    "validating_model_output",
    "dispatching_tool",
    "saving_tool_result",
    "validating_final_answer",
    "succeeded",
    "failed",
]
AgentDecisionType = Literal["tool_call", "final_answer", "invalid"]
AgentReferenceKind = Literal["episode", "segment", "chunk"]
AgentCitationValidation = Literal["not_checked", "valid", "invalid"]
AgentResultStatus = Literal["success", "failure", "empty", "duplicate"]
AgentErrorCode = Literal[
    "invalid_request",
    "model_failed",
    "model_output_invalid",
    "unknown_tool",
    "invalid_arguments",
    "not_found",
    "tool_failed",
    "empty_results",
    "duplicate_call",
    "max_steps_exceeded",
    "invalid_citation",
]
AgentTerminationReason = Literal[
    "completed",
    "invalid_request",
    "model_failed",
    "model_output_invalid",
    "unknown_tool",
    "invalid_arguments",
    "not_found",
    "tool_failed",
    "empty_results",
    "duplicate_call",
    "max_steps_exceeded",
    "invalid_citation",
]


class AgentSchema(BaseModel):
    """Strict base for data crossing the Day 22 Agent boundary."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
    )


class AgentConfig(AgentSchema):
    """Small, trusted configuration surface for one Agent workflow."""

    max_steps: int = Field(
        default=DEFAULT_MAX_AGENT_STEPS,
        ge=1,
        le=MAX_AGENT_STEPS,
    )
    require_tool_grounding: bool = False


class AgentCitation(AgentSchema):
    """One model-selected identifier; the workflow validates its provenance."""

    kind: AgentReferenceKind
    episode_id: str = Field(pattern=EPISODE_ID_PATTERN)
    reference_id: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_kind_specific_identifier_shape(self) -> Self:
        pattern = {
            "episode": EPISODE_ID_PATTERN,
            "segment": SEGMENT_ID_PATTERN,
            "chunk": CHUNK_ID_PATTERN,
        }[self.kind]
        if re.fullmatch(pattern, self.reference_id) is None:
            raise ValueError("引用标识与引用类型不匹配")
        return self


class AgentFinalAnswer(AgentSchema):
    """Strict final-answer structure returned by the model and application."""

    answer: str = Field(min_length=1, max_length=MAX_AGENT_ANSWER_CHARACTERS)
    citations: list[AgentCitation] = Field(max_length=MAX_AGENT_CITATIONS)

    @field_validator("answer")
    @classmethod
    def strip_and_reject_blank_answer(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("answer 不能为空")
        return cleaned

    @field_validator("citations")
    @classmethod
    def require_unique_citations(
        cls,
        value: list[AgentCitation],
    ) -> list[AgentCitation]:
        keys = [
            (citation.kind, citation.episode_id, citation.reference_id)
            for citation in value
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("citations 不能重复")
        return value


class ModelToolCallDecision(AgentSchema):
    """Untrusted model proposal for one Dispatcher call."""

    decision_type: Literal["tool_call"]
    tool_name: str = Field(min_length=1, max_length=128)
    arguments: dict[str, object]

    @field_validator("tool_name")
    @classmethod
    def strip_tool_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("tool_name 不能为空")
        return cleaned


class ModelFinalAnswerDecision(AgentSchema):
    """Untrusted model proposal to end the workflow with an answer."""

    decision_type: Literal["final_answer"]
    final_answer: AgentFinalAnswer


ModelDecision = ModelToolCallDecision | ModelFinalAnswerDecision
MODEL_DECISION_ADAPTER = TypeAdapter(
    Annotated[ModelDecision, Field(discriminator="decision_type")]
)


class AgentTraceEntry(AgentSchema):
    """One bounded, content-free state/action audit entry."""

    sequence: int = Field(ge=1, le=MAX_AGENT_TRACE_ENTRIES)
    model_step: int = Field(ge=0, le=MAX_AGENT_STEPS)
    state: AgentState
    decision_type: AgentDecisionType | None = None
    decision_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    argument_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    call_id: str | None = Field(default=None, pattern=r"^call_[0-9a-f]{32}$")
    tool_name: RecordedToolName | None = None
    result_status: AgentResultStatus | None = None
    result_count: int | None = Field(default=None, ge=0, le=8)
    citation_validation: AgentCitationValidation = "not_checked"
    termination_reason: AgentTerminationReason | None = None
    duration_ms: float = Field(ge=0)


class AgentRunTrace(AgentSchema):
    """Bounded trace plus a one-way summary of the user question."""

    question_length: int = Field(ge=0)
    question_sha256: str = Field(pattern=SHA256_PATTERN)
    max_steps: int = Field(ge=1, le=MAX_AGENT_STEPS)
    entries: list[AgentTraceEntry] = Field(
        min_length=1,
        max_length=MAX_AGENT_TRACE_ENTRIES,
    )

    @model_validator(mode="after")
    def require_contiguous_sequences_and_terminal_state(self) -> Self:
        if [entry.sequence for entry in self.entries] != list(
            range(1, len(self.entries) + 1)
        ):
            raise ValueError("trace sequence 必须从 1 连续递增")
        if self.entries[0].state != "received":
            raise ValueError("trace 必须从 received 开始")
        if self.entries[-1].state not in {"succeeded", "failed"}:
            raise ValueError("trace 必须以终止状态结束")
        return self


class AgentError(AgentSchema):
    code: AgentErrorCode
    message: str = Field(min_length=1, max_length=200)


class AgentSuccess(AgentSchema):
    ok: Literal[True] = True
    answer: AgentFinalAnswer
    termination_reason: Literal["completed"] = "completed"
    model_call_count: int = Field(ge=1, le=MAX_AGENT_STEPS)
    tool_call_count: int = Field(ge=0, le=MAX_AGENT_STEPS)
    trace: AgentRunTrace

    @model_validator(mode="after")
    def require_consistent_success_trace(self) -> Self:
        terminal = self.trace.entries[-1]
        if terminal.state != "succeeded" or terminal.termination_reason != "completed":
            raise ValueError("成功结果的 trace 终止状态不一致")
        if self.tool_call_count > self.model_call_count:
            raise ValueError("tool_call_count 不能超过 model_call_count")
        return self


class AgentFailure(AgentSchema):
    ok: Literal[False] = False
    error: AgentError
    termination_reason: AgentErrorCode
    model_call_count: int = Field(ge=0, le=MAX_AGENT_STEPS)
    tool_call_count: int = Field(ge=0, le=MAX_AGENT_STEPS)
    trace: AgentRunTrace

    @model_validator(mode="after")
    def require_consistent_failure_trace(self) -> Self:
        terminal = self.trace.entries[-1]
        if self.termination_reason != self.error.code:
            raise ValueError("失败原因必须与错误码一致")
        if terminal.state != "failed":
            raise ValueError("失败结果的 trace 必须以 failed 结束")
        if terminal.termination_reason != self.termination_reason:
            raise ValueError("失败结果的 trace 终止原因不一致")
        if self.tool_call_count > self.model_call_count:
            raise ValueError("tool_call_count 不能超过 model_call_count")
        return self


AgentRunResult = AgentSuccess | AgentFailure
