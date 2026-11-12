"""Strict state-adjacent and public result schemas for the Day 23 experiment."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.agent.schemas import AgentFinalAnswer, MAX_AGENT_QUESTION_CHARACTERS
from app.tooling.schemas import EPISODE_ID_PATTERN, SHA256_PATTERN


MAX_FRAMEWORK_TRACE_ENTRIES = 5

FrameworkNodeName = Literal[
    "validate_input",
    "retrieve",
    "generate_answer",
    "validate_citations",
    "safe_failure",
]
FrameworkStage = Literal[
    "received",
    "input_validated",
    "retrieved",
    "answer_generated",
    "citations_validated",
    "succeeded",
    "failed",
]
FrameworkErrorCode = Literal[
    "invalid_request",
    "unknown_tool",
    "invalid_arguments",
    "not_found",
    "tool_failed",
    "empty_results",
    "answer_failed",
    "answer_output_invalid",
    "invalid_citation",
    "workflow_failed",
]


class FrameworkSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
    )


class FrameworkRequest(FrameworkSchema):
    question: str = Field(min_length=1, max_length=MAX_AGENT_QUESTION_CHARACTERS)
    episode_id: str = Field(pattern=EPISODE_ID_PATTERN)
    top_k: int = Field(ge=1, le=8)

    @field_validator("question", "episode_id", mode="before")
    @classmethod
    def strip_string_inputs(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class FrameworkNodeCounts(FrameworkSchema):
    validate_input: int = Field(default=0, ge=0, le=1)
    retrieve: int = Field(default=0, ge=0, le=1)
    generate_answer: int = Field(default=0, ge=0, le=1)
    validate_citations: int = Field(default=0, ge=0, le=1)
    safe_failure: int = Field(default=0, ge=0, le=1)

    def increment(self, node: FrameworkNodeName) -> Self:
        values = self.model_dump()
        values[node] = getattr(self, node) + 1
        return type(self).model_validate(values)


class FrameworkTraceEntry(FrameworkSchema):
    sequence: int = Field(ge=1, le=MAX_FRAMEWORK_TRACE_ENTRIES)
    node: FrameworkNodeName
    stage: FrameworkStage
    outcome: Literal["success", "failure"]
    error_code: FrameworkErrorCode | None = None
    result_count: int | None = Field(default=None, ge=0, le=8)
    dispatcher_call_count: int = Field(ge=0, le=1)
    answer_call_count: int = Field(ge=0, le=1)
    duration_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def require_error_only_on_failure(self) -> Self:
        if self.outcome == "success" and self.error_code is not None:
            raise ValueError("成功轨迹项不能包含 error_code")
        if self.outcome == "failure" and self.error_code is None:
            raise ValueError("失败轨迹项必须包含 error_code")
        return self


class FrameworkRunTrace(FrameworkSchema):
    question_length: int = Field(ge=0)
    question_sha256: str = Field(pattern=SHA256_PATTERN)
    entries: tuple[FrameworkTraceEntry, ...] = Field(
        min_length=1,
        max_length=MAX_FRAMEWORK_TRACE_ENTRIES,
    )

    @model_validator(mode="after")
    def require_contiguous_sequences(self) -> Self:
        if [entry.sequence for entry in self.entries] != list(
            range(1, len(self.entries) + 1)
        ):
            raise ValueError("实验轨迹 sequence 必须从 1 连续递增")
        return self


class FrameworkError(FrameworkSchema):
    code: FrameworkErrorCode
    message: str = Field(min_length=1, max_length=200)


class FrameworkSuccess(FrameworkSchema):
    ok: Literal[True] = True
    answer: AgentFinalAnswer
    termination_reason: Literal["completed"] = "completed"
    dispatcher_call_count: Literal[1] = 1
    answer_call_count: Literal[1] = 1
    node_counts: FrameworkNodeCounts
    trace: FrameworkRunTrace

    @model_validator(mode="after")
    def require_success_path(self) -> Self:
        if self.node_counts != FrameworkNodeCounts(
            validate_input=1,
            retrieve=1,
            generate_answer=1,
            validate_citations=1,
        ):
            raise ValueError("成功结果必须来自固定四节点路径")
        if self.trace.entries[-1].stage != "succeeded":
            raise ValueError("成功轨迹必须以 succeeded 结束")
        return self


class FrameworkFailure(FrameworkSchema):
    ok: Literal[False] = False
    error: FrameworkError
    termination_reason: FrameworkErrorCode
    dispatcher_call_count: int = Field(ge=0, le=1)
    answer_call_count: int = Field(ge=0, le=1)
    counts_complete: bool = True
    node_counts: FrameworkNodeCounts
    trace: FrameworkRunTrace

    @model_validator(mode="after")
    def require_failure_consistency(self) -> Self:
        if self.error.code != self.termination_reason:
            raise ValueError("失败原因必须与错误码一致")
        if self.trace.entries[-1].stage != "failed":
            raise ValueError("失败轨迹必须以 failed 结束")
        if self.counts_complete and self.node_counts.safe_failure != 1:
            raise ValueError("正常失败路径必须经过 safe_failure")
        return self


FrameworkRunResult = FrameworkSuccess | FrameworkFailure
