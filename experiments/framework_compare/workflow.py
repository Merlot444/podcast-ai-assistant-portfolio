"""A finite four-stage retrieval-answer experiment orchestrated by LangGraph."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from types import MappingProxyType
from typing import Literal, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from app.agent.citations import (
    ReferenceKey,
    citations_are_valid,
    references_from_tool_success,
)
from app.agent.schemas import AgentFinalAnswer
from app.tooling.dispatcher import ToolDispatcher
from app.tooling.recording import stable_json_sha256, text_sha256
from app.tooling.schemas import SearchTranscriptData, ToolFailure, ToolSuccess
from experiments.framework_compare.generator import (
    AnswerGenerator,
    AnswerGeneratorRequest,
)
from experiments.framework_compare.schemas import (
    FrameworkError,
    FrameworkErrorCode,
    FrameworkFailure,
    FrameworkNodeCounts,
    FrameworkNodeName,
    FrameworkRequest,
    FrameworkRunResult,
    FrameworkRunTrace,
    FrameworkStage,
    FrameworkSuccess,
    FrameworkTraceEntry,
)


Clock = Callable[[], float]
Route = Literal["next", "safe_failure", "end"]

FRAMEWORK_ERROR_MESSAGES = MappingProxyType(
    {
        "invalid_request": "问题或检索范围无效，无法开始处理。",
        "unknown_tool": "检索工具不可用。",
        "invalid_arguments": "检索参数无效。",
        "not_found": "请求的节目不存在。",
        "tool_failed": "检索工具暂时无法完成调用。",
        "empty_results": "未找到可用于回答的相关内容。",
        "answer_failed": "回答生成器暂时无法完成调用。",
        "answer_output_invalid": "回答生成器返回的格式无效。",
        "invalid_citation": "回答引用未通过校验，未返回未经验证的内容。",
        "workflow_failed": "实验工作流未能安全完成。",
    }
)


class FrameworkGraphInput(TypedDict):
    question: object
    episode_id: object
    top_k: object


class FrameworkGraphOutput(TypedDict):
    result: FrameworkRunResult


class FrameworkGraphState(TypedDict, total=False):
    question: object
    episode_id: object
    top_k: object
    request: FrameworkRequest | None
    stage: FrameworkStage
    tool_arguments: dict[str, object] | None
    tool_result: ToolSuccess | None
    allowed_references: tuple[ReferenceKey, ...]
    candidate_answer: AgentFinalAnswer | None
    result: FrameworkRunResult
    error_code: FrameworkErrorCode | None
    node_counts: FrameworkNodeCounts
    dispatcher_call_count: int
    answer_call_count: int
    question_length: int
    question_sha256: str
    trace_entries: tuple[FrameworkTraceEntry, ...]


class LangGraphRetrievalQA:
    """Compile and invoke one acyclic graph with trusted dependencies in closures."""

    def __init__(
        self,
        *,
        dispatcher: ToolDispatcher,
        answer_generator: AnswerGenerator,
        clock: Clock = perf_counter,
    ) -> None:
        self.dispatcher = dispatcher
        self.answer_generator = answer_generator
        self.clock = clock
        self.graph = self._compile_graph()

    def _compile_graph(self):  # noqa: ANN202 - LangGraph return type is internal.
        builder = StateGraph(
            FrameworkGraphState,
            input_schema=FrameworkGraphInput,
            output_schema=FrameworkGraphOutput,
        )
        builder.add_node("validate_input", self._validate_input)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("generate_answer", self._generate_answer)
        builder.add_node("validate_citations", self._validate_citations)
        builder.add_node("safe_failure", self._safe_failure)
        builder.add_edge(START, "validate_input")
        builder.add_conditional_edges(
            "validate_input",
            _route_after_node,
            {"next": "retrieve", "safe_failure": "safe_failure"},
        )
        builder.add_conditional_edges(
            "retrieve",
            _route_after_node,
            {"next": "generate_answer", "safe_failure": "safe_failure"},
        )
        builder.add_conditional_edges(
            "generate_answer",
            _route_after_node,
            {"next": "validate_citations", "safe_failure": "safe_failure"},
        )
        builder.add_conditional_edges(
            "validate_citations",
            _route_after_validation,
            {"end": END, "safe_failure": "safe_failure"},
        )
        builder.add_edge("safe_failure", END)
        return builder.compile()

    def run(
        self,
        question: object,
        *,
        episode_id: object,
        top_k: object = 3,
    ) -> FrameworkRunResult:
        """Invoke the compiled graph and return only its strict, redacted result."""
        try:
            output = self.graph.invoke(
                {
                    "question": question,
                    "episode_id": episode_id,
                    "top_k": top_k,
                },
                {"recursion_limit": 10},
            )
            result = output.get("result")
            if isinstance(result, FrameworkSuccess | FrameworkFailure):
                return result
        except Exception:
            pass
        return _unexpected_failure(question)

    def _validate_input(self, state: FrameworkGraphState) -> FrameworkGraphState:
        started = self.clock()
        question = state.get("question")
        question_length = len(question) if isinstance(question, str) else 0
        question_sha256 = (
            text_sha256(question)
            if isinstance(question, str)
            else stable_json_sha256(question)
        )
        counts = FrameworkNodeCounts(validate_input=1)
        base: FrameworkGraphState = {
            "question": None,
            "episode_id": None,
            "top_k": None,
            "request": None,
            "stage": "received",
            "tool_arguments": None,
            "tool_result": None,
            "allowed_references": (),
            "candidate_answer": None,
            "error_code": None,
            "node_counts": counts,
            "dispatcher_call_count": 0,
            "answer_call_count": 0,
            "question_length": question_length,
            "question_sha256": question_sha256,
            "trace_entries": (),
        }
        try:
            request = FrameworkRequest.model_validate(
                {
                    "question": question,
                    "episode_id": state.get("episode_id"),
                    "top_k": state.get("top_k"),
                }
            )
        except Exception:
            error_code: FrameworkErrorCode = "invalid_request"
            return _with_trace(
                base,
                node="validate_input",
                stage="failed",
                outcome="failure",
                error_code=error_code,
                duration_ms=_elapsed_ms(self.clock(), started),
                updates={"error_code": error_code, "stage": "failed"},
            )
        arguments: dict[str, object] = {
            "query": request.question,
            "episode_id": request.episode_id,
            "top_k": request.top_k,
        }
        return _with_trace(
            base,
            node="validate_input",
            stage="input_validated",
            outcome="success",
            duration_ms=_elapsed_ms(self.clock(), started),
            updates={
                "request": request,
                "stage": "input_validated",
                "tool_arguments": arguments,
            },
        )

    def _retrieve(self, state: FrameworkGraphState) -> FrameworkGraphState:
        started = self.clock()
        counts = state["node_counts"].increment("retrieve")
        arguments = state.get("tool_arguments")
        if not isinstance(arguments, dict):
            return self._node_failure(
                state,
                node="retrieve",
                counts=counts,
                code="workflow_failed",
                started=started,
            )
        dispatcher_call_count = 1
        try:
            dispatch_result = self.dispatcher.dispatch(
                "search_transcript",
                arguments,
            )
        except Exception:
            return self._node_failure(
                state,
                node="retrieve",
                counts=counts,
                code="tool_failed",
                started=started,
                dispatcher_call_count=dispatcher_call_count,
            )
        if isinstance(dispatch_result, ToolFailure):
            return self._node_failure(
                state,
                node="retrieve",
                counts=counts,
                code=cast(FrameworkErrorCode, dispatch_result.error.code),
                started=started,
                dispatcher_call_count=dispatcher_call_count,
            )
        if not isinstance(dispatch_result.data, SearchTranscriptData):
            return self._node_failure(
                state,
                node="retrieve",
                counts=counts,
                code="tool_failed",
                started=started,
                dispatcher_call_count=dispatcher_call_count,
            )
        if dispatch_result.data.result_count == 0:
            return self._node_failure(
                state,
                node="retrieve",
                counts=counts,
                code="empty_results",
                started=started,
                dispatcher_call_count=dispatcher_call_count,
                result_count=0,
            )
        try:
            allowed_references = tuple(
                sorted(references_from_tool_success(dispatch_result))
            )
        except Exception:
            return self._node_failure(
                state,
                node="retrieve",
                counts=counts,
                code="tool_failed",
                started=started,
                dispatcher_call_count=dispatcher_call_count,
            )
        return _with_trace(
            state,
            node="retrieve",
            stage="retrieved",
            outcome="success",
            result_count=dispatch_result.data.result_count,
            duration_ms=_elapsed_ms(self.clock(), started),
            updates={
                "stage": "retrieved",
                "tool_result": dispatch_result,
                "allowed_references": allowed_references,
                "node_counts": counts,
                "dispatcher_call_count": dispatcher_call_count,
            },
        )

    def _generate_answer(self, state: FrameworkGraphState) -> FrameworkGraphState:
        started = self.clock()
        counts = state["node_counts"].increment("generate_answer")
        request = state.get("request")
        tool_result = state.get("tool_result")
        if (
            request is None
            or tool_result is None
            or not isinstance(tool_result.data, SearchTranscriptData)
        ):
            return self._node_failure(
                state,
                node="generate_answer",
                counts=counts,
                code="workflow_failed",
                started=started,
            )
        answer_call_count = 1
        try:
            raw_answer = self.answer_generator.generate(
                AnswerGeneratorRequest(
                    question=request.question,
                    retrieval=tool_result.data,
                    allowed_references=state["allowed_references"],
                )
            )
        except Exception:
            return self._node_failure(
                state,
                node="generate_answer",
                counts=counts,
                code="answer_failed",
                started=started,
                answer_call_count=answer_call_count,
            )
        try:
            candidate_answer = AgentFinalAnswer.model_validate(raw_answer)
        except Exception:
            return self._node_failure(
                state,
                node="generate_answer",
                counts=counts,
                code="answer_output_invalid",
                started=started,
                answer_call_count=answer_call_count,
            )
        return _with_trace(
            state,
            node="generate_answer",
            stage="answer_generated",
            outcome="success",
            duration_ms=_elapsed_ms(self.clock(), started),
            updates={
                "stage": "answer_generated",
                "candidate_answer": candidate_answer,
                "node_counts": counts,
                "answer_call_count": answer_call_count,
            },
        )

    def _validate_citations(
        self,
        state: FrameworkGraphState,
    ) -> FrameworkGraphState:
        started = self.clock()
        counts = state["node_counts"].increment("validate_citations")
        candidate_answer = state.get("candidate_answer")
        if candidate_answer is None:
            return self._node_failure(
                state,
                node="validate_citations",
                counts=counts,
                code="workflow_failed",
                started=started,
            )
        valid = citations_are_valid(
            candidate_answer.citations,
            has_tool_results=True,
            allowed_references=state["allowed_references"],
        )
        if not valid:
            return self._node_failure(
                state,
                node="validate_citations",
                counts=counts,
                code="invalid_citation",
                started=started,
                clear_candidate=True,
            )
        traced = _with_trace(
            state,
            node="validate_citations",
            stage="succeeded",
            outcome="success",
            duration_ms=_elapsed_ms(self.clock(), started),
            updates={
                "stage": "succeeded",
                "node_counts": counts,
            },
        )
        result = FrameworkSuccess(
            answer=candidate_answer,
            node_counts=counts,
            trace=_build_trace(traced),
        )
        return {**traced, "result": result}

    def _safe_failure(self, state: FrameworkGraphState) -> FrameworkGraphState:
        started = self.clock()
        code = state.get("error_code") or "workflow_failed"
        counts = state.get("node_counts", FrameworkNodeCounts()).increment(
            "safe_failure"
        )
        traced = _with_trace(
            state,
            node="safe_failure",
            stage="failed",
            outcome="failure",
            error_code=code,
            duration_ms=_elapsed_ms(self.clock(), started),
            updates={
                "question": None,
                "episode_id": None,
                "top_k": None,
                "request": None,
                "stage": "failed",
                "tool_arguments": None,
                "tool_result": None,
                "allowed_references": (),
                "candidate_answer": None,
                "error_code": code,
                "node_counts": counts,
            },
        )
        result = FrameworkFailure(
            error=FrameworkError(code=code, message=FRAMEWORK_ERROR_MESSAGES[code]),
            termination_reason=code,
            dispatcher_call_count=traced.get("dispatcher_call_count", 0),
            answer_call_count=traced.get("answer_call_count", 0),
            node_counts=counts,
            trace=_build_trace(traced),
        )
        return {**traced, "result": result}

    def _node_failure(
        self,
        state: FrameworkGraphState,
        *,
        node: FrameworkNodeName,
        counts: FrameworkNodeCounts,
        code: FrameworkErrorCode,
        started: float,
        dispatcher_call_count: int | None = None,
        answer_call_count: int | None = None,
        result_count: int | None = None,
        clear_candidate: bool = False,
    ) -> FrameworkGraphState:
        updates: FrameworkGraphState = {
            "stage": "failed",
            "error_code": code,
            "node_counts": counts,
        }
        if dispatcher_call_count is not None:
            updates["dispatcher_call_count"] = dispatcher_call_count
        if answer_call_count is not None:
            updates["answer_call_count"] = answer_call_count
        if clear_candidate:
            updates["candidate_answer"] = None
        return _with_trace(
            state,
            node=node,
            stage="failed",
            outcome="failure",
            error_code=code,
            result_count=result_count,
            duration_ms=_elapsed_ms(self.clock(), started),
            updates=updates,
        )


def _route_after_node(state: FrameworkGraphState) -> Route:
    return "safe_failure" if state.get("error_code") is not None else "next"


def _route_after_validation(state: FrameworkGraphState) -> Route:
    return "safe_failure" if state.get("error_code") is not None else "end"


def _with_trace(
    state: FrameworkGraphState,
    *,
    node: FrameworkNodeName,
    stage: FrameworkStage,
    outcome: Literal["success", "failure"],
    duration_ms: float,
    updates: FrameworkGraphState,
    error_code: FrameworkErrorCode | None = None,
    result_count: int | None = None,
) -> FrameworkGraphState:
    dispatcher_call_count = updates.get(
        "dispatcher_call_count",
        state.get("dispatcher_call_count", 0),
    )
    answer_call_count = updates.get(
        "answer_call_count",
        state.get("answer_call_count", 0),
    )
    prior_entries = state.get("trace_entries", ())
    entry = FrameworkTraceEntry(
        sequence=len(prior_entries) + 1,
        node=node,
        stage=stage,
        outcome=outcome,
        error_code=error_code,
        result_count=result_count,
        dispatcher_call_count=dispatcher_call_count,
        answer_call_count=answer_call_count,
        duration_ms=max(0.0, duration_ms),
    )
    return {**state, **updates, "trace_entries": (*prior_entries, entry)}


def _build_trace(state: FrameworkGraphState) -> FrameworkRunTrace:
    return FrameworkRunTrace(
        question_length=state.get("question_length", 0),
        question_sha256=state.get("question_sha256", stable_json_sha256(None)),
        entries=state["trace_entries"],
    )


def _unexpected_failure(question: object) -> FrameworkFailure:
    question_length = len(question) if isinstance(question, str) else 0
    question_sha256 = (
        text_sha256(question)
        if isinstance(question, str)
        else stable_json_sha256(question)
    )
    counts = FrameworkNodeCounts(safe_failure=1)
    trace = FrameworkRunTrace(
        question_length=question_length,
        question_sha256=question_sha256,
        entries=(
            FrameworkTraceEntry(
                sequence=1,
                node="safe_failure",
                stage="failed",
                outcome="failure",
                error_code="workflow_failed",
                dispatcher_call_count=0,
                answer_call_count=0,
                duration_ms=0.0,
            ),
        ),
    )
    return FrameworkFailure(
        error=FrameworkError(
            code="workflow_failed",
            message=FRAMEWORK_ERROR_MESSAGES["workflow_failed"],
        ),
        termination_reason="workflow_failed",
        dispatcher_call_count=0,
        answer_call_count=0,
        counts_complete=False,
        node_counts=counts,
        trace=trace,
    )


def _elapsed_ms(finished: float, started: float) -> float:
    return max(0.0, (finished - started) * 1_000)
