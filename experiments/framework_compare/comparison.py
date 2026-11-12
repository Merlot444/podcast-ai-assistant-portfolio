"""Shared synthetic cases and local overhead measurement for Day 22/23."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean, median
from time import perf_counter_ns

from app.agent import (
    AgentConfig,
    AgentFailure,
    AgentModelRequest,
    AgentSuccess,
    AgentWorkflow,
)
from app.schemas.retriever import RetrievedCandidate
from app.schemas.transcript import ProcessedTranscript, TranscriptSegment
from app.tooling import ToolDependencies, ToolDispatcher, ToolHandlers
from experiments.framework_compare import (
    AnswerGeneratorRequest,
    FrameworkFailure,
    FrameworkSuccess,
    LangGraphRetrievalQA,
)


EPISODE_ID = "ep_1111111111111111"
OTHER_EPISODE_ID = "ep_2222222222222222"
PRIVATE_SOURCE = "/private/synthetic/day23-secret.json"
SENSITIVE_SENTINEL = "SECRET traceback private retrieval body"


@dataclass(frozen=True)
class ComparisonScenario:
    case_id: str
    label: str
    question: object
    episode_id: str
    candidates: tuple[RetrievedCandidate, ...]
    answer_output: object
    retriever_fails: bool = False
    expected_day22: str = "completed"
    expected_day23: str = "completed"


@dataclass(frozen=True)
class ComparisonCaseResult:
    case_id: str
    label: str
    day22_outcome: str
    day23_outcome: str
    semantic_match: bool
    day22_model_calls: int
    day22_dispatcher_calls: int
    day23_answer_calls: int
    day23_dispatcher_calls: int


@dataclass(frozen=True)
class LatencySummary:
    iterations: int
    warmup_iterations: int
    day22_mean_ms: float
    day22_median_ms: float
    day22_p95_ms: float
    day23_mean_ms: float
    day23_median_ms: float
    day23_p95_ms: float


def _candidate(number: int, *, segment_index: int) -> RetrievedCandidate:
    return RetrievedCandidate(
        rank=number,
        chunk_id=f"chunk_{number:016x}",
        source_segment_ids=(f"{EPISODE_ID}:{segment_index:06d}",),
        score=1.0 / number,
        score_semantics="synthetic_score",
        retriever_name="day23_fake",
    )


DEFAULT_CANDIDATES = (
    _candidate(1, segment_index=0),
    _candidate(2, segment_index=1),
)


def _answer(reference: dict[str, str]) -> dict[str, object]:
    return {
        "answer": "A bounded synthetic answer.",
        "citations": [reference],
    }


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


COMPARISON_SCENARIOS: tuple[ComparisonScenario, ...] = (
    ComparisonScenario(
        case_id="normal_chunk",
        label="正常检索并引用返回的 Chunk",
        question="What is the first synthetic point?",
        episode_id=EPISODE_ID,
        candidates=DEFAULT_CANDIDATES,
        answer_output=_answer(_citation("chunk", "chunk_0000000000000001")),
    ),
    ComparisonScenario(
        case_id="normal_segment",
        label="另一个正常问题并引用返回的 segment",
        question="What is the second synthetic point?",
        episode_id=EPISODE_ID,
        candidates=DEFAULT_CANDIDATES,
        answer_output=_answer(
            _citation("segment", f"{EPISODE_ID}:000001")
        ),
    ),
    ComparisonScenario(
        case_id="empty_question",
        label="空问题",
        question="   ",
        episode_id=EPISODE_ID,
        candidates=DEFAULT_CANDIDATES,
        answer_output=_answer(_citation("chunk", "chunk_0000000000000001")),
        expected_day22="invalid_request",
        expected_day23="invalid_request",
    ),
    ComparisonScenario(
        case_id="overlong_question",
        label="超长问题",
        question="x" * 2_001,
        episode_id=EPISODE_ID,
        candidates=DEFAULT_CANDIDATES,
        answer_output=_answer(_citation("chunk", "chunk_0000000000000001")),
        expected_day22="invalid_request",
        expected_day23="invalid_request",
    ),
    ComparisonScenario(
        case_id="episode_not_found",
        label="episode 不存在",
        question="Find a missing synthetic episode.",
        episode_id=OTHER_EPISODE_ID,
        candidates=DEFAULT_CANDIDATES,
        answer_output=_answer(_citation("chunk", "chunk_0000000000000001")),
        expected_day22="not_found",
        expected_day23="not_found",
    ),
    ComparisonScenario(
        case_id="retrieval_failure",
        label="检索工具内部失败",
        question="Trigger the synthetic retrieval failure.",
        episode_id=EPISODE_ID,
        candidates=DEFAULT_CANDIDATES,
        answer_output=_answer(_citation("chunk", "chunk_0000000000000001")),
        retriever_fails=True,
        expected_day22="tool_failed",
        expected_day23="tool_failed",
    ),
    ComparisonScenario(
        case_id="empty_results",
        label="检索结果为空",
        question="Return no synthetic candidates.",
        episode_id=EPISODE_ID,
        candidates=(),
        answer_output=_answer(_citation("chunk", "chunk_0000000000000001")),
        expected_day22="empty_results",
        expected_day23="empty_results",
    ),
    ComparisonScenario(
        case_id="malformed_answer",
        label="回答输出格式错误",
        question="Return a malformed synthetic answer.",
        episode_id=EPISODE_ID,
        candidates=DEFAULT_CANDIDATES,
        answer_output={"answer": "Missing citations field."},
        expected_day22="model_output_invalid",
        expected_day23="answer_output_invalid",
    ),
    ComparisonScenario(
        case_id="unreturned_citation",
        label="引用未在本次检索中返回",
        question="Invent a synthetic citation.",
        episode_id=EPISODE_ID,
        candidates=DEFAULT_CANDIDATES,
        answer_output=_answer(_citation("chunk", "chunk_0000000000000099")),
        expected_day22="invalid_citation",
        expected_day23="invalid_citation",
    ),
    ComparisonScenario(
        case_id="cross_episode_citation",
        label="跨 episode 引用",
        question="Cross the synthetic episode boundary.",
        episode_id=EPISODE_ID,
        candidates=DEFAULT_CANDIDATES,
        answer_output=_answer(
            _citation(
                "segment",
                f"{OTHER_EPISODE_ID}:000000",
                episode_id=OTHER_EPISODE_ID,
            )
        ),
        expected_day22="invalid_citation",
        expected_day23="invalid_citation",
    ),
)


class SharedFakeRetriever:
    name = "day23_fake"

    def __init__(
        self,
        candidates: Sequence[RetrievedCandidate],
        *,
        fails: bool,
    ) -> None:
        self.candidates = list(candidates)
        self.fails = fails
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> list[RetrievedCandidate]:
        self.calls.append((query, top_k))
        if self.fails:
            raise RuntimeError(
                f"{PRIVATE_SOURCE} {SENSITIVE_SENTINEL} query={query}"
            )
        return self.candidates[:top_k]


class SharedFakeAnswerProgram:
    """Expose one scripted answer through both Day 22 and Day 23 interfaces."""

    def __init__(self, *, episode_id: str, answer_output: object) -> None:
        self.episode_id = episode_id
        self.answer_output = answer_output
        self.day22_requests: list[AgentModelRequest] = []
        self.day23_requests: list[AnswerGeneratorRequest] = []

    def decide(self, request: AgentModelRequest) -> object:
        self.day22_requests.append(request)
        if not request.tool_results:
            return {
                "decision_type": "tool_call",
                "tool_name": "search_transcript",
                "arguments": {
                    "query": request.question,
                    "episode_id": self.episode_id,
                    "top_k": 2,
                },
            }
        return {
            "decision_type": "final_answer",
            "final_answer": self.answer_output,
        }

    def generate(self, request: AnswerGeneratorRequest) -> object:
        self.day23_requests.append(request)
        return self.answer_output


class StableCallIdFactory:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self) -> str:
        self.count += 1
        return f"call_{self.count:032x}"


def synthetic_transcript() -> ProcessedTranscript:
    return ProcessedTranscript(
        episode_id=EPISODE_ID,
        source_file=PRIVATE_SOURCE,
        source_format="json",
        segments=[
            TranscriptSegment(
                segment_id=f"{EPISODE_ID}:{index:06d}",
                segment_index=index,
                text=f"Synthetic evidence body {index}.",
                source_file=PRIVATE_SOURCE,
                speaker="Synthetic Guest",
                section_title="Synthetic Section",
                start_time=float(index),
                end_time=float(index + 1),
            )
            for index in range(3)
        ],
    )


def build_shared_dependencies(
    scenario: ComparisonScenario,
) -> tuple[ToolDispatcher, SharedFakeRetriever, SharedFakeAnswerProgram]:
    retriever = SharedFakeRetriever(
        scenario.candidates,
        fails=scenario.retriever_fails,
    )
    dispatcher = ToolDispatcher(
        ToolHandlers(
            ToolDependencies(
                transcripts={EPISODE_ID: synthetic_transcript()},
                retrievers={EPISODE_ID: retriever},
            )
        ),
        call_id_factory=StableCallIdFactory(),
    )
    answer_program = SharedFakeAnswerProgram(
        episode_id=scenario.episode_id,
        answer_output=scenario.answer_output,
    )
    return dispatcher, retriever, answer_program


def run_comparison_cases() -> tuple[ComparisonCaseResult, ...]:
    results: list[ComparisonCaseResult] = []
    for scenario in COMPARISON_SCENARIOS:
        day22_dispatcher, _, day22_program = build_shared_dependencies(scenario)
        day22 = AgentWorkflow(
            model=day22_program,
            dispatcher=day22_dispatcher,
            config=AgentConfig(max_steps=2),
        ).run(scenario.question)

        day23_dispatcher, _, day23_program = build_shared_dependencies(scenario)
        day23 = LangGraphRetrievalQA(
            dispatcher=day23_dispatcher,
            answer_generator=day23_program,
        ).run(
            scenario.question,
            episode_id=scenario.episode_id,
            top_k=2,
        )

        day22_outcome = (
            "completed"
            if isinstance(day22, AgentSuccess)
            else day22.termination_reason
        )
        day23_outcome = (
            "completed"
            if isinstance(day23, FrameworkSuccess)
            else day23.termination_reason
        )
        results.append(
            ComparisonCaseResult(
                case_id=scenario.case_id,
                label=scenario.label,
                day22_outcome=day22_outcome,
                day23_outcome=day23_outcome,
                semantic_match=(
                    _normalized_outcome(day22_outcome)
                    == _normalized_outcome(day23_outcome)
                ),
                day22_model_calls=day22.model_call_count,
                day22_dispatcher_calls=day22.tool_call_count,
                day23_answer_calls=day23.answer_call_count,
                day23_dispatcher_calls=day23.dispatcher_call_count,
            )
        )
    return tuple(results)


def benchmark_normal_case(
    *,
    warmup_iterations: int = 100,
    iterations: int = 1_000,
) -> LatencySummary:
    if isinstance(warmup_iterations, bool) or warmup_iterations < 0:
        raise ValueError("warmup_iterations 必须是非负整数")
    if isinstance(iterations, bool) or iterations < 1:
        raise ValueError("iterations 必须是正整数")
    scenario = COMPARISON_SCENARIOS[0]
    day22_dispatcher, _, day22_program = build_shared_dependencies(scenario)
    day22_workflow = AgentWorkflow(
        model=day22_program,
        dispatcher=day22_dispatcher,
        config=AgentConfig(max_steps=2),
    )
    day23_dispatcher, _, day23_program = build_shared_dependencies(scenario)
    day23_workflow = LangGraphRetrievalQA(
        dispatcher=day23_dispatcher,
        answer_generator=day23_program,
    )

    for _ in range(warmup_iterations):
        _require_success(day22_workflow.run(scenario.question))
        _require_success(
            day23_workflow.run(
                scenario.question,
                episode_id=scenario.episode_id,
                top_k=2,
            )
        )

    day22_durations = [
        _measure_ns(lambda: day22_workflow.run(scenario.question))
        for _ in range(iterations)
    ]
    day23_durations = [
        _measure_ns(
            lambda: day23_workflow.run(
                scenario.question,
                episode_id=scenario.episode_id,
                top_k=2,
            )
        )
        for _ in range(iterations)
    ]
    return LatencySummary(
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        day22_mean_ms=mean(day22_durations) / 1_000_000,
        day22_median_ms=median(day22_durations) / 1_000_000,
        day22_p95_ms=_percentile(day22_durations, 0.95) / 1_000_000,
        day23_mean_ms=mean(day23_durations) / 1_000_000,
        day23_median_ms=median(day23_durations) / 1_000_000,
        day23_p95_ms=_percentile(day23_durations, 0.95) / 1_000_000,
    )


def _normalized_outcome(outcome: str) -> str:
    if outcome in {"model_output_invalid", "answer_output_invalid"}:
        return "answer_output_invalid"
    return outcome


def _measure_ns(call) -> int:  # noqa: ANN001
    started = perf_counter_ns()
    result = call()
    _require_success(result)
    return perf_counter_ns() - started


def _require_success(result: object) -> None:
    if not isinstance(result, AgentSuccess | FrameworkSuccess):
        reason = (
            result.termination_reason
            if isinstance(result, AgentFailure | FrameworkFailure)
            else "unexpected_result"
        )
        raise RuntimeError(f"基准正常案例意外失败：{reason}")


def _percentile(values: Sequence[int], quantile: float) -> float:
    ordered = sorted(values)
    rank = max(0, math.ceil(len(ordered) * quantile) - 1)
    return float(ordered[rank])
