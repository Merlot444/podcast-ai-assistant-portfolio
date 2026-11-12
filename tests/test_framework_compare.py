"""Offline Day 23 LangGraph experiment, parity, and safety tests."""

from __future__ import annotations

import ast
import importlib.metadata
import json
import tomllib
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from app.agent import AgentFailure, AgentSuccess
from experiments.framework_compare import FrameworkFailure, FrameworkSuccess
from experiments.framework_compare.comparison import (
    COMPARISON_SCENARIOS,
    EPISODE_ID,
    OTHER_EPISODE_ID,
    PRIVATE_SOURCE,
    SENSITIVE_SENTINEL,
    ComparisonScenario,
    SharedFakeAnswerProgram,
    benchmark_normal_case,
    build_shared_dependencies,
    run_comparison_cases,
)
from experiments.framework_compare.workflow import LangGraphRetrievalQA


PROJECT_ROOT = Path(__file__).parents[1]


def _scenario(case_id: str) -> ComparisonScenario:
    return next(item for item in COMPARISON_SCENARIOS if item.case_id == case_id)


def _workflow(
    case_id: str = "normal_chunk",
) -> tuple[
    LangGraphRetrievalQA,
    object,
    SharedFakeAnswerProgram,
    ComparisonScenario,
]:
    scenario = _scenario(case_id)
    dispatcher, retriever, answer_program = build_shared_dependencies(scenario)
    return (
        LangGraphRetrievalQA(
            dispatcher=dispatcher,
            answer_generator=answer_program,
        ),
        retriever,
        answer_program,
        scenario,
    )


def test_graph_compiles_with_fixed_nodes_and_invokes_successfully() -> None:
    workflow, retriever, answer_program, scenario = _workflow()

    graph_view = workflow.graph.get_graph()
    output = workflow.graph.invoke(
        {
            "question": scenario.question,
            "episode_id": scenario.episode_id,
            "top_k": 2,
        }
    )

    assert set(graph_view.nodes) == {
        "__start__",
        "validate_input",
        "retrieve",
        "generate_answer",
        "validate_citations",
        "safe_failure",
        "__end__",
    }
    assert {
        (edge.source, edge.target, edge.conditional) for edge in graph_view.edges
    } == {
        ("__start__", "validate_input", False),
        ("validate_input", "retrieve", True),
        ("validate_input", "safe_failure", True),
        ("retrieve", "generate_answer", True),
        ("retrieve", "safe_failure", True),
        ("generate_answer", "validate_citations", True),
        ("generate_answer", "safe_failure", True),
        ("validate_citations", "__end__", True),
        ("validate_citations", "safe_failure", True),
        ("safe_failure", "__end__", False),
    }
    assert set(output) == {"result"}
    assert isinstance(output["result"], FrameworkSuccess)
    assert retriever.calls == [(scenario.question, 2)]
    assert len(answer_program.day23_requests) == 1


def test_success_path_state_updates_and_node_counts_are_exact() -> None:
    workflow, retriever, answer_program, scenario = _workflow("normal_segment")

    result = workflow.run(
        scenario.question,
        episode_id=scenario.episode_id,
        top_k=2,
    )

    assert isinstance(result, FrameworkSuccess)
    assert [entry.node for entry in result.trace.entries] == [
        "validate_input",
        "retrieve",
        "generate_answer",
        "validate_citations",
    ]
    assert [entry.stage for entry in result.trace.entries] == [
        "input_validated",
        "retrieved",
        "answer_generated",
        "succeeded",
    ]
    assert result.node_counts.model_dump() == {
        "validate_input": 1,
        "retrieve": 1,
        "generate_answer": 1,
        "validate_citations": 1,
        "safe_failure": 0,
    }
    assert result.dispatcher_call_count == result.answer_call_count == 1
    assert len(retriever.calls) == len(answer_program.day23_requests) == 1


@pytest.mark.parametrize(
    ("question", "episode_id", "top_k"),
    [
        ("   ", EPISODE_ID, 2),
        ("x" * 2_001, EPISODE_ID, 2),
        (42, EPISODE_ID, 2),
        ("valid question", "../../private", 2),
        ("valid question", EPISODE_ID, True),
        ("valid question", EPISODE_ID, 9),
    ],
)
def test_invalid_input_stops_before_dispatcher_and_answer_generator(
    question: object,
    episode_id: object,
    top_k: object,
) -> None:
    workflow, retriever, answer_program, _ = _workflow()

    result = workflow.run(question, episode_id=episode_id, top_k=top_k)

    assert isinstance(result, FrameworkFailure)
    assert result.termination_reason == "invalid_request"
    assert result.dispatcher_call_count == result.answer_call_count == 0
    assert retriever.calls == []
    assert answer_program.day23_requests == []
    assert [entry.node for entry in result.trace.entries] == [
        "validate_input",
        "safe_failure",
    ]


def test_question_between_workflow_and_tool_limits_uses_dispatcher_error() -> None:
    workflow, retriever, answer_program, _ = _workflow()

    result = workflow.run(
        "x" * 501,
        episode_id=EPISODE_ID,
        top_k=2,
    )

    assert isinstance(result, FrameworkFailure)
    assert result.termination_reason == "invalid_arguments"
    assert result.error.message == "检索参数无效。"
    assert result.dispatcher_call_count == 1
    assert result.answer_call_count == 0
    assert retriever.calls == []
    assert answer_program.day23_requests == []
    assert [entry.node for entry in result.trace.entries] == [
        "validate_input",
        "retrieve",
        "safe_failure",
    ]


@pytest.mark.parametrize(
    ("case_id", "expected"),
    [
        ("episode_not_found", "not_found"),
        ("retrieval_failure", "tool_failed"),
        ("empty_results", "empty_results"),
    ],
)
def test_retrieval_failures_take_the_explicit_safe_terminal(
    case_id: str,
    expected: str,
) -> None:
    workflow, retriever, answer_program, scenario = _workflow(case_id)

    result = workflow.run(
        scenario.question,
        episode_id=scenario.episode_id,
        top_k=2,
    )

    assert isinstance(result, FrameworkFailure)
    assert result.termination_reason == expected
    assert result.dispatcher_call_count == 1
    assert result.answer_call_count == 0
    assert answer_program.day23_requests == []
    assert result.trace.entries[-1].node == "safe_failure"
    assert result.node_counts.safe_failure == 1
    if case_id == "episode_not_found":
        assert retriever.calls == []
    else:
        assert len(retriever.calls) == 1


def test_malformed_answer_is_rejected_without_persisting_raw_output() -> None:
    workflow, retriever, answer_program, scenario = _workflow("malformed_answer")

    result = workflow.run(
        scenario.question,
        episode_id=scenario.episode_id,
        top_k=2,
    )

    assert isinstance(result, FrameworkFailure)
    assert result.termination_reason == "answer_output_invalid"
    assert result.dispatcher_call_count == result.answer_call_count == 1
    assert len(retriever.calls) == len(answer_program.day23_requests) == 1
    serialized = result.model_dump_json()
    assert "Missing citations field" not in serialized
    assert '"answer"' not in serialized


@pytest.mark.parametrize(
    "case_id",
    ["unreturned_citation", "cross_episode_citation"],
)
def test_unreturned_and_cross_episode_citations_are_rejected(case_id: str) -> None:
    workflow, _, _, scenario = _workflow(case_id)

    result = workflow.run(
        scenario.question,
        episode_id=scenario.episode_id,
        top_k=2,
    )

    assert isinstance(result, FrameworkFailure)
    assert result.termination_reason == "invalid_citation"
    assert result.node_counts.validate_citations == 1
    assert result.node_counts.safe_failure == 1
    assert '"answer"' not in result.model_dump_json()


class RaisingAnswerGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request) -> object:  # noqa: ANN001
        self.calls += 1
        raise RuntimeError(
            f"{PRIVATE_SOURCE} {SENSITIVE_SENTINEL} question={request.question}"
        )


def test_answer_generator_exception_is_redacted_and_bounded() -> None:
    scenario = _scenario("normal_chunk")
    dispatcher, retriever, _ = build_shared_dependencies(scenario)
    generator = RaisingAnswerGenerator()
    workflow = LangGraphRetrievalQA(
        dispatcher=dispatcher,
        answer_generator=generator,
    )

    result = workflow.run(
        scenario.question,
        episode_id=scenario.episode_id,
        top_k=2,
    )

    assert isinstance(result, FrameworkFailure)
    assert result.termination_reason == "answer_failed"
    assert len(retriever.calls) == generator.calls == 1
    serialized = result.model_dump_json()
    for forbidden in (
        scenario.question,
        PRIVATE_SOURCE,
        SENSITIVE_SENTINEL,
        "RuntimeError",
    ):
        assert forbidden not in serialized


def test_unexpected_failure_marks_call_counts_incomplete_after_dispatch() -> None:
    scenario = _scenario("normal_chunk")
    dispatcher, retriever, answer_program = build_shared_dependencies(scenario)
    workflow = LangGraphRetrievalQA(
        dispatcher=dispatcher,
        answer_generator=answer_program,
    )

    class FailingAfterDispatchGraph:
        def invoke(self, graph_input, config):  # noqa: ANN001
            del config
            dispatcher.dispatch(
                "search_transcript",
                {
                    "query": graph_input["question"],
                    "episode_id": graph_input["episode_id"],
                    "top_k": graph_input["top_k"],
                },
            )
            raise RuntimeError(
                f"{PRIVATE_SOURCE} {SENSITIVE_SENTINEL} after dispatch"
            )

    workflow.graph = FailingAfterDispatchGraph()  # type: ignore[assignment]

    result = workflow.run(
        scenario.question,
        episode_id=scenario.episode_id,
        top_k=2,
    )

    assert isinstance(result, FrameworkFailure)
    assert result.termination_reason == "workflow_failed"
    assert result.counts_complete is False
    assert result.dispatcher_call_count == 0
    assert result.answer_call_count == 0
    assert retriever.calls == [(scenario.question, 2)]
    serialized = result.model_dump_json()
    for forbidden in (
        scenario.question,
        PRIVATE_SOURCE,
        SENSITIVE_SENTINEL,
        "RuntimeError",
    ):
        assert forbidden not in serialized


def test_failure_result_and_trace_exclude_question_body_and_candidate_answer() -> None:
    workflow, _, _, scenario = _workflow("cross_episode_citation")

    result = workflow.run(
        scenario.question,
        episode_id=scenario.episode_id,
        top_k=2,
    )

    assert isinstance(result, FrameworkFailure)
    serialized = result.model_dump_json()
    for forbidden in (
        str(scenario.question),
        "A bounded synthetic answer.",
        PRIVATE_SOURCE,
        OTHER_EPISODE_ID,
        "Synthetic evidence body",
    ):
        assert forbidden not in serialized
    assert result.trace.question_length == len(str(scenario.question))
    assert len(result.trace.question_sha256) == 64


def test_dispatcher_is_the_only_retrieval_entry_and_executes_once() -> None:
    workflow, retriever, answer_program, scenario = _workflow()

    result = workflow.run(
        f"  {scenario.question}  ",
        episode_id=scenario.episode_id,
        top_k=2,
    )

    assert isinstance(result, FrameworkSuccess)
    assert retriever.calls == [(scenario.question, 2)]
    assert len(answer_program.day23_requests) == 1
    request = answer_program.day23_requests[0]
    assert request.retrieval.result_count == 2
    assert all(reference[1] == EPISODE_ID for reference in request.allowed_references)


def test_runtime_has_no_dynamic_execution_file_network_or_handler_bypass() -> None:
    workflow_path = PROJECT_ROOT / "experiments/framework_compare/workflow.py"
    tree = ast.parse(workflow_path.read_text(encoding="utf-8"))
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
        "retrieve",
        "run",
        "system",
        "write_bytes",
        "write_text",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden_names
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in forbidden_attributes
        if isinstance(node, ast.ImportFrom):
            assert node.module not in {
                "app.tooling.handlers",
                "app.retrieval.base",
            }
    source = workflow_path.read_text(encoding="utf-8")
    assert source.count("self.dispatcher.dispatch(") == 1
    assert "subprocess" not in source
    assert "requests" not in source


def test_ten_shared_cases_match_expected_safe_semantics() -> None:
    results = run_comparison_cases()

    assert len(results) == len(COMPARISON_SCENARIOS) == 10
    assert [item.case_id for item in results] == [
        item.case_id for item in COMPARISON_SCENARIOS
    ]
    for scenario, result in zip(COMPARISON_SCENARIOS, results, strict=True):
        assert result.day22_outcome == scenario.expected_day22
        assert result.day23_outcome == scenario.expected_day23
        assert result.semantic_match is True


def test_comparison_call_counts_show_the_intended_scope_difference() -> None:
    results = {item.case_id: item for item in run_comparison_cases()}

    for case_id in ("normal_chunk", "normal_segment"):
        result = results[case_id]
        assert result.day22_model_calls == 2
        assert result.day22_dispatcher_calls == 1
        assert result.day23_answer_calls == 1
        assert result.day23_dispatcher_calls == 1
    for case_id in ("empty_question", "overlong_question"):
        result = results[case_id]
        assert result.day22_model_calls == result.day23_answer_calls == 0
        assert result.day22_dispatcher_calls == result.day23_dispatcher_calls == 0


def test_shared_fake_data_reaches_both_real_workflows_without_real_services() -> None:
    scenario = _scenario("normal_chunk")
    day22_dispatcher, day22_retriever, day22_program = build_shared_dependencies(
        scenario
    )
    from app.agent import AgentConfig, AgentWorkflow

    day22 = AgentWorkflow(
        model=day22_program,
        dispatcher=day22_dispatcher,
        config=AgentConfig(max_steps=2),
    ).run(scenario.question)
    day23_workflow, day23_retriever, day23_program, _ = _workflow()
    day23 = day23_workflow.run(
        scenario.question,
        episode_id=scenario.episode_id,
        top_k=2,
    )

    assert isinstance(day22, AgentSuccess)
    assert isinstance(day23, FrameworkSuccess)
    assert day22.answer == day23.answer
    assert day22_retriever.calls == day23_retriever.calls
    assert day22_program.answer_output == day23_program.answer_output


def test_local_latency_benchmark_uses_warmup_and_equal_success_case() -> None:
    summary = benchmark_normal_case(warmup_iterations=2, iterations=5)

    assert summary.warmup_iterations == 2
    assert summary.iterations == 5
    assert summary.day22_mean_ms > 0
    assert summary.day22_median_ms > 0
    assert summary.day22_p95_ms > 0
    assert summary.day23_mean_ms > 0
    assert summary.day23_median_ms > 0
    assert summary.day23_p95_ms > 0


def test_langgraph_is_optional_and_absent_from_supported_app_imports() -> None:
    configuration = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert "langgraph>=1.2,<1.3" in configuration["project"][
        "optional-dependencies"
    ]["experiment"]
    assert "experiments*" in configuration["tool"]["setuptools"]["packages"][
        "find"
    ]["include"]
    assert all(
        "langgraph" not in path.read_text(encoding="utf-8")
        for path in (PROJECT_ROOT / "app").rglob("*.py")
    )


def test_verified_langgraph_version_is_in_the_declared_minor_line() -> None:
    version = importlib.metadata.version("langgraph")
    major, minor, *_ = (int(part) for part in version.split("."))
    assert (major, minor) == (1, 2)


def test_day22_failures_remain_the_original_strict_result_type() -> None:
    results = run_comparison_cases()
    missing = _scenario("episode_not_found")
    dispatcher, _, program = build_shared_dependencies(missing)
    from app.agent import AgentConfig, AgentWorkflow

    result = AgentWorkflow(
        model=program,
        dispatcher=dispatcher,
        config=AgentConfig(max_steps=2),
    ).run(missing.question)

    assert isinstance(result, AgentFailure)
    assert result.termination_reason == "not_found"
    assert any(item.case_id == "episode_not_found" for item in results)
    assert "langgraph" not in json.dumps(result.model_dump(mode="json"))
