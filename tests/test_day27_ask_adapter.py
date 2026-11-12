"""Offline Day 27 AskService adapter tests over the bounded Agent boundary."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from app.agent import AgentConfig, AgentWorkflow
from app.api.adapters import AgentWorkflowAskService
from app.api.schemas import AskRequest
from app.api.services import AskServiceSuccess, ServiceFailure
from app.schemas.retriever import RetrievedCandidate
from app.schemas.transcript import ProcessedTranscript, TranscriptSegment
from app.tooling import ToolDependencies, ToolDispatcher, ToolHandlers


EPISODE_ID = "ep_1111111111111111"
OTHER_EPISODE_ID = "ep_2222222222222222"
QUESTION = "What does the synthetic episode say?"
SEGMENT_ID = f"{EPISODE_ID}:000000"
CHUNK_ID = "chunk_1111111111111111"


class ScriptedModel:
    def __init__(self, decisions: Sequence[object]) -> None:
        self.decisions = list(decisions)
        self.requests: list[object] = []

    def decide(self, request: object) -> object:
        self.requests.append(request)
        decision = self.decisions[len(self.requests) - 1]
        if isinstance(decision, BaseException):
            raise decision
        return decision


class RecordingRetriever:
    name = "vector"

    def __init__(self, *, return_empty: bool = False) -> None:
        self.calls: list[tuple[str, int]] = []
        self.return_empty = return_empty

    def retrieve(self, query: str, top_k: int) -> list[RetrievedCandidate]:
        self.calls.append((query, top_k))
        if self.return_empty:
            return []
        return [
            RetrievedCandidate(
                rank=1,
                chunk_id=CHUNK_ID,
                source_segment_ids=(SEGMENT_ID,),
                score=0.9,
                score_semantics="cosine_similarity",
                retriever_name="vector",
            )
        ]


def _transcript() -> ProcessedTranscript:
    return ProcessedTranscript(
        episode_id=EPISODE_ID,
        source_file="synthetic-day27.json",
        source_format="json",
        segments=[
            TranscriptSegment(
                segment_id=SEGMENT_ID,
                segment_index=0,
                text="Original synthetic evidence for an offline adapter test.",
                source_file="synthetic-day27.json",
            )
        ],
    )


def _request(*, episode_id: str = EPISODE_ID, top_k: int = 3) -> AskRequest:
    return AskRequest(question=QUESTION, episode_id=episode_id, top_k=top_k)


def _tool_call(*, episode_id: str = EPISODE_ID, top_k: int = 3) -> dict[str, object]:
    return {
        "decision_type": "tool_call",
        "tool_name": "search_transcript",
        "arguments": {
            "query": QUESTION,
            "episode_id": episode_id,
            "top_k": top_k,
        },
    }


def _final(*, reference_id: str = CHUNK_ID) -> dict[str, object]:
    return {
        "decision_type": "final_answer",
        "final_answer": {
            "answer": "A bounded synthetic answer.",
            "citations": [
                {
                    "kind": "chunk",
                    "episode_id": EPISODE_ID,
                    "reference_id": reference_id,
                }
            ],
        },
    }


def _service(
    decisions: Sequence[object],
    *,
    return_empty: bool = False,
) -> tuple[AgentWorkflowAskService, RecordingRetriever, ScriptedModel]:
    retriever = RecordingRetriever(return_empty=return_empty)
    dispatcher = ToolDispatcher(
        ToolHandlers(
            ToolDependencies(
                transcripts={EPISODE_ID: _transcript()},
                retrievers={EPISODE_ID: retriever},
            )
        )
    )
    model = ScriptedModel(decisions)
    workflow = AgentWorkflow(
        model=model,
        dispatcher=dispatcher,
        config=AgentConfig(require_tool_grounding=True),
    )
    return AgentWorkflowAskService(workflow=workflow, episode_id=EPISODE_ID), retriever, model


def test_success_maps_existing_verified_answer_and_honors_request_scope() -> None:
    service, retriever, model = _service([_tool_call(), _final()])

    result = service.ask(_request())

    assert isinstance(result, AskServiceSuccess)
    assert result.answer.answer == "A bounded synthetic answer."
    assert result.answer.citations[0].reference_id == CHUNK_ID
    assert retriever.calls == [(QUESTION, 3)]
    assert len(model.requests) == 2


def test_missing_episode_and_empty_results_keep_current_service_mappings() -> None:
    service, retriever, model = _service([_final()])

    result = service.ask(_request(episode_id=OTHER_EPISODE_ID))

    assert result == ServiceFailure(code="not_found")
    assert model.requests == []
    assert retriever.calls == []

    service, retriever, model = _service([_tool_call()], return_empty=True)

    result = service.ask(_request())

    assert result == ServiceFailure(code="service_failed")
    assert len(model.requests) == 1
    assert retriever.calls == [(QUESTION, 3)]


@pytest.mark.parametrize(
    ("decisions", "expected_retriever_calls"),
    [
        ([RuntimeError("private provider error with key")], []),
        ([_tool_call(top_k=2)], []),
        (
            [_tool_call(), _final(reference_id="chunk_9999999999999999")],
            [(QUESTION, 3)],
        ),
    ],
    ids=["provider_failure", "scope_violation", "citation_not_from_run"],
)
def test_workflow_failures_map_to_one_safe_service_failure(
    decisions: Sequence[object],
    expected_retriever_calls: list[tuple[str, int]],
) -> None:
    service, retriever, model = _service(decisions)

    result = service.ask(_request())

    assert result == ServiceFailure(code="service_failed")
    assert len(model.requests) >= 1
    assert retriever.calls == expected_retriever_calls


def test_runner_exception_and_malformed_result_are_safe_failures() -> None:
    class BrokenRunner:
        def __init__(self, result: object) -> None:
            self.result = result

        def run(self, question: object, **scope: object) -> object:
            del question, scope
            if isinstance(self.result, BaseException):
                raise self.result
            return self.result

    for value in (RuntimeError("private traceback key"), {"ok": True}):
        service = AgentWorkflowAskService(
            workflow=BrokenRunner(value),  # type: ignore[arg-type]
            episode_id=EPISODE_ID,
        )
        assert service.ask(_request()) == ServiceFailure(code="service_failed")


def test_adapter_rejects_invalid_configured_episode() -> None:
    with pytest.raises(ValueError, match="配置无效"):
        AgentWorkflowAskService(
            workflow=object(),  # type: ignore[arg-type]
            episode_id="invalid-private-path",
        )
