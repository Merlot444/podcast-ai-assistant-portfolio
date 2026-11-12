"""Offline contract tests for the Dify-facing retrieval-only API."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from app.api import create_app
from app.api.adapters import EvaluationRetrieveService
from app.api.schemas import RetrieveRequest, RetrieveResponse
from app.api.services import RetrievedChunk, RetrieveServiceSuccess, ServiceFailure
from app.schemas.chunk import ChunkSourceSpan, TranscriptChunk
from app.schemas.retriever import RetrievedCandidate


EPISODE_ID = "ep_1111111111111111"
OTHER_EPISODE_ID = "ep_2222222222222222"
CHUNK_ID = "chunk_1111111111111111"
SEGMENT_ID = f"{EPISODE_ID}:000000"
QUESTION = "Why is hybrid retrieval used?"
FINGERPRINT = "a" * 64
BEARER_TOKEN = "synthetic-bearer-token-never-log-0001"


class FakeRetrieveService:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[RetrieveRequest] = []

    def retrieve(self, request: RetrieveRequest) -> object:
        self.calls.append(request)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class RecordingRetriever:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> object:
        self.calls.append((query, top_k))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def _chunk() -> TranscriptChunk:
    return TranscriptChunk(
        chunk_id=CHUNK_ID,
        chunk_index=0,
        episode_id=EPISODE_ID,
        source_file="synthetic-dify.json",
        text="Hybrid retrieval combines complementary retrieval signals.",
        source_segment_ids=[SEGMENT_ID],
        source_segment_start_index=0,
        source_segment_end_index=0,
        source_spans=[
            ChunkSourceSpan(
                segment_id=SEGMENT_ID,
                segment_index=0,
                start_char=0,
                end_char=59,
            )
        ],
        speakers=["Host"],
        section_titles=["Retrieval"],
        start_time=12.5,
        end_time=18.0,
        time_range_complete=True,
    )


def _candidate(
    *,
    rank: int = 1,
    chunk_id: str = CHUNK_ID,
    source_segment_ids: tuple[str, ...] = (SEGMENT_ID,),
) -> RetrievedCandidate:
    return RetrievedCandidate(
        rank=rank,
        chunk_id=chunk_id,
        source_segment_ids=source_segment_ids,
        score=0.91,
        score_semantics="cosine_similarity",
        retriever_name="vector",
    )


def _success(*, chunks: tuple[RetrievedChunk, ...] | None = None) -> RetrieveServiceSuccess:
    resolved = chunks
    if resolved is None:
        resolved = (
            RetrievedChunk(
                rank=1,
                chunk_id=CHUNK_ID,
                text=_chunk().text,
                source_segment_ids=(SEGMENT_ID,),
                score=0.91,
                score_semantics="cosine_similarity",
                retriever_name="vector",
                speakers=("Host",),
                section_titles=("Retrieval",),
                start_time=12.5,
                end_time=18.0,
                time_range_complete=True,
            ),
        )
    return RetrieveServiceSuccess(
        episode_id=EPISODE_ID,
        input_chunk_sha256=FINGERPRINT,
        chunks=resolved,
    )


def _clock(*values: float):
    pending = iter(values)
    return lambda: next(pending)


def test_retrieve_route_returns_dify_ready_evidence_and_metrics() -> None:
    service = FakeRetrieveService(_success())

    with TestClient(
        create_app(
            retrieve_service=service,
            retrieval_clock=_clock(10.0, 10.012345),
        )
    ) as client:
        response = client.post(
            "/retrieve",
            json={
                "query": f"  {QUESTION}  ",
                "episode_id": EPISODE_ID,
                "top_k": 5,
            },
        )

    parsed = RetrieveResponse.model_validate_json(response.text)
    assert response.status_code == 200
    assert parsed.result_count == 1
    assert parsed.retrieval_call_count == 1
    assert parsed.duration_ms == 12.345
    assert parsed.input_chunk_sha256 == FINGERPRINT
    assert parsed.chunks[0].chunk_id == CHUNK_ID
    assert parsed.chunks[0].text == _chunk().text
    assert parsed.chunks[0].source_segment_ids == (SEGMENT_ID,)
    assert service.calls[0].query == QUESTION


def test_empty_retrieval_is_a_200_success_for_dify_branching() -> None:
    service = FakeRetrieveService(_success(chunks=()))

    with TestClient(
        create_app(
            retrieve_service=service,
            retrieval_clock=_clock(2.0, 2.001),
        )
    ) as client:
        response = client.post(
            "/retrieve",
            json={"query": QUESTION, "episode_id": EPISODE_ID, "top_k": 5},
        )

    assert response.status_code == 200
    assert response.json()["result_count"] == 0
    assert response.json()["chunks"] == []
    assert len(service.calls) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"episode_id": EPISODE_ID, "top_k": 5},
        {"query": " ", "episode_id": EPISODE_ID, "top_k": 5},
        {"query": QUESTION, "episode_id": "ep_demo_001", "top_k": 5},
        {"query": QUESTION, "episode_id": EPISODE_ID, "top_k": 0},
        {"query": QUESTION, "episode_id": EPISODE_ID, "top_k": 9},
        {"query": QUESTION, "episode_id": EPISODE_ID, "top_k": "5"},
        {"query": QUESTION, "episode_id": EPISODE_ID, "top_k": 5, "extra": True},
    ],
)
def test_invalid_retrieve_request_never_calls_service(
    payload: dict[str, object],
) -> None:
    service = FakeRetrieveService(_success())

    with TestClient(create_app(retrieve_service=service)) as client:
        response = client.post("/retrieve", json=payload)

    assert response.status_code == 422
    assert response.json() == {
        "ok": False,
        "error": {"code": "invalid_request", "message": "请求格式无效。"},
    }
    assert service.calls == []


def test_default_retrieve_service_is_explicitly_unavailable() -> None:
    with TestClient(create_app()) as client:
        response = client.post(
            "/retrieve",
            json={"query": QUESTION, "episode_id": EPISODE_ID, "top_k": 5},
        )

    assert response.status_code == 503
    assert response.json()["error"] == {
        "code": "service_unavailable",
        "message": "检索服务暂时不可用。",
    }


@pytest.mark.parametrize(
    "authorization", [None, "Bearer wrong-token-value-000000000000", b"Bearer \xff"]
)
def test_bearer_auth_rejects_missing_or_wrong_token_before_service_call(
    authorization: str | bytes | None,
) -> None:
    service = FakeRetrieveService(_success())
    headers = {} if authorization is None else {"Authorization": authorization}

    with TestClient(
        create_app(
            retrieve_service=service,
            api_bearer_token=BEARER_TOKEN,
        )
    ) as client:
        response = client.post(
            "/retrieve",
            headers=headers,
            json={"query": QUESTION, "episode_id": EPISODE_ID, "top_k": 5},
        )

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json() == {
        "ok": False,
        "error": {"code": "unauthorized", "message": "认证失败。"},
    }
    assert service.calls == []


def test_correct_bearer_auth_allows_retrieval_and_health_remains_public() -> None:
    service = FakeRetrieveService(_success())

    with TestClient(
        create_app(
            retrieve_service=service,
            api_bearer_token=BEARER_TOKEN,
            retrieval_clock=_clock(3.0, 3.01),
        )
    ) as client:
        health = client.get("/health")
        retrieve = client.post(
            "/retrieve",
            headers={"Authorization": f"Bearer {BEARER_TOKEN}"},
            json={"query": QUESTION, "episode_id": EPISODE_ID, "top_k": 5},
        )

    assert health.status_code == 200
    assert retrieve.status_code == 200
    assert len(service.calls) == 1


def test_invalid_bearer_token_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="配置无效"):
        create_app(api_bearer_token="short")


@pytest.mark.parametrize(
    ("failure", "status", "code"),
    [
        (ServiceFailure(code="invalid_request"), 422, "invalid_request"),
        (ServiceFailure(code="not_found"), 404, "not_found"),
        (ServiceFailure(code="service_unavailable"), 503, "service_unavailable"),
        (ServiceFailure(code="service_failed"), 503, "service_failed"),
    ],
)
def test_retrieve_failures_have_stable_http_statuses(
    failure: ServiceFailure,
    status: int,
    code: str,
) -> None:
    service = FakeRetrieveService(failure)

    with TestClient(
        create_app(
            retrieve_service=service,
            retrieval_clock=_clock(1.0, 1.01),
        )
    ) as client:
        response = client.post(
            "/retrieve",
            json={"query": QUESTION, "episode_id": EPISODE_ID, "top_k": 5},
        )

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert len(service.calls) == 1


def test_adapter_calls_retriever_once_and_resolves_trusted_chunk_text() -> None:
    retriever = RecordingRetriever([_candidate()])
    service = EvaluationRetrieveService(
        retriever=retriever,  # type: ignore[arg-type]
        chunks=(_chunk(),),
        episode_id=EPISODE_ID,
        input_chunk_sha256=FINGERPRINT,
    )

    result = service.retrieve(
        RetrieveRequest(query=QUESTION, episode_id=EPISODE_ID, top_k=5)
    )

    assert isinstance(result, RetrieveServiceSuccess)
    assert result.chunks[0].text == _chunk().text
    assert result.chunks[0].source_segment_ids == (SEGMENT_ID,)
    assert retriever.calls == [(QUESTION, 5)]


def test_adapter_returns_empty_success_and_rejects_other_episode_without_call() -> None:
    retriever = RecordingRetriever([])
    service = EvaluationRetrieveService(
        retriever=retriever,  # type: ignore[arg-type]
        chunks=(_chunk(),),
        episode_id=EPISODE_ID,
        input_chunk_sha256=FINGERPRINT,
    )

    empty = service.retrieve(
        RetrieveRequest(query=QUESTION, episode_id=EPISODE_ID, top_k=5)
    )
    missing = service.retrieve(
        RetrieveRequest(query=QUESTION, episode_id=OTHER_EPISODE_ID, top_k=5)
    )

    assert isinstance(empty, RetrieveServiceSuccess)
    assert empty.chunks == ()
    assert missing == ServiceFailure(code="not_found")
    assert retriever.calls == [(QUESTION, 5)]


@pytest.mark.parametrize(
    "result",
    [
        RuntimeError("private provider failure"),
        [_candidate(rank=2)],
        [_candidate(chunk_id="chunk_2222222222222222")],
        [_candidate(source_segment_ids=(f"{EPISODE_ID}:000001",))],
    ],
)
def test_adapter_fails_closed_on_retriever_or_provenance_errors(
    result: object,
) -> None:
    retriever = RecordingRetriever(result)
    service = EvaluationRetrieveService(
        retriever=retriever,  # type: ignore[arg-type]
        chunks=(_chunk(),),
        episode_id=EPISODE_ID,
        input_chunk_sha256=FINGERPRINT,
    )

    outcome = service.retrieve(
        RetrieveRequest(query=QUESTION, episode_id=EPISODE_ID, top_k=5)
    )

    assert outcome == ServiceFailure(code="service_failed")
    assert retriever.calls == [(QUESTION, 5)]


def test_production_bundle_reuses_one_assembled_retriever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import bootstrap

    ask_service = object()
    retriever = RecordingRetriever([_candidate()])
    vector_index = SimpleNamespace(
        manifest=SimpleNamespace(
            chunks=[_chunk()],
            input_chunk_sha256=FINGERPRINT,
        )
    )
    assembly = bootstrap._AgentAssembly(  # noqa: SLF001 - composition seam test
        ask_service=ask_service,  # type: ignore[arg-type]
        vector_index=vector_index,
        vector_retriever=retriever,
    )
    monkeypatch.setattr(
        bootstrap,
        "_build_agent_assembly",
        lambda *_args, **_kwargs: assembly,
    )

    services = bootstrap.build_agent_api_services(
        SimpleNamespace(episode_id=EPISODE_ID),  # type: ignore[arg-type]
    )

    assert services.ask_service is ask_service
    result = services.retrieve_service.retrieve(
        RetrieveRequest(query=QUESTION, episode_id=EPISODE_ID, top_k=5)
    )
    assert isinstance(result, RetrieveServiceSuccess)
    assert retriever.calls == [(QUESTION, 5)]
