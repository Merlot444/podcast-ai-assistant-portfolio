"""Full episode context must be authenticated, complete and provider-free."""

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from app.api import create_app
from app.api.adapters import EvaluationRetrieveService
from tests.test_dify_retrieve_api import (
    BEARER_TOKEN, EPISODE_ID, OTHER_EPISODE_ID, FINGERPRINT,
    RecordingRetriever, _chunk,
)


def test_full_context_requires_auth_and_never_calls_retriever():
    retriever = RecordingRetriever(RuntimeError("must not call"))
    service = EvaluationRetrieveService(
        retriever=retriever, chunks=(_chunk(),), episode_id=EPISODE_ID,
        input_chunk_sha256=FINGERPRINT,
    )
    client = TestClient(create_app(retrieve_service=service, api_bearer_token=BEARER_TOKEN))
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    assert client.post("/summary-context", json={"episode_id": EPISODE_ID}).status_code == 401
    assert client.post("/summary-context", headers=headers, json={"episode_id": OTHER_EPISODE_ID}).status_code == 404
    response = client.post("/summary-context", headers=headers, json={"episode_id": EPISODE_ID})
    assert response.status_code == 200
    body = response.json()
    assert body["coverage"] == "full_episode"
    assert body["retrieval_call_count"] == 0
    assert body["result_count"] == len(body["chunks"]) == 1
    assert body["chunks"][0]["text"] == _chunk().text
    assert "source_file" not in response.text
    assert retriever.calls == []
    routed = client.post(
        "/evidence/EPISODE_SUMMARY", headers=headers,
        json={"episode_id": EPISODE_ID, "query": "Summarize the episode", "top_k": 5},
    )
    assert routed.status_code == 200
    assert routed.json()["coverage"] == "full_episode"
    assert retriever.calls == []
    assert client.post(
        "/evidence/UNSUPPORTED", headers=headers,
        json={"episode_id": EPISODE_ID, "query": "hello", "top_k": 5},
    ).status_code == 422
    unauthenticated_app = TestClient(create_app(retrieve_service=service))
    assert unauthenticated_app.post("/summary-context", json={"episode_id": EPISODE_ID}).status_code == 503


def test_oversized_episode_fails_without_truncation():
    chunk = _chunk().model_copy(update={"text": "x" * 200001})
    service = EvaluationRetrieveService(
        retriever=RecordingRetriever([]), chunks=(chunk,), episode_id=EPISODE_ID,
        input_chunk_sha256=FINGERPRINT,
    )
    client = TestClient(create_app(retrieve_service=service, api_bearer_token=BEARER_TOKEN))
    response = client.post("/summary-context", headers={"Authorization": f"Bearer {BEARER_TOKEN}"}, json={"episode_id": EPISODE_ID})
    assert response.status_code == 503
    assert "x" * 100 not in response.text
