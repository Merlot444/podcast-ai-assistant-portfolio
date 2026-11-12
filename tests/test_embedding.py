"""Offline tests for the Day 7 Embedder boundary."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.indexing.embedding import (
    EmbeddingError,
    OpenAICompatibleEmbedder,
    SyntheticHashEmbedder,
)


def _response(items: list[tuple[int, list[float]]]) -> SimpleNamespace:
    return SimpleNamespace(
        data=[
            SimpleNamespace(index=index, embedding=embedding)
            for index, embedding in items
        ]
    )


def test_real_embedder_batches_documents_and_restores_response_order() -> None:
    client = Mock()
    client.embeddings.create.side_effect = [
        _response([(1, [0.0, 1.0]), (0, [1.0, 0.0])]),
        _response([(0, [0.5, 0.5])]),
    ]
    embedder = OpenAICompatibleEmbedder(
        client,
        model_name="embedding-test-model",
        batch_size=2,
        expected_dimension=2,
    )

    vectors = embedder.embed_documents(["first", "second", "third"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    assert embedder.document_batch_count == 2
    calls = client.embeddings.create.call_args_list
    assert calls[0].kwargs == {
        "model": "embedding-test-model",
        "input": ["first", "second"],
        "extra_body": {
            "task": "retrieval.passage",
            "embedding_type": "float",
            "normalized": True,
            "truncate": False,
            "dimensions": 2,
        },
    }
    assert calls[1].kwargs["input"] == ["third"]


def test_real_embedder_uses_a_separate_single_query_call() -> None:
    client = Mock()
    client.embeddings.create.return_value = _response([(0, [1.0, 2.0])])
    embedder = OpenAICompatibleEmbedder(
        client,
        model_name="embedding-test-model",
        expected_dimension=2,
    )

    vector = embedder.embed_query("question")

    assert vector == [1.0, 2.0]
    assert embedder.query_call_count == 1
    assert embedder.document_batch_count == 0
    request = client.embeddings.create.call_args.kwargs
    assert request["input"] == ["question"]
    assert request["extra_body"] == {
        "task": "retrieval.query",
        "embedding_type": "float",
        "normalized": True,
        "truncate": False,
        "dimensions": 2,
    }


def test_real_embedder_uses_explicit_jina_request_options() -> None:
    client = Mock()
    client.embeddings.create.return_value = _response([(0, [1.0])])
    embedder = OpenAICompatibleEmbedder(
        client,
        model_name="jina-embeddings-v5-text-small",
        expected_dimension=1,
        document_task="text-matching",
        query_task="classification",
        normalized=False,
        truncate=True,
    )

    embedder.embed_documents(["document"])
    document_body = client.embeddings.create.call_args.kwargs["extra_body"]
    embedder.embed_query("query")
    query_body = client.embeddings.create.call_args.kwargs["extra_body"]

    assert document_body["task"] == "text-matching"
    assert query_body["task"] == "classification"
    assert document_body["normalized"] is False
    assert document_body["truncate"] is True
    assert "encoding_format" not in client.embeddings.create.call_args.kwargs


@pytest.mark.parametrize(
    ("items", "message"),
    [
        ([(0, [1.0]), (0, [2.0])], "重复 index"),
        ([(1, [1.0])], "越界"),
        ([], "index 不连续"),
        ([(0, [1.0, 2.0])], "实际维度与配置不一致"),
    ],
)
def test_real_embedder_rejects_unmappable_or_wrong_dimension_responses(
    items: list[tuple[int, list[float]]],
    message: str,
) -> None:
    client = Mock()
    client.embeddings.create.return_value = _response(items)
    embedder = OpenAICompatibleEmbedder(
        client,
        model_name="embedding-test-model",
        expected_dimension=1,
    )

    with pytest.raises(EmbeddingError, match=message):
        embedder.embed_documents(["only input"])


def test_embedder_rejects_empty_document_and_query_before_a_request() -> None:
    client = Mock()
    embedder = OpenAICompatibleEmbedder(
        client,
        model_name="embedding-test-model",
    )

    with pytest.raises(EmbeddingError, match="列表不能为空"):
        embedder.embed_documents([])
    with pytest.raises(EmbeddingError, match="查询不能为空"):
        embedder.embed_query("   ")

    client.embeddings.create.assert_not_called()


def test_synthetic_hash_embedder_is_deterministic_and_explicitly_named() -> None:
    first = SyntheticHashEmbedder(32)
    second = SyntheticHashEmbedder(32)

    first_vectors = first.embed_documents(["planning systems", "world models"])
    second_vectors = second.embed_documents(["planning systems", "world models"])

    assert first.model_name == "synthetic-hash-v1-d32"
    assert first_vectors == second_vectors
    assert first.embed_query("planning systems") == first_vectors[0]
    assert first.document_call_count == 1
    assert first.query_call_count == 1
