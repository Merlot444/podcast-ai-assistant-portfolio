"""Small JSON Lines event boundary for the Day 10 RAG pipeline."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, TextIO


EVENT_FIELDS = frozenset(
    {
        "event",
        "stage",
        "status",
        "elapsed_seconds",
        "episode_id",
        "input_file",
        "input_kind",
        "input_segment_count",
        "chunk_count",
        "index_directory",
        "index_action",
        "top_k",
        "retrieved_count",
        "vector_count",
        "vector_dimension",
        "embedding_model",
        "answer_model",
        "query_embedding_calls",
        "document_embedding_calls",
        "document_embedding_batches",
        "answer_model_calls",
        "has_sufficient_evidence",
        "citation_count",
        "error_type",
        "error_category",
        "error_message",
    }
)


class RagEventSink(Protocol):
    """Receive one already-redacted structured pipeline event."""

    def emit(self, event: Mapping[str, object]) -> None:
        """Persist or collect one event."""


@dataclass
class JsonLineEventSink:
    """Write one compact JSON object per line to the configured stream."""

    stream: TextIO

    def emit(self, event: Mapping[str, object]) -> None:
        unexpected = set(event) - EVENT_FIELDS
        if unexpected:
            raise ValueError(
                "结构化日志包含未允许字段：" + ", ".join(sorted(unexpected))
            )
        self.stream.write(
            json.dumps(
                dict(event),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        self.stream.flush()


class NullEventSink:
    """Discard events when a caller does not need observable logging."""

    def emit(self, event: Mapping[str, object]) -> None:
        del event
