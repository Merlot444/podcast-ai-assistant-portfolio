"""Minimal retrieval boundary consumed by the Day 12 evaluator."""

from __future__ import annotations

from typing import Protocol

from app.schemas.retriever import RetrievedCandidate


class EvaluationRetriever(Protocol):
    """Return one stable ordered candidate list for an evaluation question."""

    @property
    def name(self) -> str:
        """Return a stable retriever identity for result provenance."""

    def retrieve(self, query: str, top_k: int) -> list[RetrievedCandidate]:
        """Retrieve at most ``top_k`` candidates without answering the query."""
