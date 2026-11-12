"""Provider-neutral answer dependency for the Day 23 framework experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.agent.citations import ReferenceKey
from app.tooling.schemas import SearchTranscriptData


@dataclass(frozen=True)
class AnswerGeneratorRequest:
    """Trusted, bounded input given to the injected answer generator."""

    question: str
    retrieval: SearchTranscriptData
    allowed_references: tuple[ReferenceKey, ...]


class AnswerGenerator(Protocol):
    """Minimal fake-friendly boundary; no provider SDK is required."""

    def generate(self, request: AnswerGeneratorRequest) -> object:
        """Return one untrusted candidate answer proposal."""
