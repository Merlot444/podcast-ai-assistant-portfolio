"""Provider-neutral model dependency boundary for the Day 22 Agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.tooling.schemas import ToolSuccess


@dataclass(frozen=True)
class AgentModelRequest:
    """Trusted context supplied to an injected decision model.

    Tool definitions are JSON-schema descriptions, never Python callables. Tool
    results are bounded Day 21 success envelopes from the current run only.
    The optional episode/top-K pair is a trusted request scope supplied by an
    application adapter; legacy Day 22 callers may omit both values.
    """

    question: str
    step_number: int
    max_steps: int
    tool_definitions: tuple[dict[str, object], ...]
    tool_results: tuple[ToolSuccess, ...]
    episode_id: str | None = None
    top_k: int | None = None


class AgentModel(Protocol):
    """Minimal replaceable decision interface; no provider SDK is required."""

    def decide(self, request: AgentModelRequest) -> object:
        """Return one untrusted tool-call or final-answer proposal."""
