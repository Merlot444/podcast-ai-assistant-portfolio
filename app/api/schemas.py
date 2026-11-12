"""Strict HTTP request, success, session, and error contracts for Day 24."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.agent.schemas import (
    MAX_AGENT_ANSWER_CHARACTERS,
    MAX_AGENT_CITATIONS,
    MAX_AGENT_QUESTION_CHARACTERS,
    AgentCitation,
)
from app.storage.schemas import STABLE_ID_PATTERN, StableId
from app.tooling.schemas import EPISODE_ID_PATTERN


SESSION_ID_PATTERN = STABLE_ID_PATTERN
ApiErrorCode = Literal[
    "invalid_request",
    "unauthorized",
    "not_found",
    "service_unavailable",
    "service_failed",
    "internal_error",
]


class ApiSchema(BaseModel):
    """Strict base for every JSON value crossing the HTTP boundary."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
    )


class AskRequest(ApiSchema):
    question: str = Field(min_length=1, max_length=MAX_AGENT_QUESTION_CHARACTERS)
    episode_id: str = Field(pattern=EPISODE_ID_PATTERN)
    top_k: int = Field(ge=1, le=8)
    session_id: StableId | None = None

    @field_validator("question", "episode_id", "session_id", mode="before")
    @classmethod
    def strip_string_fields(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class RetrieveRequest(ApiSchema):
    query: str = Field(min_length=1, max_length=MAX_AGENT_QUESTION_CHARACTERS)
    episode_id: str = Field(pattern=EPISODE_ID_PATTERN)
    top_k: int = Field(default=5, ge=1, le=8)

    @field_validator("query", "episode_id", mode="before")
    @classmethod
    def strip_string_fields(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class HealthResponse(ApiSchema):
    ok: Literal[True] = True
    status: Literal["ok"] = "ok"


class AskResponse(ApiSchema):
    ok: Literal[True] = True
    answer: str = Field(min_length=1, max_length=MAX_AGENT_ANSWER_CHARACTERS)
    citations: tuple[AgentCitation, ...] = Field(max_length=MAX_AGENT_CITATIONS)

    @field_validator("answer")
    @classmethod
    def strip_answer(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("answer 不能为空")
        return cleaned

    @model_validator(mode="after")
    def require_unique_citations(self) -> Self:
        keys = [
            (item.kind, item.episode_id, item.reference_id)
            for item in self.citations
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("citations 不能重复")
        return self


class RetrievedChunkResponse(ApiSchema):
    rank: int = Field(ge=1)
    chunk_id: str = Field(pattern=r"^chunk_[0-9a-f]{16}$")
    text: str = Field(min_length=1)
    source_segment_ids: tuple[str, ...] = Field(min_length=1)
    score: float = Field(allow_inf_nan=False)
    score_semantics: str = Field(min_length=1, max_length=64)
    retriever_name: str = Field(min_length=1, max_length=100)
    speakers: tuple[str, ...]
    section_titles: tuple[str, ...]
    start_time: float | None = Field(default=None, ge=0)
    end_time: float | None = Field(default=None, ge=0)
    time_range_complete: bool


class RetrieveResponse(ApiSchema):
    ok: Literal[True] = True
    episode_id: str = Field(pattern=EPISODE_ID_PATTERN)
    result_count: int = Field(ge=0)
    retrieval_call_count: Literal[1] = 1
    duration_ms: float = Field(ge=0, allow_inf_nan=False)
    input_chunk_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunks: tuple[RetrievedChunkResponse, ...] = Field(max_length=8)

    @model_validator(mode="after")
    def require_consistent_result_count(self) -> Self:
        if self.result_count != len(self.chunks):
            raise ValueError("result_count 必须与 chunks 数量一致")
        return self


class SessionMetadata(ApiSchema):
    session_id: StableId
    title: str | None = Field(default=None, max_length=500)
    status: Literal["active", "closed"]
    created_at: datetime
    updated_at: datetime

    @field_validator("created_at", "updated_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("session 时间必须包含时区")
        return value


class SessionResponse(ApiSchema):
    ok: Literal[True] = True
    session: SessionMetadata


class ApiError(ApiSchema):
    code: ApiErrorCode
    message: str = Field(min_length=1, max_length=200)


class ApiErrorResponse(ApiSchema):
    ok: Literal[False] = False
    error: ApiError
