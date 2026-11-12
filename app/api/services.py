"""Minimal synchronous application-service protocols injected into the API."""

from __future__ import annotations

from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from app.agent import AgentFinalAnswer
from app.api.schemas import AskRequest, RetrieveRequest, SessionMetadata
from app.tooling.schemas import EPISODE_ID_PATTERN


ServiceFailureCode = Literal[
    "invalid_request",
    "not_found",
    "service_unavailable",
    "service_failed",
]


class ServiceSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
    )


class AskServiceSuccess(ServiceSchema):
    ok: Literal[True] = True
    answer: AgentFinalAnswer


class RetrievedChunk(ServiceSchema):
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


class RetrieveServiceSuccess(ServiceSchema):
    ok: Literal[True] = True
    episode_id: str = Field(pattern=EPISODE_ID_PATTERN)
    input_chunk_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunks: tuple[RetrievedChunk, ...] = Field(max_length=8)


class ServiceFailure(ServiceSchema):
    """A safe code only; lower-layer messages never cross into HTTP responses."""

    ok: Literal[False] = False
    code: ServiceFailureCode


AskServiceResult = AskServiceSuccess | ServiceFailure
ASK_SERVICE_RESULT_ADAPTER = TypeAdapter(
    Annotated[AskServiceResult, Field(discriminator="ok")]
)
RetrieveServiceResult = RetrieveServiceSuccess | ServiceFailure
RETRIEVE_SERVICE_RESULT_ADAPTER = TypeAdapter(
    Annotated[RetrieveServiceResult, Field(discriminator="ok")]
)
SessionReadResult = SessionMetadata | ServiceFailure | None
SESSION_READ_RESULT_ADAPTER = TypeAdapter(SessionReadResult)


class AskService(Protocol):
    def ask(self, request: AskRequest) -> AskServiceResult:
        """Return one already business-validated answer or a safe failure code."""


class RetrieveService(Protocol):
    def retrieve(self, request: RetrieveRequest) -> RetrieveServiceResult:
        """Return ranked evidence without generating an answer."""


class SessionReader(Protocol):
    def get_session(self, session_id: str) -> SessionReadResult:
        """Read only session metadata; never create, update, or delete a session."""


class UnavailableAskService:
    def ask(self, request: AskRequest) -> ServiceFailure:
        return ServiceFailure(code="service_unavailable")


class UnavailableRetrieveService:
    def retrieve(self, request: RetrieveRequest) -> ServiceFailure:
        return ServiceFailure(code="service_unavailable")


class UnavailableSessionReader:
    def get_session(self, session_id: str) -> ServiceFailure:
        return ServiceFailure(code="service_unavailable")
