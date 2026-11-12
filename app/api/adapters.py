"""Thin read-only adapters from existing application services to API contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

from app.agent.schemas import AgentFailure, AgentRunResult, AgentSuccess
from app.api.schemas import AskRequest, RetrieveRequest, SessionMetadata
from app.api.services import (
    AskServiceSuccess,
    RetrievedChunk,
    RetrieveServiceSuccess,
    ServiceFailure,
)
from app.schemas.chunk import TranscriptChunk
from app.schemas.retriever import RetrievedCandidate
from app.storage.schemas import SessionRead
from app.tooling.schemas import EPISODE_ID_PATTERN


class ConversationSessionSource(Protocol):
    """The read-only subset implemented by ConversationPersistenceService."""

    def get(self, session_id: str) -> SessionRead | None:
        ...


class BoundedAgentRunner(Protocol):
    """The request-scoped subset exposed by the existing Agent Workflow."""

    def run(
        self,
        question: object,
        *,
        episode_id: object | None = None,
        top_k: object | None = None,
    ) -> AgentRunResult:
        ...


class EvidenceRetriever(Protocol):
    """The metadata-only retrieval subset required by the HTTP adapter."""

    def retrieve(self, query: str, top_k: int) -> list[RetrievedCandidate]:
        ...


@dataclass(frozen=True)
class AgentWorkflowAskService:
    """Map one strict Ask request to the existing bounded Agent Workflow."""

    workflow: BoundedAgentRunner
    episode_id: str

    def __post_init__(self) -> None:
        if re.fullmatch(EPISODE_ID_PATTERN, self.episode_id) is None:
            raise ValueError("Ask Service episode 配置无效")

    def ask(self, request: AskRequest) -> AskServiceSuccess | ServiceFailure:
        if request.episode_id != self.episode_id:
            return ServiceFailure(code="not_found")
        try:
            result = self.workflow.run(
                request.question,
                episode_id=request.episode_id,
                top_k=request.top_k,
            )
        except Exception:
            return ServiceFailure(code="service_failed")
        if isinstance(result, AgentSuccess):
            return AskServiceSuccess(answer=result.answer)
        if not isinstance(result, AgentFailure):
            return ServiceFailure(code="service_failed")
        if result.termination_reason == "invalid_request":
            return ServiceFailure(code="invalid_request")
        if result.termination_reason == "not_found":
            return ServiceFailure(code="not_found")
        return ServiceFailure(code="service_failed")


@dataclass(frozen=True)
class EvaluationRetrieveService:
    """Resolve metadata-only candidates against one trusted Chunk catalog."""

    retriever: EvidenceRetriever
    chunks: tuple[TranscriptChunk, ...]
    episode_id: str
    input_chunk_sha256: str
    _catalog: Mapping[str, TranscriptChunk] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if re.fullmatch(EPISODE_ID_PATTERN, self.episode_id) is None:
            raise ValueError("Retrieve Service episode 配置无效")
        if re.fullmatch(r"[0-9a-f]{64}", self.input_chunk_sha256) is None:
            raise ValueError("Retrieve Service Chunk 指纹配置无效")
        if not self.chunks:
            raise ValueError("Retrieve Service Chunk 目录不能为空")
        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Retrieve Service Chunk ID 不能重复")
        if any(chunk.episode_id != self.episode_id for chunk in self.chunks):
            raise ValueError("Retrieve Service Chunk episode 配置不一致")
        object.__setattr__(
            self,
            "_catalog",
            MappingProxyType({chunk.chunk_id: chunk for chunk in self.chunks}),
        )

    def retrieve(
        self,
        request: RetrieveRequest,
    ) -> RetrieveServiceSuccess | ServiceFailure:
        if request.episode_id != self.episode_id:
            return ServiceFailure(code="not_found")
        try:
            candidates = self.retriever.retrieve(request.query, request.top_k)
            resolved: list[RetrievedChunk] = []
            seen_chunk_ids: set[str] = set()
            for expected_rank, candidate in enumerate(candidates, start=1):
                if expected_rank > request.top_k or candidate.rank != expected_rank:
                    return ServiceFailure(code="service_failed")
                if candidate.chunk_id in seen_chunk_ids:
                    return ServiceFailure(code="service_failed")
                seen_chunk_ids.add(candidate.chunk_id)
                chunk = self._catalog.get(candidate.chunk_id)
                if chunk is None:
                    return ServiceFailure(code="service_failed")
                if tuple(chunk.source_segment_ids) != candidate.source_segment_ids:
                    return ServiceFailure(code="service_failed")
                resolved.append(
                    RetrievedChunk(
                        rank=candidate.rank,
                        chunk_id=candidate.chunk_id,
                        text=chunk.text,
                        source_segment_ids=candidate.source_segment_ids,
                        score=candidate.score,
                        score_semantics=candidate.score_semantics,
                        retriever_name=candidate.retriever_name,
                        speakers=tuple(chunk.speakers),
                        section_titles=tuple(chunk.section_titles),
                        start_time=chunk.start_time,
                        end_time=chunk.end_time,
                        time_range_complete=chunk.time_range_complete,
                    )
                )
        except Exception:
            return ServiceFailure(code="service_failed")
        return RetrieveServiceSuccess(
            episode_id=self.episode_id,
            input_chunk_sha256=self.input_chunk_sha256,
            chunks=tuple(resolved),
        )


@dataclass(frozen=True)
class ConversationSessionReader:
    """Expose only ConversationPersistenceService.get() and a fixed projection."""

    service: ConversationSessionSource

    def get_session(self, session_id: str) -> SessionMetadata | None:
        stored = self.service.get(session_id)
        if stored is None:
            return None
        return SessionMetadata(
            session_id=stored.session_id,
            title=stored.title,
            status=stored.status,
            created_at=stored.created_at,
            updated_at=stored.updated_at,
        )
