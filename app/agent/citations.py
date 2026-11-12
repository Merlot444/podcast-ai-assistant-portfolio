"""Application-owned citation provenance rules shared by bounded workflows."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import TypeAlias

from app.agent.schemas import AgentCitation, AgentReferenceKind
from app.tooling.schemas import (
    EpisodeMetadataData,
    SearchTranscriptData,
    SegmentContextData,
    ToolSuccess,
)


ReferenceKey: TypeAlias = tuple[AgentReferenceKind, str, str]


def references_from_tool_success(result: ToolSuccess) -> frozenset[ReferenceKey]:
    """Return exactly the citation identities contributed by one trusted result."""
    data = result.data
    references: set[ReferenceKey] = set()
    if isinstance(data, SearchTranscriptData):
        references.add(("episode", data.episode_id, data.episode_id))
        for hit in data.hits:
            references.add(("chunk", data.episode_id, hit.chunk_id))
            references.update(
                ("segment", data.episode_id, segment_id)
                for segment_id in hit.source_segment_ids
            )
        return frozenset(references)
    if isinstance(data, EpisodeMetadataData):
        references.add(("episode", data.episode_id, data.episode_id))
        return frozenset(references)
    if isinstance(data, SegmentContextData):
        references.add(("episode", data.episode_id, data.episode_id))
        references.update(
            ("segment", data.episode_id, segment.segment_id)
            for segment in data.segments
        )
        return frozenset(references)
    raise TypeError("工具结果类型没有显式引用策略")


def citations_are_valid(
    citations: Sequence[AgentCitation],
    *,
    has_tool_results: bool,
    allowed_references: Collection[ReferenceKey],
) -> bool:
    """Apply the Day 22 current-run citation ownership rule without side effects."""
    if has_tool_results and not citations:
        return False
    if not has_tool_results and citations:
        return False
    return all(
        (citation.kind, citation.episode_id, citation.reference_id)
        in allowed_references
        for citation in citations
    )
