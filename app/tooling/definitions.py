"""Stable Day 21 tool contracts and OpenAI-compatible definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.tooling.schemas import (
    EpisodeMetadataData,
    GetEpisodeMetadataArguments,
    GetSegmentContextArguments,
    SearchTranscriptArguments,
    SearchTranscriptData,
    SegmentContextData,
    ToolArguments,
    ToolData,
    ToolName,
    ToolSchema,
)


@dataclass(frozen=True)
class ToolContract:
    name: ToolName
    description: str
    argument_model: type[ToolArguments]
    result_model: type[ToolSchema]


TOOL_CONTRACTS: tuple[ToolContract, ...] = (
    ToolContract(
        name="search_transcript",
        description=(
            "Search one already loaded podcast episode and return citation-ready "
            "candidate metadata. This tool never accepts a file path."
        ),
        argument_model=SearchTranscriptArguments,
        result_model=SearchTranscriptData,
    ),
    ToolContract(
        name="get_episode_metadata",
        description=(
            "Return a low-risk structural summary for one already loaded episode; "
            "source paths and transcript text are excluded."
        ),
        argument_model=GetEpisodeMetadataArguments,
        result_model=EpisodeMetadataData,
    ),
    ToolContract(
        name="get_segment_context",
        description=(
            "Return one validated transcript segment with at most two already loaded "
            "neighboring segments on each side."
        ),
        argument_model=GetSegmentContextArguments,
        result_model=SegmentContextData,
    ),
)


def export_openai_tool_definitions() -> list[dict[str, Any]]:
    """Export definitions in deterministic registry order."""
    return [
        {
            "type": "function",
            "function": {
                "name": contract.name,
                "description": contract.description,
                "parameters": contract.argument_model.model_json_schema(),
                "strict": True,
            },
        }
        for contract in TOOL_CONTRACTS
    ]
