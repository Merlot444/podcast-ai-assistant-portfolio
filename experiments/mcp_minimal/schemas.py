"""Narrow MCP result contracts derived from the Day 21 tool boundary."""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from app.tooling.schemas import (
    CALL_ID_PATTERN,
    EpisodeMetadataData,
    ToolError,
    ToolSchema,
)


class McpMetadataSuccess(ToolSchema):
    """The only success shape exposed by the Day 24 MCP experiment."""

    ok: Literal[True] = True
    tool_name: Literal["get_episode_metadata"] = "get_episode_metadata"
    call_id: str = Field(pattern=CALL_ID_PATTERN)
    data: EpisodeMetadataData


class McpMetadataFailure(ToolSchema):
    """The existing Day 21 safe failure envelope, narrowed to this server."""

    ok: Literal[False] = False
    tool_name: Literal["get_episode_metadata", "unknown"]
    call_id: str = Field(pattern=CALL_ID_PATTERN)
    error: ToolError


McpMetadataResult = McpMetadataSuccess | McpMetadataFailure
MCP_METADATA_RESULT_ADAPTER = TypeAdapter(
    Annotated[McpMetadataResult, Field(discriminator="ok")]
)


def parse_metadata_result(value: object) -> McpMetadataResult:
    """Validate one JSON-shaped MCP result without parsing display text."""
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return MCP_METADATA_RESULT_ADAPTER.validate_json(encoded)
