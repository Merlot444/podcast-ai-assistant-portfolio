"""Public boundary for the one-tool Day 24 MCP experiment."""

from experiments.mcp_minimal.client import discover_and_call_metadata
from experiments.mcp_minimal.demo import (
    SYNTHETIC_EPISODE_ID,
    build_synthetic_dispatcher,
)
from experiments.mcp_minimal.schemas import (
    McpMetadataFailure,
    McpMetadataResult,
    McpMetadataSuccess,
)
from experiments.mcp_minimal.server import MCP_TOOL_NAME, create_server

__all__ = [
    "MCP_TOOL_NAME",
    "SYNTHETIC_EPISODE_ID",
    "McpMetadataFailure",
    "McpMetadataResult",
    "McpMetadataSuccess",
    "build_synthetic_dispatcher",
    "create_server",
    "discover_and_call_metadata",
]
