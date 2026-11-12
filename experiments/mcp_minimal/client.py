"""Minimal official MCP client: discover exactly one tool, then call it once."""

from __future__ import annotations

from typing import Any

from mcp import Client
from mcp.server import Server

from experiments.mcp_minimal.schemas import (
    McpMetadataFailure,
    McpMetadataSuccess,
    parse_metadata_result,
)
from experiments.mcp_minimal.server import MCP_TOOL_NAME


class MinimalMcpClientError(RuntimeError):
    """A fixed, content-free client assertion failure."""


async def discover_and_call_metadata(
    server: Server[Any],
    episode_id: str,
) -> McpMetadataSuccess:
    """Assert the discovery surface and complete one successful in-memory call."""
    async with Client(server) as client:
        discovered = await client.list_tools()
        tool_names = tuple(tool.name for tool in discovered.tools)
        if tool_names != (MCP_TOOL_NAME,):
            raise MinimalMcpClientError("MCP 工具发现结果不符合最小实验契约。")
        called = await client.call_tool(
            MCP_TOOL_NAME,
            {"episode_id": episode_id},
        )

    try:
        parsed = parse_metadata_result(called.structured_content)
    except Exception:
        raise MinimalMcpClientError("MCP 工具返回格式无效。") from None
    if called.is_error or isinstance(parsed, McpMetadataFailure):
        raise MinimalMcpClientError("MCP 工具调用未成功。")
    return parsed
