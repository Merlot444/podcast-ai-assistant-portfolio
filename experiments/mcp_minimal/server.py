"""One-tool MCP server whose only executable boundary is ToolDispatcher."""

from __future__ import annotations

from typing import Any

from mcp.server import Server, ServerRequestContext
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
    Tool,
    ToolAnnotations,
)

from app.tooling import ToolDispatcher, ToolFailure, ToolSuccess
from app.tooling.dispatcher import TOOL_FAILED_MESSAGE, UNKNOWN_TOOL_MESSAGE
from app.tooling.schemas import (
    EpisodeMetadataData,
    GetEpisodeMetadataArguments,
    ToolError,
)
from experiments.mcp_minimal.schemas import (
    MCP_METADATA_RESULT_ADAPTER,
    McpMetadataFailure,
    McpMetadataResult,
    McpMetadataSuccess,
)


MCP_SERVER_NAME = "podcast-metadata-minimal"
MCP_SERVER_VERSION = "day24"
MCP_TOOL_NAME = "get_episode_metadata"
MCP_FALLBACK_CALL_ID = "call_00000000000000000000000000000000"


def create_server(dispatcher: ToolDispatcher) -> Server:
    """Create one in-process-capable MCP server over a trusted Dispatcher."""

    async def list_tools(
        _context: ServerRequestContext[Any],
        _params: PaginatedRequestParams | None,
    ) -> ListToolsResult:
        return ListToolsResult(
            tools=[
                Tool(
                    name=MCP_TOOL_NAME,
                    description=(
                        "Return a low-risk structural summary for one already "
                        "loaded episode. Transcript text and source paths are excluded."
                    ),
                    input_schema=GetEpisodeMetadataArguments.model_json_schema(),
                    output_schema=_metadata_output_schema(),
                    annotations=ToolAnnotations(
                        read_only_hint=True,
                        destructive_hint=False,
                        idempotent_hint=True,
                        open_world_hint=False,
                    ),
                )
            ]
        )

    async def call_tool(
        _context: ServerRequestContext[Any],
        params: CallToolRequestParams,
    ) -> CallToolResult:
        if params.name != MCP_TOOL_NAME:
            return _to_call_result(
                McpMetadataFailure(
                    tool_name="unknown",
                    call_id=MCP_FALLBACK_CALL_ID,
                    error=ToolError(
                        code="unknown_tool",
                        message=UNKNOWN_TOOL_MESSAGE,
                    ),
                )
            )
        result = _dispatch_metadata(dispatcher, params.arguments or {})
        return _to_call_result(result)

    return Server(
        MCP_SERVER_NAME,
        version=MCP_SERVER_VERSION,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


def _metadata_output_schema() -> dict[str, object]:
    """Publish the narrow union with the root object type older MCP expects."""
    schema = MCP_METADATA_RESULT_ADAPTER.json_schema()
    schema["type"] = "object"
    return schema


def _dispatch_metadata(
    dispatcher: ToolDispatcher,
    arguments: object,
) -> McpMetadataResult:
    """Call the existing Dispatcher once and narrow its validated envelope."""
    try:
        result = dispatcher.dispatch(MCP_TOOL_NAME, arguments)
        if (
            isinstance(result, ToolSuccess)
            and result.tool_name == MCP_TOOL_NAME
            and isinstance(result.data, EpisodeMetadataData)
        ):
            return McpMetadataSuccess(
                tool_name=MCP_TOOL_NAME,
                call_id=result.call_id,
                data=result.data,
            )
        if isinstance(result, ToolFailure) and result.tool_name in {
            MCP_TOOL_NAME,
            "unknown",
        }:
            return McpMetadataFailure(
                tool_name=result.tool_name,
                call_id=result.call_id,
                error=result.error,
            )
    except Exception:
        # Dispatch or envelope-narrowing faults become the same safe result.
        return _tool_failed_result()
    return _tool_failed_result()


def _tool_failed_result() -> McpMetadataFailure:
    return McpMetadataFailure(
        tool_name=MCP_TOOL_NAME,
        call_id=MCP_FALLBACK_CALL_ID,
        error=ToolError(code="tool_failed", message=TOOL_FAILED_MESSAGE),
    )


def _to_call_result(result: McpMetadataResult) -> CallToolResult:
    content = result.model_dump_json()
    return CallToolResult(
        content=[TextContent(type="text", text=content)],
        structured_content=result.model_dump(mode="json"),
        is_error=isinstance(result, McpMetadataFailure),
    )
