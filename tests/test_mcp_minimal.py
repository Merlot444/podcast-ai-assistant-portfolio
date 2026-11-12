"""Offline Day 24 MCP server, discovery, client, and safety tests."""

from __future__ import annotations

import ast
import asyncio
import importlib.metadata
import json
import tomllib
from pathlib import Path
from unittest.mock import Mock

import pytest

pytest.importorskip("mcp")

from mcp import Client
from mcp.server import Server

from app.tooling import ToolDispatcher, ToolFailure, ToolNotFoundError
from app.tooling.schemas import (
    EpisodeMetadataData,
    GetEpisodeMetadataArguments,
    SearchTranscriptData,
    SegmentContextData,
)
from experiments.mcp_minimal.client import discover_and_call_metadata
from experiments.mcp_minimal.demo import (
    SYNTHETIC_EPISODE_ID,
    build_synthetic_dispatcher,
)
from experiments.mcp_minimal.schemas import (
    McpMetadataFailure,
    McpMetadataSuccess,
    parse_metadata_result,
)
from experiments.mcp_minimal.server import (
    MCP_TOOL_NAME,
    create_server,
)


PROJECT_ROOT = Path(__file__).parents[1]
EPISODE_ID = "ep_1111111111111111"
OTHER_EPISODE_ID = "ep_2222222222222222"
PRIVATE_PATH = "/private/transcripts/day24-secret.json"
TRANSCRIPT_SENTINEL = "DAY24_PRIVATE_TRANSCRIPT_BODY"
FAILURE_SENTINEL = (
    "dispatcher traceback postgresql://admin:password@localhost/private SECRET_KEY"
)


class CallIdFactory:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self) -> str:
        self.count += 1
        return f"call_{self.count:032x}"


class SpyHandlers:
    def __init__(
        self,
        *,
        metadata_failure: Exception | None = None,
    ) -> None:
        self.calls: list[tuple[str, object]] = []
        self.metadata_failure = metadata_failure

    def search_transcript(self, arguments: object) -> SearchTranscriptData:
        self.calls.append(("search_transcript", arguments))
        raise AssertionError("MCP metadata server must not call search_transcript")

    def get_episode_metadata(self, arguments: object) -> EpisodeMetadataData:
        self.calls.append(("get_episode_metadata", arguments))
        if self.metadata_failure is not None:
            raise self.metadata_failure
        episode_id = getattr(arguments, "episode_id")
        return EpisodeMetadataData(
            episode_id=episode_id,
            source_format="json",
            segment_count=3,
            timed_segment_count=2,
            speaker_count=2,
            section_count=2,
            section_names=("Opening", "Synthetic details"),
        )

    def get_segment_context(self, arguments: object) -> SegmentContextData:
        self.calls.append(("get_segment_context", arguments))
        raise AssertionError("MCP metadata server must not call get_segment_context")


def _dispatcher(
    *,
    metadata_failure: Exception | None = None,
    call_id_factory: object | None = None,
) -> tuple[ToolDispatcher, SpyHandlers]:
    handlers = SpyHandlers(metadata_failure=metadata_failure)
    dispatcher = ToolDispatcher(
        handlers,
        call_id_factory=(
            call_id_factory  # type: ignore[arg-type]
            if call_id_factory is not None
            else CallIdFactory()
        ),
    )
    return dispatcher, handlers


async def _list_and_call(
    dispatcher: ToolDispatcher,
    arguments: dict[str, object],
) -> tuple[object, object]:
    server = create_server(dispatcher)
    async with Client(server) as client:
        tools = await client.list_tools()
        result = await client.call_tool(MCP_TOOL_NAME, arguments)
    return tools, result


def test_server_can_be_created_without_network_or_model_dependencies() -> None:
    dispatcher, handlers = _dispatcher()

    server = create_server(dispatcher)

    assert isinstance(server, Server)
    assert handlers.calls == []


def test_client_discovers_exactly_one_strict_read_only_tool_and_calls_it_once() -> None:
    dispatcher, handlers = _dispatcher()

    result = asyncio.run(discover_and_call_metadata(create_server(dispatcher), EPISODE_ID))

    assert isinstance(result, McpMetadataSuccess)
    assert result.tool_name == MCP_TOOL_NAME
    assert result.data.episode_id == EPISODE_ID
    assert [name for name, _ in handlers.calls] == [MCP_TOOL_NAME]


def test_discovery_schema_is_the_existing_day21_argument_schema() -> None:
    dispatcher, _ = _dispatcher()

    tools, _ = asyncio.run(_list_and_call(dispatcher, {"episode_id": EPISODE_ID}))

    assert [tool.name for tool in tools.tools] == [MCP_TOOL_NAME]
    tool = tools.tools[0]
    assert tool.input_schema == GetEpisodeMetadataArguments.model_json_schema()
    assert tool.input_schema["additionalProperties"] is False
    assert tool.input_schema["properties"]["episode_id"]["pattern"] == (
        r"^ep_[0-9a-f]{16}$"
    )
    assert tool.output_schema is not None
    assert tool.output_schema["type"] == "object"
    serialized_output_schema = json.dumps(tool.output_schema, sort_keys=True)
    for forbidden in ("SearchTranscriptData", "SegmentContextData", "source_file"):
        assert forbidden not in serialized_output_schema
    assert tool.annotations is not None
    assert tool.annotations.read_only_hint is True
    assert tool.annotations.destructive_hint is False
    assert tool.annotations.open_world_hint is False


def test_legacy_protocol_can_discover_and_call_the_narrow_output_schema() -> None:
    dispatcher, handlers = _dispatcher()

    async def legacy_call() -> object:
        async with Client(create_server(dispatcher), mode="legacy") as client:
            listed = await client.list_tools()
            assert [tool.name for tool in listed.tools] == [MCP_TOOL_NAME]
            return await client.call_tool(
                MCP_TOOL_NAME,
                {"episode_id": EPISODE_ID},
            )

    raw_result = asyncio.run(legacy_call())
    parsed = parse_metadata_result(raw_result.structured_content)

    assert isinstance(parsed, McpMetadataSuccess)
    assert parsed.data.episode_id == EPISODE_ID
    assert [name for name, _ in handlers.calls] == [MCP_TOOL_NAME]


def test_mcp_server_calls_dispatcher_once_with_the_fixed_tool_name() -> None:
    dispatcher, handlers = _dispatcher()
    dispatch_spy = Mock(wraps=dispatcher.dispatch)
    dispatcher.dispatch = dispatch_spy  # type: ignore[method-assign]

    _, raw_result = asyncio.run(
        _list_and_call(dispatcher, {"episode_id": EPISODE_ID})
    )

    result = parse_metadata_result(raw_result.structured_content)
    assert isinstance(result, McpMetadataSuccess)
    dispatch_spy.assert_called_once_with(
        MCP_TOOL_NAME,
        {"episode_id": EPISODE_ID},
    )
    assert [name for name, _ in handlers.calls] == [MCP_TOOL_NAME]


def test_unlisted_dispatcher_tool_name_cannot_bypass_the_mcp_surface() -> None:
    dispatcher, handlers = _dispatcher()
    dispatch_spy = Mock(wraps=dispatcher.dispatch)
    dispatcher.dispatch = dispatch_spy  # type: ignore[method-assign]

    async def call_unlisted_tool() -> object:
        async with Client(create_server(dispatcher)) as client:
            return await client.call_tool(
                "search_transcript",
                {
                    "query": "private",
                    "episode_id": EPISODE_ID,
                    "top_k": 1,
                },
            )

    raw_result = asyncio.run(call_unlisted_tool())
    parsed = parse_metadata_result(raw_result.structured_content)

    assert isinstance(parsed, McpMetadataFailure)
    assert parsed.tool_name == "unknown"
    assert parsed.error.code == "unknown_tool"
    assert raw_result.is_error is True
    assert "search_transcript" not in raw_result.model_dump_json(by_alias=True)
    assert dispatch_spy.call_count == 0
    assert handlers.calls == []


@pytest.mark.parametrize(
    "arguments",
    [
        {"episode_id": "episode-invalid"},
        {"episode_id": EPISODE_ID, "path": PRIVATE_PATH},
        {"episode_id": EPISODE_ID, "url": "https://example.invalid/private"},
    ],
)
def test_invalid_arguments_return_safe_dispatcher_error_before_handlers(
    arguments: dict[str, object],
) -> None:
    dispatcher, handlers = _dispatcher()

    _, raw_result = asyncio.run(_list_and_call(dispatcher, arguments))

    result = parse_metadata_result(raw_result.structured_content)
    assert isinstance(result, McpMetadataFailure)
    assert raw_result.is_error is True
    assert result.error.code == "invalid_arguments"
    assert result.error.message == "工具参数无效。"
    assert handlers.calls == []
    serialized = raw_result.model_dump_json(by_alias=True)
    assert PRIVATE_PATH not in serialized
    assert "example.invalid" not in serialized


def test_missing_episode_returns_stable_safe_not_found() -> None:
    dispatcher, handlers = _dispatcher(
        metadata_failure=ToolNotFoundError(FAILURE_SENTINEL)
    )

    _, raw_result = asyncio.run(
        _list_and_call(dispatcher, {"episode_id": OTHER_EPISODE_ID})
    )

    result = parse_metadata_result(raw_result.structured_content)
    assert isinstance(result, McpMetadataFailure)
    assert result.error.code == "not_found"
    assert result.error.message == "请求的节目或片段不存在。"
    assert [name for name, _ in handlers.calls] == [MCP_TOOL_NAME]
    assert FAILURE_SENTINEL not in raw_result.model_dump_json(by_alias=True)


def test_handler_exception_is_wrapped_by_dispatcher_without_details() -> None:
    dispatcher, handlers = _dispatcher(
        metadata_failure=RuntimeError(FAILURE_SENTINEL)
    )

    _, raw_result = asyncio.run(
        _list_and_call(dispatcher, {"episode_id": EPISODE_ID})
    )

    result = parse_metadata_result(raw_result.structured_content)
    assert isinstance(result, McpMetadataFailure)
    assert result.error.code == "tool_failed"
    assert result.error.message == "工具暂时无法完成调用。"
    assert [name for name, _ in handlers.calls] == [MCP_TOOL_NAME]
    serialized = raw_result.model_dump_json(by_alias=True)
    for forbidden in (FAILURE_SENTINEL, "traceback", "postgresql://", "SECRET_KEY"):
        assert forbidden not in serialized


def test_dispatcher_collaborator_failure_is_caught_by_mcp_boundary() -> None:
    def fail_call_id() -> str:
        raise RuntimeError(FAILURE_SENTINEL)

    dispatcher, handlers = _dispatcher(call_id_factory=fail_call_id)

    _, raw_result = asyncio.run(
        _list_and_call(dispatcher, {"episode_id": EPISODE_ID})
    )

    result = parse_metadata_result(raw_result.structured_content)
    assert isinstance(result, McpMetadataFailure)
    assert result.error.code == "tool_failed"
    assert result.call_id == "call_00000000000000000000000000000000"
    assert handlers.calls == []
    assert FAILURE_SENTINEL not in raw_result.model_dump_json(by_alias=True)


def test_success_output_contains_only_structural_metadata() -> None:
    dispatcher, _ = _dispatcher()

    _, raw_result = asyncio.run(
        _list_and_call(dispatcher, {"episode_id": EPISODE_ID})
    )

    result = parse_metadata_result(raw_result.structured_content)
    assert isinstance(result, McpMetadataSuccess)
    serialized = raw_result.model_dump_json(by_alias=True)
    for forbidden in (
        TRANSCRIPT_SENTINEL,
        PRIVATE_PATH,
        "source_file",
        "postgresql://",
        "SECRET_KEY",
    ):
        assert forbidden not in serialized


def test_synthetic_transcript_body_and_source_path_do_not_cross_mcp_output() -> None:
    dispatcher = build_synthetic_dispatcher()

    _, raw_result = asyncio.run(
        _list_and_call(dispatcher, {"episode_id": SYNTHETIC_EPISODE_ID})
    )

    parsed = parse_metadata_result(raw_result.structured_content)
    assert isinstance(parsed, McpMetadataSuccess)
    serialized = raw_result.model_dump_json(by_alias=True)
    assert "Original synthetic introduction" not in serialized
    assert "Original synthetic metadata example" not in serialized
    assert "day24-synthetic.json" not in serialized


def test_runtime_has_no_dynamic_execution_file_url_or_handler_bypass() -> None:
    experiment_root = PROJECT_ROOT / "experiments" / "mcp_minimal"
    forbidden_names = {"open", "eval", "exec", "compile", "__import__", "getattr"}
    forbidden_attributes = {
        "open",
        "popen",
        "read_bytes",
        "read_text",
        "request",
        "run",
        "system",
        "urlopen",
        "write_bytes",
        "write_text",
    }
    dispatch_calls = 0
    for filename in ("server.py", "client.py"):
        tree = ast.parse((experiment_root / filename).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_names
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_attributes
                if node.func.attr == "dispatch":
                    dispatch_calls += 1
            if isinstance(node, ast.ImportFrom):
                assert node.module not in {
                    "app.tooling.handlers",
                    "app.storage",
                    "app.storage.crud",
                    "app.storage.repository",
                }
    assert dispatch_calls == 1


def test_mcp_optional_dependency_is_official_v2_and_separate_from_default() -> None:
    configuration = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    optional = configuration["project"]["optional-dependencies"]

    assert optional["mcp"] == ["mcp>=2,<3"]
    assert all("mcp" not in dependency.lower() for dependency in configuration["project"]["dependencies"])
    assert importlib.metadata.version("mcp").split(".", maxsplit=1)[0] == "2"
    assert "experiments*" in configuration["tool"]["setuptools"]["packages"]["find"]["include"]


def test_safe_failure_is_still_the_day21_error_envelope() -> None:
    dispatcher, _ = _dispatcher(metadata_failure=RuntimeError(FAILURE_SENTINEL))

    _, raw_result = asyncio.run(
        _list_and_call(dispatcher, {"episode_id": EPISODE_ID})
    )
    parsed = parse_metadata_result(raw_result.structured_content)

    assert isinstance(parsed, McpMetadataFailure)
    day21 = ToolFailure.model_validate(parsed.model_dump(mode="json"))
    assert day21.error.code == "tool_failed"
    assert day21.tool_name == MCP_TOOL_NAME
