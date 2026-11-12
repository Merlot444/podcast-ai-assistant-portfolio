"""Fixed-registry dispatcher for exactly one validated Day 21 tool call."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from types import MappingProxyType
from typing import Mapping, Protocol
from uuid import uuid4

from pydantic import ValidationError

from app.tooling.definitions import TOOL_CONTRACTS, ToolContract
from app.tooling.handlers import ToolHandlers, ToolNotFoundError
from app.tooling.recording import (
    InMemoryToolCallRecorder,
    ToolCallRecorder,
    stable_json_sha256,
    text_sha256,
)
from app.tooling.schemas import (
    EpisodeMetadataData,
    GetEpisodeMetadataArguments,
    GetSegmentContextArguments,
    RecordedToolName,
    SearchTranscriptArguments,
    SearchTranscriptData,
    SegmentContextData,
    ToolArguments,
    ToolCallRecord,
    ToolData,
    ToolDispatchResult,
    ToolError,
    ToolErrorCode,
    ToolFailure,
    ToolName,
    ToolSuccess,
)


UNKNOWN_TOOL_MESSAGE = "工具未注册。"
INVALID_ARGUMENTS_MESSAGE = "工具参数无效。"
NOT_FOUND_MESSAGE = "请求的节目或片段不存在。"
TOOL_FAILED_MESSAGE = "工具暂时无法完成调用。"


class HandlerSet(Protocol):
    def search_transcript(self, arguments: SearchTranscriptArguments) -> SearchTranscriptData:
        ...

    def get_episode_metadata(
        self,
        arguments: GetEpisodeMetadataArguments,
    ) -> EpisodeMetadataData:
        ...

    def get_segment_context(
        self,
        arguments: GetSegmentContextArguments,
    ) -> SegmentContextData:
        ...


@dataclass(frozen=True)
class RegisteredTool:
    contract: ToolContract
    handler: Callable[[ToolArguments], ToolData]


class ToolDispatcher:
    """Validate one named call, invoke one pre-bound handler, and record it."""

    def __init__(
        self,
        handlers: ToolHandlers | HandlerSet,
        *,
        recorder: ToolCallRecorder | None = None,
        call_id_factory: Callable[[], str] | None = None,
    ) -> None:
        handler_by_name: dict[ToolName, Callable[[ToolArguments], ToolData]] = {
            "search_transcript": handlers.search_transcript,
            "get_episode_metadata": handlers.get_episode_metadata,
            "get_segment_context": handlers.get_segment_context,
        }
        registry = {
            contract.name: RegisteredTool(
                contract=contract,
                handler=handler_by_name[contract.name],
            )
            for contract in TOOL_CONTRACTS
        }
        self._registry: Mapping[str, RegisteredTool] = MappingProxyType(registry)
        self.recorder = recorder or InMemoryToolCallRecorder()
        self.call_id_factory = call_id_factory or (
            lambda: f"call_{uuid4().hex}"
        )

    @property
    def registry(self) -> Mapping[str, RegisteredTool]:
        return self._registry

    def canonical_call_sha256(
        self,
        tool_name: object,
        arguments: object,
    ) -> str | None:
        """Return a schema-normalized call fingerprint without invoking a handler.

        Unknown tools and invalid arguments deliberately have no canonical key;
        callers must still use ``dispatch`` to receive the existing safe error
        envelope. The method exposes neither parsed values nor a handler.
        """
        registered = (
            self._registry.get(tool_name) if isinstance(tool_name, str) else None
        )
        if registered is None:
            return None
        try:
            parsed_arguments = registered.contract.argument_model.model_validate(
                arguments
            )
            normalized_arguments = parsed_arguments.model_dump(mode="json")
            return stable_json_sha256(
                {
                    "tool_name": registered.contract.name,
                    "arguments": normalized_arguments,
                }
            )
        except Exception:
            return None

    def dispatch(self, tool_name: object, arguments: object) -> ToolDispatchResult:
        """Dispatch one call only; handlers cannot select another handler."""
        call_id = self.call_id_factory()
        started_at = datetime.now(UTC)
        started = perf_counter()
        requested_tool_sha256 = stable_json_sha256(tool_name)
        argument_sha256 = stable_json_sha256(arguments)
        query_length, query_sha256 = _query_summary(arguments)

        registered = self._registry.get(tool_name) if isinstance(tool_name, str) else None
        if registered is None:
            result = _failure(
                tool_name="unknown",
                call_id=call_id,
                code="unknown_tool",
                message=UNKNOWN_TOOL_MESSAGE,
            )
            self._record(
                result=result,
                parsed_arguments=None,
                started_at=started_at,
                started=started,
                requested_tool_sha256=requested_tool_sha256,
                argument_sha256=argument_sha256,
                query_length=query_length,
                query_sha256=query_sha256,
            )
            return result

        try:
            parsed_arguments = registered.contract.argument_model.model_validate(arguments)
        except ValidationError:
            result = _failure(
                tool_name=registered.contract.name,
                call_id=call_id,
                code="invalid_arguments",
                message=INVALID_ARGUMENTS_MESSAGE,
            )
            self._record(
                result=result,
                parsed_arguments=None,
                started_at=started_at,
                started=started,
                requested_tool_sha256=requested_tool_sha256,
                argument_sha256=argument_sha256,
                query_length=query_length,
                query_sha256=query_sha256,
            )
            return result

        try:
            raw_data = registered.handler(parsed_arguments)
            data = registered.contract.result_model.model_validate(raw_data)
            result = ToolSuccess(
                tool_name=registered.contract.name,
                call_id=call_id,
                data=data,
            )
        except ToolNotFoundError:
            result = _failure(
                tool_name=registered.contract.name,
                call_id=call_id,
                code="not_found",
                message=NOT_FOUND_MESSAGE,
            )
        except Exception:
            result = _failure(
                tool_name=registered.contract.name,
                call_id=call_id,
                code="tool_failed",
                message=TOOL_FAILED_MESSAGE,
            )

        self._record(
            result=result,
            parsed_arguments=parsed_arguments,
            started_at=started_at,
            started=started,
            requested_tool_sha256=requested_tool_sha256,
            argument_sha256=argument_sha256,
            query_length=query_length,
            query_sha256=query_sha256,
        )
        return result

    def _record(
        self,
        *,
        result: ToolDispatchResult,
        parsed_arguments: ToolArguments | None,
        started_at: datetime,
        started: float,
        requested_tool_sha256: str,
        argument_sha256: str,
        query_length: int | None,
        query_sha256: str | None,
    ) -> None:
        episode_id = None
        segment_id = None
        if isinstance(
            parsed_arguments,
            (
                SearchTranscriptArguments,
                GetEpisodeMetadataArguments,
                GetSegmentContextArguments,
            ),
        ):
            episode_id = parsed_arguments.episode_id
        if isinstance(parsed_arguments, GetSegmentContextArguments):
            segment_id = parsed_arguments.segment_id
        result_count, result_ids = _result_summary(result)
        error_code: ToolErrorCode | None = None
        outcome = "success"
        if isinstance(result, ToolFailure):
            outcome = "failure"
            error_code = result.error.code
        self.recorder.record(
            ToolCallRecord(
                call_id=result.call_id,
                tool_name=result.tool_name,
                requested_tool_sha256=requested_tool_sha256,
                started_at=started_at,
                duration_ms=max(0.0, (perf_counter() - started) * 1000),
                outcome=outcome,
                error_code=error_code,
                result_count=result_count,
                episode_id=episode_id,
                segment_id=segment_id,
                result_ids=result_ids,
                argument_sha256=argument_sha256,
                result_sha256=stable_json_sha256(result),
                query_length=query_length,
                query_sha256=query_sha256,
            )
        )


def _failure(
    *,
    tool_name: RecordedToolName,
    call_id: str,
    code: ToolErrorCode,
    message: str,
) -> ToolFailure:
    return ToolFailure(
        tool_name=tool_name,
        call_id=call_id,
        error=ToolError(code=code, message=message),
    )


def _query_summary(arguments: object) -> tuple[int | None, str | None]:
    if not isinstance(arguments, Mapping):
        return None, None
    query = arguments.get("query")
    if not isinstance(query, str):
        return None, None
    return len(query), text_sha256(query)


def _result_summary(result: ToolDispatchResult) -> tuple[int, tuple[str, ...]]:
    if isinstance(result, ToolFailure):
        return 0, ()
    if isinstance(result.data, SearchTranscriptData):
        return result.data.result_count, tuple(item.chunk_id for item in result.data.hits)
    if isinstance(result.data, SegmentContextData):
        return result.data.result_count, tuple(
            item.segment_id for item in result.data.segments
        )
    return 1, (result.data.episode_id,)
