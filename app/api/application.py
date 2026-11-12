"""FastAPI application factory with injected services and safe error boundaries."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from functools import partial
from pathlib import Path as FileSystemPath
from secrets import compare_digest
from time import perf_counter
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Path
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request
from starlette.responses import Response

from app.api.lifecycle import (
    LifecycleClock,
    LifecycleEventSink,
    service_lifespan,
    write_lifecycle_event,
)
from app.api.schemas import (
    SESSION_ID_PATTERN,
    ApiError,
    ApiErrorCode,
    ApiErrorResponse,
    AskRequest,
    AskResponse,
    HealthResponse,
    RetrieveRequest,
    RetrievedChunkResponse,
    RetrieveResponse,
    SessionMetadata,
    SessionResponse,
)
from app.api.request_boundaries import (
    DEFAULT_MAX_REQUEST_BODY_BYTES,
    BoundaryErrorKind,
    HttpRequestBoundaryMiddleware,
    MonotonicClock,
    RequestEventSink,
    RequestIdFactory,
    UtcClock,
    generate_request_id,
    validate_max_request_body_bytes,
    write_request_event,
)
from app.api.services import (
    ASK_SERVICE_RESULT_ADAPTER,
    RETRIEVE_SERVICE_RESULT_ADAPTER,
    SESSION_READ_RESULT_ADAPTER,
    AskService,
    AskServiceSuccess,
    RetrieveService,
    RetrieveServiceSuccess,
    ServiceFailure,
    SessionReader,
    UnavailableAskService,
    UnavailableRetrieveService,
    UnavailableSessionReader,
)
from app.api.adapters import EvaluationRetrieveService
from app.api.summary_context import (
    SummaryContextRequest,
    SummaryContextResponse,
    build_summary_context,
)


INVALID_REQUEST_MESSAGE = "请求格式无效。"
UNAUTHORIZED_MESSAGE = "认证失败。"
ASK_NOT_FOUND_MESSAGE = "请求的节目不存在。"
SESSION_NOT_FOUND_MESSAGE = "请求的会话不存在。"
ASK_UNAVAILABLE_MESSAGE = "问答服务暂时不可用。"
ASK_FAILED_MESSAGE = "问答服务无法完成请求。"
RETRIEVE_NOT_FOUND_MESSAGE = "请求的节目不存在。"
RETRIEVE_UNAVAILABLE_MESSAGE = "检索服务暂时不可用。"
RETRIEVE_FAILED_MESSAGE = "检索服务无法完成请求。"
SESSION_UNAVAILABLE_MESSAGE = "会话服务暂时不可用。"
SESSION_FAILED_MESSAGE = "会话服务无法完成请求。"
RESOURCE_NOT_FOUND_MESSAGE = "请求的资源不存在。"
INTERNAL_ERROR_MESSAGE = "服务暂时无法完成请求。"
REQUEST_BODY_TOO_LARGE_MESSAGE = "请求体过大。"
STATIC_ROOT = FileSystemPath(__file__).with_name("static")
_BEARER_TOKEN_RE = re.compile(r"^[A-Za-z0-9._~-]{32,256}$")


class ApiAuthenticationError(Exception):
    """Raised without carrying the received or expected credential."""

ApiFailureKind = Literal[
    "invalid_request",
    "method_not_allowed",
    "unauthorized",
    "ask_not_found",
    "session_not_found",
    "ask_unavailable",
    "ask_failed",
    "retrieve_not_found",
    "retrieve_unavailable",
    "retrieve_failed",
    "session_unavailable",
    "session_failed",
    "resource_not_found",
    "internal_error",
    "body_too_large",
]
_ERROR_RESPONSES: dict[ApiFailureKind, tuple[int, ApiErrorCode, str]] = {
    "invalid_request": (422, "invalid_request", INVALID_REQUEST_MESSAGE),
    "method_not_allowed": (405, "invalid_request", INVALID_REQUEST_MESSAGE),
    "unauthorized": (401, "unauthorized", UNAUTHORIZED_MESSAGE),
    "ask_not_found": (404, "not_found", ASK_NOT_FOUND_MESSAGE),
    "session_not_found": (404, "not_found", SESSION_NOT_FOUND_MESSAGE),
    "ask_unavailable": (503, "service_unavailable", ASK_UNAVAILABLE_MESSAGE),
    "ask_failed": (503, "service_failed", ASK_FAILED_MESSAGE),
    "retrieve_not_found": (404, "not_found", RETRIEVE_NOT_FOUND_MESSAGE),
    "retrieve_unavailable": (
        503,
        "service_unavailable",
        RETRIEVE_UNAVAILABLE_MESSAGE,
    ),
    "retrieve_failed": (503, "service_failed", RETRIEVE_FAILED_MESSAGE),
    "session_unavailable": (
        503,
        "service_unavailable",
        SESSION_UNAVAILABLE_MESSAGE,
    ),
    "session_failed": (503, "service_failed", SESSION_FAILED_MESSAGE),
    "resource_not_found": (404, "not_found", RESOURCE_NOT_FOUND_MESSAGE),
    "internal_error": (500, "internal_error", INTERNAL_ERROR_MESSAGE),
    "body_too_large": (413, "invalid_request", REQUEST_BODY_TOO_LARGE_MESSAGE),
}


def create_app(
    *,
    ask_service: AskService | None = None,
    retrieve_service: RetrieveService | None = None,
    api_bearer_token: str | None = None,
    session_reader: SessionReader | None = None,
    request_id_factory: RequestIdFactory | None = None,
    event_sink: RequestEventSink | None = None,
    clock: UtcClock | None = None,
    monotonic_clock: MonotonicClock | None = None,
    retrieval_clock: MonotonicClock | None = None,
    max_request_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES,
    lifecycle_event_sink: LifecycleEventSink | None = None,
    lifecycle_clock: LifecycleClock | None = None,
) -> FastAPI:
    """Create a startable app without constructing models, tools, or databases."""
    resolved_ask_service = (
        ask_service if ask_service is not None else UnavailableAskService()
    )
    resolved_retrieve_service = (
        retrieve_service
        if retrieve_service is not None
        else UnavailableRetrieveService()
    )
    resolved_api_bearer_token = _validated_bearer_token(api_bearer_token)
    resolved_session_reader = (
        session_reader if session_reader is not None else UnavailableSessionReader()
    )
    resolved_request_id_factory = (
        request_id_factory
        if request_id_factory is not None
        else generate_request_id
    )
    resolved_event_sink = event_sink if event_sink is not None else write_request_event
    resolved_clock = clock if clock is not None else _utc_now
    resolved_monotonic_clock = (
        monotonic_clock if monotonic_clock is not None else perf_counter
    )
    resolved_retrieval_clock = (
        retrieval_clock if retrieval_clock is not None else perf_counter
    )
    resolved_lifecycle_event_sink = (
        lifecycle_event_sink
        if lifecycle_event_sink is not None
        else write_lifecycle_event
    )
    resolved_lifecycle_clock = (
        lifecycle_clock if lifecycle_clock is not None else _utc_now
    )
    resolved_max_request_body_bytes = validate_max_request_body_bytes(
        max_request_body_bytes
    )
    app = FastAPI(
        title="Podcast AI Assistant API",
        version="day27",
        debug=False,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=partial(
            service_lifespan,
            event_sink=resolved_lifecycle_event_sink,
            clock=resolved_lifecycle_clock,
        ),
    )
    app.add_middleware(
        HttpRequestBoundaryMiddleware,
        request_id_factory=resolved_request_id_factory,
        event_sink=resolved_event_sink,
        clock=resolved_clock,
        monotonic_clock=resolved_monotonic_clock,
        max_request_body_bytes=resolved_max_request_body_bytes,
        error_response_factory=_boundary_error_response,
    )

    def get_ask_service() -> AskService:
        return resolved_ask_service

    def get_retrieve_service() -> RetrieveService:
        return resolved_retrieve_service

    def require_api_access(request: Request) -> None:
        if resolved_api_bearer_token is None:
            return
        authorization = request.headers.get("authorization")
        expected = f"Bearer {resolved_api_bearer_token}"
        if (
            authorization is None
            or not authorization.isascii()
            or not compare_digest(authorization, expected)
        ):
            raise ApiAuthenticationError()

    def get_session_reader() -> SessionReader:
        return resolved_session_reader

    @app.exception_handler(RequestValidationError)
    def handle_validation_error(
        _request: Request,
        _error: RequestValidationError,
    ) -> JSONResponse:
        return _error_response("invalid_request")

    @app.exception_handler(ApiAuthenticationError)
    def handle_authentication_error(
        _request: Request,
        _error: ApiAuthenticationError,
    ) -> JSONResponse:
        return _error_response("unauthorized")

    @app.exception_handler(StarletteHTTPException)
    def handle_http_error(
        _request: Request,
        error: StarletteHTTPException,
    ) -> JSONResponse:
        if error.status_code == 404:
            return _error_response("resource_not_found")
        if error.status_code == 405:
            return _error_response("method_not_allowed")
        return _error_response("internal_error")

    @app.exception_handler(Exception)
    def handle_unexpected_error(
        _request: Request,
        _error: Exception,
    ) -> JSONResponse:
        return _error_response("internal_error")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/", include_in_schema=False)
    def page() -> FileResponse:
        return FileResponse(
            STATIC_ROOT / "index.html",
            media_type="text/html; charset=utf-8",
        )

    @app.get("/static/day25.css", include_in_schema=False)
    def page_styles() -> FileResponse:
        return FileResponse(
            STATIC_ROOT / "day25.css",
            media_type="text/css; charset=utf-8",
        )

    @app.get("/static/day25.js", include_in_schema=False)
    def page_script() -> FileResponse:
        return FileResponse(
            STATIC_ROOT / "day25.js",
            media_type="text/javascript; charset=utf-8",
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(
        request: AskRequest,
        _authorized: None = Depends(require_api_access),
        service: AskService = Depends(get_ask_service),
    ) -> AskResponse | Response:
        try:
            outcome = ASK_SERVICE_RESULT_ADAPTER.validate_python(
                service.ask(request)
            )
        except Exception:
            return _error_response("ask_failed")
        if isinstance(outcome, ServiceFailure):
            return _ask_failure_response(outcome)
        if not isinstance(outcome, AskServiceSuccess):
            return _error_response("ask_failed")
        return AskResponse(
            answer=outcome.answer.answer,
            citations=tuple(outcome.answer.citations),
        )

    @app.post("/summary-context", response_model=SummaryContextResponse)
    def summary_context(
        request: SummaryContextRequest,
        _authorized: None = Depends(require_api_access),
        service: RetrieveService = Depends(get_retrieve_service),
    ) -> SummaryContextResponse | Response:
        # Full transcript evidence must never be exposed without authentication.
        if resolved_api_bearer_token is None:
            return _error_response("retrieve_unavailable")
        if not isinstance(service, EvaluationRetrieveService):
            return _error_response("retrieve_unavailable")
        if request.episode_id != service.episode_id:
            return _error_response("retrieve_not_found")
        try:
            return build_summary_context(service)
        except Exception:
            return _error_response("retrieve_failed")

    @app.post("/retrieve", response_model=RetrieveResponse)
    def retrieve(
        request: RetrieveRequest,
        _authorized: None = Depends(require_api_access),
        service: RetrieveService = Depends(get_retrieve_service),
    ) -> RetrieveResponse | Response:
        started_at = resolved_retrieval_clock()
        try:
            outcome = RETRIEVE_SERVICE_RESULT_ADAPTER.validate_python(
                service.retrieve(request)
            )
        except Exception:
            return _error_response("retrieve_failed")
        duration_ms = max(
            0.0,
            (resolved_retrieval_clock() - started_at) * 1_000,
        )
        if isinstance(outcome, ServiceFailure):
            return _retrieve_failure_response(outcome)
        if not isinstance(outcome, RetrieveServiceSuccess):
            return _error_response("retrieve_failed")
        chunks = tuple(
            RetrievedChunkResponse.model_validate(chunk.model_dump())
            for chunk in outcome.chunks
        )
        return RetrieveResponse(
            episode_id=outcome.episode_id,
            result_count=len(chunks),
            duration_ms=round(duration_ms, 3),
            input_chunk_sha256=outcome.input_chunk_sha256,
            chunks=chunks,
        )

    @app.post("/evidence/{mode}", response_model=RetrieveResponse | SummaryContextResponse)
    def episode_evidence(
        mode: Literal["EPISODE_QA", "EPISODE_SUMMARY"],
        request: RetrieveRequest,
        _authorized: None = Depends(require_api_access),
        service: RetrieveService = Depends(get_retrieve_service),
    ) -> RetrieveResponse | SummaryContextResponse | Response:
        if mode == "EPISODE_SUMMARY":
            return summary_context(SummaryContextRequest(episode_id=request.episode_id), None, service)
        return retrieve(request, None, service)

    @app.get(
        "/sessions/{session_id}",
        response_model=SessionResponse,
    )
    def get_session(
        session_id: Annotated[
            str,
            Path(
                min_length=1,
                max_length=128,
                pattern=SESSION_ID_PATTERN,
            ),
        ],
        _authorized: None = Depends(require_api_access),
        reader: SessionReader = Depends(get_session_reader),
    ) -> SessionResponse | Response:
        try:
            outcome = SESSION_READ_RESULT_ADAPTER.validate_python(
                reader.get_session(session_id)
            )
        except Exception:
            return _error_response("session_failed")
        if outcome is None:
            return _error_response("session_not_found")
        if isinstance(outcome, ServiceFailure):
            return _session_failure_response(outcome)
        if not isinstance(outcome, SessionMetadata):
            return _error_response("session_failed")
        return SessionResponse(session=outcome)

    @app.get("/sessions/{_invalid_session_id:path}", include_in_schema=False)
    def reject_slash_containing_session_id(
        _invalid_session_id: str,
    ) -> Response:
        return _error_response("invalid_request")

    return app


def _ask_failure_response(failure: ServiceFailure) -> JSONResponse:
    if failure.code == "invalid_request":
        return _error_response("invalid_request")
    if failure.code == "not_found":
        return _error_response("ask_not_found")
    if failure.code == "service_unavailable":
        return _error_response("ask_unavailable")
    return _error_response("ask_failed")


def _retrieve_failure_response(failure: ServiceFailure) -> JSONResponse:
    if failure.code == "invalid_request":
        return _error_response("invalid_request")
    if failure.code == "not_found":
        return _error_response("retrieve_not_found")
    if failure.code == "service_unavailable":
        return _error_response("retrieve_unavailable")
    return _error_response("retrieve_failed")


def _session_failure_response(failure: ServiceFailure) -> JSONResponse:
    if failure.code == "invalid_request":
        return _error_response("invalid_request")
    if failure.code == "not_found":
        return _error_response("session_not_found")
    if failure.code == "service_unavailable":
        return _error_response("session_unavailable")
    return _error_response("session_failed")


def _error_response(
    kind: ApiFailureKind,
) -> JSONResponse:
    status_code, code, message = _ERROR_RESPONSES[kind]
    payload = ApiErrorResponse(error=ApiError(code=code, message=message))
    headers = {"WWW-Authenticate": "Bearer"} if kind == "unauthorized" else None
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json"),
        headers=headers,
    )


def _boundary_error_response(kind: BoundaryErrorKind) -> JSONResponse:
    if kind == "body_too_large":
        return _error_response("body_too_large")
    return _error_response("internal_error")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _validated_bearer_token(value: str | None) -> str | None:
    if value is None:
        return None
    if type(value) is not str or _BEARER_TOKEN_RE.fullmatch(value) is None:
        raise ValueError("api_bearer_token 配置无效")
    return value
