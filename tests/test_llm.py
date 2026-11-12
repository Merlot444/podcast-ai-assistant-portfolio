"""Offline tests for safe model-service error translation."""

from __future__ import annotations

from unittest.mock import Mock

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from app.core.llm import ModelServiceError, create_json_completion


SECRET_API_KEY = "day5-test-key-must-not-appear"


def _request() -> httpx.Request:
    return httpx.Request(
        "POST",
        "https://example.test/chat/completions",
        headers={"Authorization": f"Bearer {SECRET_API_KEY}"},
    )


def _response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, request=_request())


def _assert_safe_error(sdk_error: Exception, expected_message: str) -> None:
    client = Mock()
    client.chat.completions.create.side_effect = sdk_error

    with pytest.raises(ModelServiceError, match=expected_message) as captured:
        create_json_completion(
            client,
            model="offline-test-model",
            messages=[{"role": "user", "content": "test"}],
        )

    assert SECRET_API_KEY not in str(captured.value)
    client.chat.completions.create.assert_called_once()


def test_authentication_error_is_translated_without_leaking_api_key() -> None:
    _assert_safe_error(
        AuthenticationError(
            f"invalid credential: {SECRET_API_KEY}",
            response=_response(401),
            body=None,
        ),
        "API 鉴权失败",
    )


def test_timeout_error_is_translated_without_leaking_api_key() -> None:
    _assert_safe_error(
        APITimeoutError(_request()),
        "模型请求超时",
    )


def test_rate_limit_error_is_translated_without_leaking_api_key() -> None:
    _assert_safe_error(
        RateLimitError(
            f"rate limited credential: {SECRET_API_KEY}",
            response=_response(429),
            body=None,
        ),
        "模型服务暂时限流",
    )


def test_connection_error_is_translated_without_leaking_api_key() -> None:
    _assert_safe_error(
        APIConnectionError(
            message=f"connection failed for {SECRET_API_KEY}",
            request=_request(),
        ),
        "无法连接模型服务",
    )


def test_http_status_error_keeps_status_but_not_api_key() -> None:
    _assert_safe_error(
        APIStatusError(
            f"upstream failure for {SECRET_API_KEY}",
            response=_response(503),
            body=None,
        ),
        "模型服务返回 HTTP 503",
    )
