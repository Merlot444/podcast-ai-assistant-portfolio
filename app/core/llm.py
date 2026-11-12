"""Minimal OpenAI-compatible client wrapper for DeepSeek."""

from __future__ import annotations

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from app.core.config import Settings


class ModelServiceError(RuntimeError):
    """A safe, user-readable model service failure."""


def create_client(settings: Settings) -> OpenAI:
    """Create a client from validated settings."""
    return OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=settings.timeout_seconds,
        max_retries=0,
    )


def create_json_completion(
    client: OpenAI,
    *,
    model: str,
    messages: list[ChatCompletionMessageParam],
    max_tokens: int = 600,
) -> ChatCompletion:
    """Request a small, non-thinking JSON response with safe errors."""
    if max_tokens <= 0:
        raise ValueError("max_tokens 必须大于 0。")
    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=max_tokens,
            stream=False,
            extra_body={"thinking": {"type": "disabled"}},
        )
    except AuthenticationError:
        raise ModelServiceError(
            "API 鉴权失败。请检查本地 API Key 是否正确或仍然有效。"
        ) from None
    except APITimeoutError:
        raise ModelServiceError(
            "模型请求超时。请稍后重试或检查网络连接。"
        ) from None
    except RateLimitError:
        raise ModelServiceError(
            "模型服务暂时限流。请稍后再试。"
        ) from None
    except APIConnectionError:
        raise ModelServiceError(
            "无法连接模型服务。请检查网络和 Base URL。"
        ) from None
    except APIStatusError as error:
        raise ModelServiceError(
            f"模型服务返回 HTTP {error.status_code}，请检查请求配置。"
        ) from None
    except OpenAIError:
        raise ModelServiceError("模型服务调用失败，请检查配置后重试。") from None
