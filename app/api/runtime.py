"""Strict process runtime configuration without global environment reads."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.api.request_boundaries import (
    DEFAULT_MAX_REQUEST_BODY_BYTES,
    MAX_CONFIGURABLE_REQUEST_BODY_BYTES,
)


PODCAST_API_HOST = "PODCAST_API_HOST"
PODCAST_API_PORT = "PODCAST_API_PORT"
PODCAST_API_BEARER_TOKEN = "PODCAST_API_BEARER_TOKEN"
PODCAST_MAX_REQUEST_BODY_BYTES = "PODCAST_MAX_REQUEST_BODY_BYTES"
PODCAST_ASK_MODE = "PODCAST_ASK_MODE"
PODCAST_ASK_EPISODE_ID = "PODCAST_ASK_EPISODE_ID"
PODCAST_ASK_TRANSCRIPT_PATH = "PODCAST_ASK_TRANSCRIPT_PATH"
PODCAST_ASK_INDEX_DIRECTORY = "PODCAST_ASK_INDEX_DIRECTORY"

LLM_API_KEY = "LLM_API_KEY"
LLM_BASE_URL = "LLM_BASE_URL"
LLM_MODEL_FAST = "LLM_MODEL_FAST"
LLM_MODEL_STRONG = "LLM_MODEL_STRONG"
LLM_TIMEOUT_SECONDS = "LLM_TIMEOUT_SECONDS"

EMBEDDING_API_KEY = "EMBEDDING_API_KEY"
EMBEDDING_BASE_URL = "EMBEDDING_BASE_URL"
EMBEDDING_MODEL = "EMBEDDING_MODEL"
EMBEDDING_DIMENSION = "EMBEDDING_DIMENSION"
EMBEDDING_DOCUMENT_TASK = "EMBEDDING_DOCUMENT_TASK"
EMBEDDING_QUERY_TASK = "EMBEDDING_QUERY_TASK"
EMBEDDING_TYPE = "EMBEDDING_TYPE"
EMBEDDING_NORMALIZED = "EMBEDDING_NORMALIZED"
EMBEDDING_TRUNCATE = "EMBEDDING_TRUNCATE"
EMBEDDING_BATCH_SIZE = "EMBEDDING_BATCH_SIZE"
EMBEDDING_TIMEOUT_SECONDS = "EMBEDDING_TIMEOUT_SECONDS"

DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8000
RUNTIME_CONFIGURATION_ERROR_MESSAGE = "运行配置无效，服务未启动。"

AllowedApiHost = Literal["127.0.0.1", "0.0.0.0"]
AskMode = Literal["unavailable", "agent"]
_ASCII_DECIMAL_RE = re.compile(r"^[0-9]+$")
_ASCII_POSITIVE_FLOAT_RE = re.compile(r"^(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)$")
_EPISODE_ID_RE = re.compile(r"^ep_[0-9a-f]{16}$")
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_BEARER_TOKEN_RE = re.compile(r"^[A-Za-z0-9._~-]{32,256}$")
_MAX_REMOTE_TIMEOUT_SECONDS = 300.0


class RuntimeConfigurationError(ValueError):
    """A fixed, non-sensitive runtime configuration failure."""


@dataclass(frozen=True)
class LanguageModelRuntimeConfig:
    """Validated model settings; the API key is never shown in ``repr``."""

    api_key: str = field(repr=False)
    base_url: str
    model_fast: str
    model_strong: str
    timeout_seconds: float


@dataclass(frozen=True)
class EmbeddingRuntimeConfig:
    """Validated production Vector query settings with a hidden API key."""

    api_key: str = field(repr=False)
    base_url: str
    model: str
    expected_dimension: int
    document_task: str
    query_task: str
    embedding_type: Literal["float"]
    normalized: bool
    truncate: bool
    batch_size: int
    timeout_seconds: float


@dataclass(frozen=True)
class AgentAskRuntimeConfig:
    """Complete trusted inputs needed to construct the explicit Agent service."""

    episode_id: str
    transcript_path: str
    index_directory: str
    language_model: LanguageModelRuntimeConfig
    embedding: EmbeddingRuntimeConfig


class RuntimeConfig(BaseModel):
    """The complete, strict configuration needed to start one API process."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
    )

    host: AllowedApiHost = DEFAULT_API_HOST
    port: int = Field(default=DEFAULT_API_PORT, ge=1, le=65_535)
    max_request_body_bytes: int = Field(
        default=DEFAULT_MAX_REQUEST_BODY_BYTES,
        ge=1,
        le=MAX_CONFIGURABLE_REQUEST_BODY_BYTES,
    )
    api_bearer_token: str | None = Field(
        default=None,
        repr=False,
        exclude=True,
    )
    ask_mode: AskMode = "unavailable"
    agent_ask: AgentAskRuntimeConfig | None = Field(
        default=None,
        repr=False,
        exclude=True,
    )

    @model_validator(mode="after")
    def require_mode_specific_configuration(self) -> Self:
        if self.ask_mode == "unavailable" and self.agent_ask is not None:
            raise ValueError("unavailable 模式不能携带 Agent 配置")
        if self.ask_mode == "agent" and self.agent_ask is None:
            raise ValueError("agent 模式缺少 Agent 配置")
        if self.ask_mode == "agent" and self.api_bearer_token is None:
            raise ValueError("agent 模式缺少 API Bearer Token")
        return self


def load_runtime_config(environ: Mapping[str, object]) -> RuntimeConfig:
    """Validate allowlisted entries from an injected mapping.

    This function deliberately never imports or reads ``os.environ``. The
    executable module owns the single process-environment read and passes the
    resulting mapping here. The default unavailable branch does not even look
    up provider credentials or local artifact paths.
    """
    try:
        host = _read_host(environ.get(PODCAST_API_HOST, DEFAULT_API_HOST))
        port = _read_decimal_integer(
            environ.get(PODCAST_API_PORT, str(DEFAULT_API_PORT))
        )
        max_request_body_bytes = _read_decimal_integer(
            environ.get(
                PODCAST_MAX_REQUEST_BODY_BYTES,
                str(DEFAULT_MAX_REQUEST_BODY_BYTES),
            )
        )
        ask_mode = _read_ask_mode(environ.get(PODCAST_ASK_MODE, "unavailable"))
        api_bearer_token = (
            _read_bearer_token(environ.get(PODCAST_API_BEARER_TOKEN))
            if ask_mode == "agent"
            else None
        )
        agent_ask = (
            _load_agent_ask_config(environ) if ask_mode == "agent" else None
        )
        return RuntimeConfig(
            host=host,
            port=port,
            max_request_body_bytes=max_request_body_bytes,
            api_bearer_token=api_bearer_token,
            ask_mode=ask_mode,
            agent_ask=agent_ask,
        )
    except (RuntimeConfigurationError, ValidationError):
        # Never propagate a validation rendering that could contain an input.
        raise RuntimeConfigurationError(
            RUNTIME_CONFIGURATION_ERROR_MESSAGE
        ) from None


def _read_host(value: object) -> AllowedApiHost:
    if type(value) is not str or value not in {"127.0.0.1", "0.0.0.0"}:
        raise RuntimeConfigurationError(RUNTIME_CONFIGURATION_ERROR_MESSAGE)
    return value


def _read_ask_mode(value: object) -> AskMode:
    if type(value) is not str or value not in {"unavailable", "agent"}:
        raise RuntimeConfigurationError(RUNTIME_CONFIGURATION_ERROR_MESSAGE)
    return value


def _read_decimal_integer(value: object) -> int:
    if type(value) is not str or _ASCII_DECIMAL_RE.fullmatch(value) is None:
        raise RuntimeConfigurationError(RUNTIME_CONFIGURATION_ERROR_MESSAGE)
    try:
        return int(value, 10)
    except ValueError:
        raise RuntimeConfigurationError(
            RUNTIME_CONFIGURATION_ERROR_MESSAGE
        ) from None


def _load_agent_ask_config(
    environ: Mapping[str, object],
) -> AgentAskRuntimeConfig:
    episode_id = _required_text(environ.get(PODCAST_ASK_EPISODE_ID))
    if _EPISODE_ID_RE.fullmatch(episode_id) is None:
        raise RuntimeConfigurationError(RUNTIME_CONFIGURATION_ERROR_MESSAGE)

    language_model = LanguageModelRuntimeConfig(
        api_key=_required_text(environ.get(LLM_API_KEY)),
        base_url=_read_base_url(environ.get(LLM_BASE_URL)),
        model_fast=_read_identifier(environ.get(LLM_MODEL_FAST)),
        model_strong=_read_identifier(environ.get(LLM_MODEL_STRONG)),
        timeout_seconds=_read_positive_float(environ.get(LLM_TIMEOUT_SECONDS)),
    )
    embedding = EmbeddingRuntimeConfig(
        api_key=_required_text(environ.get(EMBEDDING_API_KEY)),
        base_url=_read_base_url(environ.get(EMBEDDING_BASE_URL)),
        model=_read_identifier(environ.get(EMBEDDING_MODEL)),
        expected_dimension=_read_positive_integer(
            environ.get(EMBEDDING_DIMENSION)
        ),
        document_task=_read_identifier(environ.get(EMBEDDING_DOCUMENT_TASK)),
        query_task=_read_identifier(environ.get(EMBEDDING_QUERY_TASK)),
        embedding_type=_read_embedding_type(environ.get(EMBEDDING_TYPE)),
        normalized=_read_boolean(environ.get(EMBEDDING_NORMALIZED)),
        truncate=_read_boolean(environ.get(EMBEDDING_TRUNCATE)),
        batch_size=_read_positive_integer(environ.get(EMBEDDING_BATCH_SIZE)),
        timeout_seconds=_read_positive_float(
            environ.get(EMBEDDING_TIMEOUT_SECONDS)
        ),
    )
    return AgentAskRuntimeConfig(
        episode_id=episode_id,
        transcript_path=_read_absolute_path(
            environ.get(PODCAST_ASK_TRANSCRIPT_PATH)
        ),
        index_directory=_read_absolute_path(
            environ.get(PODCAST_ASK_INDEX_DIRECTORY)
        ),
        language_model=language_model,
        embedding=embedding,
    )


def _required_text(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise RuntimeConfigurationError(RUNTIME_CONFIGURATION_ERROR_MESSAGE)
    return value


def _read_identifier(value: object) -> str:
    text = _required_text(value)
    if _SAFE_IDENTIFIER_RE.fullmatch(text) is None:
        raise RuntimeConfigurationError(RUNTIME_CONFIGURATION_ERROR_MESSAGE)
    return text


def _read_bearer_token(value: object) -> str:
    text = _required_text(value)
    if _BEARER_TOKEN_RE.fullmatch(text) is None:
        raise RuntimeConfigurationError(RUNTIME_CONFIGURATION_ERROR_MESSAGE)
    return text


def _read_absolute_path(value: object) -> str:
    text = _required_text(value)
    if "\x00" in text or not Path(text).is_absolute():
        raise RuntimeConfigurationError(RUNTIME_CONFIGURATION_ERROR_MESSAGE)
    return text


def _read_base_url(value: object) -> str:
    text = _required_text(value)
    try:
        parsed = urlsplit(text)
    except ValueError:
        raise RuntimeConfigurationError(
            RUNTIME_CONFIGURATION_ERROR_MESSAGE
        ) from None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeConfigurationError(RUNTIME_CONFIGURATION_ERROR_MESSAGE)
    return text.rstrip("/")


def _read_positive_integer(value: object) -> int:
    parsed = _read_decimal_integer(value)
    if parsed <= 0:
        raise RuntimeConfigurationError(RUNTIME_CONFIGURATION_ERROR_MESSAGE)
    return parsed


def _read_positive_float(value: object) -> float:
    if (
        type(value) is not str
        or _ASCII_POSITIVE_FLOAT_RE.fullmatch(value) is None
    ):
        raise RuntimeConfigurationError(RUNTIME_CONFIGURATION_ERROR_MESSAGE)
    parsed = float(value)
    if not isfinite(parsed) or not 0 < parsed <= _MAX_REMOTE_TIMEOUT_SECONDS:
        raise RuntimeConfigurationError(RUNTIME_CONFIGURATION_ERROR_MESSAGE)
    return parsed


def _read_boolean(value: object) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise RuntimeConfigurationError(RUNTIME_CONFIGURATION_ERROR_MESSAGE)


def _read_embedding_type(value: object) -> Literal["float"]:
    if value != "float":
        raise RuntimeConfigurationError(RUNTIME_CONFIGURATION_ERROR_MESSAGE)
    return "float"
