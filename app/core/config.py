"""Safe configuration loading for model and embedding APIs."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


EMBEDDING_TASKS = frozenset(
    {
        "retrieval.query",
        "retrieval.passage",
        "text-matching",
        "clustering",
        "classification",
    }
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(RuntimeError):
    """Raised when required application configuration is invalid."""


def _project_relative_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


@dataclass(frozen=True)
class Settings:
    """Validated settings needed by the language-model client."""

    api_key: str = field(repr=False)
    base_url: str
    model_fast: str
    model_strong: str
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> "Settings":
        """Load settings without exposing secret values in error messages."""
        if env_file is not None:
            load_dotenv(_project_relative_path(env_file), override=False)

        names = (
            "LLM_API_KEY",
            "LLM_BASE_URL",
            "LLM_MODEL_FAST",
            "LLM_MODEL_STRONG",
        )
        values = {name: os.getenv(name, "").strip() for name in names}
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise ConfigError(
                "缺少必要配置：" + ", ".join(missing) + "。请检查本地 .env 文件。"
            )

        base_url = values["LLM_BASE_URL"].rstrip("/")
        if not base_url.startswith(("https://", "http://")):
            raise ConfigError("LLM_BASE_URL 必须以 https:// 或 http:// 开头。")

        timeout_text = os.getenv("LLM_TIMEOUT_SECONDS", "30").strip()
        try:
            timeout_seconds = float(timeout_text)
        except ValueError as error:
            raise ConfigError("LLM_TIMEOUT_SECONDS 必须是数字。") from error
        if timeout_seconds <= 0:
            raise ConfigError("LLM_TIMEOUT_SECONDS 必须大于 0。")

        return cls(
            api_key=values["LLM_API_KEY"],
            base_url=base_url,
            model_fast=values["LLM_MODEL_FAST"],
            model_strong=values["LLM_MODEL_STRONG"],
            timeout_seconds=timeout_seconds,
        )


@dataclass(frozen=True)
class EmbeddingSettings:
    """Validated settings for a separate OpenAI-compatible Embedding service."""

    api_key: str = field(repr=False)
    base_url: str
    model: str
    expected_dimension: int | None = None
    batch_size: int = 16
    timeout_seconds: float = 30.0
    document_task: str = "retrieval.passage"
    query_task: str = "retrieval.query"
    embedding_type: str = "float"
    normalized: bool = True
    truncate: bool = False

    @classmethod
    def from_env(
        cls,
        env_file: str | Path | None = ".env",
    ) -> "EmbeddingSettings":
        """Load Embedding settings without exposing secret values."""
        if env_file is not None:
            load_dotenv(_project_relative_path(env_file), override=False)

        names = (
            "EMBEDDING_API_KEY",
            "EMBEDDING_BASE_URL",
            "EMBEDDING_MODEL",
        )
        values = {name: os.getenv(name, "").strip() for name in names}
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise ConfigError(
                "缺少必要配置：" + ", ".join(missing) + "。请检查本地 .env 文件。"
            )

        base_url = values["EMBEDDING_BASE_URL"].rstrip("/")
        if not base_url.startswith(("https://", "http://")):
            raise ConfigError(
                "EMBEDDING_BASE_URL 必须以 https:// 或 http:// 开头。"
            )

        dimension_text = os.getenv("EMBEDDING_DIMENSION", "").strip()
        expected_dimension = _optional_positive_int(
            dimension_text,
            name="EMBEDDING_DIMENSION",
        )
        batch_size = _positive_int_from_env("EMBEDDING_BATCH_SIZE", default=16)
        timeout_seconds = _positive_float_from_env(
            "EMBEDDING_TIMEOUT_SECONDS",
            default=30.0,
        )
        document_task = _embedding_task_from_env(
            "EMBEDDING_DOCUMENT_TASK",
            default="retrieval.passage",
        )
        query_task = _embedding_task_from_env(
            "EMBEDDING_QUERY_TASK",
            default="retrieval.query",
        )
        embedding_type = os.getenv("EMBEDDING_TYPE", "float").strip()
        if embedding_type != "float":
            raise ConfigError(
                "EMBEDDING_TYPE 当前只支持 float，"
                "因为 FAISS 建库需要稠密浮点向量。"
            )
        normalized = _boolean_from_env("EMBEDDING_NORMALIZED", default=True)
        truncate = _boolean_from_env("EMBEDDING_TRUNCATE", default=False)

        return cls(
            api_key=values["EMBEDDING_API_KEY"],
            base_url=base_url,
            model=values["EMBEDDING_MODEL"],
            expected_dimension=expected_dimension,
            batch_size=batch_size,
            timeout_seconds=timeout_seconds,
            document_task=document_task,
            query_task=query_task,
            embedding_type=embedding_type,
            normalized=normalized,
            truncate=truncate,
        )


def _optional_positive_int(value: str, *, name: str) -> int | None:
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise ConfigError(f"{name} 必须是整数。") from error
    if parsed <= 0:
        raise ConfigError(f"{name} 必须大于 0。")
    return parsed


def _positive_int_from_env(name: str, *, default: int) -> int:
    value = os.getenv(name, str(default)).strip()
    try:
        parsed = int(value)
    except ValueError as error:
        raise ConfigError(f"{name} 必须是整数。") from error
    if parsed <= 0:
        raise ConfigError(f"{name} 必须大于 0。")
    return parsed


def _positive_float_from_env(name: str, *, default: float) -> float:
    value = os.getenv(name, str(default)).strip()
    try:
        parsed = float(value)
    except ValueError as error:
        raise ConfigError(f"{name} 必须是数字。") from error
    if parsed <= 0:
        raise ConfigError(f"{name} 必须大于 0。")
    return parsed


def _embedding_task_from_env(name: str, *, default: str) -> str:
    value = os.getenv(name, default).strip()
    if value not in EMBEDDING_TASKS:
        allowed = ", ".join(sorted(EMBEDDING_TASKS))
        raise ConfigError(f"{name} 不受支持；可选值：{allowed}。")
    return value


def _boolean_from_env(name: str, *, default: bool) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ConfigError(f"{name} 必须是 true 或 false。")


_LEGACY_STORAGE_EXPORTS = frozenset(
    {
        "DatabaseSettings",
        "DEFAULT_DATABASE_URL",
        "SUPPORTED_DATABASE_DRIVERS",
    }
)


def __getattr__(name: str) -> object:
    """Load legacy database exports only when an old import requests them."""
    if name in _LEGACY_STORAGE_EXPORTS:
        from app.storage import config as storage_config

        return getattr(storage_config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
