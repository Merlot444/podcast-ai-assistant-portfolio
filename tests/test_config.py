"""Tests for environment-based configuration."""

from pathlib import Path
import subprocess
import sys

import pytest

from app.core.config import PROJECT_ROOT, ConfigError, EmbeddingSettings, Settings
from app.core.errors import safe_error_message


REQUIRED_NAMES = (
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL_FAST",
    "LLM_MODEL_STRONG",
    "LLM_TIMEOUT_SECONDS",
)


def test_safe_error_message_does_not_trust_foreign_exception_text() -> None:
    secret = "raw query or model output"

    message = safe_error_message(RuntimeError(secret))

    assert secret not in message
    assert "RuntimeError" in message


def test_safe_error_message_preserves_local_safe_error() -> None:
    message = safe_error_message(ConfigError("缺少配置名称"))

    assert message == "缺少配置名称"


def test_settings_load_valid_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key-not-a-secret")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com/")
    monkeypatch.setenv("LLM_MODEL_FAST", "deepseek-v4-flash")
    monkeypatch.setenv("LLM_MODEL_STRONG", "deepseek-v4-pro")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "15")

    settings = Settings.from_env(env_file=None)

    assert settings.base_url == "https://api.deepseek.com"
    assert settings.model_fast == "deepseek-v4-flash"
    assert settings.timeout_seconds == 15
    assert "test-key-not-a-secret" not in repr(settings)


def test_model_settings_resolve_default_env_from_project_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    captured: list[Path] = []

    def fake_load_dotenv(path: Path, *, override: bool) -> bool:
        captured.append(path)
        return True

    monkeypatch.setattr("app.core.config.load_dotenv", fake_load_dotenv)
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test")
    monkeypatch.setenv("LLM_MODEL_FAST", "fast")
    monkeypatch.setenv("LLM_MODEL_STRONG", "strong")

    Settings.from_env()

    assert captured == [(PROJECT_ROOT / ".env").resolve()]


def test_importing_model_config_does_not_require_sqlalchemy() -> None:
    script = """
import builtins

original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "sqlalchemy" or name.startswith("sqlalchemy."):
        raise ModuleNotFoundError("blocked by test")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from app.core.config import EmbeddingSettings, Settings
assert Settings is not None and EmbeddingSettings is not None
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_missing_settings_report_names_not_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in REQUIRED_NAMES:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ConfigError) as captured:
        Settings.from_env(env_file=None)

    message = str(captured.value)
    assert "LLM_API_KEY" in message
    assert "LLM_BASE_URL" in message
    assert "test-key" not in message


def test_embedding_settings_are_independent_and_dimension_is_optional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_API_KEY", "embedding-test-key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://embedding.example/v1/")
    monkeypatch.setenv("EMBEDDING_MODEL", "embedding-model")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "8")
    monkeypatch.setenv("EMBEDDING_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("EMBEDDING_DOCUMENT_TASK", "retrieval.passage")
    monkeypatch.setenv("EMBEDDING_QUERY_TASK", "retrieval.query")
    monkeypatch.setenv("EMBEDDING_TYPE", "float")
    monkeypatch.setenv("EMBEDDING_NORMALIZED", "true")
    monkeypatch.setenv("EMBEDDING_TRUNCATE", "false")

    settings = EmbeddingSettings.from_env(env_file=None)

    assert settings.base_url == "https://embedding.example/v1"
    assert settings.model == "embedding-model"
    assert settings.expected_dimension is None
    assert settings.batch_size == 8
    assert settings.timeout_seconds == 12.5
    assert settings.document_task == "retrieval.passage"
    assert settings.query_task == "retrieval.query"
    assert settings.embedding_type == "float"
    assert settings.normalized is True
    assert settings.truncate is False
    assert "embedding-test-key" not in repr(settings)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("EMBEDDING_DIMENSION", "0", "EMBEDDING_DIMENSION"),
        ("EMBEDDING_BATCH_SIZE", "not-an-int", "EMBEDDING_BATCH_SIZE"),
        ("EMBEDDING_TIMEOUT_SECONDS", "-1", "EMBEDDING_TIMEOUT_SECONDS"),
    ],
)
def test_invalid_embedding_numeric_configuration_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    message: str,
) -> None:
    monkeypatch.setenv("EMBEDDING_API_KEY", "embedding-test-key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://embedding.example/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "embedding-model")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "2")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "8")
    monkeypatch.setenv("EMBEDDING_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("EMBEDDING_DOCUMENT_TASK", "retrieval.passage")
    monkeypatch.setenv("EMBEDDING_QUERY_TASK", "retrieval.query")
    monkeypatch.setenv("EMBEDDING_TYPE", "float")
    monkeypatch.setenv("EMBEDDING_NORMALIZED", "true")
    monkeypatch.setenv("EMBEDDING_TRUNCATE", "false")
    monkeypatch.setenv(name, value)

    with pytest.raises(ConfigError, match=message):
        EmbeddingSettings.from_env(env_file=None)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("EMBEDDING_DOCUMENT_TASK", "unknown", "EMBEDDING_DOCUMENT_TASK"),
        ("EMBEDDING_QUERY_TASK", "", "EMBEDDING_QUERY_TASK"),
        ("EMBEDDING_TYPE", "base64", "只支持 float"),
        ("EMBEDDING_NORMALIZED", "sometimes", "必须是 true 或 false"),
        ("EMBEDDING_TRUNCATE", "sometimes", "必须是 true 或 false"),
    ],
)
def test_invalid_jina_embedding_options_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    message: str,
) -> None:
    monkeypatch.setenv("EMBEDDING_API_KEY", "embedding-test-key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://api.jina.ai/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "jina-embeddings-v5-text-small")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "1024")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "16")
    monkeypatch.setenv("EMBEDDING_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("EMBEDDING_DOCUMENT_TASK", "retrieval.passage")
    monkeypatch.setenv("EMBEDDING_QUERY_TASK", "retrieval.query")
    monkeypatch.setenv("EMBEDDING_TYPE", "float")
    monkeypatch.setenv("EMBEDDING_NORMALIZED", "true")
    monkeypatch.setenv("EMBEDDING_TRUNCATE", "false")
    monkeypatch.setenv(name, value)

    with pytest.raises(ConfigError, match=message):
        EmbeddingSettings.from_env(env_file=None)
