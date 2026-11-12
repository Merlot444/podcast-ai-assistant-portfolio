"""Mocked tests for the model-facing analysis workflow."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.core.config import Settings
from app.generation.analyzer import AnalysisOutputError, analyze_transcript


def _settings() -> Settings:
    return Settings(
        api_key="test-key-not-a-secret",
        base_url="https://example.test",
        model_fast="fast-test-model",
        model_strong="strong-test-model",
    )


def test_analyze_transcript_uses_json_mode_without_real_api_call() -> None:
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    content=(
                        '{"summary":"Scaling and research both matter.",'
                        '"keywords":["scaling","architecture","research"],'
                        '"tags":["AI","model development"]}'
                    )
                ),
            )
        ]
    )
    settings = Settings(
        api_key="test-key-not-a-secret",
        base_url="https://api.deepseek.com",
        model_fast="deepseek-v4-flash",
        model_strong="deepseek-v4-pro",
    )

    result = analyze_transcript(client, settings, "A short transcript excerpt.")

    assert result.tags == ["AI", "model development"]
    request = client.chat.completions.create.call_args.kwargs
    assert request["model"] == "deepseek-v4-flash"
    assert request["response_format"] == {"type": "json_object"}
    assert request["temperature"] == 0.2
    assert request["max_tokens"] == 600
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "JSON" in request["messages"][0]["content"]


def test_analyze_transcript_rejects_empty_model_content() -> None:
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="   "),
            )
        ]
    )

    with pytest.raises(AnalysisOutputError, match="空内容"):
        analyze_transcript(client, _settings(), "A short transcript excerpt.")


def test_analyze_transcript_rejects_response_without_choices() -> None:
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(choices=[])

    with pytest.raises(AnalysisOutputError, match="没有可用结果"):
        analyze_transcript(client, _settings(), "A short transcript excerpt.")
