"""Tests for parsing and validating model output."""

import pytest

from app.generation.analyzer import AnalysisOutputError, parse_analysis


def test_parse_valid_analysis() -> None:
    raw_content = """{
      "summary": "The speakers discuss scaling and architectural research.",
      "keywords": ["scaling", "architecture", "research"],
      "tags": ["AI", "future systems"]
    }"""

    result = parse_analysis(raw_content)

    assert result.summary.startswith("The speakers")
    assert result.keywords == ["scaling", "architecture", "research"]


@pytest.mark.parametrize(
    "raw_content",
    (
        "not json",
        '{"summary":"Only a summary"}',
        '{"summary":"x","keywords":["one"],"tags":["AI"]}',
    ),
)
def test_parse_rejects_invalid_model_output(raw_content: str) -> None:
    with pytest.raises(AnalysisOutputError):
        parse_analysis(raw_content)
