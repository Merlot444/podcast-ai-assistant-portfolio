"""Loading and validation for the five versioned Day 10 smoke questions."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from app.core.errors import safe_error_message
from app.schemas.rag_pipeline import Day10RegressionSuite


DEFAULT_REGRESSION_SUITE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "evaluation"
    / "day10-regression.json"
)


class RegressionSuiteError(ValueError):
    """Raised when the versioned Day 10 question file cannot be trusted."""


def load_day10_regression_suite(
    path: str | Path = DEFAULT_REGRESSION_SUITE,
) -> Day10RegressionSuite:
    """Read exactly five metadata-only regression cases without network access."""
    source = Path(path)
    if not source.is_file():
        raise RegressionSuiteError(f"找不到 Day 10 回归问题文件：{source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise RegressionSuiteError("Day 10 回归问题文件必须使用 UTF-8") from error
    except json.JSONDecodeError as error:
        raise RegressionSuiteError(
            f"Day 10 回归 JSON 格式错误（第 {error.lineno} 行，第 {error.colno} 列）"
        ) from error
    try:
        return Day10RegressionSuite.model_validate(payload)
    except ValidationError as error:
        raise RegressionSuiteError(
            f"Day 10 回归问题校验失败：{safe_error_message(error)}"
        ) from error
