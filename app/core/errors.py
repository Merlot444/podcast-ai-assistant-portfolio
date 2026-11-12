"""Shared helpers for keeping untrusted input out of user-facing errors."""

from __future__ import annotations

from pydantic import ValidationError


def safe_error_message(error: Exception) -> str:
    """Return diagnostics without trusting third-party exception text."""
    if isinstance(error, ValidationError):
        count = error.error_count()
        return f"输入契约校验失败（{count} 项字段错误）"
    if error.__class__.__module__.startswith("app."):
        return str(error)
    return f"{type(error).__name__}；详细内容已脱敏"
