"""Deterministic text and timestamp normalization for transcripts."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any


_NOISE_MARKER = re.compile(
    r"[\[【(（]\s*(?:音乐|掌声|笑声|广告|片头|片尾|听不清|杂音|"
    r"music|applause|laughter|inaudible|crosstalk|background\s+noise)"
    r"\s*[\]】)）]",
    flags=re.IGNORECASE,
)
_WHITESPACE = re.compile(r"\s+")


class TimestampError(ValueError):
    """Raised when a source timestamp cannot be normalized."""


def clean_text(value: str) -> str:
    """Normalize Unicode/whitespace and remove explicit noise markers."""
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\u200b", "").replace("\ufeff", "")
    without_noise = _NOISE_MARKER.sub(" ", normalized)
    return _WHITESPACE.sub(" ", without_noise).strip()


def clean_speaker(value: Any) -> str | None:
    """Normalize an optional speaker label without inventing one."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("speaker 必须是字符串")
    return clean_text(value) or None


def parse_timestamp(value: Any, *, field_name: str) -> float | None:
    """Convert seconds or MM:SS/HH:MM:SS text to seconds."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise TimestampError(f"{field_name} 不是有效时间")
    if isinstance(value, (int, float)):
        seconds = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if ":" not in text:
            try:
                seconds = float(text)
            except ValueError as error:
                raise TimestampError(f"{field_name} 不是有效时间") from error
        else:
            parts = text.split(":")
            if len(parts) not in (2, 3):
                raise TimestampError(f"{field_name} 不是有效时间")
            try:
                numbers = [float(part) for part in parts]
            except ValueError as error:
                raise TimestampError(f"{field_name} 不是有效时间") from error
            if any(number < 0 for number in numbers):
                raise TimestampError(f"{field_name} 不能为负数")
            if numbers[-1] >= 60 or (len(numbers) == 3 and numbers[-2] >= 60):
                raise TimestampError(f"{field_name} 的分或秒必须小于 60")
            seconds = (
                numbers[0] * 60 + numbers[1]
                if len(numbers) == 2
                else numbers[0] * 3600 + numbers[1] * 60 + numbers[2]
            )
    else:
        raise TimestampError(
            f"{field_name} 必须是秒数或 MM:SS/HH:MM:SS 格式"
        )

    if not math.isfinite(seconds):
        raise TimestampError(f"{field_name} 必须是有限数字")
    if seconds < 0:
        raise TimestampError(f"{field_name} 不能为负数")
    return seconds
