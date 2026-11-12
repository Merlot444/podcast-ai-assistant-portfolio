"""Content-free recording primitives for one Day 21 tool call."""

from __future__ import annotations

import json
import math
from hashlib import sha256
from typing import Mapping, Protocol, Sequence

from pydantic import BaseModel

from app.tooling.schemas import ToolCallRecord


MAX_AUDIT_HASH_DEPTH = 64
_CYCLE_MARKER = "<cycle>"
_DEPTH_MARKER = "<maximum-depth>"
_NORMALIZATION_ERROR_MARKER = "<normalization-error>"


class ToolCallRecorder(Protocol):
    def record(self, record: ToolCallRecord) -> None:
        """Save one already redacted call summary."""


class InMemoryToolCallRecorder:
    """Small default recorder suitable for one process and deterministic tests."""

    def __init__(self) -> None:
        self._records: list[ToolCallRecord] = []

    def record(self, record: ToolCallRecord) -> None:
        self._records.append(record)

    @property
    def records(self) -> tuple[ToolCallRecord, ...]:
        return tuple(self._records)


def stable_json_sha256(value: object) -> str:
    """Hash JSON-like content without persisting or reflecting its raw text."""
    try:
        normalized = _safe_json_value(value, depth=0, active_ids=set())
    except Exception:
        normalized = _NORMALIZATION_ERROR_MARKER
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def text_sha256(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _safe_json_value(
    value: object,
    *,
    depth: int,
    active_ids: set[int],
) -> object:
    if depth >= MAX_AUDIT_HASH_DEPTH:
        return _DEPTH_MARKER
    if isinstance(value, BaseModel):
        return _safe_json_value(
            value.model_dump(mode="json"),
            depth=depth + 1,
            active_ids=active_ids,
        )
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else f"<{type(value).__name__}>"
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active_ids:
            return _CYCLE_MARKER
        active_ids.add(identity)
        try:
            normalized: dict[str, object] = {}
            for key, item in value.items():
                safe_key = (
                    key if isinstance(key, str) else f"<{type(key).__name__}>"
                )
                normalized[safe_key] = _safe_json_value(
                    item,
                    depth=depth + 1,
                    active_ids=active_ids,
                )
            return normalized
        finally:
            active_ids.remove(identity)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        identity = id(value)
        if identity in active_ids:
            return _CYCLE_MARKER
        active_ids.add(identity)
        try:
            return [
                _safe_json_value(
                    item,
                    depth=depth + 1,
                    active_ids=active_ids,
                )
                for item in value
            ]
        finally:
            active_ids.remove(identity)
    return f"<{type(value).__name__}>"
