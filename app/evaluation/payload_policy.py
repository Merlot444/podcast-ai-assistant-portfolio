"""Shared forbidden-field policy for persisted evaluation artifacts."""

from __future__ import annotations

from collections.abc import Collection


FORBIDDEN_PERSISTED_CONTENT_KEYS = frozenset(
    {
        "answer",
        "api_key",
        "authorization",
        "chunk_text",
        "expected_chunk_ids",
        "expected_source_segment_ids",
        "messages",
        "prompt",
        "question",
        "raw_response",
        "reference_answer",
        "secret",
        "text",
        "transcript",
    }
)

# Day 16 deliberately retained generated answers for auditable human scoring.
# This exception is disclosed in the README and Day 16 report; Day 17 has none.
DAY16_DISCLOSED_CONTENT_KEY_EXCEPTIONS = frozenset({"answer"})

# The expanded retrieval artifact retains stable truth IDs needed to recompute
# metrics. It contains no question, transcript, Chunk text, or generated answer.
V2_RETRIEVAL_METADATA_KEY_EXCEPTIONS = frozenset(
    {"expected_source_segment_ids"}
)


def find_unsafe_payload_path(
    value: object,
    *,
    allowed_keys: Collection[str] = (),
    path: str = "$",
) -> str | None:
    """Return the first forbidden field/value path, or ``None`` when safe."""
    allowed = {key.casefold() for key in allowed_keys}
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = key.casefold()
            item_path = f"{path}.{key}"
            if (
                normalized_key in FORBIDDEN_PERSISTED_CONTENT_KEYS
                and normalized_key not in allowed
            ):
                return item_path
            unsafe = find_unsafe_payload_path(
                item,
                allowed_keys=allowed,
                path=item_path,
            )
            if unsafe is not None:
                return unsafe
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            unsafe = find_unsafe_payload_path(
                item,
                allowed_keys=allowed,
                path=f"{path}[{index}]",
            )
            if unsafe is not None:
                return unsafe
    elif isinstance(value, str) and value.casefold().startswith("bearer "):
        return path
    return None
