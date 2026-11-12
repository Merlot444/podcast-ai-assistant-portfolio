"""Safe persistence for complete Day 12 baseline artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from app.schemas.evaluation_result import RetrievalBaselineResult


class BaselinePersistenceError(RuntimeError):
    """Raised when a baseline cannot be safely written."""


def save_retrieval_baseline(
    result: RetrievalBaselineResult,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically save a complete result and refuse silent replacement."""
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise BaselinePersistenceError(
            f"目标基线已存在：{destination}；不会静默覆盖"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            result.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, destination)
    except OSError as error:
        raise BaselinePersistenceError("基线结果写入失败") from error
    return destination
