"""File loading and orchestration for the Day 7 index build."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from app.core.errors import safe_error_message
from app.indexing.embedding import Embedder
from app.indexing.index import (
    PersistentVectorIndex,
    build_vector_index,
    save_vector_index,
)
from app.schemas.chunk import ChunkedTranscript


class IndexingInputError(ValueError):
    """Raised when a Day 6 Chunk file cannot be validated."""


def load_chunked_transcript(path: str | Path) -> ChunkedTranscript:
    """Read one Git-ignored Day 6 result without contacting any service."""
    source = Path(path)
    if not source.is_file():
        raise IndexingInputError(f"找不到 Day 6 Chunk 文件：{source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise IndexingInputError("Day 6 Chunk 文件必须使用 UTF-8 编码") from error
    except json.JSONDecodeError as error:
        raise IndexingInputError(
            f"Day 6 Chunk JSON 格式错误（第 {error.lineno} 行，第 {error.colno} 列）"
        ) from error
    try:
        return ChunkedTranscript.model_validate(payload)
    except ValidationError as error:
        raise IndexingInputError(
            f"Day 6 Chunk 数据校验失败：{safe_error_message(error)}"
        ) from error


def build_and_save_index(
    input_path: str | Path,
    output_directory: str | Path,
    *,
    embedder: Embedder,
    overwrite: bool = False,
) -> tuple[PersistentVectorIndex, Path, Path]:
    """Build from a validated Chunk file and persist both required artifacts."""
    source = Path(input_path)
    chunked = load_chunked_transcript(source)
    vector_index = build_vector_index(
        chunked,
        embedder=embedder,
        input_chunk_file=source.name,
    )
    index_path, manifest_path = save_vector_index(
        vector_index,
        output_directory,
        overwrite=overwrite,
    )
    return vector_index, index_path, manifest_path
