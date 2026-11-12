"""Offline loading and persistence for Day 6 chunk generation."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from app.chunking.core import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    ChunkingError,
    chunk_transcript,
)
from app.core.errors import safe_error_message
from app.core.files import atomic_write_text
from app.schemas.chunk import ChunkedTranscript
from app.schemas.transcript import ProcessedTranscript


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_processed_transcript(path: str | Path) -> ProcessedTranscript:
    """Read and validate one Day 3 processed transcript without network access."""
    source = Path(path)
    if not source.is_file():
        raise ChunkingError(f"找不到处理后的转写文件：{source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise ChunkingError("处理后的转写文件必须使用 UTF-8 编码") from error
    except json.JSONDecodeError as error:
        raise ChunkingError(
            f"处理后的转写 JSON 格式错误（第 {error.lineno} 行，第 {error.colno} 列）"
        ) from error
    try:
        return ProcessedTranscript.model_validate(payload)
    except ValidationError as error:
        raise ChunkingError(
            f"处理后的转写数据校验失败：{safe_error_message(error)}"
        ) from error


def default_output_path(input_path: str | Path) -> Path:
    """Place generated chunks under the Git-ignored processed-data directory."""
    source = Path(input_path)
    return _PROJECT_ROOT / "data" / "processed" / f"{source.stem}-chunks.json"


def save_chunked_transcript(
    result: ChunkedTranscript,
    output_path: str | Path,
) -> Path:
    """Save deterministic, readable UTF-8 JSON."""
    destination = Path(output_path)
    atomic_write_text(
        destination,
        json.dumps(
            result.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def process_chunks(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> tuple[ChunkedTranscript, Path]:
    """Load a Day 3 file, create chunks, and save the validated result."""
    transcript = load_processed_transcript(input_path)
    result = chunk_transcript(
        transcript,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    destination = (
        Path(output_path) if output_path else default_output_path(input_path)
    )
    return result, save_chunked_transcript(result, destination)
