"""Persistence pipeline for cleaned transcript data."""

from __future__ import annotations

import json
from pathlib import Path

from app.core.files import atomic_write_text
from app.ingestion.transcripts import load_transcript
from app.schemas.transcript import ProcessedTranscript


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def default_output_path(input_path: str | Path) -> Path:
    """Place generated data under data/processed with a stable filename."""
    source_path = Path(input_path)
    return _PROJECT_ROOT / "data" / "processed" / f"{source_path.stem}.json"


def save_processed_transcript(
    transcript: ProcessedTranscript,
    output_path: str | Path,
) -> Path:
    """Write deterministic UTF-8 JSON without timestamps or random values."""
    destination = Path(output_path)
    payload = transcript.model_dump(mode="json")
    atomic_write_text(
        destination,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def process_transcript(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> tuple[ProcessedTranscript, Path]:
    """Load, clean, validate, and save one transcript."""
    transcript = load_transcript(input_path)
    destination = Path(output_path) if output_path else default_output_path(input_path)
    return transcript, save_processed_transcript(transcript, destination)
