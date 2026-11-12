"""Synthetic, offline wiring for the one-call Day 24 MCP demonstration."""

from __future__ import annotations

from app.schemas.transcript import ProcessedTranscript, TranscriptSegment
from app.tooling import ToolDependencies, ToolDispatcher, ToolHandlers


SYNTHETIC_EPISODE_ID = "ep_1111111111111111"


def build_synthetic_dispatcher() -> ToolDispatcher:
    """Build trusted in-memory dependencies without a model, file read, or network."""
    source_file = "day24-synthetic.json"
    transcript = ProcessedTranscript(
        episode_id=SYNTHETIC_EPISODE_ID,
        source_file=source_file,
        source_format="json",
        segments=[
            TranscriptSegment(
                segment_id=f"{SYNTHETIC_EPISODE_ID}:000000",
                segment_index=0,
                text="Original synthetic introduction used only in memory.",
                source_file=source_file,
                speaker="Host",
                section_title="Opening",
                start_time=0.0,
                end_time=4.0,
            ),
            TranscriptSegment(
                segment_id=f"{SYNTHETIC_EPISODE_ID}:000001",
                segment_index=1,
                text="Original synthetic metadata example used only in memory.",
                source_file=source_file,
                speaker="Guest",
                section_title="Metadata",
                start_time=4.0,
                end_time=8.0,
            ),
        ],
    )
    dependencies = ToolDependencies(
        transcripts={SYNTHETIC_EPISODE_ID: transcript},
    )
    return ToolDispatcher(ToolHandlers(dependencies))
