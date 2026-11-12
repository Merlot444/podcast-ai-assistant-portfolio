"""Validated input and output structures."""

from app.schemas.analysis import PodcastAnalysis

__all__ = ["PodcastAnalysis"]
"""Validated application data structures."""

from app.schemas.analysis import PodcastAnalysis
from app.schemas.transcript import ProcessedTranscript, TranscriptSegment

__all__ = ["PodcastAnalysis", "ProcessedTranscript", "TranscriptSegment"]
