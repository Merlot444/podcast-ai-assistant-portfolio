"""Thin projection from the existing Day 8 vector retriever."""

from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.corpus import RetrievalCorpusIdentity
from app.retrieval.core import Retriever
from app.schemas.retriever import RetrievedCandidate


@dataclass(frozen=True)
class VectorRetrieverAdapter:
    """Delegate once and preserve the Day 8 result order without ``row_index``."""

    retriever: Retriever
    name: str = "vector"
    score_semantics: str = "cosine_similarity"

    @property
    def corpus_identity(self) -> RetrievalCorpusIdentity:
        """Expose the validated ordered catalog needed for safe hybrid fusion."""
        return RetrievalCorpusIdentity.from_manifest(
            self.retriever.vector_index.manifest
        )

    def retrieve(self, query: str, top_k: int) -> list[RetrievedCandidate]:
        results = self.retriever.retrieve(query, top_k)
        return [
            RetrievedCandidate(
                rank=result.rank,
                chunk_id=result.chunk_id,
                source_segment_ids=tuple(result.source_segment_ids),
                score=result.score,
                score_semantics=self.score_semantics,
                retriever_name=self.name,
            )
            for result in results
        ]
