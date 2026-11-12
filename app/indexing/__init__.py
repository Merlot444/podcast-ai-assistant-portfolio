"""Day 7 Embedding and persistent FAISS indexing."""

from app.indexing.embedding import (
    Embedder,
    EmbeddingError,
    OpenAICompatibleEmbedder,
    SyntheticHashEmbedder,
)
from app.indexing.index import (
    PersistentVectorIndex,
    SimilarityCheckResult,
    build_vector_index,
    load_vector_index,
    save_vector_index,
)
from app.indexing.pipeline import build_and_save_index, load_chunked_transcript

__all__ = [
    "Embedder",
    "EmbeddingError",
    "OpenAICompatibleEmbedder",
    "PersistentVectorIndex",
    "SimilarityCheckResult",
    "SyntheticHashEmbedder",
    "build_and_save_index",
    "build_vector_index",
    "load_chunked_transcript",
    "load_vector_index",
    "save_vector_index",
]
