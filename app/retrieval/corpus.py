"""Strict corpus identity shared by independent evaluation retrievers."""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.index import VectorIndexManifest


@dataclass(frozen=True)
class RetrievalCorpusIdentity:
    """Identify one ordered Chunk catalog without carrying Chunk text."""

    episode_id: str
    source_file: str
    chunking_algorithm: str
    chunk_size: int
    overlap: int
    input_chunk_sha256: str
    chunk_id_order: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.episode_id or not self.source_file:
            raise ValueError("检索语料 identity 缺少 episode 或 source_file")
        if self.chunk_size <= 0 or self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("检索语料 identity 的 Chunk 配置无效")
        if len(self.input_chunk_sha256) != 64:
            raise ValueError("检索语料 identity 的 input_chunk_sha256 无效")
        if not self.chunk_id_order or len(self.chunk_id_order) != len(
            set(self.chunk_id_order)
        ):
            raise ValueError("检索语料 identity 的 Chunk catalog 必须非空且唯一")

    @classmethod
    def from_manifest(cls, manifest: VectorIndexManifest) -> RetrievalCorpusIdentity:
        """Project the complete ordered-corpus identity from a validated manifest."""
        return cls(
            episode_id=manifest.episode_id,
            source_file=manifest.source_file,
            chunking_algorithm=manifest.chunking_algorithm,
            chunk_size=manifest.chunk_size,
            overlap=manifest.overlap,
            input_chunk_sha256=manifest.input_chunk_sha256,
            chunk_id_order=tuple(manifest.chunk_id_order),
        )

    @property
    def chunk_position_by_id(self) -> dict[str, int]:
        """Return the predeclared stable tie-breaking position for every Chunk."""
        return {chunk_id: index for index, chunk_id in enumerate(self.chunk_id_order)}
