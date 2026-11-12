"""Validated manifest for one persistent Day 7 FAISS index."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.chunk import TranscriptChunk


class VectorIndexManifest(BaseModel):
    """The reproducibility and row-mapping data stored beside a FAISS index."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    index_type: Literal["faiss_index_flat_ip"] = "faiss_index_flat_ip"
    distance_metric: Literal["cosine"] = "cosine"
    normalized: Literal[True] = True
    embedding_model: str = Field(min_length=1)
    embedding_document_task: str | None = None
    embedding_query_task: str | None = None
    embedding_type: str | None = None
    provider_normalized: bool | None = None
    provider_truncate: bool | None = None
    vector_dimension: int = Field(gt=0)
    vector_count: int = Field(gt=0)
    chunk_id_order: list[str] = Field(min_length=1)
    input_chunk_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    index_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    input_chunk_file: str = Field(min_length=1)
    chunking_algorithm: Literal["fixed_character_windows_v1"]
    episode_id: str = Field(pattern=r"^ep_[0-9a-f]{16}$")
    source_file: str = Field(min_length=1)
    input_segment_count: int = Field(gt=0)
    chunk_size: int = Field(gt=0)
    overlap: int = Field(ge=0)
    chunks: list[TranscriptChunk] = Field(min_length=1)

    @field_validator("embedding_model", "input_chunk_file", "source_file")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("字段不能为空")
        return cleaned

    @field_validator(
        "embedding_document_task",
        "embedding_query_task",
        "embedding_type",
    )
    @classmethod
    def strip_optional_embedding_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @model_validator(mode="after")
    def validate_mapping_and_source(self) -> Self:
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap 必须小于 chunk_size")
        if self.vector_count != len(self.chunks):
            raise ValueError("vector_count 必须与 chunks 数量一致")
        if self.vector_count != len(self.chunk_id_order):
            raise ValueError("vector_count 必须与 chunk_id_order 数量一致")

        expected_indexes = list(range(self.vector_count))
        if [chunk.chunk_index for chunk in self.chunks] != expected_indexes:
            raise ValueError("Chunk 顺序必须与 FAISS 行号连续对应")
        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        if self.chunk_id_order != chunk_ids:
            raise ValueError("chunk_id_order 必须与 chunks 顺序一致")
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("chunk_id_order 不能包含重复 ID")

        for chunk in self.chunks:
            if chunk.episode_id != self.episode_id:
                raise ValueError("Chunk 的 episode_id 与 manifest 不一致")
            if chunk.source_file != self.source_file:
                raise ValueError("Chunk 的 source_file 与 manifest 不一致")
        return self
