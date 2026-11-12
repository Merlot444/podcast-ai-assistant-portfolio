"""Strict schemas for the multi-episode v2 evaluation pilot."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from app.schemas.evaluation import (
    EVALUATION_CATEGORIES,
    EvaluationCase,
    EvaluationChunkBaseline,
    evaluation_cases_sha256,
)


class BaseDatasetReference(BaseModel):
    """Immutable pointer to the original v1 benchmark."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: Literal["eval_dataset_v1"]
    artifact_path: Literal["data/evaluation/eval_dataset_v1.json"]
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cases_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_count: Literal[20]


class DraftGenerationBudget(BaseModel):
    """Provider usage recorded for draft creation, not semantic review."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["OpenAI-compatible DeepSeek service"]
    model: str = Field(min_length=1, max_length=100)
    fast_model_calls: StrictInt = Field(ge=0, le=8)
    strong_model_calls: Literal[0]
    embedding_calls: Literal[0]
    rerank_calls: Literal[0]
    input_tokens: StrictInt = Field(ge=0)
    output_tokens: StrictInt = Field(ge=0)
    total_tokens: StrictInt = Field(ge=0)
    batch_response_sha256: tuple[str, ...] = Field(min_length=8, max_length=8)

    @model_validator(mode="after")
    def validate_usage(self) -> Self:
        if self.input_tokens + self.output_tokens != self.total_tokens:
            raise ValueError("total_tokens 必须等于 input_tokens 与 output_tokens 之和")
        if len(set(self.batch_response_sha256)) != len(
            self.batch_response_sha256
        ):
            raise ValueError("批次响应摘要必须唯一")
        return self


class PilotEpisodeDataset(BaseModel):
    """Thirty reviewed cases and their source identities for one episode."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_slug: str = Field(pattern=r"^[a-z][a-z0-9_]{2,31}$")
    episode_id: str = Field(pattern=r"^ep_[0-9a-f]{16}$")
    title: str = Field(min_length=1, max_length=300)
    source_url: str = Field(pattern=r"^https://(?:www\.)?dwarkesh\.com/p/[a-z0-9-]+$")
    article_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_transcript_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    processed_transcript_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunk_catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    timestamp_semantics: Literal["speaker_turn", "chapter_start_only"]
    chunk_baseline: EvaluationChunkBaseline
    cases_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cases: tuple[EvaluationCase, ...] = Field(min_length=30, max_length=30)

    @model_validator(mode="after")
    def validate_episode(self) -> Self:
        case_ids = [case.case_id for case in self.cases]
        questions = [" ".join(case.question.casefold().split()) for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("单期节目内 case_id 必须唯一")
        if len(questions) != len(set(questions)):
            raise ValueError("单期节目内 question 必须唯一")
        counts = Counter(case.category for case in self.cases)
        expected_counts = {category: 6 for category in EVALUATION_CATEGORIES}
        if counts != expected_counts:
            raise ValueError(f"单期节目必须五类各 6 题：{counts}")
        prefix = f"{self.episode_id}:"
        if any(
            not source_id.startswith(prefix)
            for case in self.cases
            for source_id in case.expected_source_segment_ids
        ):
            raise ValueError("案例来源片段不属于声明的 episode_id")
        if evaluation_cases_sha256(self.cases) != self.cases_sha256:
            raise ValueError("cases_sha256 与单期节目案例不一致")
        return self


class EvaluationDatasetV2Pilot(BaseModel):
    """An additive 60-case pilot that preserves the frozen 20-case v1 set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: Literal["eval_dataset_v2_pilot"]
    schema_version: Literal[2]
    parent: BaseDatasetReference
    source_annotation_unit: Literal["source_segment_id"]
    review_status: Literal["single_agent_semantic_review_complete"]
    independent_human_review_required: Literal[True]
    new_case_count: Literal[60]
    expanded_case_count: Literal[80]
    answerable_new_cases: Literal[48]
    unanswerable_new_cases: Literal[12]
    draft_generation: DraftGenerationBudget
    episode_datasets: tuple[PilotEpisodeDataset, ...] = Field(
        min_length=2, max_length=2
    )
    collection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_collection(self) -> Self:
        episode_ids = [item.episode_id for item in self.episode_datasets]
        slugs = [item.episode_slug for item in self.episode_datasets]
        if len(set(episode_ids)) != 2 or len(set(slugs)) != 2:
            raise ValueError("试点必须包含两期不同节目")
        cases = [case for item in self.episode_datasets for case in item.cases]
        case_ids = [case.case_id for case in cases]
        questions = [" ".join(case.question.casefold().split()) for case in cases]
        if len(cases) != self.new_case_count:
            raise ValueError("新增题数与 new_case_count 不一致")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("跨节目 case_id 必须唯一")
        if len(questions) != len(set(questions)):
            raise ValueError("跨节目 question 必须唯一")
        if sum(case.answerable for case in cases) != self.answerable_new_cases:
            raise ValueError("可回答题数量不一致")
        if collection_sha256(self) != self.collection_sha256:
            raise ValueError("collection_sha256 与集合内容不一致")
        return self


def collection_sha256(dataset: EvaluationDatasetV2Pilot | dict[str, object]) -> str:
    """Hash the complete collection except its self-referential digest."""
    if isinstance(dataset, BaseModel):
        payload = dataset.model_dump(mode="json")
    else:
        payload = dict(dataset)
    payload.pop("collection_sha256", None)
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    """Return deterministic first-seen metadata order."""
    return tuple(dict.fromkeys(values))
