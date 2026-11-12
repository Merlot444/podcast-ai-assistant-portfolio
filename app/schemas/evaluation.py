"""Strict, immutable schemas for the frozen Day 11 evaluation dataset."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)


EvaluationCategory = Literal[
    "fact",
    "multi_segment",
    "timestamp",
    "similar_term_distractor",
    "unanswerable",
]

EVALUATION_CATEGORIES = frozenset(
    {
        "fact",
        "multi_segment",
        "timestamp",
        "similar_term_distractor",
        "unanswerable",
    }
)
MINIMUM_CASES = 20
MINIMUM_CASES_PER_CATEGORY = 3

_SEGMENT_ID_PATTERN = re.compile(r"^ep_[0-9a-f]{16}:\d{6}$")
_CHUNK_ID_PATTERN = re.compile(r"^chunk_[0-9a-f]{16}$")


class EvaluationTimeRange(BaseModel):
    """Expected source time metadata for one stable transcript segment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_segment_id: str = Field(pattern=r"^ep_[0-9a-f]{16}:\d{6}$")
    start_time: float | None = Field(default=None, ge=0, strict=True)
    end_time: float | None = Field(default=None, ge=0, strict=True)
    time_range_complete: StrictBool

    @model_validator(mode="after")
    def validate_time_range(self) -> Self:
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.end_time < self.start_time
        ):
            raise ValueError("end_time 不能早于 start_time")
        if self.time_range_complete and (
            self.start_time is None or self.end_time is None
        ):
            raise ValueError("完整时间范围必须同时包含 start_time 和 end_time")
        return self


class EvaluationChunkBaseline(BaseModel):
    """The one Chunk configuration to which expected_chunk_ids belong."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    algorithm: Literal["fixed_character_windows_v1"]
    chunk_size: StrictInt = Field(gt=0)
    overlap: StrictInt = Field(ge=0)
    input_chunk_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_chunking_parameters(self) -> Self:
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap 必须小于 chunk_size")
        return self


class EvaluationCase(BaseModel):
    """One human-reviewed question and its metadata-only source annotation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    category: EvaluationCategory
    question: str = Field(min_length=1, max_length=500)
    answerable: StrictBool
    expected_source_segment_ids: tuple[str, ...]
    expected_chunk_ids: tuple[str, ...]
    expected_speakers: tuple[str, ...] = ()
    expected_section_titles: tuple[str, ...] = ()
    expected_time_ranges: tuple[EvaluationTimeRange, ...] = ()
    reference_answer: str | None = Field(default=None, max_length=500)
    notes: str = Field(min_length=1, max_length=800)

    @field_validator("question", "notes")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("字段不能为空")
        if cleaned != value:
            raise ValueError("冻结评测字段不能包含首尾空白")
        return value

    @field_validator("reference_answer")
    @classmethod
    def strip_reference_answer(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("空 reference_answer 必须显式写为 null")
        if cleaned != value:
            raise ValueError("冻结评测字段不能包含首尾空白")
        return value

    @field_validator("expected_source_segment_ids")
    @classmethod
    def validate_source_segment_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        _reject_duplicates(value, "expected_source_segment_ids")
        for segment_id in value:
            if not _SEGMENT_ID_PATTERN.fullmatch(segment_id):
                raise ValueError(
                    "expected_source_segment_ids 只能保存稳定的节目片段 ID，"
                    f"不能保存临时证据编号：{segment_id}"
                )
        return value

    @field_validator("expected_chunk_ids")
    @classmethod
    def validate_chunk_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        _reject_duplicates(value, "expected_chunk_ids")
        for chunk_id in value:
            if not _CHUNK_ID_PATTERN.fullmatch(chunk_id):
                raise ValueError(
                    "expected_chunk_ids 只能保存当前基线的稳定 chunk_<16 hex>，"
                    f"不能保存临时证据编号：{chunk_id}"
                )
        return value

    @field_validator("expected_speakers", "expected_section_titles")
    @classmethod
    def clean_metadata(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(item.strip() for item in value)
        if any(not item for item in cleaned):
            raise ValueError("来源元数据不能包含空字符串")
        if cleaned != value:
            raise ValueError("冻结来源元数据不能包含首尾空白")
        _reject_duplicates(cleaned, "来源元数据")
        return value

    @model_validator(mode="after")
    def validate_answerability_and_sources(self) -> Self:
        is_unanswerable_category = self.category == "unanswerable"
        if self.answerable == is_unanswerable_category:
            raise ValueError("answerable 必须与 unanswerable 类别一致")

        provenance_fields = (
            self.expected_source_segment_ids,
            self.expected_chunk_ids,
            self.expected_speakers,
            self.expected_section_titles,
            self.expected_time_ranges,
        )
        if self.answerable:
            if not self.expected_source_segment_ids:
                raise ValueError("可回答题必须至少有一个稳定来源片段 ID")
            if not self.expected_chunk_ids:
                raise ValueError("可回答题必须至少有一个当前基线 Chunk ID")
            time_range_ids = tuple(
                item.source_segment_id for item in self.expected_time_ranges
            )
            _reject_duplicates(time_range_ids, "expected_time_ranges")
            source_ids = set(self.expected_source_segment_ids)
            if any(segment_id not in source_ids for segment_id in time_range_ids):
                raise ValueError(
                    "expected_time_ranges 只能引用本题的稳定来源片段"
                )
            time_range_id_set = set(time_range_ids)
            expected_time_order = tuple(
                segment_id
                for segment_id in self.expected_source_segment_ids
                if segment_id in time_range_id_set
            )
            if time_range_ids != expected_time_order:
                raise ValueError("expected_time_ranges 必须遵循稳定来源片段顺序")
        else:
            if any(provenance_fields):
                raise ValueError("无答案题不能保存任何伪造来源或来源元数据")
            if self.reference_answer is not None:
                raise ValueError("无答案题不能保存伪造的 reference_answer")

        if self.category == "multi_segment" and len(
            self.expected_source_segment_ids
        ) < 2:
            raise ValueError("multi_segment 题必须至少标注两个必要来源片段")
        if self.category == "timestamp" and not any(
            (
                self.expected_speakers,
                self.expected_section_titles,
                self.expected_time_ranges,
            )
        ):
            raise ValueError("timestamp 题必须至少标注说话人、章节或时间来源")
        return self


class EvaluationDataset(BaseModel):
    """The frozen v1 question set used by all later retrieval experiments."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: Literal["eval_dataset_v1"]
    schema_version: Literal[1]
    episode_id: str = Field(pattern=r"^ep_[0-9a-f]{16}$")
    annotation_source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_annotation_unit: Literal["source_segment_id"]
    chunk_baseline: EvaluationChunkBaseline
    cases_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cases: tuple[EvaluationCase, ...] = Field(min_length=MINIMUM_CASES)

    @model_validator(mode="after")
    def validate_frozen_dataset(self) -> Self:
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("case_id 必须唯一")
        normalized_questions = [
            " ".join(case.question.casefold().split()) for case in self.cases
        ]
        if len(normalized_questions) != len(set(normalized_questions)):
            raise ValueError("question 必须唯一，不能复制同一问题凑数")

        category_counts = Counter(case.category for case in self.cases)
        missing = sorted(EVALUATION_CATEGORIES - set(category_counts))
        underrepresented = {
            category: category_counts[category]
            for category in sorted(EVALUATION_CATEGORIES)
            if category_counts[category] < MINIMUM_CASES_PER_CATEGORY
        }
        if missing or underrepresented:
            raise ValueError(
                "评测类别覆盖不足；"
                f"missing={missing}, underrepresented={underrepresented}"
            )

        expected_prefix = f"{self.episode_id}:"
        for case in self.cases:
            if any(
                not segment_id.startswith(expected_prefix)
                for segment_id in case.expected_source_segment_ids
            ):
                raise ValueError(
                    f"案例 {case.case_id} 的来源片段不属于数据集 episode_id"
                )

        actual_digest = evaluation_cases_sha256(self.cases)
        if actual_digest != self.cases_sha256:
            raise ValueError(
                "cases_sha256 与当前题目或标注不一致；"
                "eval_dataset_v1 不允许在加载时静默改题"
            )
        return self


class EvaluationValidationReport(BaseModel):
    """Read-only integrity statistics; deliberately contains no retrieval metric."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: Literal["eval_dataset_v1"]
    schema_version: Literal[1]
    episode_id: str
    cases_sha256: str
    total_cases: int = Field(ge=MINIMUM_CASES)
    category_counts: dict[str, int]
    answerable_cases: int = Field(ge=0)
    unanswerable_cases: int = Field(ge=0)
    single_source_cases: int = Field(ge=0)
    multi_source_cases: int = Field(ge=0)
    timestamp_cases: int = Field(ge=0)
    incomplete_time_range_cases: int = Field(ge=0)
    source_segment_annotations: int = Field(ge=0)
    baseline_chunk_annotations: int = Field(ge=0)
    all_source_segment_ids_exist: Literal[True]
    all_baseline_chunk_ids_exist: Literal[True]
    episode_ids_match: Literal[True]
    all_provided_source_metadata_matches: Literal[True]
    baseline_configuration_matches: Literal[True]
    full_chunk_text_absent: Literal[True] = True
    temporary_evidence_source_ids_absent: Literal[True] = True


def evaluation_cases_sha256(cases: Sequence[EvaluationCase]) -> str:
    """Hash normalized cases, including their order, questions, and annotations."""
    canonical = json.dumps(
        [case.model_dump(mode="json") for case in cases],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _reject_duplicates(value: Sequence[str], field_name: str) -> None:
    if len(value) != len(set(value)):
        raise ValueError(f"{field_name} 不能包含重复值")
