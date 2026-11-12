"""Strict external input and output contracts for Day 18 storage operations."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)


STABLE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$"
StableId = Annotated[
    str,
    Field(min_length=1, max_length=128, pattern=STABLE_ID_PATTERN),
]
MessageRole = Literal["user", "assistant", "system"]
SessionStatus = Literal["active", "closed"]
EvalRunStatus = Literal["pending", "running", "completed", "failed"]
JsonObject = dict[str, JsonValue]


class StorageSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DocumentCreate(StorageSchema):
    document_id: StableId
    episode_id: StableId
    source_file: str = Field(min_length=1, max_length=512)
    source_format: Literal["txt", "json"]
    title: str | None = Field(default=None, max_length=500)
    input_summary: str | None = Field(default=None, max_length=2_000)
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("source_file")
    @classmethod
    def strip_source_file(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("source_file 不能为空")
        return cleaned

    @field_validator("title", "input_summary")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None


class DocumentMetadataUpdate(StorageSchema):
    title: str | None = Field(default=None, max_length=500)
    input_summary: str | None = Field(default=None, max_length=2_000)
    metadata: JsonObject | None = None

    @field_validator("title", "input_summary")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None

    @model_validator(mode="after")
    def require_update_field(self) -> Self:
        if not self.model_fields_set:
            raise ValueError("至少提供一个可更新字段")
        return self


class DocumentRead(DocumentCreate):
    created_at: datetime
    updated_at: datetime


class SessionCreate(StorageSchema):
    session_id: StableId
    title: str | None = Field(default=None, max_length=500)
    status: SessionStatus = "active"

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None


class SessionUpdate(StorageSchema):
    title: str | None = Field(default=None, max_length=500)
    status: SessionStatus | None = None

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None

    @model_validator(mode="after")
    def require_update_field(self) -> Self:
        if not self.model_fields_set:
            raise ValueError("至少提供一个可更新字段")
        return self


class SessionRead(SessionCreate):
    created_at: datetime
    updated_at: datetime


class MessageCreate(StorageSchema):
    message_id: StableId
    role: MessageRole
    content: str = Field(min_length=1, max_length=100_000)

    @field_validator("content")
    @classmethod
    def strip_content(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("content 不能为空")
        return cleaned


class MessageRead(MessageCreate):
    session_id: StableId
    sequence: int = Field(ge=0)
    created_at: datetime


class EvalRunCreate(StorageSchema):
    run_id: StableId
    evaluation_name: str = Field(min_length=1, max_length=256)
    schema_version: int = Field(ge=1)
    status: EvalRunStatus = "pending"
    input_artifact_ref: str | None = Field(default=None, max_length=1_024)
    input_summary: str | None = Field(default=None, max_length=2_000)
    config_metadata: JsonObject = Field(default_factory=dict)

    @field_validator("evaluation_name")
    @classmethod
    def strip_evaluation_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("evaluation_name 不能为空")
        return cleaned

    @field_validator("input_artifact_ref", "input_summary")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None


class EvalRunStatusUpdate(StorageSchema):
    status: EvalRunStatus


class EvalRunRead(EvalRunCreate):
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None


class EvalResultCreate(StorageSchema):
    result_id: StableId
    case_id: StableId
    method_name: str = Field(min_length=1, max_length=128)
    metrics_metadata: JsonObject = Field(default_factory=dict)

    @field_validator("method_name")
    @classmethod
    def strip_method_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("method_name 不能为空")
        return cleaned


class EvalResultRead(EvalResultCreate):
    eval_run_id: StableId
    created_at: datetime
