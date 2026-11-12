"""Strict contracts crossing the Day 20 Text2SQL trust boundary."""

from __future__ import annotations

import math
import re
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
)


ParameterValue = StrictStr | StrictInt | StrictFloat | StrictBool | None
ParameterMap = dict[str, ParameterValue]
ResultScalar = str | int | float | bool | None
_PARAMETER_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
MAX_PARAMETERS = 20
MAX_PARAMETER_STRING_LENGTH = 512


class Text2SQLSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
    )


class Text2SQLRequest(Text2SQLSchema):
    question: str = Field(min_length=1, max_length=2_000)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question 不能为空")
        return cleaned


class GeneratedQuery(Text2SQLSchema):
    """Untrusted structured output produced by a model or offline stub."""

    sql: str = Field(min_length=1, max_length=20_000)
    parameters: ParameterMap = Field(default_factory=dict)

    @field_validator("sql")
    @classmethod
    def normalize_sql(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("sql 不能为空")
        return cleaned

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, value: ParameterMap) -> ParameterMap:
        if len(value) > MAX_PARAMETERS:
            raise ValueError(f"parameters 最多允许 {MAX_PARAMETERS} 个")
        for name, parameter in value.items():
            if not _PARAMETER_NAME.fullmatch(name):
                raise ValueError(f"参数名无效：{name}")
            if isinstance(parameter, str) and len(parameter) > MAX_PARAMETER_STRING_LENGTH:
                raise ValueError(
                    f"字符串参数最长 {MAX_PARAMETER_STRING_LENGTH} 个字符"
                )
            if isinstance(parameter, float) and not math.isfinite(parameter):
                raise ValueError("浮点参数必须是有限值")
        return dict(value)


class ValidatedQuery(Text2SQLSchema):
    """Normalized query plus an in-process capability set only by the validator."""

    normalized_sql: str
    parameters: ParameterMap
    referenced_tables: tuple[str, ...]
    referenced_columns: tuple[str, ...]
    effective_limit: int = Field(ge=1)
    _approval: Any = PrivateAttr(default=None)

    def _approve(self, token: object) -> None:
        object.__setattr__(self, "_approval", token)

    def _is_approved_by(self, token: object) -> bool:
        return self._approval is token


class Text2SQLResult(Text2SQLSchema):
    columns: tuple[str, ...]
    rows: tuple[tuple[ResultScalar, ...], ...]
    row_count: int = Field(ge=0)
    truncated: bool
    elapsed_seconds: float = Field(ge=0)


def new_approval_token() -> object:
    """Create a validator/executor capability without exposing it in query JSON."""
    return object()
