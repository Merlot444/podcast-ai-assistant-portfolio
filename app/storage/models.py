"""SQLAlchemy models for the minimal Day 18 relational data boundary."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator


def utc_now() -> datetime:
    """Return an aware UTC timestamp for ORM defaults and lifecycle updates."""
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator[datetime]):
    """Use SQLite UTC text or PostgreSQL native timezone-aware timestamps."""

    impl = String(27)
    cache_ok = True

    def load_dialect_impl(self, dialect: Any) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(DateTime(timezone=True))
        return dialect.type_descriptor(String(27))

    def process_bind_param(
        self,
        value: datetime | None,
        dialect: Any,
    ) -> str | datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("数据库时间必须包含时区")
        normalized = value.astimezone(UTC)
        if dialect.name == "postgresql":
            return normalized
        return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")

    def process_result_value(
        self,
        value: str | datetime | None,
        dialect: Any,
    ) -> datetime | None:
        if value is None:
            return None
        parsed = (
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            if isinstance(value, str)
            else value
        )
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("数据库返回了无时区时间")
        return parsed.astimezone(UTC)


JSON_OBJECT = JSON().with_variant(JSONB(), "postgresql")


def _json_object_constraints(column_name: str, *, prefix: str) -> tuple[Any, Any]:
    sqlite_constraint = CheckConstraint(
        f"json_type({column_name}) = 'object'",
        name=f"ck_{prefix}_object",
    ).ddl_if(dialect="sqlite")
    postgres_constraint = CheckConstraint(
        f"jsonb_typeof({column_name}) = 'object'",
        name=f"ck_{prefix}_object_postgresql",
    ).ddl_if(dialect="postgresql")
    return sqlite_constraint, postgres_constraint


class Base(DeclarativeBase):
    """Declarative base kept private to the storage package."""


class DocumentModel(Base):
    """Metadata about one stable podcast document, without transcript text."""

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("episode_id", name="uq_documents_episode_id"),
        CheckConstraint("length(source_file) > 0", name="ck_documents_source_file"),
        CheckConstraint(
            "source_format IN ('txt', 'json')",
            name="ck_documents_source_format",
        ),
        *_json_object_constraints("metadata", prefix="documents_metadata"),
    )

    document_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    episode_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_file: Mapped[str] = mapped_column(String(512), nullable=False)
    source_format: Mapped[str] = mapped_column(String(16), nullable=False)
    title: Mapped[str | None] = mapped_column(String(500))
    input_summary: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSON_OBJECT,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=utc_now,
    )


class SessionModel(Base):
    """One conversation container."""

    __tablename__ = "sessions"
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'closed')",
            name="ck_sessions_status",
        ),
    )

    session_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    title: Mapped[str | None] = mapped_column(String(500))
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=utc_now,
    )
    messages: Mapped[list[MessageModel]] = relationship(
        back_populates="session_parent",
        passive_deletes=True,
    )


class MessageModel(Base):
    """One ordered message belonging to a conversation session."""

    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint("sequence >= 0", name="ck_messages_sequence"),
        CheckConstraint(
            "role IN ('user', 'assistant', 'system')",
            name="ck_messages_role",
        ),
        CheckConstraint("length(content) > 0", name="ck_messages_content"),
        Index(
            "ix_messages_session_sequence",
            "session_id",
            "sequence",
            unique=True,
        ),
    )

    message_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=utc_now,
    )
    session_parent: Mapped[SessionModel] = relationship(back_populates="messages")


class EvalRunModel(Base):
    """One versioned evaluation execution or imported result group."""

    __tablename__ = "eval_runs"
    __table_args__ = (
        CheckConstraint("length(evaluation_name) > 0", name="ck_eval_runs_name"),
        CheckConstraint("schema_version >= 1", name="ck_eval_runs_schema_version"),
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')",
            name="ck_eval_runs_status",
        ),
        *_json_object_constraints(
            "config_metadata",
            prefix="eval_runs_config_metadata",
        ),
    )

    run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    evaluation_name: Mapped[str] = mapped_column(String(256), nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    input_artifact_ref: Mapped[str | None] = mapped_column(String(1024))
    input_summary: Mapped[str | None] = mapped_column(Text)
    config_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSON_OBJECT,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=utc_now,
    )
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    results: Mapped[list[EvalResultModel]] = relationship(
        back_populates="eval_run",
        passive_deletes=True,
    )


class EvalResultModel(Base):
    """Metadata-only result for one case and one method in an evaluation run."""

    __tablename__ = "eval_results"
    __table_args__ = (
        CheckConstraint("length(case_id) > 0", name="ck_eval_results_case_id"),
        CheckConstraint("length(method_name) > 0", name="ck_eval_results_method"),
        *_json_object_constraints(
            "metrics_metadata",
            prefix="eval_results_metrics_metadata",
        ),
        Index(
            "ix_eval_results_run_case_method",
            "eval_run_id",
            "case_id",
            "method_name",
            unique=True,
        ),
    )

    result_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    eval_run_id: Mapped[str] = mapped_column(
        ForeignKey("eval_runs.run_id", ondelete="CASCADE"),
        nullable=False,
    )
    case_id: Mapped[str] = mapped_column(String(128), nullable=False)
    method_name: Mapped[str] = mapped_column(String(128), nullable=False)
    metrics_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSON_OBJECT,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=utc_now,
    )
    eval_run: Mapped[EvalRunModel] = relationship(back_populates="results")
