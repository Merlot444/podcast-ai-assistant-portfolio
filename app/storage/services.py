"""Business mappings, lifecycle rules, and transaction boundaries for Day 19."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy.exc import IntegrityError, OperationalError

from app.schemas.evaluation_result import RetrievalBaselineResult
from app.schemas.transcript import ProcessedTranscript
from app.storage.crud import StorageNotFoundError
from app.storage.database import Database
from app.storage.repository import StorageRepository
from app.storage.schemas import (
    DocumentCreate,
    DocumentRead,
    EvalResultCreate,
    EvalResultRead,
    EvalRunCreate,
    EvalRunRead,
    EvalRunStatus,
    EvalRunStatusUpdate,
    MessageCreate,
    MessageRead,
    SessionCreate,
    SessionRead,
    SessionUpdate,
)


class StorageConflictError(RuntimeError):
    """Raised for a duplicate identity or concurrent integrity conflict."""


class StorageLifecycleError(RuntimeError):
    """Raised when a business state transition is not allowed."""


@dataclass(frozen=True)
class StoredEvaluation:
    run: EvalRunRead
    results: tuple[EvalResultRead, ...]


class DocumentPersistenceService:
    def __init__(self, database: Database) -> None:
        self.database = database

    def save_processed_transcript(
        self,
        transcript: ProcessedTranscript,
        *,
        title: str | None = None,
        language: str | None = None,
        input_summary: str | None = None,
    ) -> DocumentRead:
        """Save only source and structural metadata, never segment text."""
        source_file = Path(transcript.source_file).name
        metadata: dict[str, Any] = {
            "segment_count": len(transcript.segments),
            "timed_segment_count": sum(
                segment.start_time is not None for segment in transcript.segments
            ),
            "section_count": len(
                {
                    segment.section_title
                    for segment in transcript.segments
                    if segment.section_title is not None
                }
            ),
        }
        if language is not None:
            cleaned_language = language.strip()
            if cleaned_language:
                metadata["language"] = cleaned_language
        payload = DocumentCreate(
            document_id=f"document:{transcript.episode_id}",
            episode_id=transcript.episode_id,
            source_file=source_file,
            source_format=transcript.source_format,
            title=title,
            input_summary=_safe_transcript_summary(transcript, input_summary),
            metadata=metadata,
        )
        try:
            with self.database.transaction() as session:
                repository = StorageRepository(session)
                if repository.get_document(payload.document_id) is not None:
                    raise StorageConflictError(
                        f"document 已存在：{payload.document_id}"
                    )
                existing = repository.get_document_by_episode(payload.episode_id)
                if existing is not None:
                    raise StorageConflictError(
                        f"episode 已绑定 document：{payload.episode_id}"
                    )
                return repository.create_document(payload)
        except IntegrityError as error:
            raise StorageConflictError("document 身份发生并发冲突") from error
        except OperationalError as error:
            _raise_if_sqlite_lock_conflict(
                self.database,
                error,
                message="document 保存遇到 SQLite 并发写锁冲突",
            )
            raise

    def get(self, document_id: str) -> DocumentRead | None:
        with self.database.transaction() as session:
            return StorageRepository(session).get_document(document_id)

    def delete(self, document_id: str) -> bool:
        with self.database.transaction() as session:
            return StorageRepository(session).delete_document(document_id)


class ConversationPersistenceService:
    def __init__(self, database: Database) -> None:
        self.database = database

    def create(self, payload: SessionCreate) -> SessionRead:
        try:
            with self.database.transaction() as session:
                return StorageRepository(session).create_session(payload)
        except IntegrityError as error:
            raise StorageConflictError(f"session 已存在：{payload.session_id}") from error

    def get(self, session_id: str) -> SessionRead | None:
        with self.database.transaction() as session:
            return StorageRepository(session).get_session(session_id)

    def list(self) -> list[SessionRead]:
        with self.database.transaction() as session:
            return StorageRepository(session).list_sessions()

    def update(self, session_id: str, payload: SessionUpdate) -> SessionRead:
        with self.database.transaction() as session:
            repository = StorageRepository(session)
            current = repository.get_session(session_id)
            if current is None:
                raise StorageNotFoundError(f"找不到 session：{session_id}")
            if (
                current.status == "closed"
                and payload.status is not None
                and payload.status != "closed"
            ):
                raise StorageLifecycleError("closed session 不能重新变为 active")
            return repository.update_session(session_id, payload)

    def append(self, session_id: str, payload: MessageCreate) -> MessageRead:
        try:
            with self.database.transaction(sqlite_immediate=True) as session:
                repository = StorageRepository(session)
                current = repository.lock_session_for_append(session_id)
                if current.status != "active":
                    raise StorageLifecycleError("closed session 不能追加消息")
                return repository.append_message(session_id, payload)
        except IntegrityError as error:
            raise StorageConflictError("消息序号或数据库写锁发生并发冲突") from error
        except OperationalError as error:
            _raise_if_sqlite_lock_conflict(
                self.database,
                error,
                message="消息追加遇到 SQLite 并发写锁冲突",
            )
            raise

    def messages(self, session_id: str) -> list[MessageRead]:
        with self.database.transaction() as session:
            return StorageRepository(session).list_messages(session_id)

    def delete(self, session_id: str) -> bool:
        with self.database.transaction() as session:
            return StorageRepository(session).delete_session(session_id)


class EvaluationPersistenceService:
    _ALLOWED_TRANSITIONS: dict[EvalRunStatus, frozenset[EvalRunStatus]] = {
        "pending": frozenset({"running", "completed", "failed"}),
        "running": frozenset({"completed", "failed"}),
        "completed": frozenset(),
        "failed": frozenset(),
    }

    def __init__(self, database: Database) -> None:
        self.database = database

    def save_run(
        self,
        run: EvalRunCreate,
        results: Sequence[EvalResultCreate],
    ) -> StoredEvaluation:
        """Persist one run and every result as a single atomic business unit."""
        try:
            with self.database.transaction() as session:
                repository = StorageRepository(session)
                stored_run = repository.create_eval_run(run)
                stored_results = tuple(
                    repository.create_eval_result(run.run_id, result)
                    for result in results
                )
                return StoredEvaluation(run=stored_run, results=stored_results)
        except IntegrityError as error:
            raise StorageConflictError("eval run/result 身份或完整性冲突") from error
        except OperationalError as error:
            _raise_if_sqlite_lock_conflict(
                self.database,
                error,
                message="eval run/results 保存遇到 SQLite 并发写锁冲突",
            )
            raise

    def save_retrieval_baseline(
        self,
        baseline: RetrievalBaselineResult,
        *,
        input_artifact_ref: str | None = None,
    ) -> StoredEvaluation:
        """Adapt the supported RetrievalBaselineResult without questions or Chunks."""
        run = EvalRunCreate(
            run_id=baseline.evaluation_id,
            evaluation_name="retrieval_baseline_result_v1",
            schema_version=baseline.schema_version,
            status="completed",
            input_artifact_ref=input_artifact_ref,
            input_summary=(
                f"{baseline.query_count} cases; "
                f"{baseline.answerable_case_count} answerable"
            ),
            config_metadata={
                "adapter": "retrieval_baseline_result_v1",
                "dataset_id": baseline.dataset_id,
                "dataset_cases_sha256": baseline.dataset_cases_sha256,
                "episode_id": baseline.episode_id,
                "retriever_name": baseline.retriever_name,
                "retriever_config": baseline.retriever_config.model_dump(mode="json"),
                "index_config": baseline.index_config.model_dump(mode="json"),
                "embedding_model": baseline.embedding_model,
                "k_values": list(baseline.k_values),
                "query_embedding_calls": baseline.query_embedding_calls,
                "document_embedding_calls": baseline.document_embedding_calls,
                "answer_model_calls": baseline.answer_model_calls,
                "input_artifact_hashes": baseline.input_artifact_hashes.model_dump(
                    mode="json"
                ),
                "aggregate_metrics": {
                    str(k): value.model_dump(mode="json")
                    for k, value in baseline.aggregate_metrics.items()
                },
                "category_metrics": {
                    category: {
                        str(k): value.model_dump(mode="json")
                        for k, value in metrics.items()
                    }
                    for category, metrics in baseline.category_metrics.items()
                },
                "failure_reviews": [
                    review.model_dump(mode="json")
                    for review in baseline.failure_reviews
                ],
            },
        )
        results = [
            EvalResultCreate(
                result_id=_eval_result_id(
                    baseline.evaluation_id,
                    case.case_id,
                    baseline.retriever_name,
                ),
                case_id=case.case_id,
                method_name=baseline.retriever_name,
                metrics_metadata={
                    "category": case.category,
                    "answerable": case.answerable,
                    "scored": case.scored,
                    "expected_source_segment_ids": list(
                        case.expected_source_segment_ids
                    ),
                    "metrics_by_k": {
                        str(k): value.model_dump(mode="json")
                        for k, value in case.metrics_by_k.items()
                    },
                    "elapsed_seconds": case.elapsed_seconds,
                },
            )
            for case in baseline.cases
        ]
        return self.save_run(run, results)

    def get(self, run_id: str) -> StoredEvaluation | None:
        with self.database.transaction() as session:
            repository = StorageRepository(session)
            run = repository.get_eval_run(run_id)
            if run is None:
                return None
            return StoredEvaluation(
                run=run,
                results=tuple(repository.list_eval_results(run_id)),
            )

    def update_status(self, run_id: str, status: EvalRunStatus) -> EvalRunRead:
        with self.database.transaction() as session:
            repository = StorageRepository(session)
            current = repository.get_eval_run(run_id)
            if current is None:
                raise StorageNotFoundError(f"找不到 eval run：{run_id}")
            if status == current.status:
                return current
            if status not in self._ALLOWED_TRANSITIONS[current.status]:
                raise StorageLifecycleError(
                    f"不允许 eval run 状态转换：{current.status} -> {status}"
                )
            return repository.update_eval_run_status(
                run_id,
                EvalRunStatusUpdate(status=status),
            )

    def delete(self, run_id: str) -> bool:
        with self.database.transaction() as session:
            return StorageRepository(session).delete_eval_run(run_id)


def _eval_result_id(run_id: str, case_id: str, method_name: str) -> str:
    digest = sha256(f"{run_id}\0{case_id}\0{method_name}".encode()).hexdigest()[:32]
    return f"eval_result:{digest}"


def _raise_if_sqlite_lock_conflict(
    database: Database,
    error: OperationalError,
    *,
    message: str,
) -> None:
    """Map only SQLite BUSY/LOCKED errors; preserve unrelated operational faults."""
    code = getattr(error.orig, "sqlite_errorcode", None)
    is_lock_error = (
        database.settings.dialect_name == "sqlite"
        and isinstance(code, int)
        and code & 0xFF in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}
    )
    if is_lock_error:
        raise StorageConflictError(message) from error


def _safe_transcript_summary(
    transcript: ProcessedTranscript,
    input_summary: str | None,
) -> str | None:
    if input_summary is None:
        return None
    cleaned = input_summary.strip()
    complete_text = "\n".join(segment.text for segment in transcript.segments)
    if complete_text and complete_text in cleaned:
        raise ValueError("input_summary 不能包含完整转写正文")
    if any(
        cleaned == segment.text.strip()
        or (len(segment.text) >= 40 and segment.text in cleaned)
        for segment in transcript.segments
    ):
        raise ValueError("input_summary 不能包含完整 segment 正文")
    return input_summary
