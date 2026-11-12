"""Offline Day 19 configuration, service, restart, and PostgreSQL design tests."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier

import pytest
from sqlalchemy import func, select
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.schema import CreateTable
from sqlalchemy.exc import OperationalError

from app.core.config import PROJECT_ROOT, ConfigError
from app.storage.config import (
    DEFAULT_DATABASE_URL,
    DatabaseSettings,
)
from app.schemas.evaluation_result import (
    CaseMetricsAtK,
    EvaluationCaseResult,
    FailureReview,
    IndexConfiguration,
    InputArtifactHashes,
    MetricSummaryAtK,
    RetrievalBaselineResult,
    SavedCandidate,
    VectorRetrieverConfiguration,
)
from app.schemas.transcript import ProcessedTranscript, TranscriptSegment
from app.storage.database import create_database
from app.storage.models import Base, EvalResultModel, MessageModel, UTCDateTime
from app.storage.schemas import (
    EvalResultCreate,
    EvalRunCreate,
    MessageCreate,
    SessionCreate,
    SessionUpdate,
)
from app.storage.services import (
    ConversationPersistenceService,
    DocumentPersistenceService,
    EvaluationPersistenceService,
    StorageConflictError,
    StorageLifecycleError,
)


EPISODE_ID = "ep_1111111111111111"


def _settings(path: Path) -> DatabaseSettings:
    return DatabaseSettings(database_url=f"sqlite+pysqlite:///{path}")


def _transcript() -> ProcessedTranscript:
    source_file = "/private/input/synthetic.json"
    return ProcessedTranscript(
        episode_id=EPISODE_ID,
        source_file=source_file,
        source_format="json",
        segments=[
            TranscriptSegment(
                segment_id=f"{EPISODE_ID}:000000",
                segment_index=0,
                text="FULL_TRANSCRIPT secret-key SYSTEM PROMPT raw_response",
                source_file=source_file,
                speaker="Guest",
                section_title="Synthetic section",
                start_time=1.0,
                end_time=2.0,
            ),
            TranscriptSegment(
                segment_id=f"{EPISODE_ID}:000001",
                segment_index=1,
                text="Second private transcript segment.",
                source_file=source_file,
                speaker="Host",
                section_title="Synthetic section",
            ),
        ],
    )


def test_legacy_core_database_settings_export_is_lazy_compatible() -> None:
    from app.core import config as core_config

    assert core_config.DatabaseSettings is DatabaseSettings
    assert core_config.DEFAULT_DATABASE_URL == DEFAULT_DATABASE_URL


def _baseline() -> RetrievalBaselineResult:
    case_metric = CaseMetricsAtK(
        scored=True,
        hit_at_k=True,
        recall_at_k=1.0,
        correct_segment_count=1,
        expected_segment_count=1,
    )
    summary = MetricSummaryAtK(
        hit_count=1,
        hit_denominator=1,
        hit_at_k=1.0,
        macro_recall_sum=1.0,
        macro_recall_denominator=1,
        macro_recall_at_k=1.0,
    )
    case = EvaluationCaseResult(
        case_id="synthetic_case",
        category="fact",
        answerable=True,
        scored=True,
        expected_source_segment_ids=(f"{EPISODE_ID}:000000",),
        retrieved_candidates=(
            SavedCandidate(
                rank=1,
                chunk_id="chunk_1111111111111111",
                source_segment_ids=(f"{EPISODE_ID}:000000",),
                score=0.9,
                score_semantics="cosine_similarity",
            ),
        ),
        metrics_by_k={1: case_metric},
        elapsed_seconds=0.01,
    )
    return RetrievalBaselineResult(
        schema_version=1,
        evaluation_id="day19-synthetic-baseline",
        created_at_utc=datetime(2026, 8, 13, tzinfo=UTC),
        dataset_id="eval_dataset_v1",
        dataset_cases_sha256="1" * 64,
        episode_id=EPISODE_ID,
        retriever_name="vector",
        retriever_config=VectorRetrieverConfiguration(
            adapter="vector_retriever_adapter_v1",
            underlying_retriever="day8_vector_retriever",
            score_semantics="cosine_similarity",
            query_strategy="one_top_8_call_then_prefixes",
            equal_score_tiebreaker="faiss_row_index_ascending",
        ),
        index_config=IndexConfiguration(
            index_type="faiss_index_flat_ip",
            distance_metric="cosine",
            vector_count=1,
            vector_dimension=8,
            chunking_algorithm="fixed_character_windows_v1",
            chunk_size=1000,
            overlap=200,
        ),
        input_artifact_hashes=InputArtifactHashes(
            dataset_file_sha256="2" * 64,
            manifest_file_sha256="3" * 64,
            faiss_file_sha256="4" * 64,
        ),
        embedding_model="synthetic-model",
        k_values=(1,),
        answerable_case_count=1,
        unanswerable_case_count=0,
        query_count=1,
        query_embedding_calls=0,
        document_embedding_calls=0,
        answer_model_calls=0,
        elapsed_seconds=0.01,
        aggregate_metrics={1: summary},
        category_metrics={"fact": {1: summary}},
        cases=(case,),
        failure_reviews=(
            FailureReview(
                case_id="synthetic_case",
                review_type="dangerous_distractor",
                expected_source_segment_ids=(f"{EPISODE_ID}:000000",),
                correct_candidate_ranks={"vector": (1,)},
                retrieved_expected_segment_ids=(f"{EPISODE_ID}:000000",),
                missing_expected_segment_ids=(),
                diagnosis="distractor_risk",
                possible_factors=("none_observed",),
                summary="Synthetic metadata-only review.",
                next_step="Keep the stable regression case.",
            ),
        ),
    )


def test_database_settings_default_and_explicit_environment_are_isolated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "sqlite+pysqlite:///should-not-be-read.sqlite3")
    monkeypatch.chdir(tmp_path)

    settings = DatabaseSettings.from_env(env_file=None, environ={})

    assert settings.database_url == DEFAULT_DATABASE_URL
    assert settings.sqlite_path == (
        PROJECT_ROOT / "data/podcast-ai-assistant.sqlite3"
    ).resolve()


def test_database_settings_loads_relative_env_file_from_project_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    captured: list[Path] = []

    def fake_load_dotenv(path: Path, *, override: bool) -> bool:
        captured.append(path)
        return True

    monkeypatch.setattr("app.storage.config.load_dotenv", fake_load_dotenv)

    settings = DatabaseSettings.from_env()

    assert captured == [(PROJECT_ROOT / ".env").resolve()]
    assert settings.database_url == DEFAULT_DATABASE_URL


def test_database_settings_preserves_explicit_absolute_env_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_path = tmp_path / "custom.env"
    captured: list[Path] = []

    def fake_load_dotenv(path: Path, *, override: bool) -> bool:
        captured.append(path)
        return True

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr("app.storage.config.load_dotenv", fake_load_dotenv)

    DatabaseSettings.from_env(env_file=env_path)

    assert captured == [env_path.resolve()]


@pytest.mark.parametrize(
    ("url", "dialect"),
    [
        ("sqlite+pysqlite:///:memory:", "sqlite"),
        ("sqlite+pysqlite:///work/day19.sqlite3", "sqlite"),
        ("postgresql+psycopg://user:password@host:5432/app", "postgresql"),
    ],
)
def test_supported_database_urls_are_parsed_by_sqlalchemy(
    url: str,
    dialect: str,
) -> None:
    assert DatabaseSettings(database_url=url).dialect_name == dialect


@pytest.mark.parametrize(
    "url",
    [
        "sqlite:///work/implicit-driver.sqlite3",
        "postgresql://user:password@host/app",
        "mysql+pymysql://user:password@host/app",
        "sqlite+pysqlite://",
        "postgresql+psycopg://user:password@host:invalid/app",
        "postgresql+psycopg://user:password@host:70000/app",
    ],
)
def test_unsupported_or_incomplete_database_urls_are_rejected(url: str) -> None:
    with pytest.raises(ConfigError) as caught:
        DatabaseSettings(database_url=url)
    assert "password" not in str(caught.value)


def test_postgres_password_is_redacted_from_display_repr_and_errors() -> None:
    secret = "do-not-leak"
    settings = DatabaseSettings(
        database_url=f"postgresql+psycopg://user:{secret}@host:5432/app"
    )

    assert secret not in settings.safe_url
    assert secret not in repr(settings)
    assert "***" in settings.safe_url

    with pytest.raises(ConfigError) as caught:
        DatabaseSettings(
            database_url=f"postgresql+psycopg://user:{secret}@host:5432"
        )
    assert secret not in str(caught.value)


def test_sqlite_directory_is_created_only_when_schema_is_initialized(
    tmp_path: Path,
) -> None:
    target = tmp_path / "new-parent" / "day19.sqlite3"
    database = create_database(_settings(target))
    try:
        assert not target.parent.exists()
        assert not target.exists()
        database.initialize_schema()
        assert target.exists()
    finally:
        database.close()


def test_postgresql_ddl_compiles_all_tables_without_sqlite_expressions() -> None:
    dialect = postgresql.dialect()
    ddls = {
        table.name: str(CreateTable(table).compile(dialect=dialect))
        for table in Base.metadata.sorted_tables
    }
    combined = "\n".join(ddls.values()).lower()

    assert set(ddls) == {
        "documents",
        "sessions",
        "messages",
        "eval_runs",
        "eval_results",
    }
    assert "json_type" not in combined
    assert "pragma" not in combined
    assert "sqlite" not in combined
    assert "jsonb_typeof" in combined
    assert combined.count("jsonb") >= 3
    assert "timestamp with time zone" in combined


def test_sqlite_ddl_retains_json_object_checks_and_utc_text() -> None:
    dialect = sqlite.dialect()
    combined = "\n".join(
        str(CreateTable(table).compile(dialect=dialect))
        for table in Base.metadata.sorted_tables
    ).lower()

    assert combined.count("json_type") == 3
    assert "jsonb_typeof" not in combined
    assert "timestamp with time zone" not in combined
    assert "varchar(27)" in combined


def test_utc_type_rejects_naive_values_for_both_dialects() -> None:
    value = UTCDateTime()
    naive = datetime(2026, 8, 13)

    with pytest.raises(ValueError, match="时区"):
        value.process_bind_param(naive, sqlite.dialect())
    with pytest.raises(ValueError, match="时区"):
        value.process_bind_param(naive, postgresql.dialect())


def test_document_service_maps_metadata_without_transcript_text_and_rejects_duplicate(
    tmp_path: Path,
) -> None:
    database = create_database(_settings(tmp_path / "document.sqlite3"))
    database.initialize_schema()
    service = DocumentPersistenceService(database)
    try:
        document = service.save_processed_transcript(
            _transcript(),
            title="Synthetic title",
            language="en",
            input_summary="Safe structural summary.",
        )

        assert document.document_id == f"document:{EPISODE_ID}"
        assert document.source_file == "synthetic.json"
        assert document.metadata == {
            "segment_count": 2,
            "timed_segment_count": 1,
            "section_count": 1,
            "language": "en",
        }
        assert "FULL_TRANSCRIPT" not in document.model_dump_json()
        with pytest.raises(ValueError, match="segment 正文"):
            service.save_processed_transcript(
                _transcript().model_copy(
                    update={"episode_id": "ep_2222222222222222"}
                ),
                input_summary=_transcript().segments[0].text,
            )
        with pytest.raises(StorageConflictError, match="document 已存在"):
            service.save_processed_transcript(_transcript())
    finally:
        database.close()


def test_conversation_lifecycle_stable_order_and_terminal_closed_state(
    tmp_path: Path,
) -> None:
    database = create_database(_settings(tmp_path / "conversation.sqlite3"))
    database.initialize_schema()
    service = ConversationPersistenceService(database)
    try:
        service.create(SessionCreate(session_id="session_day19", title="Synthetic"))
        first = service.append(
            "session_day19",
            MessageCreate(message_id="message_user", role="user", content="Hello"),
        )
        second = service.append(
            "session_day19",
            MessageCreate(
                message_id="message_assistant",
                role="assistant",
                content="Synthetic reply",
            ),
        )
        service.update("session_day19", SessionUpdate(status="closed"))

        assert (first.sequence, second.sequence) == (0, 1)
        assert [item.message_id for item in service.messages("session_day19")] == [
            "message_user",
            "message_assistant",
        ]
        with pytest.raises(StorageLifecycleError, match="不能追加"):
            service.append(
                "session_day19",
                MessageCreate(message_id="message_late", role="user", content="late"),
            )
        with pytest.raises(StorageLifecycleError, match="不能重新"):
            service.update("session_day19", SessionUpdate(status="active"))
    finally:
        database.close()


def test_concurrent_sqlite_message_appends_are_serialized(tmp_path: Path) -> None:
    database = create_database(_settings(tmp_path / "concurrent.sqlite3"))
    database.initialize_schema()
    service = ConversationPersistenceService(database)
    service.create(SessionCreate(session_id="session_concurrent"))
    barrier = Barrier(2)

    def append(number: int) -> int:
        barrier.wait()
        return service.append(
            "session_concurrent",
            MessageCreate(
                message_id=f"message_concurrent_{number}",
                role="user",
                content=f"message {number}",
            ),
        ).sequence

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            sequences = sorted(executor.map(append, (1, 2)))
        assert sequences == [0, 1]
        assert [item.sequence for item in service.messages("session_concurrent")] == [
            0,
            1,
        ]
    finally:
        database.close()


def test_eval_run_results_are_atomic_and_failure_rolls_back(tmp_path: Path) -> None:
    database = create_database(_settings(tmp_path / "eval-atomic.sqlite3"))
    database.initialize_schema()
    service = EvaluationPersistenceService(database)
    run = EvalRunCreate(
        run_id="run_atomic",
        evaluation_name="synthetic",
        schema_version=1,
    )
    duplicate_results = [
        EvalResultCreate(
            result_id="result_1",
            case_id="case_same",
            method_name="vector",
        ),
        EvalResultCreate(
            result_id="result_2",
            case_id="case_same",
            method_name="vector",
        ),
    ]
    try:
        with pytest.raises(StorageConflictError):
            service.save_run(run, duplicate_results)
        assert service.get("run_atomic") is None
        with database.transaction() as session:
            assert session.scalar(select(func.count()).select_from(EvalResultModel)) == 0
    finally:
        database.close()


@pytest.mark.parametrize("service_name", ["document", "conversation", "evaluation"])
def test_sqlite_lock_errors_are_mapped_to_storage_conflicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    service_name: str,
) -> None:
    database = create_database(_settings(tmp_path / f"{service_name}-locked.sqlite3"))
    database.initialize_schema()
    conversation_service = ConversationPersistenceService(database)
    if service_name == "conversation":
        conversation_service.create(SessionCreate(session_id="session_locked"))
    original = sqlite3.OperationalError("database is locked")
    original.sqlite_errorcode = sqlite3.SQLITE_BUSY
    original.sqlite_errorname = "SQLITE_BUSY"
    locked = OperationalError("BEGIN IMMEDIATE", {}, original)

    def fail_transaction(*args: object, **kwargs: object) -> object:
        raise locked

    monkeypatch.setattr(database, "transaction", fail_transaction)

    try:
        with pytest.raises(StorageConflictError, match="SQLite 并发写锁冲突"):
            if service_name == "document":
                DocumentPersistenceService(database).save_processed_transcript(
                    _transcript()
                )
            elif service_name == "conversation":
                conversation_service.append(
                    "session_locked",
                    MessageCreate(
                        message_id="message_locked",
                        role="user",
                        content="locked",
                    ),
                )
            else:
                EvaluationPersistenceService(database).save_run(
                    EvalRunCreate(
                        run_id="run_locked",
                        evaluation_name="synthetic",
                        schema_version=1,
                    ),
                    [],
                )
    finally:
        database.close()


def test_unrelated_operational_errors_are_not_hidden_as_conflicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = create_database(_settings(tmp_path / "operational-error.sqlite3"))
    database.initialize_schema()
    original = sqlite3.OperationalError("disk I/O error")
    original.sqlite_errorcode = sqlite3.SQLITE_IOERR
    original.sqlite_errorname = "SQLITE_IOERR"
    io_error = OperationalError("INSERT", {}, original)

    def fail_transaction(*args: object, **kwargs: object) -> object:
        raise io_error

    monkeypatch.setattr(database, "transaction", fail_transaction)

    try:
        with pytest.raises(OperationalError) as caught:
            EvaluationPersistenceService(database).save_run(
                EvalRunCreate(
                    run_id="run_io_error",
                    evaluation_name="synthetic",
                    schema_version=1,
                ),
                [],
            )
        assert caught.value is io_error
    finally:
        database.close()


def test_eval_status_transition_and_completed_at_rules(tmp_path: Path) -> None:
    database = create_database(_settings(tmp_path / "eval-status.sqlite3"))
    database.initialize_schema()
    service = EvaluationPersistenceService(database)
    try:
        service.save_run(
            EvalRunCreate(
                run_id="run_status",
                evaluation_name="synthetic",
                schema_version=1,
            ),
            [],
        )
        running = service.update_status("run_status", "running")
        completed = service.update_status("run_status", "completed")
        repeated = service.update_status("run_status", "completed")

        assert running.completed_at is None
        assert completed.completed_at is not None
        assert repeated.completed_at == completed.completed_at
        with pytest.raises(StorageLifecycleError, match="completed -> pending"):
            service.update_status("run_status", "pending")
    finally:
        database.close()


def test_business_restart_smoke_persists_reads_and_cascades(tmp_path: Path) -> None:
    target = tmp_path / "restart.sqlite3"
    settings = _settings(target)
    first = create_database(settings)
    first.initialize_schema()
    DocumentPersistenceService(first).save_processed_transcript(
        _transcript(),
        language="en",
        input_summary="Safe structural summary.",
    )
    conversations = ConversationPersistenceService(first)
    conversations.create(SessionCreate(session_id="session_restart"))
    conversations.append(
        "session_restart",
        MessageCreate(message_id="message_restart_1", role="user", content="Hello"),
    )
    conversations.append(
        "session_restart",
        MessageCreate(
            message_id="message_restart_2",
            role="assistant",
            content="Offline answer",
        ),
    )
    EvaluationPersistenceService(first).save_retrieval_baseline(
        _baseline(),
        input_artifact_ref="synthetic://day19/baseline",
    )
    first.close()

    second = create_database(settings)
    try:
        document_service = DocumentPersistenceService(second)
        conversation_service = ConversationPersistenceService(second)
        evaluation_service = EvaluationPersistenceService(second)

        document = document_service.get(f"document:{EPISODE_ID}")
        messages = conversation_service.messages("session_restart")
        evaluation = evaluation_service.get("day19-synthetic-baseline")

        assert document is not None and document.metadata["segment_count"] == 2
        assert [item.sequence for item in messages] == [0, 1]
        assert evaluation is not None and len(evaluation.results) == 1
        assert evaluation.run.status == "completed"
        assert evaluation.run.completed_at is not None
        config = evaluation.run.config_metadata
        assert config["retriever_config"] == _baseline().retriever_config.model_dump(
            mode="json"
        )
        assert config["index_config"] == _baseline().index_config.model_dump(
            mode="json"
        )
        assert config["category_metrics"] == {
            "fact": {
                "1": _baseline()
                .category_metrics["fact"][1]
                .model_dump(mode="json")
            }
        }
        assert config["failure_reviews"] == [
            _baseline().failure_reviews[0].model_dump(mode="json")
        ]

        with second.engine.connect() as connection:
            metadata_dump = " ".join(
                str(value)
                for value in (
                    connection.exec_driver_sql(
                        "SELECT metadata FROM documents"
                    ).scalar_one(),
                    connection.exec_driver_sql(
                        "SELECT config_metadata FROM eval_runs"
                    ).scalar_one(),
                    connection.exec_driver_sql(
                        "SELECT metrics_metadata FROM eval_results"
                    ).scalar_one(),
                )
            )
        for forbidden in (
            "FULL_TRANSCRIPT",
            "private transcript",
            "SYSTEM PROMPT",
            "secret-key",
            "raw_response",
            "chunk_1111111111111111",
        ):
            assert forbidden not in metadata_dump

        assert document_service.delete(f"document:{EPISODE_ID}") is True
        assert conversation_service.delete("session_restart") is True
        assert evaluation_service.delete("day19-synthetic-baseline") is True
        with second.transaction() as session:
            assert session.scalar(select(func.count()).select_from(MessageModel)) == 0
            assert session.scalar(select(func.count()).select_from(EvalResultModel)) == 0
    finally:
        second.close()
