"""Completely offline SQLite CRUD and integrity acceptance for Day 18."""

from __future__ import annotations

from datetime import UTC
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import func, inspect, select
from sqlalchemy.exc import IntegrityError

from app.storage.crud import (
    StorageNotFoundError,
    append_message,
    create_document,
    create_eval_result,
    create_eval_run,
    create_session,
    delete_document,
    delete_eval_run,
    delete_session,
    get_document,
    get_eval_run,
    get_session,
    list_documents,
    list_eval_results,
    list_messages,
    list_sessions,
    update_document_metadata,
    update_eval_run_status,
    update_session,
)
from app.storage.database import SqliteDatabase
from app.storage.models import (
    DocumentModel,
    EvalResultModel,
    EvalRunModel,
    MessageModel,
)
from app.storage.schemas import (
    DocumentCreate,
    DocumentMetadataUpdate,
    EvalResultCreate,
    EvalRunCreate,
    EvalRunStatusUpdate,
    MessageCreate,
    SessionCreate,
    SessionUpdate,
)


@pytest.fixture
def database(tmp_path: Path):
    value = SqliteDatabase(tmp_path / "day18.sqlite")
    value.initialize_schema()
    yield value
    value.close()


def _document(document_id: str = "doc_001", episode_id: str = "ep_001"):
    return DocumentCreate(
        document_id=document_id,
        episode_id=episode_id,
        source_file="synthetic-episode.txt",
        source_format="txt",
        title="Synthetic episode",
        input_summary="Metadata-only synthetic input.",
        metadata={"language": "en", "segment_count": 2},
    )


def _eval_run(run_id: str = "run_001"):
    return EvalRunCreate(
        run_id=run_id,
        evaluation_name="synthetic retrieval check",
        schema_version=1,
        status="pending",
        input_artifact_ref="synthetic://day18/input",
        input_summary="Synthetic metadata only.",
        config_metadata={"k_values": [1, 3]},
    )


def test_schema_initialization_is_idempotent_and_has_five_tables_and_indexes(
    database: SqliteDatabase,
) -> None:
    database.initialize_schema()
    schema = inspect(database.engine)

    assert set(schema.get_table_names()) == {
        "documents",
        "sessions",
        "messages",
        "eval_runs",
        "eval_results",
    }
    assert {item["name"] for item in schema.get_indexes("messages")} == {
        "ix_messages_session_sequence"
    }
    assert {item["name"] for item in schema.get_indexes("eval_results")} == {
        "ix_eval_results_run_case_method"
    }
    assert {
        item["name"] for item in schema.get_check_constraints("documents")
    } >= {"ck_documents_metadata_object"}
    assert {
        item["name"] for item in schema.get_check_constraints("eval_runs")
    } >= {"ck_eval_runs_config_metadata_object"}
    assert {
        item["name"] for item in schema.get_check_constraints("eval_results")
    } >= {"ck_eval_results_metrics_metadata_object"}


@pytest.mark.parametrize(
    ("model_name", "invalid_model"),
    [
        (
            "documents.metadata",
            DocumentModel(
                document_id="doc_invalid_json",
                episode_id="ep_invalid_json",
                source_file="synthetic.txt",
                source_format="txt",
                metadata_json="not-an-object",
            ),
        ),
        (
            "eval_runs.config_metadata",
            EvalRunModel(
                run_id="run_invalid_json",
                evaluation_name="synthetic",
                schema_version=1,
                status="pending",
                config_metadata="not-an-object",
            ),
        ),
        (
            "eval_results.metrics_metadata",
            EvalResultModel(
                result_id="result_invalid_json",
                eval_run_id="run_parent",
                case_id="case_invalid_json",
                method_name="vector",
                metrics_metadata="not-an-object",
            ),
        ),
    ],
)
def test_json_metadata_columns_reject_non_objects_from_direct_orm_writes(
    database: SqliteDatabase,
    model_name: str,
    invalid_model: object,
) -> None:
    if model_name == "eval_results.metrics_metadata":
        with database.transaction() as session:
            create_eval_run(session, _eval_run("run_parent"))

    with pytest.raises(IntegrityError):
        with database.transaction() as session:
            session.add(invalid_model)
            session.flush()


def test_document_create_get_list_update_and_delete(database: SqliteDatabase) -> None:
    with database.transaction() as session:
        created = create_document(session, _document())
        create_document(session, _document("doc_002", "ep_002"))

    with database.transaction() as session:
        updated = update_document_metadata(
            session,
            "doc_001",
            DocumentMetadataUpdate(
                title="Updated synthetic title",
                metadata={"language": "en", "reviewed": True},
            ),
        )
        found = get_document(session, "doc_001")
        documents = list_documents(session)

    assert created.created_at.tzinfo is not None
    assert updated.title == "Updated synthetic title"
    assert updated.metadata == {"language": "en", "reviewed": True}
    assert found == updated
    assert [item.document_id for item in documents] == ["doc_001", "doc_002"]

    with database.transaction() as session:
        assert delete_document(session, "doc_001") is True
        assert delete_document(session, "doc_001") is False
        assert get_document(session, "doc_001") is None


def test_document_unique_episode_failure_rolls_back_whole_transaction(
    database: SqliteDatabase,
) -> None:
    with pytest.raises(IntegrityError):
        with database.transaction() as session:
            create_document(session, _document("doc_first", "ep_same"))
            create_document(session, _document("doc_second", "ep_same"))

    with database.transaction() as session:
        assert list_documents(session) == []


def test_session_messages_are_appended_and_read_in_stable_sequence(
    database: SqliteDatabase,
) -> None:
    with database.transaction() as session:
        create_session(
            session,
            SessionCreate(session_id="session_001", title="Synthetic chat"),
        )
        user = append_message(
            session,
            "session_001",
            MessageCreate(
                message_id="message_001",
                role="user",
                content="What did the synthetic guest say?",
            ),
        )
        assistant = append_message(
            session,
            "session_001",
            MessageCreate(
                message_id="message_002",
                role="assistant",
                content="Only synthetic metadata is stored here.",
            ),
        )

    with database.transaction() as session:
        messages = list_messages(session, "session_001")
        current = get_session(session, "session_001")
        sessions = list_sessions(session)

    assert (user.sequence, assistant.sequence) == (0, 1)
    assert [item.message_id for item in messages] == ["message_001", "message_002"]
    assert current is not None
    assert current.updated_at >= current.created_at
    assert [item.session_id for item in sessions] == ["session_001"]


def test_message_content_with_quotes_round_trips_through_parameterized_orm(
    database: SqliteDatabase,
) -> None:
    content = "Guest said: 'SQLite'; DELETE FROM sessions; -- as plain text"
    with database.transaction() as session:
        create_session(session, SessionCreate(session_id="session_quotes"))
        append_message(
            session,
            "session_quotes",
            MessageCreate(message_id="message_quotes", role="user", content=content),
        )

    with database.transaction() as session:
        assert list_messages(session, "session_quotes")[0].content == content
        assert get_session(session, "session_quotes") is not None


def test_invalid_message_role_is_rejected_by_schema_and_database(
    database: SqliteDatabase,
) -> None:
    with pytest.raises(ValidationError):
        MessageCreate(message_id="message_bad", role="tool", content="invalid")

    with database.transaction() as session:
        create_session(session, SessionCreate(session_id="session_role"))

    with pytest.raises(IntegrityError):
        with database.transaction() as session:
            session.add(
                MessageModel(
                    message_id="message_db_bad",
                    session_id="session_role",
                    role="tool",
                    content="invalid",
                    sequence=0,
                )
            )
            session.flush()


def test_missing_session_cannot_accept_message(database: SqliteDatabase) -> None:
    with pytest.raises(StorageNotFoundError, match="session_missing"):
        with database.transaction() as session:
            append_message(
                session,
                "session_missing",
                MessageCreate(
                    message_id="message_orphan",
                    role="user",
                    content="orphan",
                ),
            )


def test_foreign_keys_are_enabled_and_reject_direct_orphan_write(
    database: SqliteDatabase,
) -> None:
    assert database.foreign_keys_enabled() is True

    with pytest.raises(IntegrityError):
        with database.transaction() as session:
            session.add(
                MessageModel(
                    message_id="message_fk_bad",
                    session_id="session_absent",
                    role="user",
                    content="orphan",
                    sequence=0,
                )
            )
            session.flush()


def test_session_update_and_delete_cascade_messages(database: SqliteDatabase) -> None:
    with database.transaction() as session:
        create_session(session, SessionCreate(session_id="session_delete"))
        append_message(
            session,
            "session_delete",
            MessageCreate(message_id="message_delete", role="user", content="hello"),
        )
        updated = update_session(
            session,
            "session_delete",
            SessionUpdate(title="Closed chat", status="closed"),
        )
        assert updated.status == "closed"
        assert delete_session(session, "session_delete") is True

    with database.transaction() as session:
        assert session.scalar(select(func.count()).select_from(MessageModel)) == 0


def test_eval_run_result_query_and_completed_lifecycle(
    database: SqliteDatabase,
) -> None:
    with database.transaction() as session:
        create_eval_run(session, _eval_run())
        result = create_eval_result(
            session,
            "run_001",
            EvalResultCreate(
                result_id="result_001",
                case_id="case_001",
                method_name="vector",
                metrics_metadata={"hit_at_1": True, "recall_at_3": 1.0},
            ),
        )
        completed = update_eval_run_status(
            session,
            "run_001",
            EvalRunStatusUpdate(status="completed"),
        )

    with database.transaction() as session:
        loaded_run = get_eval_run(session, "run_001")
        results = list_eval_results(session, "run_001")

    assert result.eval_run_id == "run_001"
    assert completed.completed_at is not None
    assert completed.completed_at.tzinfo == UTC
    assert loaded_run == completed
    assert results == [result]


def test_missing_eval_run_cannot_accept_result(database: SqliteDatabase) -> None:
    with pytest.raises(StorageNotFoundError, match="run_missing"):
        with database.transaction() as session:
            create_eval_result(
                session,
                "run_missing",
                EvalResultCreate(
                    result_id="result_orphan",
                    case_id="case_orphan",
                    method_name="vector",
                ),
            )


def test_eval_result_business_identity_is_unique(database: SqliteDatabase) -> None:
    with database.transaction() as session:
        create_eval_run(session, _eval_run("run_unique"))
        create_eval_result(
            session,
            "run_unique",
            EvalResultCreate(
                result_id="result_unique_1",
                case_id="case_same",
                method_name="vector",
            ),
        )

    with pytest.raises(IntegrityError):
        with database.transaction() as session:
            create_eval_result(
                session,
                "run_unique",
                EvalResultCreate(
                    result_id="result_unique_2",
                    case_id="case_same",
                    method_name="vector",
                ),
            )


def test_eval_run_delete_cascades_results(database: SqliteDatabase) -> None:
    with database.transaction() as session:
        create_eval_run(session, _eval_run("run_delete"))
        create_eval_result(
            session,
            "run_delete",
            EvalResultCreate(
                result_id="result_delete",
                case_id="case_delete",
                method_name="bm25",
            ),
        )
        assert delete_eval_run(session, "run_delete") is True

    with database.transaction() as session:
        assert session.scalar(select(func.count()).select_from(EvalResultModel)) == 0


def test_sqlite_stores_utc_as_iso_text_and_returns_aware_datetime(
    database: SqliteDatabase,
) -> None:
    with database.transaction() as session:
        created = create_document(session, _document("doc_utc", "ep_utc"))

    with database.engine.connect() as connection:
        raw = connection.exec_driver_sql(
            "SELECT created_at FROM documents WHERE document_id = ?",
            ("doc_utc",),
        ).scalar_one()

    assert raw.endswith("Z")
    assert created.created_at.tzinfo == UTC
    assert created.created_at.utcoffset().total_seconds() == 0


def test_multiple_database_files_are_isolated(tmp_path: Path) -> None:
    first = SqliteDatabase(tmp_path / "first.sqlite")
    second = SqliteDatabase(tmp_path / "second.sqlite")
    try:
        first.initialize_schema()
        second.initialize_schema()
        with first.transaction() as session:
            create_document(session, _document("doc_isolated", "ep_isolated"))
        with second.transaction() as session:
            assert list_documents(session) == []
    finally:
        first.close()
        second.close()


def test_complete_real_sqlite_crud_smoke_flow(tmp_path: Path) -> None:
    target = tmp_path / "smoke.sqlite3"
    with SqliteDatabase(target) as database:
        database.initialize_schema()
        with database.transaction() as session:
            create_document(session, _document("doc_smoke", "ep_smoke"))
            update_document_metadata(
                session,
                "doc_smoke",
                DocumentMetadataUpdate(metadata={"smoke": True}),
            )
            create_session(session, SessionCreate(session_id="session_smoke"))
            append_message(
                session,
                "session_smoke",
                MessageCreate(message_id="message_smoke_1", role="user", content="Hi"),
            )
            append_message(
                session,
                "session_smoke",
                MessageCreate(
                    message_id="message_smoke_2",
                    role="assistant",
                    content="Hello",
                ),
            )
            create_eval_run(session, _eval_run("run_smoke"))
            create_eval_result(
                session,
                "run_smoke",
                EvalResultCreate(
                    result_id="result_smoke",
                    case_id="case_smoke",
                    method_name="hybrid_rrf",
                    metrics_metadata={"hit_at_5": True},
                ),
            )

        with database.transaction() as session:
            assert get_document(session, "doc_smoke").metadata == {"smoke": True}
            sequences = [
                item.sequence for item in list_messages(session, "session_smoke")
            ]
            assert sequences == [0, 1]
            assert len(list_eval_results(session, "run_smoke")) == 1
            delete_document(session, "doc_smoke")
            delete_session(session, "session_smoke")
            delete_eval_run(session, "run_smoke")

        with database.transaction() as session:
            assert list_documents(session) == []
            assert session.scalar(select(func.count()).select_from(MessageModel)) == 0
            result_count = session.scalar(
                select(func.count()).select_from(EvalResultModel)
            )
            assert result_count == 0
        assert database.foreign_keys_enabled() is True

    assert target.exists()
