from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import OperationalError

from app.storage.database import SqliteDatabase
from app.text2sql import (
    DAY20_CATALOG,
    GeneratedQuery,
    MappingQueryGenerator,
    OpenAICompatibleQueryGenerator,
    QueryExecutionError,
    QueryGenerationError,
    QueryTimeoutError,
    QueryValidationError,
    QueryValidator,
    Text2SQLRequest,
    Text2SQLService,
    ValidatedQuery,
    ValidationPolicy,
    build_postgres_read_only_plan,
    create_sqlite_text2sql_service,
)
from app.text2sql.cli import main as cli_main
from app.text2sql.schemas import new_approval_token


NORMAL_QUERIES = {
    "当前有多少文档？": GeneratedQuery(
        sql="SELECT COUNT(*) AS document_count FROM documents",
        parameters={},
    ),
    "active 和 closed 会话分别有多少？": GeneratedQuery(
        sql=(
            "SELECT status, COUNT(*) AS session_count FROM sessions "
            "GROUP BY status ORDER BY status"
        ),
        parameters={},
    ),
    "最近创建的 5 个会话是什么？": GeneratedQuery(
        sql=(
            "SELECT session_id, created_at FROM sessions "
            "ORDER BY created_at DESC LIMIT 5"
        ),
        parameters={},
    ),
    "completed 的评测运行有多少？": GeneratedQuery(
        sql=(
            "SELECT COUNT(*) AS completed_count FROM eval_runs "
            "WHERE status = :status"
        ),
        parameters={"status": "completed"},
    ),
    "每种评测方法分别有多少结果？": GeneratedQuery(
        sql=(
            "SELECT method_name, COUNT(*) AS result_count FROM eval_results "
            "GROUP BY method_name ORDER BY method_name"
        ),
        parameters={},
    ),
}

EXPECTED_RESULTS = {
    "当前有多少文档？": ((3,),),
    "active 和 closed 会话分别有多少？": (("active", 3), ("closed", 3)),
    "最近创建的 5 个会话是什么？": (
        ("session-6", "2026-08-06T00:00:00Z"),
        ("session-5", "2026-08-05T00:00:00Z"),
        ("session-4", "2026-08-04T00:00:00Z"),
        ("session-3", "2026-08-03T00:00:00Z"),
        ("session-2", "2026-08-02T00:00:00Z"),
    ),
    "completed 的评测运行有多少？": ((2,),),
    "每种评测方法分别有多少结果？": (("hybrid", 2), ("vector", 3)),
}


@pytest.fixture
def synthetic_database(tmp_path: Path) -> Path:
    path = tmp_path / "day20-synthetic.sqlite3"
    database = SqliteDatabase(path)
    database.initialize_schema()
    database.close()
    engine = create_engine(f"sqlite+pysqlite:///{path}")
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO documents "
                "(document_id, episode_id, source_file, source_format, title, "
                "input_summary, metadata, created_at, updated_at) VALUES "
                "(:document_id, :episode_id, :source_file, :source_format, :title, "
                ":input_summary, :metadata, :created_at, :updated_at)"
            ),
            [
                {
                    "document_id": f"document-{index}",
                    "episode_id": f"episode-{index}",
                    "source_file": f"private-{index}.txt",
                    "source_format": "txt",
                    "title": f"Synthetic {index}",
                    "input_summary": "synthetic only",
                    "metadata": "{}",
                    "created_at": f"2026-07-0{index}T00:00:00Z",
                    "updated_at": f"2026-07-0{index}T00:00:00Z",
                }
                for index in range(1, 4)
            ],
        )
        connection.execute(
            text(
                "INSERT INTO sessions "
                "(session_id, title, status, created_at, updated_at) VALUES "
                "(:session_id, :title, :status, :created_at, :updated_at)"
            ),
            [
                {
                    "session_id": f"session-{index}",
                    "title": f"private-title-{index}",
                    "status": "active" if index % 2 == 0 else "closed",
                    "created_at": f"2026-08-0{index}T00:00:00Z",
                    "updated_at": f"2026-08-0{index}T00:00:00Z",
                }
                for index in range(1, 7)
            ],
        )
        connection.execute(
            text(
                "INSERT INTO messages "
                "(message_id, session_id, role, content, sequence, created_at) "
                "VALUES ('message-1', 'session-2', 'user', "
                "'never expose this message', 0, '2026-08-02T00:00:00Z')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO eval_runs "
                "(run_id, evaluation_name, schema_version, status, "
                "input_artifact_ref, input_summary, config_metadata, created_at, "
                "updated_at, completed_at) VALUES "
                "(:run_id, :evaluation_name, 1, :status, NULL, NULL, '{}', "
                ":created_at, :created_at, :completed_at)"
            ),
            [
                {
                    "run_id": "run-1",
                    "evaluation_name": "synthetic-1",
                    "status": "completed",
                    "created_at": "2026-08-01T00:00:00Z",
                    "completed_at": "2026-08-01T01:00:00Z",
                },
                {
                    "run_id": "run-2",
                    "evaluation_name": "synthetic-2",
                    "status": "completed",
                    "created_at": "2026-08-02T00:00:00Z",
                    "completed_at": "2026-08-02T01:00:00Z",
                },
                {
                    "run_id": "run-3",
                    "evaluation_name": "synthetic-3",
                    "status": "running",
                    "created_at": "2026-08-03T00:00:00Z",
                    "completed_at": None,
                },
            ],
        )
        connection.execute(
            text(
                "INSERT INTO eval_results "
                "(result_id, eval_run_id, case_id, method_name, metrics_metadata, "
                "created_at) VALUES "
                "(:result_id, :eval_run_id, :case_id, :method_name, '{}', "
                "'2026-08-04T00:00:00Z')"
            ),
            [
                {
                    "result_id": f"vector-{index}",
                    "eval_run_id": "run-1",
                    "case_id": f"case-{index}",
                    "method_name": "vector",
                }
                for index in range(1, 4)
            ]
            + [
                {
                    "result_id": f"hybrid-{index}",
                    "eval_run_id": "run-2",
                    "case_id": f"case-{index}",
                    "method_name": "hybrid",
                }
                for index in range(1, 3)
            ],
        )
    engine.dispose()
    return path


@pytest.fixture
def validator() -> QueryValidator:
    return QueryValidator(new_approval_token())


def validate(
    validator: QueryValidator,
    sql: str,
    parameters: dict[str, str | int | float | bool | None] | None = None,
) -> ValidatedQuery:
    return validator.validate(
        GeneratedQuery(sql=sql, parameters=parameters or {})
    )


def test_catalog_exposes_only_four_minimal_tables() -> None:
    assert DAY20_CATALOG.table_names == {
        "documents",
        "sessions",
        "eval_runs",
        "eval_results",
    }
    assert DAY20_CATALOG.table("sessions").columns == {
        "session_id",
        "status",
        "created_at",
        "updated_at",
    }


def test_catalog_prompt_omits_sensitive_schema() -> None:
    prompt = DAY20_CATALOG.prompt_text()
    for forbidden in (
        "messages",
        "content",
        "source_file",
        "input_summary",
        "config_metadata",
        "metrics_metadata",
        "sqlite_master",
    ):
        assert forbidden not in prompt


def test_catalog_declares_only_eval_relationship() -> None:
    assert len(DAY20_CATALOG.relationships) == 1
    assert DAY20_CATALOG.relationships[0].endpoints == {
        ("eval_results", "eval_run_id"),
        ("eval_runs", "run_id"),
    }


@pytest.mark.parametrize("question", NORMAL_QUERIES)
def test_five_normal_questions_complete_offline_round_trip(
    question: str,
    synthetic_database: Path,
) -> None:
    generator = MappingQueryGenerator(NORMAL_QUERIES)
    service = create_sqlite_text2sql_service(generator, synthetic_database)
    try:
        result = service.run(Text2SQLRequest(question=question))
    finally:
        service.close()
    assert result.rows == EXPECTED_RESULTS[question]
    assert result.row_count == len(EXPECTED_RESULTS[question])
    assert generator.call_count == 1


def test_dry_run_validates_without_execution(synthetic_database: Path) -> None:
    question = "当前有多少文档？"
    generator = MappingQueryGenerator(NORMAL_QUERIES)
    service = create_sqlite_text2sql_service(generator, synthetic_database)
    service.executor.execute = pytest.fail  # type: ignore[method-assign]
    try:
        result = service.run(Text2SQLRequest(question=question), dry_run=True)
    finally:
        service.close()
    assert isinstance(result, ValidatedQuery)
    assert result.normalized_sql.endswith("LIMIT 20")


def test_openai_generator_accepts_strict_json_and_receives_only_catalog() -> None:
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=(
                        '{"sql":"SELECT session_id FROM sessions '
                        'WHERE status = :status","parameters":{"status":"active"}}'
                    )
                )
            )
        ]
    )
    generator = OpenAICompatibleQueryGenerator(object(), model="mock-model")
    request = Text2SQLRequest(question="active sessions")
    with patch(
        "app.text2sql.generator.create_json_completion",
        return_value=completion,
    ) as create_completion:
        candidate = generator.generate(request, DAY20_CATALOG)
    assert candidate.parameters == {"status": "active"}
    messages = create_completion.call_args.kwargs["messages"]
    assert DAY20_CATALOG.prompt_text() in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "active sessions"}
    assert "messages" not in messages[0]["content"]
    assert "source_file" not in messages[0]["content"]


def test_openai_generator_rejects_markdown_wrapped_json() -> None:
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='```json\n{"sql":"SELECT session_id FROM sessions","parameters":{}}\n```'
                )
            )
        ]
    )
    generator = OpenAICompatibleQueryGenerator(object(), model="mock-model")
    with patch(
        "app.text2sql.generator.create_json_completion",
        return_value=completion,
    ):
        with pytest.raises(QueryGenerationError, match="严格契约"):
            generator.generate(Text2SQLRequest(question="sessions"), DAY20_CATALOG)


def test_openai_generator_rejects_extra_fields() -> None:
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=(
                        '{"sql":"SELECT session_id FROM sessions",'
                        '"parameters":{},"explanation":"unsafe extra"}'
                    )
                )
            )
        ]
    )
    generator = OpenAICompatibleQueryGenerator(object(), model="mock-model")
    with patch(
        "app.text2sql.generator.create_json_completion",
        return_value=completion,
    ):
        with pytest.raises(QueryGenerationError, match="严格契约"):
            generator.generate(Text2SQLRequest(question="sessions"), DAY20_CATALOG)


def test_openai_generator_rejects_empty_content() -> None:
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=""))]
    )
    generator = OpenAICompatibleQueryGenerator(object(), model="mock-model")
    with patch(
        "app.text2sql.generator.create_json_completion",
        return_value=completion,
    ):
        with pytest.raises(QueryGenerationError, match="没有返回"):
            generator.generate(Text2SQLRequest(question="sessions"), DAY20_CATALOG)


def test_allowed_catalog_join_passes(validator: QueryValidator) -> None:
    query = validate(
        validator,
        "SELECT e.method_name, r.status FROM eval_results AS e "
        "JOIN eval_runs AS r ON e.eval_run_id = r.run_id "
        "WHERE r.status = :status",
        {"status": "completed"},
    )
    assert query.referenced_tables == ("eval_results", "eval_runs")
    assert "eval_results.eval_run_id" in query.referenced_columns


def test_explicit_inner_join_passes(validator: QueryValidator) -> None:
    query = validate(
        validator,
        "SELECT e.result_id, r.run_id FROM eval_results AS e "
        "INNER JOIN eval_runs AS r ON e.eval_run_id = r.run_id",
    )
    assert query.referenced_tables == ("eval_results", "eval_runs")


def test_limit_is_added_with_ast(validator: QueryValidator) -> None:
    query = validate(validator, "SELECT session_id FROM sessions")
    assert query.normalized_sql == "SELECT session_id FROM sessions LIMIT 20"
    assert query.effective_limit == 20


def test_limit_is_tightened_with_ast(validator: QueryValidator) -> None:
    query = validate(validator, "SELECT session_id FROM sessions LIMIT 999")
    assert query.normalized_sql.endswith("LIMIT 100")
    assert query.effective_limit == 100


def test_smaller_limit_is_preserved(validator: QueryValidator) -> None:
    query = validate(validator, "SELECT session_id FROM sessions LIMIT 5")
    assert query.normalized_sql.endswith("LIMIT 5")
    assert query.effective_limit == 5


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO sessions (session_id) VALUES ('x')",
        "UPDATE sessions SET status = 'closed'",
        "DELETE FROM sessions",
        "DROP TABLE sessions",
        "ALTER TABLE sessions ADD COLUMN danger TEXT",
        "CREATE TABLE danger (value TEXT)",
        "TRUNCATE TABLE sessions",
        "MERGE INTO sessions USING sessions ON 1 = 1 WHEN MATCHED THEN DELETE",
        "REPLACE INTO sessions (session_id) VALUES ('x')",
        "ATTACH DATABASE 'x' AS other",
        "DETACH DATABASE other",
        "PRAGMA table_info(sessions)",
        "VACUUM",
        "BEGIN TRANSACTION",
        "COMMIT",
    ],
)
def test_write_and_control_statements_are_rejected(
    validator: QueryValidator,
    sql: str,
) -> None:
    with pytest.raises(QueryValidationError):
        validate(validator, sql)


def test_multiple_statements_are_rejected(validator: QueryValidator) -> None:
    with pytest.raises(QueryValidationError, match="一条"):
        validate(
            validator,
            "SELECT session_id FROM sessions; DROP TABLE sessions",
        )


@pytest.mark.parametrize("table", ["messages", "sqlite_master", "pragma_table_info"])
def test_unauthorized_and_system_tables_are_rejected(
    validator: QueryValidator,
    table: str,
) -> None:
    with pytest.raises(QueryValidationError, match="未授权表"):
        validate(validator, f"SELECT name FROM {table}")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT title FROM sessions",
        "SELECT source_file FROM documents",
        "SELECT input_summary FROM eval_runs",
        "SELECT metrics_metadata FROM eval_results",
    ],
)
def test_sensitive_fields_are_rejected(
    validator: QueryValidator,
    sql: str,
) -> None:
    with pytest.raises(QueryValidationError, match="字段"):
        validate(validator, sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM sessions",
        "SELECT sessions.* FROM sessions",
    ],
)
def test_select_star_is_rejected(validator: QueryValidator, sql: str) -> None:
    with pytest.raises(QueryValidationError, match="星号|SELECT"):
        validate(validator, sql)


def test_count_star_is_the_only_allowed_star(validator: QueryValidator) -> None:
    assert validate(validator, "SELECT COUNT(*) AS total FROM sessions")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT session_id FROM sessions UNION SELECT message_id FROM messages",
        "WITH hidden AS (SELECT content FROM messages) SELECT content FROM hidden",
        "SELECT session_id FROM sessions WHERE session_id = "
        "(SELECT session_id FROM sessions LIMIT 1)",
        "WITH changed AS (DELETE FROM sessions RETURNING session_id) "
        "SELECT session_id FROM changed",
    ],
)
def test_union_cte_and_subquery_bypasses_are_rejected(
    validator: QueryValidator,
    sql: str,
) -> None:
    with pytest.raises(QueryValidationError):
        validate(validator, sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT random() FROM sessions",
        "SELECT load_extension(:path) FROM sessions",
        "SELECT sqlite_version() FROM sessions",
    ],
)
def test_unapproved_functions_are_rejected(
    validator: QueryValidator,
    sql: str,
) -> None:
    with pytest.raises(QueryValidationError):
        validate(validator, sql, {"path": "danger"} if ":path" in sql else {})


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT /* hidden */ session_id FROM sessions",
        "SELECT session_id FROM sessions -- trailing",
        "SELECT session_id /*+ hint */ FROM sessions",
    ],
)
def test_comments_and_hints_are_rejected(
    validator: QueryValidator,
    sql: str,
) -> None:
    with pytest.raises(QueryValidationError, match="注释|Hint"):
        validate(validator, sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT session_id INTO copied_sessions FROM sessions",
        "SELECT session_id FROM sessions FOR UPDATE",
        "SELECT session_id FROM sessions FOR SHARE",
    ],
)
def test_select_into_and_locking_reads_are_rejected(
    validator: QueryValidator,
    sql: str,
) -> None:
    with pytest.raises(QueryValidationError):
        validate(validator, sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT sessions.session_id FROM sessions CROSS JOIN documents",
        "SELECT sessions.session_id FROM sessions JOIN documents "
        "ON sessions.session_id = documents.document_id",
    ],
)
def test_cartesian_and_unrelated_joins_are_rejected(
    validator: QueryValidator,
    sql: str,
) -> None:
    with pytest.raises(QueryValidationError, match="JOIN|关系"):
        validate(validator, sql)


def test_cross_schema_identifier_is_rejected(validator: QueryValidator) -> None:
    with pytest.raises(QueryValidationError, match="Schema|Catalog"):
        validate(validator, "SELECT session_id FROM main.sessions")


def test_ambiguous_unqualified_column_is_rejected(validator: QueryValidator) -> None:
    with pytest.raises(QueryValidationError, match="唯一"):
        validate(
            validator,
            "SELECT created_at FROM eval_results e JOIN eval_runs r "
            "ON e.eval_run_id = r.run_id",
        )


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({}, "完全对应"),
        ({"status": "active", "unused": "x"}, "完全对应"),
    ],
)
def test_missing_and_extra_parameters_are_rejected(
    validator: QueryValidator,
    parameters: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(QueryValidationError, match=message):
        validate(
            validator,
            "SELECT session_id FROM sessions WHERE status = :status",
            parameters,
        )


def test_unnamed_parameter_is_rejected(validator: QueryValidator) -> None:
    with pytest.raises(QueryValidationError, match="命名参数"):
        validate(validator, "SELECT session_id FROM sessions WHERE status = ?")


def test_literal_condition_value_is_rejected(validator: QueryValidator) -> None:
    with pytest.raises(QueryValidationError, match="命名参数"):
        validate(
            validator,
            "SELECT session_id FROM sessions WHERE status = 'active'",
        )


def test_where_rejects_aggregate_while_having_allows_it(
    validator: QueryValidator,
) -> None:
    with pytest.raises(QueryValidationError, match="左侧"):
        validate(
            validator,
            "SELECT status, COUNT(*) AS total FROM sessions "
            "WHERE COUNT(*) > :minimum GROUP BY status",
            {"minimum": 1},
        )
    query = validate(
        validator,
        "SELECT status, COUNT(*) AS total FROM sessions GROUP BY status "
        "HAVING COUNT(*) > :minimum",
        {"minimum": 1},
    )
    assert "HAVING COUNT(*) > :minimum" in query.normalized_sql


def test_parameter_outside_predicate_is_rejected(validator: QueryValidator) -> None:
    with pytest.raises(QueryValidationError):
        validate(validator, "SELECT :value FROM sessions", {"value": "x"})


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT session_id FROM sessions LIMIT 0",
        "SELECT session_id FROM sessions LIMIT :limit",
        "SELECT session_id FROM sessions LIMIT 5 OFFSET 101",
        "SELECT session_id FROM sessions LIMIT 5 OFFSET -1",
    ],
)
def test_limit_and_offset_must_be_controlled_integers(
    validator: QueryValidator,
    sql: str,
) -> None:
    with pytest.raises(QueryValidationError):
        validate(validator, sql, {"limit": 5} if ":limit" in sql else {})


def test_sql_length_is_limited(validator: QueryValidator) -> None:
    sql = "SELECT" + (" " * 4_100) + "session_id FROM sessions"
    with pytest.raises(QueryValidationError, match="长度"):
        validate(validator, sql)


def test_table_and_join_counts_are_limited(validator: QueryValidator) -> None:
    with pytest.raises(QueryValidationError, match="表数量|JOIN"):
        validate(
            validator,
            "SELECT e.result_id FROM eval_results e "
            "JOIN eval_runs r ON e.eval_run_id = r.run_id "
            "JOIN sessions s ON r.status = s.status",
        )


def test_ast_depth_is_limited() -> None:
    validator = QueryValidator(
        new_approval_token(),
        policy=ValidationPolicy(max_ast_depth=8),
    )
    predicate = "status = :status"
    for _ in range(12):
        predicate = f"({predicate} OR status = :status)"
    with pytest.raises(QueryValidationError, match="深度"):
        validate(
            validator,
            f"SELECT session_id FROM sessions WHERE {predicate}",
            {"status": "active"},
        )


@pytest.mark.parametrize(
    "parameters",
    [
        {"bad-name": "active"},
        {"1status": "active"},
        {"status": "x" * 513},
        {"status": ["active"]},
        {"status": {"nested": "active"}},
        {"status": float("nan")},
    ],
)
def test_parameter_contract_rejects_invalid_names_sizes_and_types(
    parameters: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        GeneratedQuery(
            sql="SELECT session_id FROM sessions WHERE status = :status",
            parameters=parameters,
        )


def test_validation_error_hides_parameter_input() -> None:
    with pytest.raises(ValidationError) as captured:
        GeneratedQuery(
            sql="SELECT session_id FROM sessions WHERE status = :status",
            parameters={"status": "secret" * 100},
        )
    assert "secret" not in str(captured.value)


def test_injection_text_is_bound_as_an_ordinary_value(
    synthetic_database: Path,
) -> None:
    question = "injection"
    candidate = GeneratedQuery(
        sql="SELECT session_id FROM sessions WHERE status = :status",
        parameters={"status": "active' OR 1=1 --"},
    )
    service = create_sqlite_text2sql_service(
        MappingQueryGenerator({question: candidate}),
        synthetic_database,
    )
    try:
        result = service.run(Text2SQLRequest(question=question))
    finally:
        service.close()
    assert result.rows == ()
    writable = create_engine(f"sqlite+pysqlite:///{synthetic_database}")
    with writable.connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM sessions")).scalar_one() == 6
    writable.dispose()


def test_executor_rejects_manually_constructed_validated_query(
    synthetic_database: Path,
) -> None:
    service = create_sqlite_text2sql_service(
        MappingQueryGenerator({}),
        synthetic_database,
    )
    forged = ValidatedQuery(
        normalized_sql="SELECT session_id FROM sessions LIMIT 1",
        parameters={},
        referenced_tables=("sessions",),
        referenced_columns=("sessions.session_id",),
        effective_limit=1,
    )
    try:
        with pytest.raises(QueryExecutionError, match="Validator"):
            service.executor.execute(forged)
    finally:
        service.close()


def test_invalid_query_never_reaches_executor(synthetic_database: Path) -> None:
    question = "delete everything"
    approval = new_approval_token()
    validator = QueryValidator(approval)

    class NeverExecutor:
        def execute(self, query: ValidatedQuery) -> None:
            pytest.fail("invalid query reached executor")

        def close(self) -> None:
            pass

    service = Text2SQLService(
        MappingQueryGenerator(
            {question: GeneratedQuery(sql="DELETE FROM sessions", parameters={})}
        ),
        validator,
        NeverExecutor(),  # type: ignore[arg-type]
    )
    with pytest.raises(QueryValidationError):
        service.run(Text2SQLRequest(question=question))


def test_read_only_mode_and_query_only_are_independent_layers(
    synthetic_database: Path,
) -> None:
    service = create_sqlite_text2sql_service(
        MappingQueryGenerator({}),
        synthetic_database,
    )
    try:
        assert service.executor.query_only_enabled() is True
        with service.executor._engine.connect() as connection:  # noqa: SLF001
            connection.exec_driver_sql("PRAGMA query_only=OFF")
            assert connection.exec_driver_sql("PRAGMA query_only").scalar_one() == 0
            with pytest.raises(OperationalError, match="readonly"):
                connection.exec_driver_sql(
                    "INSERT INTO sessions "
                    "(session_id, status, created_at, updated_at) VALUES "
                    "('bypass', 'active', '2026-01-01', '2026-01-01')"
                )
        assert service.executor.query_only_enabled() is True
    finally:
        service.close()


def test_query_timeout_interrupts_sqlite_execution(synthetic_database: Path) -> None:
    question = "timeout"
    generator = MappingQueryGenerator(
        {
            question: GeneratedQuery(
                sql="SELECT session_id FROM sessions ORDER BY created_at",
                parameters={},
            )
        }
    )
    service = create_sqlite_text2sql_service(
        generator,
        synthetic_database,
        timeout_seconds=1e-12,
    )
    service.executor._progress_operations = 1  # noqa: SLF001
    try:
        with pytest.raises(QueryTimeoutError, match="时间"):
            service.run(Text2SQLRequest(question=question))
    finally:
        service.close()


def test_result_row_limit_sets_truncated(synthetic_database: Path) -> None:
    question = "two rows"
    service = create_sqlite_text2sql_service(
        MappingQueryGenerator(
            {
                question: GeneratedQuery(
                    sql="SELECT session_id FROM sessions ORDER BY created_at LIMIT 2",
                    parameters={},
                )
            }
        ),
        synthetic_database,
    )
    try:
        result = service.run(Text2SQLRequest(question=question))
    finally:
        service.close()
    assert result.row_count == 2
    assert result.truncated is True


def test_result_body_size_sets_truncated(synthetic_database: Path) -> None:
    question = "small body"
    service = create_sqlite_text2sql_service(
        MappingQueryGenerator(
            {
                question: GeneratedQuery(
                    sql="SELECT session_id FROM sessions ORDER BY created_at",
                    parameters={},
                )
            }
        ),
        synthetic_database,
        max_result_bytes=3,
    )
    try:
        result = service.run(Text2SQLRequest(question=question))
    finally:
        service.close()
    assert result.rows == ()
    assert result.truncated is True


def test_execution_error_does_not_leak_path_sql_or_parameter(tmp_path: Path) -> None:
    path = tmp_path / "password=hunter2.sqlite3"
    engine = create_engine(f"sqlite+pysqlite:///{path}")
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE sessions (session_id TEXT, status TEXT)"
        )
    engine.dispose()
    question = "broken schema"
    service = create_sqlite_text2sql_service(
        MappingQueryGenerator(
            {
                question: GeneratedQuery(
                    sql="SELECT created_at FROM sessions WHERE status = :status",
                    parameters={"status": "secret-value"},
                )
            }
        ),
        path,
    )
    try:
        with pytest.raises(QueryExecutionError) as captured:
            service.run(Text2SQLRequest(question=question))
    finally:
        service.close()
    message = str(captured.value)
    assert "hunter2" not in message
    assert "created_at" not in message
    assert "secret-value" not in message


def test_postgres_read_only_design_compiles_without_connecting() -> None:
    plan = build_postgres_read_only_plan(statement_timeout_ms=2_000)
    all_sql = (*plan.permission_statements, *plan.transaction_statements)
    compiled = [str(text(statement).compile(dialect=postgresql.dialect())) for statement in all_sql]
    joined = "\n".join(compiled)
    assert "GRANT SELECT (" in joined
    assert "messages" not in joined
    assert "source_file" not in joined
    assert "REVOKE EXECUTE" in joined
    assert "default_transaction_read_only = on" in joined
    assert "SET TRANSACTION READ ONLY" in joined
    assert "statement_timeout = '2000ms'" in joined


@pytest.mark.parametrize("timeout", [0, 3_001])
def test_postgres_timeout_design_is_bounded(timeout: int) -> None:
    with pytest.raises(ValueError, match="statement_timeout"):
        build_postgres_read_only_plan(statement_timeout_ms=timeout)


def test_cli_dry_run_is_offline_and_structured(
    synthetic_database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate = GeneratedQuery(
        sql="SELECT COUNT(*) AS total FROM documents",
        parameters={},
    )
    status = cli_main(
        [
            "count documents",
            "--candidate-json",
            candidate.model_dump_json(),
            "--read-only-sqlite",
            str(synthetic_database),
            "--dry-run",
        ]
    )
    assert status == 0
    output = json.loads(capsys.readouterr().out)
    assert output["effective_limit"] == 20
    assert output["referenced_tables"] == ["documents"]


def test_cli_contract_error_does_not_echo_input(
    synthetic_database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = cli_main(
        [
            "count documents",
            "--candidate-json",
            '{"sql":"SELECT session_id FROM sessions","parameters":{"x":"password-secret"},"extra":1}',
            "--read-only-sqlite",
            str(synthetic_database),
        ]
    )
    assert status == 2
    output = capsys.readouterr().out
    assert json.loads(output) == {"error": "输入契约无效。"}
    assert "password-secret" not in output


def test_cli_token_error_is_safe_and_does_not_echo_sql(
    synthetic_database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret_fragment = "password-secret-fragment"
    candidate = GeneratedQuery(
        sql=(
            "SELECT session_id FROM sessions WHERE status = '"
            + secret_fragment
        ),
        parameters={},
    )
    status = cli_main(
        [
            "broken SQL",
            "--candidate-json",
            candidate.model_dump_json(),
            "--read-only-sqlite",
            str(synthetic_database),
        ]
    )
    output = capsys.readouterr().out
    assert status == 2
    assert json.loads(output) == {"error": "SQL 无法按 SQLite 方言解析。"}
    assert secret_fragment not in output
    assert "Traceback" not in output


def test_cli_maps_query_generation_error_without_traceback(
    synthetic_database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FailingService:
        def run(self, request: Text2SQLRequest, *, dry_run: bool = False) -> None:
            del request, dry_run
            raise QueryGenerationError("SQL 候选生成失败。")

        def close(self) -> None:
            pass

    candidate = GeneratedQuery(
        sql="SELECT session_id FROM sessions",
        parameters={},
    )
    with patch(
        "app.text2sql.cli.create_sqlite_text2sql_service",
        return_value=FailingService(),
    ):
        status = cli_main(
            [
                "sessions",
                "--candidate-json",
                candidate.model_dump_json(),
                "--read-only-sqlite",
                str(synthetic_database),
            ]
        )
    output = capsys.readouterr().out
    assert status == 2
    assert json.loads(output) == {"error": "SQL 候选生成失败。"}
    assert "Traceback" not in output
