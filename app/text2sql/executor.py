"""Capability-gated execution against a dedicated read-only SQLite file."""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from time import monotonic
from typing import Any
from urllib.parse import quote

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from app.text2sql.schemas import ResultScalar, Text2SQLResult, ValidatedQuery


class QueryExecutionError(RuntimeError):
    """Safe execution failure that omits SQL, parameters, and connection details."""


class QueryTimeoutError(QueryExecutionError):
    pass


class ReadOnlySQLiteExecutor:
    def __init__(
        self,
        database_path: str | Path,
        approval_token: object,
        *,
        timeout_seconds: float = 2.0,
        max_result_bytes: int = 64_000,
        progress_operations: int = 100,
    ) -> None:
        path = Path(database_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError("Text2SQL 数据库必须是已存在的专用 SQLite 文件。")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须大于 0。")
        if max_result_bytes <= 0 or progress_operations <= 0:
            raise ValueError("结果体积和 progress_operations 上限必须大于 0。")
        self._approval_token = approval_token
        self.timeout_seconds = timeout_seconds
        self.max_result_bytes = max_result_bytes
        self._progress_operations = progress_operations
        database_uri = f"file:{quote(path.as_posix(), safe='/')}?mode=ro"
        self._engine = create_engine(
            "sqlite+pysqlite://",
            creator=lambda: sqlite3.connect(
                database_uri,
                uri=True,
                check_same_thread=False,
            ),
            hide_parameters=True,
        )
        self._enable_query_only(self._engine)

    @staticmethod
    def _enable_query_only(engine: Engine) -> None:
        @event.listens_for(engine, "checkout")
        def set_query_only(
            dbapi_connection: Any,
            connection_record: Any,
            connection_proxy: Any,
        ) -> None:
            del connection_record, connection_proxy
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA query_only=ON")
                cursor.execute("PRAGMA query_only")
                if cursor.fetchone() != (1,):
                    raise RuntimeError("SQLite query_only 未成功启用")
            finally:
                cursor.close()

    def execute(self, query: ValidatedQuery) -> Text2SQLResult:
        if not isinstance(query, ValidatedQuery) or not query._is_approved_by(
            self._approval_token
        ):
            raise QueryExecutionError("Executor 只接受匹配 Validator 批准的查询。")
        started = monotonic()
        deadline = started + self.timeout_seconds
        try:
            with self._engine.connect() as connection:
                raw = connection.connection.driver_connection
                raw.set_progress_handler(
                    lambda: 1 if monotonic() >= deadline else 0,
                    self._progress_operations,
                )
                try:
                    result = connection.execute(
                        text(query.normalized_sql),
                        dict(query.parameters),
                    )
                    columns = tuple(result.keys())
                    rows, truncated = self._bounded_rows(
                        result,
                        query.effective_limit,
                        columns,
                    )
                finally:
                    raw.set_progress_handler(None, 0)
        except OperationalError as error:
            if getattr(error.orig, "sqlite_errorcode", None) == sqlite3.SQLITE_INTERRUPT:
                raise QueryTimeoutError("只读查询超过执行时间上限。") from None
            raise QueryExecutionError("只读查询执行失败。") from None
        except SQLAlchemyError:
            raise QueryExecutionError("只读查询执行失败。") from None
        elapsed = max(0.0, monotonic() - started)
        return Text2SQLResult(
            columns=columns,
            rows=tuple(rows),
            row_count=len(rows),
            truncated=truncated,
            elapsed_seconds=elapsed,
        )

    def _bounded_rows(
        self,
        result: Any,
        effective_limit: int,
        columns: tuple[str, ...],
    ) -> tuple[list[tuple[ResultScalar, ...]], bool]:
        rows: list[tuple[ResultScalar, ...]] = []
        total_bytes = len(
            json.dumps(
                {"columns": columns, "rows": []},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        )
        if total_bytes > self.max_result_bytes:
            return rows, True
        truncated = False
        for raw_row in result:
            row = tuple(_json_scalar(value) for value in raw_row)
            encoded_size = len(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode()
            )
            if total_bytes + encoded_size > self.max_result_bytes:
                truncated = True
                break
            rows.append(row)
            total_bytes += encoded_size + 1
            if len(rows) >= effective_limit:
                truncated = True
                break
        return rows, truncated

    def query_only_enabled(self) -> bool:
        with self._engine.connect() as connection:
            return connection.exec_driver_sql("PRAGMA query_only").scalar_one() == 1

    def close(self) -> None:
        self._engine.dispose()

    def __enter__(self) -> ReadOnlySQLiteExecutor:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def _json_scalar(value: Any) -> ResultScalar:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise QueryExecutionError("查询结果包含非有限浮点数。")
        return value
    if isinstance(value, Decimal):
        converted = float(value)
        if not math.isfinite(converted):
            raise QueryExecutionError("查询结果包含非有限数值。")
        return converted
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise QueryExecutionError("查询结果包含不支持的值类型。")
