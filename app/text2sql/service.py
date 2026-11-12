"""Text2SQL orchestration with explicit generation, validation, and execution phases."""

from __future__ import annotations

from pathlib import Path

from app.text2sql.catalog import DAY20_CATALOG, SchemaCatalog
from app.text2sql.executor import ReadOnlySQLiteExecutor
from app.text2sql.generator import QueryGenerator
from app.text2sql.schemas import (
    Text2SQLRequest,
    Text2SQLResult,
    ValidatedQuery,
    new_approval_token,
)
from app.text2sql.validator import QueryValidator, ValidationPolicy


class Text2SQLService:
    def __init__(
        self,
        generator: QueryGenerator,
        validator: QueryValidator,
        executor: ReadOnlySQLiteExecutor,
        *,
        catalog: SchemaCatalog = DAY20_CATALOG,
    ) -> None:
        self.generator = generator
        self.validator = validator
        self.executor = executor
        self.catalog = catalog

    def validate(self, request: Text2SQLRequest) -> ValidatedQuery:
        candidate = self.generator.generate(request, self.catalog)
        return self.validator.validate(candidate)

    def run(
        self,
        request: Text2SQLRequest,
        *,
        dry_run: bool = False,
    ) -> ValidatedQuery | Text2SQLResult:
        validated = self.validate(request)
        if dry_run:
            return validated
        return self.executor.execute(validated)

    def close(self) -> None:
        self.executor.close()


def create_sqlite_text2sql_service(
    generator: QueryGenerator,
    database_path: str | Path,
    *,
    catalog: SchemaCatalog = DAY20_CATALOG,
    policy: ValidationPolicy = ValidationPolicy(),
    timeout_seconds: float = 2.0,
    max_result_bytes: int = 64_000,
) -> Text2SQLService:
    approval = new_approval_token()
    validator = QueryValidator(approval, catalog=catalog, policy=policy)
    executor = ReadOnlySQLiteExecutor(
        database_path,
        approval,
        timeout_seconds=timeout_seconds,
        max_result_bytes=max_result_bytes,
    )
    return Text2SQLService(
        generator,
        validator,
        executor,
        catalog=catalog,
    )
