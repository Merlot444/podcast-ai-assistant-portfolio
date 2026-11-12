"""Day 20 constrained, read-only Text2SQL boundary."""

from app.text2sql.catalog import DAY20_CATALOG, SchemaCatalog
from app.text2sql.executor import (
    QueryExecutionError,
    QueryTimeoutError,
    ReadOnlySQLiteExecutor,
)
from app.text2sql.generator import (
    MappingQueryGenerator,
    OpenAICompatibleQueryGenerator,
    QueryGenerationError,
    QueryGenerator,
)
from app.text2sql.postgresql import PostgresReadOnlyPlan, build_postgres_read_only_plan
from app.text2sql.schemas import (
    GeneratedQuery,
    Text2SQLRequest,
    Text2SQLResult,
    ValidatedQuery,
)
from app.text2sql.service import Text2SQLService, create_sqlite_text2sql_service
from app.text2sql.validator import (
    QueryValidationError,
    QueryValidator,
    ValidationPolicy,
)

__all__ = [
    "DAY20_CATALOG",
    "GeneratedQuery",
    "MappingQueryGenerator",
    "OpenAICompatibleQueryGenerator",
    "PostgresReadOnlyPlan",
    "QueryExecutionError",
    "QueryGenerationError",
    "QueryGenerator",
    "QueryTimeoutError",
    "QueryValidationError",
    "QueryValidator",
    "ReadOnlySQLiteExecutor",
    "SchemaCatalog",
    "Text2SQLRequest",
    "Text2SQLResult",
    "Text2SQLService",
    "ValidatedQuery",
    "ValidationPolicy",
    "create_sqlite_text2sql_service",
    "build_postgres_read_only_plan",
]
