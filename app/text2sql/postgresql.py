"""Compile-only PostgreSQL least-privilege plan; this module never connects."""

from __future__ import annotations

from dataclasses import dataclass

from app.text2sql.catalog import DAY20_CATALOG, SchemaCatalog


@dataclass(frozen=True)
class PostgresReadOnlyPlan:
    role: str
    schema: str
    permission_statements: tuple[str, ...]
    transaction_statements: tuple[str, ...]


def build_postgres_read_only_plan(
    *,
    catalog: SchemaCatalog = DAY20_CATALOG,
    role: str = "text2sql_reader",
    schema: str = "public",
    statement_timeout_ms: int = 2_000,
) -> PostgresReadOnlyPlan:
    """Return fixed-identifier design SQL for review/compilation, not execution."""
    if role != "text2sql_reader" or schema != "public":
        raise ValueError("Day 20 设计仅允许固定的隔离角色和 public schema。")
    if not 1 <= statement_timeout_ms <= 3_000:
        raise ValueError("statement_timeout 必须为 1～3000 ms。")
    permissions = [
        f"REVOKE ALL ON SCHEMA {schema} FROM {role}",
        f"REVOKE ALL ON ALL TABLES IN SCHEMA {schema} FROM {role}",
        f"REVOKE ALL ON ALL SEQUENCES IN SCHEMA {schema} FROM {role}",
        f"REVOKE EXECUTE ON ALL FUNCTIONS IN SCHEMA {schema} FROM {role}",
        f"GRANT USAGE ON SCHEMA {schema} TO {role}",
    ]
    for table in catalog.tables:
        columns = ", ".join(sorted(table.columns))
        permissions.append(
            f"GRANT SELECT ({columns}) ON {schema}.{table.name} TO {role}"
        )
    permissions.extend(
        (
            f"REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER "
            f"ON ALL TABLES IN SCHEMA {schema} FROM {role}",
            f"ALTER ROLE {role} SET default_transaction_read_only = on",
        )
    )
    return PostgresReadOnlyPlan(
        role=role,
        schema=schema,
        permission_statements=tuple(permissions),
        transaction_statements=(
            "SET TRANSACTION READ ONLY",
            f"SET LOCAL statement_timeout = '{statement_timeout_ms}ms'",
        ),
    )
