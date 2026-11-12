"""Least-privilege schema information visible to Text2SQL generators."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CatalogTable:
    name: str
    columns: frozenset[str]


@dataclass(frozen=True)
class CatalogRelationship:
    left_table: str
    left_column: str
    right_table: str
    right_column: str

    @property
    def endpoints(self) -> frozenset[tuple[str, str]]:
        return frozenset(
            {
                (self.left_table, self.left_column),
                (self.right_table, self.right_column),
            }
        )


@dataclass(frozen=True)
class SchemaCatalog:
    tables: tuple[CatalogTable, ...]
    relationships: tuple[CatalogRelationship, ...]
    aggregate_functions: frozenset[str]

    def table(self, name: str) -> CatalogTable | None:
        return next((table for table in self.tables if table.name == name), None)

    @property
    def table_names(self) -> frozenset[str]:
        return frozenset(table.name for table in self.tables)

    def prompt_text(self) -> str:
        """Return only explicitly public schema facts, never reflected database DDL."""
        lines = ["Allowed read-only schema:"]
        for table in self.tables:
            lines.append(f"- {table.name}: {', '.join(sorted(table.columns))}")
        lines.append("Allowed relationships:")
        for relationship in self.relationships:
            lines.append(
                "- "
                f"{relationship.left_table}.{relationship.left_column} = "
                f"{relationship.right_table}.{relationship.right_column}"
            )
        lines.append(
            "Allowed aggregate functions: "
            + ", ".join(sorted(self.aggregate_functions))
        )
        return "\n".join(lines)


DAY20_CATALOG = SchemaCatalog(
    tables=(
        CatalogTable(
            "documents",
            frozenset(
                {
                    "document_id",
                    "episode_id",
                    "source_format",
                    "title",
                    "created_at",
                    "updated_at",
                }
            ),
        ),
        CatalogTable(
            "sessions",
            frozenset(
                {
                    "session_id",
                    "status",
                    "created_at",
                    "updated_at",
                }
            ),
        ),
        CatalogTable(
            "eval_runs",
            frozenset(
                {
                    "run_id",
                    "evaluation_name",
                    "schema_version",
                    "status",
                    "created_at",
                    "updated_at",
                    "completed_at",
                }
            ),
        ),
        CatalogTable(
            "eval_results",
            frozenset(
                {
                    "result_id",
                    "eval_run_id",
                    "case_id",
                    "method_name",
                    "created_at",
                }
            ),
        ),
    ),
    relationships=(
        CatalogRelationship(
            "eval_results",
            "eval_run_id",
            "eval_runs",
            "run_id",
        ),
    ),
    aggregate_functions=frozenset({"COUNT", "SUM", "AVG", "MIN", "MAX"}),
)
