"""Explicit database engine, schema initialization, and transaction lifecycle."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from app.storage.config import DatabaseSettings
from app.storage.models import Base


class Database:
    """Own one supported SQLAlchemy engine without global mutable connections."""

    def __init__(self, settings: DatabaseSettings) -> None:
        self.settings = settings
        self.path = settings.sqlite_path
        self.engine = create_engine(settings.url, hide_parameters=True)
        if settings.dialect_name == "sqlite":
            self._enable_sqlite_foreign_keys(self.engine)
        self._session_factory = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
        )

    @staticmethod
    def _enable_sqlite_foreign_keys(engine: Engine) -> None:
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA foreign_keys")
                if cursor.fetchone() != (1,):
                    raise RuntimeError("SQLite 外键约束未成功启用")
            finally:
                cursor.close()

    def initialize_schema(self) -> None:
        """Create missing tables and indexes; safe to call repeatedly."""
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        Base.metadata.create_all(self.engine)

    @contextmanager
    def transaction(self, *, sqlite_immediate: bool = False) -> Iterator[Session]:
        """Commit all work together or roll the entire unit back on failure."""
        with self._session_factory() as session:
            try:
                if sqlite_immediate and self.settings.dialect_name == "sqlite":
                    session.connection().exec_driver_sql("BEGIN IMMEDIATE")
                yield session
                session.commit()
            except BaseException:
                session.rollback()
                raise

    def foreign_keys_enabled(self) -> bool:
        """Verify the PRAGMA on a newly checked-out connection."""
        if self.settings.dialect_name != "sqlite":
            raise RuntimeError("foreign_keys_enabled 只适用于 SQLite")
        with self.engine.connect() as connection:
            return connection.execute(text("PRAGMA foreign_keys")).scalar_one() == 1

    def close(self) -> None:
        """Release pooled connections and underlying SQLite handles."""
        self.engine.dispose()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


class SqliteDatabase(Database):
    """Backward-compatible Day 18 entry point for an explicit SQLite path."""

    def __init__(self, path: str | Path) -> None:
        database_path = Path(path).expanduser().resolve()
        super().__init__(
            DatabaseSettings(database_url=f"sqlite+pysqlite:///{database_path}")
        )


def create_database(settings: DatabaseSettings | None = None) -> Database:
    """Create an uninitialized database from explicit settings or DATABASE_URL."""
    return Database(settings or DatabaseSettings.from_env())
