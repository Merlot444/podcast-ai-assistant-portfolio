"""Database-only configuration kept outside the model-client core."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv
from sqlalchemy.engine import URL, make_url

from app.core.config import ConfigError, PROJECT_ROOT


DEFAULT_DATABASE_URL = "sqlite+pysqlite:///data/podcast-ai-assistant.sqlite3"
SUPPORTED_DATABASE_DRIVERS = frozenset(
    {"sqlite+pysqlite", "postgresql+psycopg"}
)


@dataclass(frozen=True)
class DatabaseSettings:
    """Strict, secret-safe database configuration independent of model settings."""

    database_url: str = field(default=DEFAULT_DATABASE_URL, repr=False)

    def __post_init__(self) -> None:
        try:
            parsed = make_url(self.database_url)
        except Exception:
            raise ConfigError("DATABASE_URL 格式无效。") from None

        if parsed.drivername not in SUPPORTED_DATABASE_DRIVERS:
            allowed = ", ".join(sorted(SUPPORTED_DATABASE_DRIVERS))
            raise ConfigError(
                "DATABASE_URL 使用不受支持的 dialect/driver；"
                f"仅支持：{allowed}。"
            )
        if parsed.drivername == "sqlite+pysqlite":
            parsed = _normalize_sqlite_url(parsed)
        elif not parsed.database:
            raise ConfigError("PostgreSQL DATABASE_URL 必须指定数据库名称。")
        else:
            try:
                port = parsed.port
            except ValueError:
                raise ConfigError("PostgreSQL DATABASE_URL 端口无效。") from None
            if port is not None and not 1 <= port <= 65_535:
                raise ConfigError("PostgreSQL DATABASE_URL 端口必须为 1～65535。")

        object.__setattr__(self, "_url", parsed)

    @property
    def url(self) -> URL:
        """Return SQLAlchemy's parsed URL; callers must not log its password."""
        return self._url

    @property
    def safe_url(self) -> str:
        """Return a display-safe URL with any password hidden."""
        return self.url.render_as_string(hide_password=True)

    @property
    def dialect_name(self) -> str:
        return self.url.get_backend_name()

    @property
    def sqlite_path(self) -> Path | None:
        if self.dialect_name != "sqlite" or self.url.database == ":memory:":
            return None
        return Path(self.url.database)

    @classmethod
    def from_env(
        cls,
        env_file: str | Path | None = ".env",
        *,
        environ: Mapping[str, str] | None = None,
    ) -> "DatabaseSettings":
        """Load only DATABASE_URL; explicit mappings keep tests isolated from .env."""
        if env_file is not None and environ is None:
            load_dotenv(_project_relative_path(env_file), override=False)
        source = os.environ if environ is None else environ
        value = source.get("DATABASE_URL", DEFAULT_DATABASE_URL).strip()
        if not value:
            raise ConfigError("DATABASE_URL 不能为空。")
        return cls(database_url=value)


def _normalize_sqlite_url(url: URL) -> URL:
    database = url.database
    if not database:
        raise ConfigError("SQLite DATABASE_URL 必须指定文件路径或 :memory:。")
    if url.username or url.password or url.host or url.port:
        raise ConfigError("SQLite DATABASE_URL 不允许用户名、密码、主机或端口。")
    if database == ":memory:":
        return url
    path = Path(database).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if path.exists() and path.is_dir():
        raise ConfigError("SQLite DATABASE_URL 的路径不能是目录。")
    return url.set(database=str(path))


def _project_relative_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()
