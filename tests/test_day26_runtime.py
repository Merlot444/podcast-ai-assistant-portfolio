"""Day 26 strict process runtime configuration and module-entry tests."""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

pytest.importorskip("fastapi")

from app.api import create_app
from app.api import __main__ as api_main
from app.api.runtime import (
    DEFAULT_API_HOST,
    DEFAULT_API_PORT,
    DEFAULT_MAX_REQUEST_BODY_BYTES,
    PODCAST_API_HOST,
    PODCAST_API_PORT,
    PODCAST_MAX_REQUEST_BODY_BYTES,
    RUNTIME_CONFIGURATION_ERROR_MESSAGE,
    RuntimeConfig,
    RuntimeConfigurationError,
    load_runtime_config,
)


PROJECT_ROOT = Path(__file__).parents[1]
PRIVATE_VALUE = "postgresql://admin:password@localhost/private"


def test_local_defaults_are_fixed_and_do_not_read_the_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PODCAST_API_HOST, "invalid-private-host")
    monkeypatch.setenv(PODCAST_API_PORT, PRIVATE_VALUE)
    monkeypatch.setenv(PODCAST_MAX_REQUEST_BODY_BYTES, "/private/limit")

    config = load_runtime_config({})
    app = create_app(
        event_sink=lambda _event: None,
        lifecycle_event_sink=lambda _event: None,
    )

    assert config == RuntimeConfig(
        host=DEFAULT_API_HOST,
        port=DEFAULT_API_PORT,
        max_request_body_bytes=DEFAULT_MAX_REQUEST_BODY_BYTES,
    )
    assert app is not None


def test_container_host_and_legal_numeric_values_are_parsed_exactly() -> None:
    config = load_runtime_config(
        {
            PODCAST_API_HOST: "0.0.0.0",
            PODCAST_API_PORT: "65535",
            PODCAST_MAX_REQUEST_BODY_BYTES: "1048576",
        }
    )

    assert config.host == "0.0.0.0"
    assert config.port == 65_535
    assert config.max_request_body_bytes == 1_048_576


def test_numeric_lower_bounds_and_decimal_leading_zeroes_are_accepted() -> None:
    config = load_runtime_config(
        {
            PODCAST_API_PORT: "00001",
            PODCAST_MAX_REQUEST_BODY_BYTES: "00001",
        }
    )

    assert config.port == 1
    assert config.max_request_body_bytes == 1


@pytest.mark.parametrize(
    ("name", "value"),
    [
        (PODCAST_API_HOST, ""),
        (PODCAST_API_HOST, "localhost"),
        (PODCAST_API_HOST, " 127.0.0.1"),
        (PODCAST_API_HOST, "0.0.0.0 "),
        (PODCAST_API_HOST, PRIVATE_VALUE),
        (PODCAST_API_PORT, ""),
        (PODCAST_API_PORT, "0"),
        (PODCAST_API_PORT, "65536"),
        (PODCAST_API_PORT, "+8000"),
        (PODCAST_API_PORT, "-1"),
        (PODCAST_API_PORT, "8000.0"),
        (PODCAST_API_PORT, " 8000"),
        (PODCAST_API_PORT, "8000 "),
        (PODCAST_API_PORT, "8_000"),
        (PODCAST_API_PORT, "８０００"),
        (PODCAST_API_PORT, True),
        (PODCAST_API_PORT, 8000),
        (PODCAST_API_PORT, 8000.0),
        (PODCAST_MAX_REQUEST_BODY_BYTES, ""),
        (PODCAST_MAX_REQUEST_BODY_BYTES, "0"),
        (PODCAST_MAX_REQUEST_BODY_BYTES, "1048577"),
        (PODCAST_MAX_REQUEST_BODY_BYTES, "+16384"),
        (PODCAST_MAX_REQUEST_BODY_BYTES, "16384.0"),
        (PODCAST_MAX_REQUEST_BODY_BYTES, " 16384"),
        (PODCAST_MAX_REQUEST_BODY_BYTES, False),
        (PODCAST_MAX_REQUEST_BODY_BYTES, 16384),
    ],
)
def test_invalid_environment_values_fail_closed_with_one_fixed_exception(
    name: str,
    value: object,
) -> None:
    with pytest.raises(RuntimeConfigurationError) as caught:
        load_runtime_config({name: value})

    assert str(caught.value) == RUNTIME_CONFIGURATION_ERROR_MESSAGE
    assert PRIVATE_VALUE not in str(caught.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"host": "localhost"},
        {"port": True},
        {"port": "8000"},
        {"max_request_body_bytes": False},
        {"max_request_body_bytes": 1.0},
        {"unexpected": "value"},
    ],
)
def test_runtime_schema_is_strict_and_forbids_extra_fields(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        RuntimeConfig(**kwargs)  # type: ignore[arg-type]


def test_module_entry_constructs_the_configured_app_and_one_uvicorn_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel_app = object()
    constructed: list[int] = []
    calls: list[tuple[object, dict[str, object]]] = []

    def fake_create_app(*, max_request_body_bytes: int) -> object:
        constructed.append(max_request_body_bytes)
        return sentinel_app

    def fake_runner(app: object, **kwargs: object) -> None:
        calls.append((app, kwargs))

    monkeypatch.setattr(api_main, "create_app", fake_create_app)

    result = api_main.main(
        environ={
            PODCAST_API_HOST: "0.0.0.0",
            PODCAST_API_PORT: "8123",
            PODCAST_MAX_REQUEST_BODY_BYTES: "4096",
        },
        run_server=fake_runner,
    )

    assert result == 0
    assert constructed == [4096]
    assert calls == [
        (
            sentinel_app,
            {
                "host": "0.0.0.0",
                "port": 8123,
                "workers": 1,
                "reload": False,
                "access_log": False,
            },
        )
    ]


def test_module_entry_resolves_uvicorn_runner_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(api_main.uvicorn, "run", lambda app, **_kwargs: calls.append(app))

    result = api_main.main(environ={})

    assert result == 0
    assert len(calls) == 1


def test_invalid_config_stops_before_app_construction_or_uvicorn_and_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = io.StringIO()
    app_calls = 0
    server_calls = 0

    def forbidden_create_app(**_kwargs: object) -> object:
        nonlocal app_calls
        app_calls += 1
        raise AssertionError("app construction must not happen")

    def forbidden_runner(*_args: object, **_kwargs: object) -> None:
        nonlocal server_calls
        server_calls += 1
        raise AssertionError("server start must not happen")

    monkeypatch.setattr(api_main, "create_app", forbidden_create_app)
    result = api_main.main(
        environ={PODCAST_API_PORT: PRIVATE_VALUE},
        run_server=forbidden_runner,
        stderr=stream,
    )

    assert result != 0
    assert app_calls == 0
    assert server_calls == 0
    assert stream.getvalue() == RUNTIME_CONFIGURATION_ERROR_MESSAGE + "\n"
    for forbidden in (PRIVATE_VALUE, "traceback", "ValidationError", "PODCAST_API_PORT"):
        assert forbidden not in stream.getvalue()


def test_python_module_invalid_config_exits_nonzero_without_a_traceback() -> None:
    environment = os.environ.copy()
    environment[PODCAST_API_PORT] = PRIVATE_VALUE

    completed = subprocess.run(
        [sys.executable, "-m", "app.api"],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )

    assert completed.returncode != 0
    assert completed.stdout == ""
    assert completed.stderr == RUNTIME_CONFIGURATION_ERROR_MESSAGE + "\n"
    assert PRIVATE_VALUE not in completed.stderr
    assert "Traceback" not in completed.stderr
