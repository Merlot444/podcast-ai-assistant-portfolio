"""Start one strictly configured API process with an explicit Ask mode."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping
from typing import TextIO

import uvicorn

from app.api.application import create_app
from app.api.runtime import (
    RUNTIME_CONFIGURATION_ERROR_MESSAGE,
    AgentAskRuntimeConfig,
    RuntimeConfigurationError,
    load_runtime_config,
)
from app.api.services import AskService
from app.bootstrap import (
    AskServiceBootstrapError,
    build_agent_api_services,
)


ServerRunner = Callable[..., object]
AskServiceBuilder = Callable[[AgentAskRuntimeConfig], AskService]


def main(
    *,
    environ: Mapping[str, object] | None = None,
    run_server: ServerRunner | None = None,
    ask_service_builder: AskServiceBuilder | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Validate runtime settings before constructing or starting Uvicorn."""
    environment = os.environ if environ is None else environ
    error_stream = sys.stderr if stderr is None else stderr
    try:
        config = load_runtime_config(environment)
        if config.ask_mode == "agent":
            if config.agent_ask is None:
                raise RuntimeConfigurationError(
                    RUNTIME_CONFIGURATION_ERROR_MESSAGE
                )
            if ask_service_builder is None:
                services = build_agent_api_services(config.agent_ask)
                app = create_app(
                    ask_service=services.ask_service,
                    retrieve_service=services.retrieve_service,
                    api_bearer_token=config.api_bearer_token,
                    max_request_body_bytes=config.max_request_body_bytes,
                )
            else:
                # Keep the focused Ask-only injection seam used by offline tests.
                ask_service = ask_service_builder(config.agent_ask)
                app = create_app(
                    ask_service=ask_service,
                    api_bearer_token=config.api_bearer_token,
                    max_request_body_bytes=config.max_request_body_bytes,
                )
        else:
            # Preserve the exact Day 24-26 safe default. In particular, do not
            # resolve or call any Agent/provider factory on this branch.
            app = create_app(
                max_request_body_bytes=config.max_request_body_bytes
            )
    except (RuntimeConfigurationError, AskServiceBootstrapError):
        error_stream.write(RUNTIME_CONFIGURATION_ERROR_MESSAGE + "\n")
        error_stream.flush()
        return 2

    resolved_runner = uvicorn.run if run_server is None else run_server
    resolved_runner(
        app,
        host=config.host,
        port=config.port,
        workers=1,
        reload=False,
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
