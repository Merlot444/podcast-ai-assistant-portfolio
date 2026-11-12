"""Run the complete synthetic MCP discovery-and-one-call experiment."""

from __future__ import annotations

import asyncio

from experiments.mcp_minimal import (
    SYNTHETIC_EPISODE_ID,
    build_synthetic_dispatcher,
    create_server,
    discover_and_call_metadata,
)


async def _run() -> None:
    dispatcher = build_synthetic_dispatcher()
    result = await discover_and_call_metadata(
        create_server(dispatcher),
        SYNTHETIC_EPISODE_ID,
    )
    print(result.model_dump_json(indent=2))


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
