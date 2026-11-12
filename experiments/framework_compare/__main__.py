"""Print the content-free ten-case comparison and optional local benchmark."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from experiments.framework_compare.comparison import (
    benchmark_normal_case,
    run_comparison_cases,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline Day 23 comparison.")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1_000)
    parser.add_argument("--skip-benchmark", action="store_true")
    arguments = parser.parse_args()
    payload: dict[str, object] = {
        "cases": [asdict(item) for item in run_comparison_cases()],
    }
    if not arguments.skip_benchmark:
        payload["latency"] = asdict(
            benchmark_normal_case(
                warmup_iterations=arguments.warmup,
                iterations=arguments.iterations,
            )
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
