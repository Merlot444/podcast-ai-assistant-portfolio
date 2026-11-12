"""Command-line entry point for the Day 4 analysis baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from app.core.config import ConfigError, Settings
from app.core.errors import safe_error_message
from app.core.llm import ModelServiceError, create_client
from app.generation.day4 import (
    Day4AnalysisError,
    analyze_processed_transcript,
    load_processed_transcript,
    save_day4_analysis,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_INPUT = _PROJECT_ROOT / "data" / "processed" / "demis-hassabis.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="对 Day 3 处理后的播客转写稿运行 Day 4 内容分析基线。",
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=_DEFAULT_INPUT,
        help="处理后的转写 JSON 路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="分析结果路径；默认保存到 data/processed/",
    )
    parser.add_argument(
        "--short-segments",
        type=int,
        help="只分析开头 N 段，用于验证短内容直接分析",
    )
    parser.add_argument(
        "--direct-max-characters",
        type=int,
        default=10_000,
        help="渲染后文本不超过该字符数时只调用一次模型（不是 Token 数）",
    )
    parser.add_argument(
        "--batch-max-characters",
        type=int,
        default=10_000,
        help="长文本每个分析批次的最大字符预算（不是 Token 数）",
    )
    return parser


def _default_output(input_path: Path, *, short: bool) -> Path:
    suffix = "-short-analysis.json" if short else "-analysis.json"
    return _PROJECT_ROOT / "data" / "processed" / f"{input_path.stem}{suffix}"


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        transcript = load_processed_transcript(args.input)
        if args.short_segments is not None:
            if args.short_segments <= 0:
                raise Day4AnalysisError("--short-segments 必须大于 0。")
            transcript = transcript.model_copy(
                update={"segments": transcript.segments[: args.short_segments]}
            )
        settings = Settings.from_env()
        client = create_client(settings)
        result = analyze_processed_transcript(
            client,
            settings,
            transcript,
            direct_max_characters=args.direct_max_characters,
            batch_max_characters=args.batch_max_characters,
        )
        output = args.output or _default_output(
            args.input,
            short=args.short_segments is not None,
        )
        save_day4_analysis(result, output)
    except (
        ConfigError,
        Day4AnalysisError,
        ModelServiceError,
        OSError,
        ValidationError,
        ValueError,
    ) as error:
        print(f"Day 4 分析失败：{safe_error_message(error)}", file=sys.stderr)
        raise SystemExit(1) from None

    metrics = result.metrics
    print(f"分析模式：{result.mode}")
    print(f"输入字符数：{metrics.input_characters}")
    print(f"原始片段数：{metrics.segment_count}")
    print(f"分析批次数：{metrics.batch_count}")
    print(f"模型调用次数：{metrics.model_call_count}")
    print(f"模型输出字符数：{metrics.output_characters}")
    print(f"总耗时：{metrics.total_duration_seconds:.3f} 秒")
    print(f"结果已保存：{output}")


if __name__ == "__main__":
    main()
