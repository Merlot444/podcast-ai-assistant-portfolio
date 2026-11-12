"""Command-line entry point for the offline Day 6 chunking pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from app.chunking.core import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, ChunkingError
from app.chunking.pipeline import process_chunks
from app.core.errors import safe_error_message


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 Day 3 处理后转写稿切分为可追溯的 Day 6 Chunk。",
    )
    parser.add_argument("input", type=Path, help="Day 3 处理后的 JSON 文件")
    parser.add_argument(
        "--output",
        type=Path,
        help="输出路径；默认写入 data/processed/<输入名>-chunks.json",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"每个 Chunk 的最大字符数（默认 {DEFAULT_CHUNK_SIZE}）",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help=f"同一章节相邻窗口的重叠字符数（默认 {DEFAULT_OVERLAP}）",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        result, destination = process_chunks(
            args.input,
            args.output,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
    except (ChunkingError, OSError, ValidationError, ValueError) as error:
        print(f"Day 6 切分失败：{safe_error_message(error)}", file=sys.stderr)
        raise SystemExit(1) from None

    print(f"切分完成：{result.input_segment_count} 个原始片段")
    print(f"生成 Chunk：{len(result.chunks)} 个")
    print(f"参数：chunk_size={result.chunk_size}, overlap={result.overlap}")
    print(f"输出文件：{destination}")


if __name__ == "__main__":
    main()
