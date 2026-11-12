"""Command-line entry point for the Day 3 ingestion pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.core.errors import safe_error_message
from app.ingestion.pipeline import process_transcript
from app.ingestion.transcripts import TranscriptFormatError


def build_parser() -> argparse.ArgumentParser:
    """Create the small, explicit Day 3 command-line interface."""
    parser = argparse.ArgumentParser(
        description="读取并清洗 TXT/JSON 播客转写，保存为统一 JSON。"
    )
    parser.add_argument("input", type=Path, help="UTF-8 编码的 .txt 或 .json 文件")
    parser.add_argument("--output", type=Path, help="输出路径，默认写入 data/processed/")
    return parser


def main() -> None:
    """Run ingestion and report a concise, user-readable result."""
    args = build_parser().parse_args()
    try:
        transcript, destination = process_transcript(args.input, args.output)
    except (TranscriptFormatError, OSError) as error:
        print(f"处理失败：{safe_error_message(error)}", file=sys.stderr)
        raise SystemExit(1) from None
    print(f"处理完成：{len(transcript.segments)} 个片段")
    print(f"输出文件：{destination}")


if __name__ == "__main__":
    main()
