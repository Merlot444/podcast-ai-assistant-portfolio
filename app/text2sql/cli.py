"""Offline Day 20 CLI for an explicit candidate and synthetic read-only copy."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from app.core.errors import safe_error_message
from app.text2sql.executor import QueryExecutionError
from app.text2sql.generator import MappingQueryGenerator, QueryGenerationError
from app.text2sql.schemas import GeneratedQuery, Text2SQLRequest
from app.text2sql.service import create_sqlite_text2sql_service
from app.text2sql.validator import QueryValidationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and optionally execute one fixed SQL candidate against an "
            "explicit synthetic read-only SQLite copy."
        )
    )
    parser.add_argument("question")
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--read-only-sqlite", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        request = Text2SQLRequest(question=args.question)
        candidate = GeneratedQuery.model_validate_json(args.candidate_json)
        generator = MappingQueryGenerator({request.question: candidate})
        service = create_sqlite_text2sql_service(
            generator,
            args.read_only_sqlite,
        )
        try:
            output = service.run(request, dry_run=args.dry_run)
        finally:
            service.close()
    except ValidationError:
        print(json.dumps({"error": "输入契约无效。"}, ensure_ascii=False))
        return 2
    except (
        ValueError,
        QueryGenerationError,
        QueryValidationError,
        QueryExecutionError,
    ) as error:
        print(json.dumps({"error": safe_error_message(error)}, ensure_ascii=False))
        return 2
    print(output.model_dump_json())
    return 0
