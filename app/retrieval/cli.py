"""Command-line entry point for Day 8 Top-K retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

from pydantic import ValidationError

from app.core.config import ConfigError, EmbeddingSettings
from app.core.errors import safe_error_message
from app.indexing.embedding import (
    EmbeddingError,
    EmbeddingServiceError,
    OpenAICompatibleEmbedder,
    SyntheticHashEmbedder,
    create_embedding_client,
)
from app.indexing.index import IndexPersistenceError, VectorIndexError, load_vector_index
from app.retrieval.core import RetrievalError, Retriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="加载已有 Day 7 索引并执行 Day 8 Top-K 语义检索。",
    )
    parser.add_argument("index_directory", type=Path, help="已有索引目录")
    parser.add_argument("query", help="查询文本")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--embedder",
        choices=("api", "synthetic-hash"),
        default="api",
        help="必须与建库时的 Embedder 类型一致",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="真实 Embedding 配置文件（默认 .env）",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        _run_retrieval(args)
    except (
        ConfigError,
        EmbeddingError,
        EmbeddingServiceError,
        IndexPersistenceError,
        OSError,
        RetrievalError,
        ValidationError,
        ValueError,
        VectorIndexError,
    ) as error:
        print(f"Day 8 检索失败：{safe_error_message(error)}", file=sys.stderr)
        raise SystemExit(1) from None


def _run_retrieval(args: argparse.Namespace) -> None:
    loaded = load_vector_index(args.index_directory)
    if args.embedder == "synthetic-hash":
        embedder = SyntheticHashEmbedder(loaded.manifest.vector_dimension)
    else:
        settings = EmbeddingSettings.from_env(args.env_file)
        embedder = OpenAICompatibleEmbedder(
            create_embedding_client(settings),
            model_name=settings.model,
            batch_size=settings.batch_size,
            expected_dimension=settings.expected_dimension,
            document_task=settings.document_task,
            query_task=settings.query_task,
            embedding_type=settings.embedding_type,
            normalized=settings.normalized,
            truncate=settings.truncate,
        )

    started = perf_counter()
    results = Retriever(loaded, embedder).retrieve(args.query, args.top_k)
    elapsed = perf_counter() - started
    payload = {
        "query": args.query,
        "embedding_model": loaded.manifest.embedding_model,
        "query_embedding_calls": getattr(embedder, "query_call_count", 0),
        "document_embeddings_recomputed": False,
        "requested_top_k": args.top_k,
        "returned_count": len(results),
        "elapsed_seconds": elapsed,
        "token_usage": "not_recorded",
        "cost": "not_recorded",
        "score_semantics": "cosine_similarity_not_probability",
        "results": [result.model_dump(mode="json") for result in results],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
