"""Command-line build and minimal similarity-check entry points for Day 7."""

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
from app.indexing.index import (
    IndexPersistenceError,
    VectorIndexError,
    load_vector_index,
)
from app.indexing.pipeline import (
    IndexingInputError,
    build_and_save_index,
    load_chunked_transcript,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="构建或加载 Day 7 FAISS 向量索引。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="为 Day 6 Chunk 建立并保存索引")
    build.add_argument("input", type=Path, help="Day 6 Chunk JSON 文件")
    build.add_argument("output", type=Path, help="索引输出目录")
    build.add_argument(
        "--embedder",
        choices=("api", "synthetic-hash"),
        default="synthetic-hash",
        help="默认 synthetic-hash 离线验收；api 需显式选择并会调用真实 Embedding",
    )
    build.add_argument(
        "--synthetic-dimension",
        type=int,
        default=256,
        help="合成 Hash Embedder 的维度（默认 256）",
    )
    build.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="真实 Embedding 配置文件（默认 .env）",
    )
    build.add_argument(
        "--overwrite",
        action="store_true",
        help="显式覆盖已存在的同名索引产物",
    )

    query = subparsers.add_parser(
        "query",
        help="重新加载索引并运行一个最小相似检查",
    )
    query.add_argument("index_directory", type=Path, help="索引目录")
    query.add_argument("query", help="查询文本")
    query.add_argument("--top-k", type=int, default=3)
    query.add_argument(
        "--embedder",
        choices=("api", "synthetic-hash"),
        default="synthetic-hash",
        help="默认 synthetic-hash；api 需显式选择且必须与建库配置一致",
    )
    query.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="真实 Embedding 配置文件（默认 .env）",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build":
            _run_build(args)
        else:
            _run_query(args)
    except (
        ConfigError,
        EmbeddingError,
        EmbeddingServiceError,
        IndexPersistenceError,
        IndexingInputError,
        OSError,
        ValidationError,
        ValueError,
        VectorIndexError,
    ) as error:
        print(f"Day 7 索引操作失败：{safe_error_message(error)}", file=sys.stderr)
        raise SystemExit(1) from None


def _run_build(args: argparse.Namespace) -> None:
    embedder = _build_embedder(args)
    if args.embedder == "synthetic-hash":
        print(
            "警告：当前使用 synthetic-hash 合成向量，"
            "它不是语义 Embedding 模型，结果不能冒充真实实验。",
            file=sys.stderr,
        )

    started = perf_counter()
    vector_index, index_path, manifest_path = build_and_save_index(
        args.input,
        args.output,
        embedder=embedder,
        overwrite=args.overwrite,
    )
    elapsed = perf_counter() - started
    chunked = load_chunked_transcript(args.input)
    reloaded = load_vector_index(
        args.output,
        expected_chunks=chunked,
        expected_embedding_model=embedder.model_name,
        expected_dimension=vector_index.manifest.vector_dimension,
    )

    batch_count = getattr(
        embedder,
        "document_batch_count",
        getattr(embedder, "document_call_count", 0),
    )
    print(f"输入 Chunk：{reloaded.manifest.vector_count}")
    print(f"Embedding 模型：{reloaded.manifest.embedding_model}")
    print(f"向量维度：{reloaded.manifest.vector_dimension}")
    print(f"Embedding 批次：{batch_count}")
    print(f"索引类型：{reloaded.manifest.index_type}")
    print(f"距离度量：{reloaded.manifest.distance_metric}")
    print(f"L2 归一化：{reloaded.manifest.normalized}")
    print(f"构建和保存耗时：{elapsed:.6f} 秒")
    print(f"FAISS 索引：{index_path}（{index_path.stat().st_size} bytes）")
    print(f"Manifest：{manifest_path}（{manifest_path.stat().st_size} bytes）")
    print("重新加载：成功；未调用 embed_documents()")


def _run_query(args: argparse.Namespace) -> None:
    loaded = load_vector_index(args.index_directory)
    if args.embedder == "synthetic-hash":
        embedder = SyntheticHashEmbedder(loaded.manifest.vector_dimension)
    else:
        settings = EmbeddingSettings.from_env(args.env_file)
        load_vector_index(
            args.index_directory,
            expected_embedding_model=settings.model,
            expected_dimension=settings.expected_dimension,
            expected_document_task=settings.document_task,
            expected_query_task=settings.query_task,
            expected_embedding_type=settings.embedding_type,
            expected_provider_normalized=settings.normalized,
            expected_provider_truncate=settings.truncate,
        )
        embedder = OpenAICompatibleEmbedder(
            create_embedding_client(settings),
            model_name=settings.model,
            batch_size=settings.batch_size,
            expected_dimension=loaded.manifest.vector_dimension,
            document_task=settings.document_task,
            query_task=settings.query_task,
            embedding_type=settings.embedding_type,
            normalized=settings.normalized,
            truncate=settings.truncate,
        )

    results = loaded.similarity_check(
        args.query,
        embedder=embedder,
        top_k=args.top_k,
    )
    payload = {
        "query": args.query,
        "embedding_model": loaded.manifest.embedding_model,
        "document_embeddings_recomputed": False,
        "results": [result.metadata_dict() for result in results],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _build_embedder(args: argparse.Namespace):
    if args.embedder == "synthetic-hash":
        return SyntheticHashEmbedder(args.synthetic_dimension)
    settings = EmbeddingSettings.from_env(args.env_file)
    return OpenAICompatibleEmbedder(
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


if __name__ == "__main__":
    main()
