"""Discoverable Day 10 command for one complete RAG 1.0 run."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from pydantic import ValidationError

from app.chunking.core import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, ChunkingError
from app.core.config import ConfigError, EmbeddingSettings, Settings
from app.core.llm import ModelServiceError, create_client
from app.indexing.embedding import (
    Embedder,
    EmbeddingError,
    EmbeddingServiceError,
    OpenAICompatibleEmbedder,
    SyntheticHashEmbedder,
    create_embedding_client,
)
from app.indexing.index import IndexPersistenceError, VectorIndexError
from app.ingestion.transcripts import TranscriptFormatError
from app.rag.core import GroundedAnswerError
from app.rag.events import JsonLineEventSink
from app.rag.pipeline import RagPipelineRequest, run_rag_pipeline
from app.retrieval.core import RetrievalError
from app.schemas.rag import GroundedCitation
from app.schemas.rag_pipeline import RagRunResult


@dataclass(frozen=True)
class _Runtime:
    embedder: Embedder
    answer_client: object
    answer_model: str


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit source/index-policy Day 10 interface."""
    parser = argparse.ArgumentParser(
        description=(
            "串联导入、Chunk、索引、Top-K 检索和受证据约束回答。"
            "业务结果写 stdout，JSONL 结构化日志写 stderr。"
        )
    )
    parser.add_argument("input", type=Path, help="原始 TXT/JSON 或 Day 3 JSON")
    parser.add_argument("question", help="本次只基于节目证据回答的问题")
    parser.add_argument(
        "--input-kind",
        choices=("raw", "processed"),
        default="raw",
        help="raw 使用 Day 3 导入；processed 加载已有统一 Schema（默认 raw）",
    )
    parser.add_argument(
        "--index-directory",
        type=Path,
        required=True,
        help="要复用或构建的 Day 7 索引目录",
    )
    parser.add_argument(
        "--index-mode",
        choices=("reuse", "build"),
        default="reuse",
        help="reuse 安全复用匹配索引；build 显式新建（默认 reuse）",
    )
    parser.add_argument(
        "--overwrite-index",
        action="store_true",
        help=(
            "仅配合 --index-mode build 显式替换；"
            "中断时两文件不能保证原子切换，优先使用新目录"
        ),
    )
    parser.add_argument("--top-k", type=_positive_int, default=5)
    parser.add_argument(
        "--chunk-size",
        type=_positive_int,
        default=DEFAULT_CHUNK_SIZE,
    )
    parser.add_argument("--overlap", type=_nonnegative_int, default=DEFAULT_OVERLAP)
    parser.add_argument(
        "--embedder",
        choices=("api", "synthetic-hash"),
        default="api",
        help=(
            "synthetic-hash 只替换 Embedding 并用于机械验收；"
            "回答仍使用配置的模型服务"
        ),
    )
    parser.add_argument(
        "--synthetic-dimension",
        type=_positive_int,
        default=256,
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="真实服务配置文件（默认 .env）",
    )
    parser.add_argument(
        "--answer-model",
        help="覆盖 LLM_MODEL_STRONG；默认使用 .env 中的强模型",
    )
    parser.add_argument(
        "--answer-max-tokens",
        type=_positive_int,
        default=1_200,
    )
    parser.add_argument(
        "--output-format",
        choices=("json", "text"),
        default="json",
        help="stdout 业务输出格式（默认 json）",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the CLI with safe categorized failures and refusal exit code zero."""
    cli_started = perf_counter()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.overwrite_index and args.index_mode != "build":
        parser.error("--overwrite-index 只能与 --index-mode build 一起使用")
    if args.overlap >= args.chunk_size:
        parser.error("--overlap 必须小于 --chunk-size")

    sink = JsonLineEventSink(sys.stderr)
    try:
        runtime = _create_runtime(args)
        result = run_rag_pipeline(
            RagPipelineRequest(
                input_path=args.input,
                input_kind=args.input_kind,
                query=args.question,
                top_k=args.top_k,
                index_directory=args.index_directory,
                index_mode=args.index_mode,
                answer_model=runtime.answer_model,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                overwrite_index=args.overwrite_index,
                answer_max_tokens=args.answer_max_tokens,
            ),
            embedder=runtime.embedder,
            answer_client=runtime.answer_client,  # type: ignore[arg-type]
            event_sink=sink,
        )
    except Exception as error:
        exit_code, category, safe_message = _classify_error(error)
        sink.emit(
            {
                "event": "rag_cli_error",
                "stage": "cli",
                "status": "failed",
                "elapsed_seconds": max(0.0, perf_counter() - cli_started),
                "error_type": type(error).__name__,
                "error_category": category,
                "error_message": safe_message,
            }
        )
        raise SystemExit(exit_code) from None

    if args.output_format == "json":
        print(
            json.dumps(
                result.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(_format_text_result(result))


def _create_runtime(args: argparse.Namespace) -> _Runtime:
    answer_settings = Settings.from_env(args.env_file)
    answer_model = (
        args.answer_model.strip()
        if args.answer_model is not None
        else answer_settings.model_strong
    )
    answer_client = create_client(answer_settings)

    if args.embedder == "synthetic-hash":
        embedder: Embedder = SyntheticHashEmbedder(args.synthetic_dimension)
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
    return _Runtime(
        embedder=embedder,
        answer_client=answer_client,
        answer_model=answer_model,
    )


def _format_text_result(result: RagRunResult) -> str:
    answer = result.answer
    lines = [
        f"问题：{result.question}",
        f"回答：{answer.answer}",
        "证据充分：" + ("是" if answer.has_sufficient_evidence else "否"),
        f"索引：{result.index_action}，{result.vector_count} 条，"
        f"{result.vector_dimension} 维",
        f"检索：Top-{result.top_k}，返回 {result.retrieved_count} 条",
        (
            "调用：document_embedding="
            f"{result.metrics.document_embedding_calls}，"
            f"query_embedding={result.metrics.query_embedding_calls}，"
            f"answer_model={result.metrics.answer_model_calls}"
        ),
        f"引用：{len(answer.citations)} 条",
    ]
    for citation in answer.citations:
        lines.extend(_format_citation(citation))
    return "\n".join(lines)


def _format_citation(citation: GroundedCitation) -> list[str]:
    speakers = "；".join(citation.speakers) or "未知"
    sections = "；".join(citation.section_titles) or "未知"
    source_ranges = "；".join(
        (
            f"{span.segment_id}"
            f"[{span.start_char}:{span.end_char})"
        )
        for span in citation.source_spans
    )
    completeness = (
        "完整"
        if citation.time_range_complete
        else "不完整；结束值只表示最后已知来源覆盖点"
    )
    return [
        (
            f"- {citation.evidence_id}: {citation.chunk_id}，"
            f"rank={citation.rank}，cosine_score={citation.score:.6f}"
            "（非概率）"
        ),
        f"  说话人：{speakers}",
        f"  章节：{sections}",
        (
            "  来源时间："
            f"{_format_seconds(citation.start_time)}～"
            f"{_format_seconds(citation.end_time)}（{completeness}）"
        ),
        (
            f"  来源：{citation.source_file}，片段 "
            f"{citation.source_segment_start_index}～"
            f"{citation.source_segment_end_index}"
        ),
        f"  字符范围：{source_ranges}",
    ]


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "未知"
    whole_seconds = int(value)
    hours, remainder = divmod(whole_seconds, 3_600)
    minutes, seconds = divmod(remainder, 60)
    fraction = value - whole_seconds
    if fraction:
        return f"{hours:02d}:{minutes:02d}:{seconds + fraction:06.3f}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("必须是整数") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("必须是整数") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("不能小于 0")
    return parsed


def _classify_error(error: Exception) -> tuple[int, str, str]:
    if isinstance(error, ConfigError):
        return (
            2,
            "configuration_failure",
            "真实服务配置缺失或无效；请按 .env.example 检查本地配置。",
        )
    if isinstance(error, EmbeddingServiceError):
        return (
            3,
            "embedding_service_failure",
            "查询或文档 Embedding 服务调用失败；未生成业务答案。",
        )
    if isinstance(error, EmbeddingError):
        return (
            3,
            "embedding_contract_failure",
            "Embedding 返回数量、维度或数值未通过校验。",
        )
    if isinstance(error, RetrievalError):
        return (
            3,
            "retrieval_failure",
            "检索输入、查询向量或索引配置不匹配。",
        )
    if isinstance(error, (IndexPersistenceError, VectorIndexError)):
        return (
            3,
            "index_failure",
            "索引缺失、损坏、已存在或与当前输入/模型不匹配。",
        )
    if isinstance(error, ModelServiceError):
        return (
            4,
            "answer_model_service_failure",
            "回答模型服务调用失败；未把它当作证据不足拒答。",
        )
    if isinstance(error, GroundedAnswerError):
        return (
            4,
            "answer_contract_failure",
            "回答模型输出未通过证据编号或业务结构校验。",
        )
    if isinstance(
        error,
        (
            TranscriptFormatError,
            ChunkingError,
            ValidationError,
            OSError,
            ValueError,
        ),
    ):
        return (
            2,
            "input_failure",
            "输入文件、问题或 Chunk 参数无效；请检查命令参数和数据格式。",
        )
    return (
        1,
        "unexpected_failure",
        "RAG 运行出现未预期错误；已隐藏可能敏感的内部细节。",
    )


if __name__ == "__main__":
    main()
