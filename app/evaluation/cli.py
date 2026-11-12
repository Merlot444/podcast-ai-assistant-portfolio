"""CLI for frozen Day 11 validation and explicit Day 12–16 experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import dotenv_values
from pydantic import ValidationError

from app.chunking.core import ChunkingError
from app.chunking.pipeline import load_processed_transcript
from app.core.config import ConfigError, EmbeddingSettings, Settings
from app.core.errors import safe_error_message
from app.core.llm import ModelServiceError, create_client
from app.evaluation.answer_quality import (
    DEFAULT_ANSWER_QUALITY_OUTPUT,
    AnswerQualityExperimentError,
    AnswerQualityPersistenceError,
    finalize_answer_quality_experiment,
    run_answer_generation,
)
from app.evaluation.baseline import build_vector_baseline_result, file_sha256
from app.evaluation.chunk_experiment import (
    DEFAULT_CHUNK_EXPERIMENT_OUTPUT,
    ChunkExperimentError,
    ChunkExperimentPersistenceError,
    run_chunk_parameter_experiment,
)
from app.evaluation.core import (
    DEFAULT_EVALUATION_DATASET,
    DEFAULT_EVALUATION_INDEX_DIRECTORY,
    DEFAULT_RETRIEVAL_BASELINE,
    DEFAULT_EVALUATION_TRANSCRIPT,
    EvaluationDatasetError,
    EvaluationSourceError,
    load_evaluation_dataset,
    validate_evaluation_sources,
)
from app.evaluation.persistence import BaselinePersistenceError, save_retrieval_baseline
from app.evaluation.retrieval_experiment import (
    DEFAULT_CHUNK_CATALOG,
    DEFAULT_DAY13_EXPERIMENT,
    DEFAULT_RETRIEVAL_EXPERIMENT_OUTPUT,
    RetrievalExperimentError,
    RetrievalExperimentPersistenceError,
    run_retrieval_experiment,
)
from app.evaluation.runner import RetrievalEvaluationError, run_retrieval_evaluation
from app.evaluation.rerank_experiment import (
    DEFAULT_RERANK_EXPERIMENT_OUTPUT,
    RerankExperimentError,
    RerankExperimentPersistenceError,
    run_rerank_experiment,
)
from app.evaluation.rag_v2_freeze import (
    DEFAULT_DAY17_INPUTS,
    DEFAULT_RAG_V2_FREEZE_OUTPUT,
    RagV2FreezeError,
    RagV2FreezePersistenceError,
    freeze_rag_v2,
)
from app.evaluation.v2_retrieval_experiment import (
    DEFAULT_EXPANDED_EPISODE_FILES,
    DEFAULT_V2_DATASET,
    DEFAULT_V2_RETRIEVAL_OUTPUT,
    ExpandedEpisodeFiles,
    V2RetrievalExperimentError,
    V2RetrievalExperimentPersistenceError,
    run_v2_retrieval_experiment,
)
from app.indexing import OpenAICompatibleEmbedder, load_vector_index
from app.indexing.embedding import EmbeddingError, EmbeddingServiceError, create_embedding_client
from app.indexing.index import INDEX_FILENAME, MANIFEST_FILENAME, IndexPersistenceError
from app.retrieval import RetrievalError, Retriever
from app.retrieval.rerank import RerankError, RerankServiceError
from app.retrieval.vector import VectorRetrieverAdapter
from app.schemas.evaluation_result import InputArtifactHashes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="校验 Day 11 固定评测集，运行 Day 12～16 实验，或离线冻结 Day 17。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser(
        "validate",
        help="校验 Schema、稳定来源片段和当前基线 Chunk，不计算检索指标",
    )
    validate.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_EVALUATION_DATASET,
        help="评测集 JSON；默认 data/evaluation/eval_dataset_v1.json",
    )
    validate.add_argument(
        "--transcript",
        type=Path,
        default=DEFAULT_EVALUATION_TRANSCRIPT,
        help="处理后转写 JSON；只读且默认使用当前真实语料",
    )
    validate.add_argument(
        "--index-directory",
        type=Path,
        default=DEFAULT_EVALUATION_INDEX_DIRECTORY,
        help="当前基线索引目录；只读加载并校验 manifest 与 FAISS",
    )
    baseline = subparsers.add_parser(
        "baseline",
        help="显式加载 .env，在冻结题集上运行一次真实 Top-8 向量检索基线",
    )
    baseline.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_EVALUATION_DATASET,
    )
    baseline.add_argument(
        "--transcript",
        type=Path,
        default=DEFAULT_EVALUATION_TRANSCRIPT,
    )
    baseline.add_argument(
        "--index-directory",
        type=Path,
        default=DEFAULT_EVALUATION_INDEX_DIRECTORY,
    )
    baseline.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="仅此正式基线子命令加载的 Embedding 配置",
    )
    baseline.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RETRIEVAL_BASELINE,
    )
    baseline.add_argument(
        "--overwrite",
        action="store_true",
        help="显式允许替换已有结果；默认安全失败",
    )
    experiment = subparsers.add_parser(
        "chunk-experiment",
        help="显式加载 .env，构建两组新索引并运行 Day 13 三配置真实对照",
    )
    experiment.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_EVALUATION_DATASET,
    )
    experiment.add_argument(
        "--transcript",
        type=Path,
        default=DEFAULT_EVALUATION_TRANSCRIPT,
    )
    experiment.add_argument(
        "--index-directory",
        type=Path,
        default=DEFAULT_EVALUATION_INDEX_DIRECTORY,
        help="只读复用的 1000/200 Day 12 基线索引目录",
    )
    experiment.add_argument(
        "--baseline-result",
        type=Path,
        default=DEFAULT_RETRIEVAL_BASELINE,
    )
    experiment.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="仅此正式实验子命令加载的 Embedding 配置",
    )
    experiment.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CHUNK_EXPERIMENT_OUTPUT,
    )
    experiment.add_argument(
        "--overwrite",
        action="store_true",
        help="显式允许替换已有正式结果；不会覆盖 alternative 索引",
    )
    retrieval_experiment = subparsers.add_parser(
        "retrieval-experiment",
        help="运行 Day 14 Vector/BM25/Hybrid-RRF 固定三路真实检索对照",
    )
    retrieval_experiment.add_argument("--dataset", type=Path, default=DEFAULT_EVALUATION_DATASET)
    retrieval_experiment.add_argument("--transcript", type=Path, default=DEFAULT_EVALUATION_TRANSCRIPT)
    retrieval_experiment.add_argument(
        "--chunk-catalog", type=Path, default=DEFAULT_CHUNK_CATALOG
    )
    retrieval_experiment.add_argument(
        "--index-directory", type=Path, default=DEFAULT_EVALUATION_INDEX_DIRECTORY
    )
    retrieval_experiment.add_argument(
        "--baseline-result", type=Path, default=DEFAULT_RETRIEVAL_BASELINE
    )
    retrieval_experiment.add_argument(
        "--day13-result", type=Path, default=DEFAULT_DAY13_EXPERIMENT
    )
    retrieval_experiment.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="仅此正式实验子命令加载的 Embedding 配置",
    )
    retrieval_experiment.add_argument(
        "--output", type=Path, default=DEFAULT_RETRIEVAL_EXPERIMENT_OUTPUT
    )
    retrieval_experiment.add_argument(
        "--overwrite",
        action="store_true",
        help="显式允许替换已有正式结果；默认安全失败",
    )
    expanded = subparsers.add_parser(
        "expanded-retrieval-experiment",
        help="参数化运行三节目 80 题 Vector/BM25/Hybrid-RRF 对照",
    )
    expanded.add_argument("--dataset-v1", type=Path, default=DEFAULT_EVALUATION_DATASET)
    expanded.add_argument("--dataset-v2", type=Path, default=DEFAULT_V2_DATASET)
    expanded.add_argument("--env-file", type=Path, default=Path(".env"))
    for files in DEFAULT_EXPANDED_EPISODE_FILES:
        prefix = files.episode_slug.replace("25", "-25")
        expanded.add_argument(
            f"--{prefix}-transcript",
            dest=f"{files.episode_slug}_transcript",
            type=Path,
            default=files.processed_transcript,
        )
        expanded.add_argument(
            f"--{prefix}-chunks",
            dest=f"{files.episode_slug}_chunks",
            type=Path,
            default=files.chunk_catalog,
        )
        expanded.add_argument(
            f"--{prefix}-index",
            dest=f"{files.episode_slug}_index",
            type=Path,
            default=files.index_directory,
        )
    expanded.add_argument("--output", type=Path, default=DEFAULT_V2_RETRIEVAL_OUTPUT)
    expanded.add_argument("--overwrite", action="store_true")
    rerank_experiment = subparsers.add_parser(
        "rerank-experiment",
        help="复用 Day 14 两路冻结 Top-20，运行 Day 15 Jina Rerank 对照",
    )
    rerank_experiment.add_argument("--dataset", type=Path, default=DEFAULT_EVALUATION_DATASET)
    rerank_experiment.add_argument("--transcript", type=Path, default=DEFAULT_EVALUATION_TRANSCRIPT)
    rerank_experiment.add_argument("--chunk-catalog", type=Path, default=DEFAULT_CHUNK_CATALOG)
    rerank_experiment.add_argument(
        "--index-directory", type=Path, default=DEFAULT_EVALUATION_INDEX_DIRECTORY
    )
    rerank_experiment.add_argument(
        "--baseline-result", type=Path, default=DEFAULT_RETRIEVAL_BASELINE
    )
    rerank_experiment.add_argument(
        "--day13-result", type=Path, default=DEFAULT_DAY13_EXPERIMENT
    )
    rerank_experiment.add_argument(
        "--day14-result", type=Path, default=DEFAULT_RETRIEVAL_EXPERIMENT_OUTPUT
    )
    rerank_experiment.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="仅此正式实验子命令读取 RERANK_API_KEY 或既有 EMBEDDING_API_KEY",
    )
    rerank_experiment.add_argument(
        "--output", type=Path, default=DEFAULT_RERANK_EXPERIMENT_OUTPUT
    )
    rerank_experiment.add_argument(
        "--overwrite",
        action="store_true",
        help="显式允许替换已有正式结果；默认安全失败",
    )
    rerank_experiment.add_argument(
        "--prior-failed-rerank-attempts",
        type=int,
        default=0,
        help="记录正式成功运行前失败的 Rerank endpoint 尝试次数",
    )
    rerank_experiment.add_argument(
        "--prior-successful-synthetic-diagnostic-requests",
        type=int,
        default=0,
        help="记录正式运行前成功但不计入质量实验的合成诊断请求次数",
    )
    answer_generation = subparsers.add_parser(
        "answer-quality-generate",
        help="只复用 Day 14 Vector Top-5，顺序生成并冻结 Day 16 的 20 份回答",
    )
    _add_day16_frozen_input_arguments(answer_generation)
    answer_generation.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="仅生成阶段读取的既有 DeepSeek 回答模型配置",
    )
    answer_generation.add_argument(
        "--generation-output",
        type=Path,
        required=True,
        help="评分前的无真值生成冻结；默认拒绝覆盖",
    )
    answer_generation.add_argument(
        "--overwrite",
        action="store_true",
        help="显式允许替换生成冻结；正式运行不应使用",
    )
    answer_finalize = subparsers.add_parser(
        "answer-quality-finalize",
        help="在全部回答冻结后载入单审阅者复核，生成严格 Day 16 正式结果",
    )
    _add_day16_frozen_input_arguments(answer_finalize)
    answer_finalize.add_argument("--generation", type=Path, required=True)
    answer_finalize.add_argument("--review", type=Path, required=True)
    answer_finalize.add_argument(
        "--output", type=Path, default=DEFAULT_ANSWER_QUALITY_OUTPUT
    )
    answer_finalize.add_argument(
        "--overwrite",
        action="store_true",
        help="显式允许替换已有正式结果；默认安全失败",
    )
    rag_v2_freeze = subparsers.add_parser(
        "freeze-rag-v2",
        help="完全离线复用 Day 12～16 与 80 题结果，生成 Day 17 RAG 2.0 冻结",
    )
    rag_v2_freeze.add_argument(
        "--dataset-v1", type=Path, default=DEFAULT_DAY17_INPUTS["eval_dataset_v1"]
    )
    rag_v2_freeze.add_argument(
        "--day12", type=Path, default=DEFAULT_DAY17_INPUTS["day12_vector_baseline"]
    )
    rag_v2_freeze.add_argument(
        "--day13", type=Path, default=DEFAULT_DAY17_INPUTS["day13_chunk_experiment"]
    )
    rag_v2_freeze.add_argument(
        "--day14", type=Path, default=DEFAULT_DAY17_INPUTS["day14_retrieval_experiment"]
    )
    rag_v2_freeze.add_argument(
        "--day15", type=Path, default=DEFAULT_DAY17_INPUTS["day15_rerank_experiment"]
    )
    rag_v2_freeze.add_argument(
        "--day16", type=Path, default=DEFAULT_DAY17_INPUTS["day16_answer_quality"]
    )
    rag_v2_freeze.add_argument(
        "--dataset-v2", type=Path, default=DEFAULT_DAY17_INPUTS["eval_dataset_v2_pilot"]
    )
    rag_v2_freeze.add_argument(
        "--v2-retrieval",
        type=Path,
        default=DEFAULT_DAY17_INPUTS["eval_dataset_v2_retrieval_experiment"],
    )
    rag_v2_freeze.add_argument(
        "--output", type=Path, default=DEFAULT_RAG_V2_FREEZE_OUTPUT
    )
    rag_v2_freeze.add_argument(
        "--overwrite",
        action="store_true",
        help="显式允许替换已有正式结果；默认安全失败",
    )
    return parser


def _add_day16_frozen_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, default=DEFAULT_EVALUATION_DATASET)
    parser.add_argument("--transcript", type=Path, default=DEFAULT_EVALUATION_TRANSCRIPT)
    parser.add_argument("--chunk-catalog", type=Path, default=DEFAULT_CHUNK_CATALOG)
    parser.add_argument(
        "--index-directory", type=Path, default=DEFAULT_EVALUATION_INDEX_DIRECTORY
    )
    parser.add_argument("--baseline-result", type=Path, default=DEFAULT_RETRIEVAL_BASELINE)
    parser.add_argument("--day13-result", type=Path, default=DEFAULT_DAY13_EXPERIMENT)
    parser.add_argument(
        "--day14-result", type=Path, default=DEFAULT_RETRIEVAL_EXPERIMENT_OUTPUT
    )
    parser.add_argument(
        "--day15-result", type=Path, default=DEFAULT_RERANK_EXPERIMENT_OUTPUT
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            payload = _run_validation(args)
        elif args.command == "baseline":
            payload = _run_baseline(args)
        elif args.command == "chunk-experiment":
            payload = _run_chunk_experiment(args)
        elif args.command == "retrieval-experiment":
            payload = _run_retrieval_experiment(args)
        elif args.command == "expanded-retrieval-experiment":
            payload = _run_v2_retrieval_experiment(args)
        elif args.command == "rerank-experiment":
            payload = _run_rerank_experiment(args)
        elif args.command == "answer-quality-generate":
            payload = _run_answer_generation(args)
        elif args.command == "answer-quality-finalize":
            payload = _run_answer_finalize(args)
        else:
            payload = _run_rag_v2_freeze(args)
    except (
        BaselinePersistenceError,
        AnswerQualityExperimentError,
        AnswerQualityPersistenceError,
        ChunkingError,
        ChunkExperimentError,
        ChunkExperimentPersistenceError,
        ConfigError,
        EmbeddingError,
        EmbeddingServiceError,
        EvaluationDatasetError,
        RetrievalEvaluationError,
        RetrievalExperimentError,
        RetrievalExperimentPersistenceError,
        V2RetrievalExperimentError,
        V2RetrievalExperimentPersistenceError,
        RerankError,
        RerankExperimentError,
        RerankExperimentPersistenceError,
        RagV2FreezeError,
        RagV2FreezePersistenceError,
        RerankServiceError,
        RetrievalError,
        EvaluationSourceError,
        IndexPersistenceError,
        ModelServiceError,
        OSError,
        ValidationError,
    ) as error:
        label = {
            "validate": "Day 11 评测集校验",
            "baseline": "Day 12 基线",
            "chunk-experiment": "Day 13 Chunk 实验",
            "retrieval-experiment": "Day 14 检索实验",
            "expanded-retrieval-experiment": "80 题扩展检索实验",
            "rerank-experiment": "Day 15 Rerank 实验",
            "answer-quality-generate": "Day 16 回答生成",
            "answer-quality-finalize": "Day 16 回答质量评测",
            "freeze-rag-v2": "Day 17 RAG 2.0 冻结",
        }[args.command]
        print(f"{label}失败：{safe_error_message(error)}", file=sys.stderr)
        raise SystemExit(1) from None

    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _run_validation(args: argparse.Namespace) -> dict[str, object]:
    dataset = load_evaluation_dataset(args.dataset)
    transcript = load_processed_transcript(args.transcript)
    vector_index = load_vector_index(args.index_directory)
    report = validate_evaluation_sources(
        dataset,
        transcript=transcript,
        manifest=vector_index.manifest,
    )
    return report.model_dump(mode="json")


def _run_baseline(args: argparse.Namespace) -> dict[str, object]:
    dataset_path = Path(args.dataset)
    transcript_path = Path(args.transcript)
    index_directory = Path(args.index_directory)
    manifest_path = index_directory / MANIFEST_FILENAME
    faiss_path = index_directory / INDEX_FILENAME
    output_path = Path(args.output)
    protected_inputs = {
        path.resolve()
        for path in (dataset_path, transcript_path, manifest_path, faiss_path)
    }
    if output_path.resolve() in protected_inputs:
        raise BaselinePersistenceError("基线输出不能覆盖任何评测输入产物")
    if output_path.exists() and not args.overwrite:
        raise BaselinePersistenceError(
            f"目标基线已存在：{output_path}；不会静默覆盖"
        )
    before_hashes = InputArtifactHashes(
        dataset_file_sha256=file_sha256(dataset_path),
        manifest_file_sha256=file_sha256(manifest_path),
        faiss_file_sha256=file_sha256(faiss_path),
    )

    dataset = load_evaluation_dataset(dataset_path)
    transcript = load_processed_transcript(transcript_path)
    vector_index = load_vector_index(index_directory)
    validate_evaluation_sources(
        dataset,
        transcript=transcript,
        manifest=vector_index.manifest,
    )

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
    retriever = VectorRetrieverAdapter(Retriever(vector_index, embedder))
    run = run_retrieval_evaluation(dataset, retriever=retriever)

    after_hashes = InputArtifactHashes(
        dataset_file_sha256=file_sha256(dataset_path),
        manifest_file_sha256=file_sha256(manifest_path),
        faiss_file_sha256=file_sha256(faiss_path),
    )
    if after_hashes != before_hashes:
        raise ValueError("正式运行前后输入产物哈希发生变化，拒绝保存基线")

    result = build_vector_baseline_result(
        dataset=dataset,
        run=run,
        manifest=vector_index.manifest,
        input_artifact_hashes=before_hashes,
        query_embedding_calls=embedder.query_call_count,
        document_embedding_calls=embedder.document_batch_count,
    )
    destination = save_retrieval_baseline(
        result,
        args.output,
        overwrite=args.overwrite,
    )
    return {
        "evaluation_id": result.evaluation_id,
        "output": str(destination),
        "query_count": result.query_count,
        "query_embedding_calls": result.query_embedding_calls,
        "document_embedding_calls": result.document_embedding_calls,
        "answer_model_calls": result.answer_model_calls,
        "elapsed_seconds": result.elapsed_seconds,
        "aggregate_metrics": {
            str(k): metric.model_dump(mode="json")
            for k, metric in result.aggregate_metrics.items()
        },
    }


def _run_chunk_experiment(args: argparse.Namespace) -> dict[str, object]:
    settings = EmbeddingSettings.from_env(args.env_file)
    result = run_chunk_parameter_experiment(
        settings=settings,
        dataset_path=args.dataset,
        transcript_path=args.transcript,
        baseline_index_directory=args.index_directory,
        baseline_result_path=args.baseline_result,
        output_path=args.output,
        overwrite=args.overwrite,
    )
    return {
        "experiment_id": result.experiment_id,
        "output": str(Path(args.output)),
        "selected_default": result.selected_default.model_dump(mode="json"),
        "elapsed_seconds": result.elapsed_seconds,
        "configurations": [
            {
                "config_id": item.config_id,
                "chunk_size": item.chunk_size,
                "overlap": item.overlap,
                "chunk_count": item.chunk_count,
                "document_embedding_calls": item.document_embedding_calls,
                "query_embedding_calls": item.query_embedding_calls,
                "build_elapsed_seconds": item.build_elapsed_seconds,
                "query_elapsed_seconds": item.query_elapsed_seconds,
                "aggregate_metrics": {
                    str(k): metric.model_dump(mode="json")
                    for k, metric in item.aggregate_metrics.items()
                },
            }
            for item in result.configurations
        ],
    }


def _run_retrieval_experiment(args: argparse.Namespace) -> dict[str, object]:
    settings = EmbeddingSettings.from_env(args.env_file)
    result = run_retrieval_experiment(
        settings=settings,
        dataset_path=args.dataset,
        transcript_path=args.transcript,
        chunk_catalog_path=args.chunk_catalog,
        index_directory=args.index_directory,
        day12_baseline_path=args.baseline_result,
        day13_experiment_path=args.day13_result,
        output_path=args.output,
        overwrite=args.overwrite,
    )
    return {
        "experiment_id": result.experiment_id,
        "output": str(Path(args.output)),
        "recommendation": result.recommendation.model_dump(mode="json"),
        "call_counts": result.call_counts.model_dump(mode="json"),
        "latency_and_size": result.latency_and_size.model_dump(mode="json"),
        "methods": [
            {
                "retriever_name": method.retriever_name,
                "aggregate_metrics": {
                    str(k): metric.model_dump(mode="json")
                    for k, metric in method.aggregate_metrics.items()
                },
            }
            for method in result.methods
        ],
    }


def _run_v2_retrieval_experiment(args: argparse.Namespace) -> dict[str, object]:
    settings = EmbeddingSettings.from_env(args.env_file)
    files = tuple(
        ExpandedEpisodeFiles(
            episode_slug=slug,
            processed_transcript=getattr(args, f"{slug}_transcript"),
            chunk_catalog=getattr(args, f"{slug}_chunks"),
            index_directory=getattr(args, f"{slug}_index"),
        )
        for slug in ("demis", "ilya25", "mark25")
    )
    result = run_v2_retrieval_experiment(
        settings=settings,
        base_dataset_path=args.dataset_v1,
        v2_dataset_path=args.dataset_v2,
        episode_files=files,
        output_path=args.output,
        overwrite=args.overwrite,
    )
    return {
        "experiment_id": result.experiment_id,
        "output": str(Path(args.output)),
        "call_counts": result.call_counts.model_dump(mode="json"),
        "methods": {
            method.retriever_name: {
                str(k): metric.model_dump(mode="json")
                for k, metric in method.aggregate_metrics.items()
            }
            for method in result.methods
        },
    }


def _run_rerank_experiment(args: argparse.Namespace) -> dict[str, object]:
    values = dotenv_values(args.env_file)
    api_key = values.get("RERANK_API_KEY") or values.get("EMBEDDING_API_KEY")
    if not api_key:
        raise ConfigError("缺少 RERANK_API_KEY；也没有可复用的 EMBEDDING_API_KEY")
    result = run_rerank_experiment(
        api_key=api_key,
        dataset_path=args.dataset,
        transcript_path=args.transcript,
        chunk_catalog_path=args.chunk_catalog,
        index_directory=args.index_directory,
        day12_baseline_path=args.baseline_result,
        day13_experiment_path=args.day13_result,
        day14_experiment_path=args.day14_result,
        output_path=args.output,
        prior_failed_rerank_attempts=args.prior_failed_rerank_attempts,
        prior_successful_synthetic_diagnostic_requests=(
            args.prior_successful_synthetic_diagnostic_requests
        ),
        overwrite=args.overwrite,
    )
    return {
        "experiment_id": result.experiment_id,
        "output": str(Path(args.output)),
        "recommendation": result.recommendation.model_dump(mode="json"),
        "call_counts": result.call_counts.model_dump(mode="json"),
        "latency_and_cost": result.latency_and_cost.model_dump(mode="json"),
        "candidate_pool_limits": [
            item.model_dump(mode="json") for item in result.candidate_pool_limits
        ],
        "methods": [
            {
                "method_name": method.method_name,
                "aggregate_metrics": {
                    str(k): metric.model_dump(mode="json")
                    for k, metric in method.aggregate_metrics.items()
                },
            }
            for method in result.methods
        ],
    }


def _run_answer_generation(args: argparse.Namespace) -> dict[str, object]:
    settings = Settings.from_env(args.env_file)
    artifact = run_answer_generation(
        settings=settings,
        answer_client=create_client(settings),
        generation_output_path=args.generation_output,
        dataset_path=args.dataset,
        transcript_path=args.transcript,
        chunk_catalog_path=args.chunk_catalog,
        index_directory=args.index_directory,
        day12_baseline_path=args.baseline_result,
        day13_experiment_path=args.day13_result,
        day14_experiment_path=args.day14_result,
        day15_experiment_path=args.day15_result,
        overwrite=args.overwrite,
    )
    return {
        "generation_id": artifact.generation_id,
        "output": str(Path(args.generation_output)),
        "generation_freeze_sha256": artifact.generation_freeze_sha256,
        "vector_top5_identity": artifact.vector_top5_identity.model_dump(mode="json"),
        "answer_model_configuration": artifact.answer_model_configuration.model_dump(
            mode="json"
        ),
        "call_counts": artifact.call_counts.model_dump(mode="json"),
        "latency_and_cost": artifact.latency_and_cost.model_dump(mode="json"),
    }


def _run_answer_finalize(args: argparse.Namespace) -> dict[str, object]:
    result = finalize_answer_quality_experiment(
        generation_path=args.generation,
        review_path=args.review,
        output_path=args.output,
        dataset_path=args.dataset,
        transcript_path=args.transcript,
        chunk_catalog_path=args.chunk_catalog,
        index_directory=args.index_directory,
        day12_baseline_path=args.baseline_result,
        day13_experiment_path=args.day13_result,
        day14_experiment_path=args.day14_result,
        day15_experiment_path=args.day15_result,
        overwrite=args.overwrite,
    )
    return {
        "experiment_id": result.experiment_id,
        "output": str(Path(args.output)),
        "generation_freeze_sha256": result.generation_freeze_sha256,
        "metrics": result.metrics.model_dump(mode="json"),
        "call_counts": result.call_counts.model_dump(mode="json"),
        "latency_and_cost": result.latency_and_cost.model_dump(mode="json"),
    }


def _run_rag_v2_freeze(args: argparse.Namespace) -> dict[str, object]:
    result = freeze_rag_v2(
        output_path=args.output,
        input_paths={
            "eval_dataset_v1": args.dataset_v1,
            "day12_vector_baseline": args.day12,
            "day13_chunk_experiment": args.day13,
            "day14_retrieval_experiment": args.day14,
            "day15_rerank_experiment": args.day15,
            "day16_answer_quality": args.day16,
            "eval_dataset_v2_pilot": args.dataset_v2,
            "eval_dataset_v2_retrieval_experiment": args.v2_retrieval,
        },
        overwrite=args.overwrite,
    )
    return {
        "freeze_id": result.freeze_id,
        "output": str(Path(args.output)),
        "retrieval_default": result.retrieval_default.model_dump(mode="json"),
        "rerank_enabled": result.rerank_enabled,
        "production_default_changed": result.production_default_changed,
        "provider_call_counts": result.provider_call_counts.model_dump(mode="json"),
    }
