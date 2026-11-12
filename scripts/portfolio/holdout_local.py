"""Evaluate the private holdout with local embeddings and optional local generation."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from time import perf_counter
from types import SimpleNamespace

from app.evaluation.runner import run_precomputed_case_evaluation
from app.indexing.index import build_vector_index, chunked_transcript_fingerprint
from app.rag.prompts import SOURCE_AWARE_SYSTEM_PROMPT
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.core import Retriever
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.vector import VectorRetrieverAdapter
from app.schemas.chunk import ChunkedTranscript
from app.schemas.evaluation import EvaluationCase, evaluation_cases_sha256
from app.schemas.retrieval import RetrievalResult
from app.schemas.transcript import ProcessedTranscript
from scripts.portfolio.compare_answers import generate, write_private

METHODS = ("vector", "hybrid_rrf")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
ANSWER_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
SEED = 20261112
MAX_LOCAL_TOKENS = 384


class LocalEmbedder:
    model_name = EMBEDDING_MODEL
    expected_dimension = 384

    def __init__(self, weights):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(
            self.model_name, specific_model_path=str(weights.resolve()), local_files_only=True, threads=2,
        )

    def embed_documents(self, texts):
        return [row.tolist() for row in self.model.passage_embed(texts)]

    def embed_query(self, text):
        return next(self.model.query_embed(text)).tolist()


class LocalAnswerClient:
    def __init__(self, model_path):
        from mlx_lm import load

        self.model, self.tokenizer = load(str(model_path.resolve()))
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, *, messages, temperature, max_tokens, **_):
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        mx.random.seed(SEED)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        parts = []
        for response in stream_generate(
            self.model, self.tokenizer, prompt, max_tokens=min(max_tokens, MAX_LOCAL_TOKENS),
            sampler=make_sampler(temp=temperature),
        ):
            parts.append(response.text)
        choice = SimpleNamespace(
            finish_reason=response.finish_reason,
            message=SimpleNamespace(content="".join(parts)),
        )
        self.usage = {
            "input_tokens": response.prompt_tokens,
            "output_tokens": response.generation_tokens,
        }
        return SimpleNamespace(choices=[choice])


def load_inputs(processed):
    data = json.loads((processed / "holdout/dataset.json").read_bytes())
    if data["dataset_id"] != "dwarkesh_holdout_v1":
        raise ValueError("Unexpected holdout ID")
    seen = set()
    for episode in data["episodes"]:
        slug = episode["slug"]
        chunk_path = processed / f"{slug}-chunks.json"
        raw = chunk_path.read_bytes()
        chunks = ChunkedTranscript.model_validate_json(raw)
        if sha256(raw).hexdigest() != episode["chunk_file_sha256"] or (
            chunked_transcript_fingerprint(chunks) != episode["chunk_baseline"]["input_chunk_sha256"]
        ) or chunks.episode_id != episode["episode_id"]:
            raise ValueError(f"Holdout catalog drift: {slug}")
        cases = tuple(EvaluationCase.model_validate(item) for item in episode["cases"])
        if len(cases) != 20 or evaluation_cases_sha256(cases) != episode["cases_sha256"]:
            raise ValueError(f"Holdout case drift: {slug}")
        for case in cases:
            if case.case_id in seen:
                raise ValueError("Duplicate case ID")
            seen.add(case.case_id)
        yield slug, chunks, cases
    if len(seen) != 60:
        raise ValueError("Expected 60 holdout cases")


def rank_episodes(inputs, embedder, output, input_identity):
    results = {}
    for slug, catalog, cases in inputs:
        index = build_vector_index(catalog, embedder=embedder, input_chunk_file=f"{slug}-chunks.json")
        vector = VectorRetrieverAdapter(Retriever(index, embedder))
        bm25 = BM25Retriever.from_manifest(index.manifest)
        rankings = {method: {} for method in (*METHODS, "bm25")}
        for case in cases:
            v = vector.retrieve(case.question, 20)
            b = bm25.retrieve(case.question, 20)
            rankings["vector"][case.case_id] = v
            rankings["bm25"][case.case_id] = b
            rankings["hybrid_rrf"][case.case_id] = reciprocal_rank_fusion(
                v, b, corpus_identity=vector.corpus_identity, top_k=20,
            )
        results[slug] = {
            method: run_precomputed_case_evaluation(
                cases, retriever_name=method, candidates_by_case_id=ranking, candidate_depth=20,
            ).model_dump(mode="json") for method, ranking in rankings.items()
        }
        print(f"{slug}: ranked {len(cases)} cases", flush=True)
    write_private(output, {"embedding_model": embedder.model_name, "inputs": input_identity, "episodes": results})
    return results


def make_review(inputs, ranked, client, out, limit, case_ids, transcripts):
    rng = Random(SEED)
    review_path, key_path = out / "review", out / "key"
    for path in (review_path, key_path):
        path.mkdir(mode=0o700, exist_ok=True)
    done = 0
    for slug, catalog, cases in inputs:
        by_id = {chunk.chunk_id: chunk for chunk in catalog.chunks}
        methods = {
            method: {case["case_id"]: case for case in ranked[slug][method]["cases"]}
            for method in METHODS
        }
        for case in cases:
            order = METHODS if rng.randrange(2) == 0 else METHODS[::-1]
            if case_ids and case.case_id not in case_ids:
                continue
            review_file, key_file = review_path / f"{case.case_id}.json", key_path / f"{case.case_id}.json"
            if review_file.exists() and key_file.exists():
                continue
            if review_file.exists() or key_file.exists():
                raise ValueError(f"Partially saved pair: {case.case_id}")
            review = {"episode": slug, "case_id": case.case_id, "question": case.question,
                      "answerable": case.answerable, "reference": case.reference_answer}
            key = {}
            for label, method in zip("AB", order, strict=True):
                evidence = []
                for item in methods[method][case.case_id]["retrieved_candidates"][:5]:
                    chunk = by_id[item["chunk_id"]]
                    if tuple(item["source_segment_ids"]) != tuple(chunk.source_segment_ids):
                        raise ValueError(f"Ranking source drift: {case.case_id}")
                    evidence.append(RetrievalResult.from_chunk(
                        rank=item["rank"], score=0, row_index=chunk.chunk_index, chunk=chunk,
                    ))
                client.usage = None
                result, duration = generate(client, ANSWER_MODEL, case, evidence, transcript=transcripts[slug])
                if not result["error"]:
                    for citation in result["citations"]:
                        chunk = by_id[citation["chunk"]]
                        citation["start_time"] = chunk.start_time
                        citation["end_time"] = chunk.end_time
                review[label] = result
                key[label] = {"method": method, "seconds": duration, "usage": client.usage}
            write_private(review_file, review)
            write_private(key_file, key)
            done += 1
            if done % 5 == 0:
                print(f"Generated {done} new pairs", flush=True)
            if limit is not None and done >= limit:
                return


def file_hash(path):
    with path.open("rb") as stream:
        from hashlib import file_digest

        return file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--embedding-weights", type=Path)
    parser.add_argument("--answer-weights", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-id", action="append", default=[])
    args = parser.parse_args()
    args.out.mkdir(mode=0o700, exist_ok=True)
    inputs = tuple(load_inputs(args.processed))
    if set(args.case_id) - {case.case_id for _, _, cases in inputs for case in cases}:
        parser.error("--case-id must identify a case in the holdout")
    input_identity = {
        slug: {"chunk": chunked_transcript_fingerprint(catalog), "cases": evaluation_cases_sha256(cases)}
        for slug, catalog, cases in inputs
    }
    ranking_path = args.out / "ranking.json"
    if ranking_path.exists():
        saved = json.loads(ranking_path.read_bytes())
        if saved["inputs"] != input_identity or saved["embedding_model"] != EMBEDDING_MODEL:
            raise ValueError("Saved ranking does not match the frozen inputs")
        results = saved["episodes"]
    else:
        if args.embedding_weights is None or not args.embedding_weights.is_dir():
            parser.error("--embedding-weights must point to local files for a new ranking")
        results = rank_episodes(inputs, LocalEmbedder(args.embedding_weights), ranking_path, input_identity)
    if args.answer_weights:
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        if not args.answer_weights.is_dir():
            parser.error("--answer-weights must point to local files")
        snapshot = json.loads((args.processed / "holdout/dataset.json").read_bytes())
        transcripts = {}
        transcript_hashes = {}
        for episode in snapshot["episodes"]:
            slug = episode["slug"]
            raw = (args.processed / f"{slug}.json").read_bytes()
            fingerprint = sha256(raw).hexdigest()
            if fingerprint != episode["processed_sha256"]:
                raise ValueError(f"Frozen transcript drift: {slug}")
            transcripts[slug] = ProcessedTranscript.model_validate_json(raw)
            transcript_hashes[slug] = fingerprint
        configuration = {
            "evidence_format": "source_excerpts_v1",
            "transcript_sha256": transcript_hashes,
            "model": ANSWER_MODEL,
            "weights_sha256": file_hash(args.answer_weights / "model.safetensors"),
            "ranking_sha256": file_hash(ranking_path),
            "system_prompt_sha256": sha256(SOURCE_AWARE_SYSTEM_PROMPT.encode()).hexdigest(),
            "temperature": 0.2,
            "max_tokens": MAX_LOCAL_TOKENS,
            "seed": SEED,
        }
        configuration_path = args.out / "generation-config.json"
        if not configuration_path.exists():
            write_private(configuration_path, configuration)
        elif json.loads(configuration_path.read_bytes()) != configuration:
            raise ValueError("Answer generation configuration differs from saved run")
        started = perf_counter()
        make_review(inputs, results, LocalAnswerClient(args.answer_weights), args.out, args.limit, set(args.case_id), transcripts)
        print(f"Local generation took {perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
