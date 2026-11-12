"""Prepare or run a private, blinded Vector vs Hybrid answer comparison."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
from secrets import randbelow
from time import perf_counter

from app.schemas.chunk import ChunkedTranscript
from app.indexing.index import chunked_transcript_fingerprint
from app.core.config import ConfigError, Settings
from app.core.llm import ModelServiceError, create_client
from app.rag.core import GroundedAnswerError, generate_grounded_answer
from app.rag.prompts import GROUNDED_ANSWER_SYSTEM_PROMPT
from app.schemas.evaluation import EvaluationDataset, EVALUATION_CATEGORIES
from app.schemas.evaluation_v2 import EvaluationDatasetV2Pilot
from app.schemas.evaluation_v2_retrieval import EvaluationV2RetrievalExperiment
from app.schemas.retrieval import RetrievalResult

METHODS = ("vector", "hybrid_rrf")
EPISODE_FILES = {
    "demis": "demis-hassabis",
    "ilya25": "ilya-sutskever-2025",
    "mark25": "mark-zuckerberg-2025",
}
EVAL_DIR = Path(__file__).resolve().parents[2] / "data/evaluation"


def digest(content):
    return sha256(content).hexdigest()


def frozen_inputs(artifact_root):
    rankings_path = EVAL_DIR / "eval_dataset_v2-pilot-retrieval-experiment.json"
    ranking_bytes = rankings_path.read_bytes()
    rankings = EvaluationV2RetrievalExperiment.model_validate_json(ranking_bytes)

    def checked(relative_path, root):
        raw = (root / relative_path).read_bytes()
        if digest(raw) != rankings.input_hashes[relative_path]:
            raise ValueError(f"Frozen input changed: {relative_path}")
        return raw

    base = EvaluationDataset.model_validate_json(checked(
        "data/evaluation/eval_dataset_v1.json", EVAL_DIR.parents[1],
    ))
    pilot = EvaluationDatasetV2Pilot.model_validate_json(checked(
        "data/evaluation/eval_dataset_v2-pilot.json", EVAL_DIR.parents[1],
    ))
    episodes = {"demis": base, **{episode.episode_slug: episode for episode in pilot.episode_datasets}}
    ranked = {
        method.retriever_name: {case.case_id: case for case in method.cases}
        for method in rankings.methods if method.retriever_name in METHODS
    }
    selected = []
    for identity in rankings.episode_identities:
        slug = identity.episode_slug
        relative_path = f"data/processed/{EPISODE_FILES[slug]}-chunks.json"
        raw = checked(relative_path, artifact_root)
        catalog = ChunkedTranscript.model_validate_json(raw)
        if chunked_transcript_fingerprint(catalog) != identity.input_chunk_sha256:
            raise ValueError(f"Chunk fingerprint mismatch: {slug}")
        episode = episodes[slug]
        if catalog.episode_id != episode.episode_id or catalog.episode_id != identity.episode_id:
            raise ValueError(f"Episode mismatch: {slug}")
        if (catalog.chunk_size, catalog.overlap, len(catalog.chunks)) != (
            identity.chunk_size, identity.overlap, identity.chunk_count,
        ):
            raise ValueError(f"Chunk configuration mismatch: {slug}")
        chunks = {chunk.chunk_id: chunk for chunk in catalog.chunks}

        # Two cases per category, in frozen dataset order. Selection never reads outcomes.
        for category in EVALUATION_CATEGORIES:
            cases = [case for case in episode.cases if case.category == category][:2]
            if len(cases) != 2:
                raise ValueError(f"Incomplete selection: {slug}/{category}")
            for case in cases:
                evidence = {}
                for method in METHODS:
                    saved = ranked[method][case.case_id]
                    if (saved.category, saved.answerable, saved.expected_source_segment_ids) != (
                        case.category, case.answerable, case.expected_source_segment_ids,
                    ):
                        raise ValueError(f"Case provenance mismatch: {case.case_id}")
                    evidence[method] = []
                    for item in saved.retrieved_candidates[:5]:
                        chunk = chunks[item.chunk_id]
                        if tuple(chunk.source_segment_ids) != item.source_segment_ids:
                            raise ValueError(f"Evidence provenance mismatch: {case.case_id}")
                        # Scores are in incompatible units; only evidence content and order vary.
                        evidence[method].append(RetrievalResult.from_chunk(
                            rank=item.rank, score=0.0, row_index=chunk.chunk_index, chunk=chunk,
                        ))
                    if len(evidence[method]) != 5:
                        raise ValueError(f"Incomplete Top-5: {case.case_id}/{method}")
                selected.append((slug, case, evidence))
    return digest(ranking_bytes), selected


def generate(client, model, case, evidence, *, transcript=None):
    started = perf_counter()
    try:
        answer = generate_grounded_answer(
            client, model=model, query=case.question, evidence=evidence, transcript=transcript,
        )
        return {
            "answer": answer.answer,
            "answered": answer.has_sufficient_evidence,
            "citations": [
                {"id": item.evidence_id, "chunk": item.chunk_id, "text": evidence[item.rank - 1].text}
                for item in answer.citations
            ],
            "error": None,
        }, round(perf_counter() - started, 3)
    except (ModelServiceError, GroundedAnswerError) as error:
        return {"error": type(error).__name__}, round(perf_counter() - started, 3)


def write_private(path, data):
    # Exclusive create and owner-only permissions. Keep outside any repo.
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as out:
        json.dump(data, out, ensure_ascii=False, indent=2)
        out.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New private output directory")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--live", action="store_true", help="Send private questions and Top-5 evidence to the configured model")
    args = parser.parse_args()
    try:
        if not args.live and args.env_file:
            parser.error("--env-file requires --live")
        ranking_hash, selected = frozen_inputs(args.artifact_root)
        settings = Settings.from_env(args.env_file) if args.live else None
        args.output.mkdir(mode=0o700)
        write_private(args.output / "plan.json", {
            "rankings_sha256": ranking_hash,
            "prompt_sha256": digest(GROUNDED_ANSWER_SYSTEM_PROMPT.encode()),
            "model": settings.model_strong if settings else None,
            "generation": {"max_tokens": 1200, "temperature": 0.2, "retries": 0},
            "cases": [
                {"episode": slug, "case_id": case.case_id, "category": case.category,
                 "evidence": {name: [item.chunk_id for item in items] for name, items in evidence.items()}}
                for slug, case, evidence in selected
            ],
        })
        print(f"Frozen {len(selected)} pairs across three episodes.", flush=True)
        if settings is None:
            return

        def run_pair(selection):
            slug, case, evidence = selection
            order = METHODS if randbelow(2) == 0 else METHODS[::-1]
            review = {"episode": slug, "case_id": case.case_id, "question": case.question,
                      "answerable": case.answerable, "reference": case.reference_answer}
            key = {}
            for label, method in zip(("A", "B"), order, strict=True):
                result, seconds = generate(client, settings.model_strong, case, evidence[method])
                review[label] = result
                key[label] = {"method": method, "generation_seconds": seconds}
            write_private(args.output / "review" / f"{case.case_id}.json", review)
            write_private(args.output / "key" / f"{case.case_id}.json", key)
            return sum(review[label]["error"] is not None for label in ("A", "B"))

        for directory in ("review", "key"):
            (args.output / directory).mkdir(mode=0o700)
        with create_client(settings) as client, ThreadPoolExecutor(max_workers=4) as pool:
            errors = sum(pool.map(run_pair, selected))
        print(f"Generated {2 * len(selected) - errors}/{2 * len(selected)} answers/refusals. Awaiting blind review.")
        if errors:
            parser.exit(1)
    except (ConfigError, OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Comparison stopped: {type(error).__name__}\n")


if __name__ == "__main__":
    main()
