"""Run three portfolio scenarios with original samples and an optional live LLM."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from openai import OpenAI

from app.api.adapters import EvaluationRetrieveService
from app.api.summary_context import build_summary_context
from app.chunking.core import chunk_transcript
from app.core.config import ConfigError, Settings
from app.core.llm import ModelServiceError, create_client
from app.ingestion.transcripts import load_transcript
from app.rag.core import GroundedAnswerError, generate_grounded_answer
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.corpus import RetrievalCorpusIdentity
from app.schemas.retrieval import RetrievalResult

SAMPLES = Path(__file__).with_name("samples")


class ScriptedClient:
    """Return one clearly labeled, predetermined OpenAI-shaped response."""

    def __init__(self, answer, evidence_ids):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.content = json.dumps({
            "answer": answer,
            "citations": [{"evidence_id": item} for item in evidence_ids],
            "has_sufficient_evidence": bool(evidence_ids),
        })

    def create(self, **_):
        return SimpleNamespace(choices=[
            SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=self.content))
        ])


def load_episode(name):
    transcript = load_transcript(SAMPLES / f"{name}.json")
    catalog = chunk_transcript(transcript)
    identity = RetrievalCorpusIdentity(
        episode_id=catalog.episode_id,
        source_file=catalog.source_file,
        chunking_algorithm=catalog.algorithm,
        chunk_size=catalog.chunk_size,
        overlap=catalog.overlap,
        input_chunk_sha256=sha256(catalog.model_dump_json().encode()).hexdigest(),
        chunk_id_order=tuple(chunk.chunk_id for chunk in catalog.chunks),
    )
    return catalog, BM25Retriever(tuple(catalog.chunks), identity)


def run(scenario, live_client=None, model="scripted-demo"):
    episode = "repair-cafe" if scenario == "summary" else "rooftop-garden"
    catalog, retriever = load_episode(episode)
    chunks = {chunk.chunk_id: chunk for chunk in catalog.chunks}
    if scenario == "summary":
        context = build_summary_context(EvaluationRetrieveService(
            retriever=retriever,
            chunks=tuple(catalog.chunks),
            episode_id=catalog.episode_id,
            input_chunk_sha256=retriever.corpus_identity.input_chunk_sha256,
        ))
        query = "Summarize the full episode in English, covering intake and handoff."
        ranking = [(item.rank, item.score, item.chunk_id) for item in context.chunks]
        scripted = ScriptedClient(
            "Visitors tag the item and describe its symptom. Volunteers teach a maintenance step and record the repair at handoff.",
            ["E1", "E2"],
        )
    else:
        query = {
            "qa": "How large was the basil pot?",
            "refusal": "What was the rooftop garden's annual budget?",
        }[scenario]
        ranking = [(item.rank, item.score, item.chunk_id) for item in retriever.retrieve(query, 5)]
        scripted = ScriptedClient(
            "The basil pot held twelve liters." if scenario == "qa" else "Insufficient evidence.",
            [f"E{rank}" for rank, _, key in ranking
             if "Choosing a container" in chunks[key].section_titles] if scenario == "qa" else [],
        )
    evidence = [
        RetrievalResult.from_chunk(
            rank=rank, score=score, row_index=chunks[key].chunk_index, chunk=chunks[key],
        )
        for rank, score, key in ranking
    ]
    answer = generate_grounded_answer(
        live_client if live_client is not None else cast(OpenAI, scripted),
        model=model, query=query, evidence=evidence,
    )
    print(f"\n[{scenario}] {query}\n{answer.answer}")
    for citation in answer.citations:
        print(f"  {citation.evidence_id}: {citation.source_file} "
              f"{citation.start_time:g}–{citation.end_time:g}s · "
              f"{', '.join(citation.speakers)} · {' / '.join(citation.section_titles)}")
        print(f"     {chunks[citation.chunk_id].text}")
    if not answer.citations:
        print("  Citations: none")
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", nargs="?", default="all", choices=["all", "qa", "refusal", "summary"])
    parser.add_argument("--live", action="store_true", help="Send original sample text to the configured LLM")
    args = parser.parse_args()
    try:
        if args.live:
            settings = Settings.from_env(env_file=None)
            print("Live model mode · original samples · local retrieval")
            with create_client(settings) as client:
                for scenario in ("qa", "refusal", "summary") if args.scenario == "all" else (args.scenario,):
                    run(scenario, client, settings.model_strong)
        else:
            print("Offline mode · scripted model output · real ingestion, retrieval and citation validation")
            for scenario in ("qa", "refusal", "summary") if args.scenario == "all" else (args.scenario,):
                run(scenario)
    except (ConfigError, GroundedAnswerError, ModelServiceError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
