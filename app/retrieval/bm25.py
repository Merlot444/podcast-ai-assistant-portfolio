"""Deterministic Unicode tokenization and a small auditable BM25Okapi retriever."""

from __future__ import annotations

import math
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from time import perf_counter

from app.retrieval.corpus import RetrievalCorpusIdentity
from app.schemas.chunk import TranscriptChunk
from app.schemas.index import VectorIndexManifest
from app.schemas.retriever import RetrievedCandidate


TOKENIZER_VERSION = "unicode_nfkc_casefold_alnum_joiners_v1"
BM25_K1 = 1.5
BM25_B = 0.75
_APOSTROPHES = frozenset({"'", "\u2019", "\u02bc"})
_HYPHENS = frozenset({"-", "\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"})


def tokenize_unicode(value: str) -> tuple[str, ...]:
    """Tokenize with NFKC/casefold, preserving internal apostrophes and hyphens."""
    if not isinstance(value, str):
        raise TypeError("tokenizer 输入必须是字符串")
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = "".join(
        "'" if character in _APOSTROPHES else "-" if character in _HYPHENS else character
        for character in normalized
    )
    tokens: list[str] = []
    current: list[str] = []
    for index, character in enumerate(normalized):
        category = unicodedata.category(character)
        if category.startswith(("L", "N")):
            current.append(character)
            continue
        is_internal_joiner = (
            character in {"'", "-"}
            and bool(current)
            and index + 1 < len(normalized)
            and unicodedata.category(normalized[index + 1]).startswith(("L", "N"))
        )
        if is_internal_joiner:
            current.append(character)
            continue
        if current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tuple(tokens)


@dataclass
class BM25Retriever:
    """Build a deterministic in-memory BM25Okapi index from one Chunk catalog."""

    chunks: tuple[TranscriptChunk, ...]
    corpus_identity: RetrievalCorpusIdentity
    k1: float = BM25_K1
    b: float = BM25_B
    name: str = "bm25"
    score_semantics: str = "bm25_okapi"
    retrieval_call_count: int = field(default=0, init=False)
    build_elapsed_seconds: float = field(default=0, init=False)
    document_token_count: int = field(default=0, init=False)
    vocabulary_size: int = field(default=0, init=False)
    posting_count: int = field(default=0, init=False)
    approximate_memory_bytes: int = field(default=0, init=False)
    _term_frequencies: tuple[Counter[str], ...] = field(default=(), init=False, repr=False)
    _document_lengths: tuple[int, ...] = field(default=(), init=False, repr=False)
    _document_frequencies: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _average_document_length: float = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        started = perf_counter()
        if not math.isfinite(self.k1) or self.k1 <= 0:
            raise ValueError("BM25 k1 必须是有限正数")
        if not math.isfinite(self.b) or not 0 <= self.b <= 1:
            raise ValueError("BM25 b 必须位于 0 到 1")
        if tuple(chunk.chunk_id for chunk in self.chunks) != self.corpus_identity.chunk_id_order:
            raise ValueError("BM25 Chunk catalog 顺序与 corpus identity 不一致")

        frequencies = tuple(Counter(tokenize_unicode(chunk.text)) for chunk in self.chunks)
        lengths = tuple(sum(items.values()) for items in frequencies)
        document_frequencies: Counter[str] = Counter()
        for items in frequencies:
            document_frequencies.update(items.keys())

        self._term_frequencies = frequencies
        self._document_lengths = lengths
        self._document_frequencies = dict(document_frequencies)
        self._average_document_length = sum(lengths) / len(lengths)
        self.document_token_count = sum(lengths)
        self.vocabulary_size = len(document_frequencies)
        self.posting_count = sum(len(items) for items in frequencies)
        self.approximate_memory_bytes = _deep_size(
            (frequencies, lengths, self._document_frequencies)
        )
        self.build_elapsed_seconds = perf_counter() - started

    @classmethod
    def from_manifest(cls, manifest: VectorIndexManifest) -> BM25Retriever:
        """Build from the already validated ordered 1000/200 manifest catalog."""
        return cls(
            chunks=tuple(manifest.chunks),
            corpus_identity=RetrievalCorpusIdentity.from_manifest(manifest),
        )

    def retrieve(self, query: str, top_k: int) -> list[RetrievedCandidate]:
        """Return positive-score matches with catalog-position tie-breaking."""
        _validate_top_k(top_k)
        if not isinstance(query, str):
            raise TypeError("BM25 查询必须是字符串")
        self.retrieval_call_count += 1
        query_tokens = tuple(dict.fromkeys(tokenize_unicode(query)))
        if not query_tokens:
            return []

        ranked: list[tuple[float, int]] = []
        document_count = len(self.chunks)
        for position, (frequencies, length) in enumerate(
            zip(self._term_frequencies, self._document_lengths, strict=True)
        ):
            score = 0.0
            for token in query_tokens:
                term_frequency = frequencies.get(token, 0)
                if term_frequency == 0:
                    continue
                document_frequency = self._document_frequencies[token]
                inverse_document_frequency = math.log(
                    1
                    + (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                length_normalization = 1 - self.b + self.b * (
                    length / self._average_document_length
                )
                score += inverse_document_frequency * (
                    term_frequency * (self.k1 + 1)
                ) / (term_frequency + self.k1 * length_normalization)
            if score > 0:
                ranked.append((score, position))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [
            RetrievedCandidate(
                rank=rank,
                chunk_id=self.chunks[position].chunk_id,
                source_segment_ids=tuple(self.chunks[position].source_segment_ids),
                score=score,
                score_semantics=self.score_semantics,
                retriever_name=self.name,
            )
            for rank, (score, position) in enumerate(ranked[:top_k], start=1)
        ]


def _validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k 必须是大于 0 的整数")


def _deep_size(value: object) -> int:
    """Estimate this process's Python container footprint without Chunk text."""
    seen: set[int] = set()

    def visit(item: object) -> int:
        identity = id(item)
        if identity in seen:
            return 0
        seen.add(identity)
        size = sys.getsizeof(item)
        if isinstance(item, dict):
            size += sum(visit(key) + visit(child) for key, child in item.items())
        elif isinstance(item, (tuple, list, set, frozenset, Counter)):
            size += sum(visit(child) for child in item)
        return size

    return visit(value)
