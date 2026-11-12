"""Strict remote reranking boundary for the frozen Day 15 experiment."""

from __future__ import annotations

import json
import math
import socket
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Protocol

import httpx

JINA_RERANK_ENDPOINT = "https://api.jina.ai/v1/rerank"
JINA_RERANK_MODEL = "jina-reranker-v3"
RERANK_CANDIDATE_DEPTH = 20
RERANK_RETURNED_DEPTH = 20
RERANK_PRODUCT_FOCUS_K = 5
RERANK_EVALUATION_K_VALUES = (1, 3, 5, 8)
RERANK_MAX_INPUT_TOKENS = 131_072
RERANK_MAX_DOCUMENT_CHARACTERS = 1_000
RERANK_TIMEOUT_SECONDS = 60.0
RERANK_MAX_RETRIES = 0


class RerankError(ValueError):
    """Raised when a rerank request or response violates the local contract."""


class RerankServiceError(RuntimeError):
    """Raised when the remote service cannot produce a trustworthy ranking."""


@dataclass(frozen=True)
class RerankInput:
    """One candidate with text available only at the in-memory model boundary."""

    chunk_id: str
    text: str = field(repr=False)
    source_segment_ids: tuple[str, ...]
    original_retriever: str
    original_rank: int
    original_score: float
    original_score_semantics: str
    catalog_position: int

    def __post_init__(self) -> None:
        if not self.chunk_id or not self.text:
            raise RerankError("Rerank 候选必须包含 Chunk ID 和正文")
        if not self.source_segment_ids or len(self.source_segment_ids) != len(
            set(self.source_segment_ids)
        ):
            raise RerankError("Rerank 候选的来源 segment 必须非空且唯一")
        if self.original_rank <= 0 or self.catalog_position < 0:
            raise RerankError("Rerank 候选的原始 rank 或 catalog position 无效")
        if not math.isfinite(self.original_score):
            raise RerankError("Rerank 候选的原始分数必须有限")
        if not self.original_retriever or not self.original_score_semantics:
            raise RerankError("Rerank 候选缺少原始 provenance")


@dataclass(frozen=True)
class RerankedCandidate:
    """A validated rerank result retaining complete metadata-only provenance."""

    chunk_id: str
    source_segment_ids: tuple[str, ...]
    original_retriever: str
    original_rank: int
    original_score: float
    original_score_semantics: str
    rerank_rank: int
    rerank_score: float
    rerank_score_semantics: str


@dataclass(frozen=True)
class RerankRequestTelemetry:
    """Non-sensitive accounting for one successful provider request."""

    elapsed_seconds: float
    document_count: int
    query_characters: int
    document_characters: int
    provider_total_tokens: int | None
    provider_model: str


class Reranker(Protocol):
    """Minimal Day 15 reranker protocol independent of retrieval backends."""

    @property
    def name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankInput],
        top_n: int,
    ) -> list[RerankedCandidate]: ...


Transport = Callable[[str, str, Mapping[str, object], float], Mapping[str, object]]


@dataclass
class JinaAPIReranker:
    """Call Jina's fixed rerank endpoint and enforce deterministic local ordering."""

    api_key: str = field(repr=False)
    model: str = JINA_RERANK_MODEL
    endpoint: str = JINA_RERANK_ENDPOINT
    timeout_seconds: float = RERANK_TIMEOUT_SECONDS
    max_retries: int = RERANK_MAX_RETRIES
    max_document_characters: int = RERANK_MAX_DOCUMENT_CHARACTERS
    transport: Transport = field(default=None, repr=False)  # type: ignore[assignment]
    telemetry: list[RerankRequestTelemetry] = field(default_factory=list, init=False)
    request_attempt_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not self.api_key:
            raise RerankError("Jina Rerank API Key 不能为空")
        if self.model != JINA_RERANK_MODEL:
            raise RerankError("Day 15 正式实验只允许预先固定的 jina-reranker-v3")
        if self.endpoint != JINA_RERANK_ENDPOINT:
            raise RerankError("Day 15 正式实验只允许固定 Jina Rerank endpoint")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise RerankError("Rerank timeout 必须是有限正数")
        if self.max_retries != 0:
            raise RerankError("Day 15 正式实验预先固定为不重试")
        if self.max_document_characters != RERANK_MAX_DOCUMENT_CHARACTERS:
            raise RerankError("Day 15 Chunk 截断上限必须固定为 1000 字符")
        if self.transport is None:
            self.transport = _default_transport

    @property
    def name(self) -> str:
        return "jina_api_reranker"

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def score_semantics(self) -> str:
        return "jina_relevance_score_higher_is_better"

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankInput],
        top_n: int,
    ) -> list[RerankedCandidate]:
        items = tuple(candidates)
        _validate_request(query, items, top_n)
        documents = [item.text[: self.max_document_characters] for item in items]
        payload: Mapping[str, object] = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": False,
        }
        started = perf_counter()
        self.request_attempt_count += 1
        try:
            response = self.transport(
                self.endpoint,
                self.api_key,
                payload,
                self.timeout_seconds,
            )
        except RerankServiceError:
            raise
        except (TimeoutError, socket.timeout) as error:
            raise RerankServiceError("Rerank 服务请求超时；未重试") from error
        except Exception as error:
            raise RerankServiceError(
                f"Rerank 服务请求失败；未重试：{type(error).__name__}"
            ) from error
        elapsed = perf_counter() - started
        ranked, usage_tokens, provider_model = self._validate_response(
            response,
            items=items,
            top_n=top_n,
        )
        self.telemetry.append(
            RerankRequestTelemetry(
                elapsed_seconds=elapsed,
                document_count=len(items),
                query_characters=len(query),
                document_characters=sum(len(document) for document in documents),
                provider_total_tokens=usage_tokens,
                provider_model=provider_model,
            )
        )
        return ranked

    def _validate_response(
        self,
        response: Mapping[str, object],
        *,
        items: tuple[RerankInput, ...],
        top_n: int,
    ) -> list[RerankedCandidate] | tuple[list[RerankedCandidate], int | None, str]:
        if not isinstance(response, Mapping):
            raise RerankServiceError("Rerank 响应不是 JSON object")
        provider_model = response.get("model")
        if provider_model != self.model:
            raise RerankServiceError("Rerank 响应模型身份与预先固定模型不一致")
        raw_results = response.get("results")
        if not isinstance(raw_results, list) or len(raw_results) != top_n:
            raise RerankServiceError("Rerank 响应必须返回固定数量的完整候选")

        scores_by_index: dict[int, float] = {}
        for raw in raw_results:
            if not isinstance(raw, Mapping):
                raise RerankServiceError("Rerank result 不是 object")
            index = raw.get("index")
            score = raw.get("relevance_score")
            if isinstance(index, bool) or not isinstance(index, int):
                raise RerankServiceError("Rerank result index 必须是整数")
            if index < 0 or index >= len(items):
                raise RerankServiceError("Rerank 返回了输入候选之外的 index")
            if index in scores_by_index:
                raise RerankServiceError("Rerank 返回了重复候选")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise RerankServiceError("Rerank relevance_score 必须是数值")
            numeric_score = float(score)
            if not math.isfinite(numeric_score):
                raise RerankServiceError("Rerank relevance_score 必须是有限数")
            scores_by_index[index] = numeric_score
        if set(scores_by_index) != set(range(len(items))):
            raise RerankServiceError("Rerank 响应缺少输入候选")

        ordered = sorted(
            enumerate(items),
            key=lambda pair: (
                -scores_by_index[pair[0]],
                pair[1].original_rank,
                pair[1].catalog_position,
                pair[1].chunk_id,
            ),
        )
        reranked = [
            RerankedCandidate(
                chunk_id=item.chunk_id,
                source_segment_ids=item.source_segment_ids,
                original_retriever=item.original_retriever,
                original_rank=item.original_rank,
                original_score=item.original_score,
                original_score_semantics=item.original_score_semantics,
                rerank_rank=rank,
                rerank_score=scores_by_index[index],
                rerank_score_semantics=self.score_semantics,
            )
            for rank, (index, item) in enumerate(ordered, start=1)
        ]
        usage_tokens = _usage_tokens(response.get("usage"))
        return reranked, usage_tokens, provider_model


def _validate_request(
    query: str,
    candidates: tuple[RerankInput, ...],
    top_n: int,
) -> None:
    if not isinstance(query, str) or not query.strip():
        raise RerankError("Rerank query 不能为空")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
        raise RerankError("Rerank top_n 必须是正整数")
    if len(candidates) != RERANK_CANDIDATE_DEPTH or top_n != RERANK_RETURNED_DEPTH:
        raise RerankError("Day 15 必须固定执行完整 Top-20 → Top-20 重排")
    ranks = [item.original_rank for item in candidates]
    if ranks != list(range(1, RERANK_CANDIDATE_DEPTH + 1)):
        raise RerankError("Rerank 输入必须按原始 rank 1..20 连续排列")
    chunk_ids = [item.chunk_id for item in candidates]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise RerankError("Rerank 输入不能包含重复 Chunk")
    retrievers = {item.original_retriever for item in candidates}
    semantics = {item.original_score_semantics for item in candidates}
    if len(retrievers) != 1 or len(semantics) != 1:
        raise RerankError("一次 Rerank 不能混合不同候选源或原始分数语义")


def _usage_tokens(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RerankServiceError("Rerank usage 必须是 object 或 null")
    total = value.get("total_tokens")
    if total is None:
        return None
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise RerankServiceError("Rerank usage.total_tokens 无效")
    return total


def _default_transport(
    endpoint: str,
    api_key: str,
    payload: Mapping[str, object],
    timeout_seconds: float,
) -> Mapping[str, object]:
    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json",
                    "User-Agent": "podcast-ai-assistant-day15/0.1",
                },
                json=payload,
            )
    except httpx.TimeoutException as error:
        raise RerankServiceError("Rerank 服务请求超时；未重试") from error
    except httpx.HTTPError as error:
        raise RerankServiceError("Rerank 服务网络失败；未重试") from error
    if response.status_code >= 400:
        labels = {
            401: "鉴权失败",
            403: "鉴权或权限失败",
            429: "触发限流",
        }
        label = labels.get(response.status_code, f"HTTP {response.status_code}")
        raise RerankServiceError(f"Rerank 服务{label}；未重试")
    try:
        parsed = response.json()
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RerankServiceError("Rerank 服务返回了非法 JSON") from error
    if not isinstance(parsed, Mapping):
        raise RerankServiceError("Rerank 服务返回的 JSON 不是 object")
    return parsed
