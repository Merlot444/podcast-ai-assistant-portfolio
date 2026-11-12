"""Formal Day 9 grounded-answer interface over Day 8 retrieval results."""

from __future__ import annotations

import json

from openai import OpenAI
from pydantic import ValidationError

from app.core.errors import safe_error_message
from app.core.llm import create_json_completion
from app.rag.prompts import (
    NumberedEvidence,
    SourceEvidenceError,
    build_grounded_answer_messages,
    number_evidence,
)
from app.schemas.rag import (
    GroundedAnswer,
    GroundedCitation,
    ModelGroundedAnswer,
)
from app.schemas.retrieval import RetrievalResult
from app.schemas.transcript import ProcessedTranscript


REFUSAL_ANSWER = "现有候选证据不足以回答该问题。"


class GroundedAnswerError(RuntimeError):
    """Raised when a Day 9 request or model output cannot be trusted."""


def generate_grounded_answer(
    client: OpenAI,
    *,
    model: str,
    query: str,
    evidence: list[RetrievalResult],
    max_tokens: int = 1_200,
    transcript: ProcessedTranscript | None = None,
) -> GroundedAnswer:
    """Generate once, validate selected IDs, and map trusted provenance."""
    cleaned_query = _validate_request(
        model=model,
        query=query,
        evidence=evidence,
        max_tokens=max_tokens,
    )
    numbered = number_evidence(evidence)
    if not numbered:
        return _refusal()

    try:
        messages = build_grounded_answer_messages(cleaned_query, numbered, transcript=transcript)
    except SourceEvidenceError as error:
        raise GroundedAnswerError("检索片段与原文不一致。") from error

    response = create_json_completion(
        client,
        model=model.strip(),
        messages=messages,
        max_tokens=max_tokens,
    )
    if not response.choices:
        raise GroundedAnswerError("模型响应中没有可用结果。")

    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise GroundedAnswerError("模型输出达到长度上限，JSON 可能被截断。")
    raw_output = _parse_model_output(choice.message.content or "")
    return _build_final_answer(raw_output, numbered)


def _validate_request(
    *,
    model: str,
    query: str,
    evidence: list[RetrievalResult],
    max_tokens: int,
) -> str:
    if not isinstance(query, str) or not query.strip():
        raise GroundedAnswerError("问题不能为空。")
    if not isinstance(model, str) or not model.strip():
        raise GroundedAnswerError("回答模型名称不能为空。")
    if not isinstance(evidence, list):
        raise GroundedAnswerError("evidence 必须是 list[RetrievalResult]。")
    if any(not isinstance(item, RetrievalResult) for item in evidence):
        raise GroundedAnswerError("evidence 只能包含 RetrievalResult。")
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
    ):
        raise GroundedAnswerError("max_tokens 必须是大于 0 的整数。")
    return query.strip()


def _parse_model_output(raw_content: str) -> ModelGroundedAnswer:
    if not raw_content or not raw_content.strip():
        raise GroundedAnswerError("模型返回了空内容。")
    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError:
        raise GroundedAnswerError("模型返回的内容不是合法 JSON。") from None
    try:
        return ModelGroundedAnswer.model_validate(payload)
    except ValidationError as error:
        raise GroundedAnswerError(
            f"模型输出不符合 Day 9 业务结构：{safe_error_message(error)}"
        ) from error


def _build_final_answer(
    raw_output: ModelGroundedAnswer,
    numbered_evidence: list[NumberedEvidence],
) -> GroundedAnswer:
    if not raw_output.has_sufficient_evidence:
        if raw_output.citations:
            raise GroundedAnswerError("证据不足的模型输出不能包含引用。")
        return _refusal()

    if not raw_output.citations:
        raise GroundedAnswerError("证据充分的模型输出必须至少包含一个引用。")

    evidence_by_id = {
        item.evidence_id: item.result for item in numbered_evidence
    }
    selected_ids: list[str] = []
    for citation in raw_output.citations:
        evidence_id = citation.evidence_id
        if evidence_id not in evidence_by_id:
            raise GroundedAnswerError(f"模型引用了不存在的证据编号：{evidence_id}")
        if evidence_id not in selected_ids:
            selected_ids.append(evidence_id)

    citations = [
        _map_citation(evidence_id, evidence_by_id[evidence_id])
        for evidence_id in selected_ids
    ]
    return GroundedAnswer(
        answer=raw_output.answer,
        citations=citations,
        has_sufficient_evidence=True,
    )


def _map_citation(
    evidence_id: str,
    result: RetrievalResult,
) -> GroundedCitation:
    return GroundedCitation(
        evidence_id=evidence_id,
        rank=result.rank,
        score=result.score,
        row_index=result.row_index,
        chunk_id=result.chunk_id,
        episode_id=result.episode_id,
        source_file=result.source_file,
        speakers=list(result.speakers),
        section_titles=list(result.section_titles),
        start_time=result.start_time,
        end_time=result.end_time,
        time_range_complete=result.time_range_complete,
        source_segment_ids=list(result.source_segment_ids),
        source_segment_start_index=result.source_segment_start_index,
        source_segment_end_index=result.source_segment_end_index,
        source_spans=[
            span.model_copy(deep=True) for span in result.source_spans
        ],
    )


def _refusal() -> GroundedAnswer:
    return GroundedAnswer(
        answer=REFUSAL_ANSWER,
        citations=[],
        has_sufficient_evidence=False,
    )
