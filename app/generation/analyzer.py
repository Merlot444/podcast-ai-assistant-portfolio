"""Generate and validate a structured analysis for a short transcript."""

from __future__ import annotations

import json

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import ValidationError

from app.core.config import Settings
from app.core.llm import create_json_completion
from app.generation.prompts import ANALYSIS_SYSTEM_PROMPT
from app.schemas.analysis import PodcastAnalysis


class AnalysisOutputError(RuntimeError):
    """Raised when model output cannot become a valid analysis."""


def build_messages(transcript: str) -> list[ChatCompletionMessageParam]:
    """Build the role-separated prompt for one transcript excerpt."""
    text = transcript.strip()
    if not text:
        raise ValueError("待分析的播客文字不能为空。")
    return [
        {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "请把下面的播客片段分析为指定 JSON：\n\n" + text,
        },
    ]


def parse_analysis(raw_content: str) -> PodcastAnalysis:
    """Parse JSON syntax, then validate its business structure."""
    if not raw_content or not raw_content.strip():
        raise AnalysisOutputError("模型返回了空内容，请调整 Prompt 后重试。")
    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError:
        raise AnalysisOutputError("模型返回的内容不是合法 JSON。") from None

    try:
        return PodcastAnalysis.model_validate(payload)
    except ValidationError:
        raise AnalysisOutputError(
            "模型返回的 JSON 不符合摘要、关键词和标签结构。"
        ) from None


def analyze_transcript(
    client: OpenAI,
    settings: Settings,
    transcript: str,
) -> PodcastAnalysis:
    """Call the fast model and return only validated data."""
    response = create_json_completion(
        client,
        model=settings.model_fast,
        messages=build_messages(transcript),
    )
    if not response.choices:
        raise AnalysisOutputError("模型响应中没有可用结果。")

    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise AnalysisOutputError("模型输出达到长度上限，JSON 可能被截断。")

    return parse_analysis(choice.message.content or "")
