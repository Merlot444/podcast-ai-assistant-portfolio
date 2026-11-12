"""Day 2 扩展：演示摘要、标签、标题与简介的结构化生成。"""

from __future__ import annotations

import json
import sys
from typing import Any, TypeVar

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError

from app.core.config import ConfigError, Settings
from app.core.errors import safe_error_message
from app.core.llm import ModelServiceError, create_client, create_json_completion
from app.generation.prompts import (
    build_summary_messages,
    build_tag_messages,
    build_title_intro_messages,
)
from app.schemas.analysis import SummaryOutput, TagOutput, TitleIntroOutput


SAMPLE_TRANSCRIPT = """主持人：大模型应用为什么需要清晰的 Prompt？
嘉宾：因为角色、任务、约束和输出格式越明确，模型的回答通常越稳定。
主持人：Few-shot 又有什么作用？
嘉宾：它通过少量示例告诉模型我们期待的输入输出形式。
"""

OutputModel = TypeVar("OutputModel", bound=BaseModel)


def parse_json_result(raw_content: str) -> dict[str, Any] | None:
    """使用 json.loads 解析结果；失败时不回显模型原文。"""
    try:
        result = json.loads(raw_content)
    except json.JSONDecodeError:
        print("JSON 解析失败：模型返回的内容不是合法 JSON。", file=sys.stderr)
        return None

    if not isinstance(result, dict):
        print("JSON 解析失败：模型返回的顶层内容不是对象。", file=sys.stderr)
        return None
    return result


def parse_validated_result(
    raw_content: str,
    output_model: type[OutputModel],
) -> OutputModel | None:
    """Parse JSON and validate the fields required by one Prompt task."""
    payload = parse_json_result(raw_content)
    if payload is None:
        return None
    try:
        return output_model.model_validate(payload)
    except ValidationError:
        print("JSON 结构校验失败。", file=sys.stderr)
        return None


def run_task(
    client: OpenAI,
    settings: Settings,
    task_name: str,
    messages: list[ChatCompletionMessageParam],
    output_model: type[OutputModel],
) -> OutputModel | None:
    """执行一个生成任务；单项失败不会中断其余演示。"""
    print(f"\n=== {task_name} ===")
    try:
        response = create_json_completion(
            client,
            model=settings.model_fast,
            messages=messages,
        )
    except ModelServiceError as error:
        print(f"模型调用失败：{safe_error_message(error)}", file=sys.stderr)
        return None

    if not response.choices:
        print("模型调用失败：响应中没有可用结果。", file=sys.stderr)
        return None

    choice = response.choices[0]
    if choice.finish_reason == "length":
        print("模型输出达到长度上限，JSON 可能被截断。", file=sys.stderr)
        return None

    raw_content = choice.message.content or ""
    result = parse_validated_result(raw_content, output_model)
    if result is not None:
        print(
            json.dumps(
                result.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
        )
    return result


def main() -> None:
    """加载一次客户端，依次运行四个 Prompt 练习。"""
    try:
        settings = Settings.from_env()
        client = create_client(settings)
    except ConfigError as error:
        print(f"启动失败：{safe_error_message(error)}", file=sys.stderr)
        return

    tasks = [
        (
            "播客摘要（Zero-shot）",
            build_summary_messages(SAMPLE_TRANSCRIPT),
            SummaryOutput,
        ),
        (
            "播客标签（Zero-shot）",
            build_tag_messages(SAMPLE_TRANSCRIPT),
            TagOutput,
        ),
        (
            "播客标签（Few-shot）",
            build_tag_messages(SAMPLE_TRANSCRIPT, few_shot=True),
            TagOutput,
        ),
        (
            "播客标题与简介（Zero-shot）",
            build_title_intro_messages(SAMPLE_TRANSCRIPT),
            TitleIntroOutput,
        ),
    ]
    for task_name, messages, output_model in tasks:
        run_task(client, settings, task_name, messages, output_model)


if __name__ == "__main__":
    main()
