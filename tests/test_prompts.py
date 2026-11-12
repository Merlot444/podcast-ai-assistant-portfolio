"""Day 3 Prompt 的结构测试。"""

import json

import pytest

from app.generation.prompts import (
    build_summary_messages,
    build_tag_messages,
    build_title_intro_messages,
)


@pytest.mark.parametrize(
    "builder",
    [build_summary_messages, build_tag_messages, build_title_intro_messages],
)
def test_prompts_include_four_required_parts(builder) -> None:
    messages = builder("一段播客文字")
    system_prompt = messages[0]["content"]

    assert "【角色】" in system_prompt
    assert "【任务】" in system_prompt
    assert "【约束】" in system_prompt
    assert "【输出格式】" in system_prompt
    assert "JSON" in system_prompt
    assert "Markdown 代码块" in system_prompt


def test_tag_zero_shot_has_no_example() -> None:
    messages = build_tag_messages("一段播客文字")

    assert [message["role"] for message in messages] == ["system", "user"]


def test_tag_few_shot_has_valid_json_example() -> None:
    messages = build_tag_messages("一段播客文字", few_shot=True)

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    example = json.loads(messages[2]["content"])
    assert len(example["tags"]) == 3


def test_empty_transcript_is_rejected() -> None:
    with pytest.raises(ValueError, match="不能为空"):
        build_summary_messages("  ")
