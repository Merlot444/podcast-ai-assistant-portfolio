"""Day 2 Prompt 扩展练习的 JSON 解析行为测试。"""

from app.prompt_demo import parse_json_result, parse_validated_result
from app.schemas.analysis import SummaryOutput, TagOutput, TitleIntroOutput


def test_parse_json_result_returns_dictionary() -> None:
    result = parse_json_result('{"summary":"测试摘要"}')

    assert result == {"summary": "测试摘要"}


def test_parse_json_result_does_not_report_raw_content(capsys) -> None:
    raw_content = "这不是合法 JSON"

    result = parse_json_result(raw_content)

    captured = capsys.readouterr()
    assert result is None
    assert "JSON 解析失败" in captured.err
    assert raw_content not in captured.err


def test_parse_json_result_rejects_non_object(capsys) -> None:
    result = parse_json_result('["标签"]')

    captured = capsys.readouterr()
    assert result is None
    assert "顶层内容不是对象" in captured.err


def test_standalone_prompt_outputs_are_validated_by_task_schema() -> None:
    summary = parse_validated_result(
        '{"summary":"结构正确的摘要"}',
        SummaryOutput,
    )
    tags = parse_validated_result(
        '{"tags":["大模型","应用开发","提示词"]}',
        TagOutput,
    )
    title_intro = parse_validated_result(
        '{"title":"Prompt 为什么重要","introduction":"讨论清晰约束和示例的作用。"}',
        TitleIntroOutput,
    )

    assert summary is not None
    assert summary.summary == "结构正确的摘要"
    assert tags is not None
    assert tags.tags == ["大模型", "应用开发", "提示词"]
    assert title_intro is not None
    assert title_intro.title == "Prompt 为什么重要"


def test_task_schema_rejects_wrong_fields_without_reporting_raw_content(capsys) -> None:
    raw_content = '{"tags":["只有一个"],"unexpected":true}'

    result = parse_validated_result(raw_content, TagOutput)

    captured = capsys.readouterr()
    assert result is None
    assert "JSON 结构校验失败" in captured.err
    assert raw_content not in captured.err
