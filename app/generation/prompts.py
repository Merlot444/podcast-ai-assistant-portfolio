"""集中管理播客分析所使用的 Prompt。"""

from __future__ import annotations

from openai.types.chat import ChatCompletionMessageParam


ANALYSIS_SYSTEM_PROMPT = """你是一名严谨的播客内容编辑。
请只分析用户提供的文字，不补充外部事实或猜测。
请仅输出一个 JSON 对象，不要输出 Markdown 代码块或额外说明。

JSON 格式示例：
{
  "summary": "用一到两句话概括片段，不超过 120 个汉字",
  "keywords": ["关键词1", "关键词2", "关键词3"],
  "tags": ["主题标签1", "主题标签2"]
}

要求：
1. summary 必须有内容；
2. keywords 返回 3 到 5 个字符串；
3. tags 返回 2 到 4 个字符串；
4. 不得添加输入中没有出现或无法从输入直接归纳的事实。
"""


SUMMARY_SYSTEM_PROMPT = """【角色】
你是一名严谨的中文播客内容编辑。

【任务】
根据用户提供的播客文字，写出一段准确、简洁的摘要。

【约束】
1. 只使用输入中明确出现的信息，不补充外部事实，不猜测。
2. 摘要使用中文，控制在 80 个汉字以内。
3. 不要输出解释、前缀或 Markdown 代码块。

【输出格式】
只返回以下结构的合法 JSON 对象：
{"summary": "摘要内容"}
"""


TAG_SYSTEM_PROMPT = """【角色】
你是一名负责内容分类的中文播客编辑。

【任务】
根据用户提供的播客文字生成主题标签。

【约束】
1. 只根据输入内容归纳，不补充无关主题。
2. 返回 3 到 5 个标签，每个标签不超过 8 个汉字。
3. 标签应简洁、具体，并去除重复项。
4. 不要输出解释、前缀或 Markdown 代码块。

【输出格式】
只返回以下结构的合法 JSON 对象：
{"tags": ["标签1", "标签2", "标签3"]}
"""


TITLE_INTRO_SYSTEM_PROMPT = """【角色】
你是一名擅长内容包装、但不夸大事实的中文播客编辑。

【任务】
根据用户提供的播客文字生成一个标题和一段节目简介。

【约束】
1. 只使用输入中的信息，不使用标题党，不虚构嘉宾观点。
2. 标题不超过 24 个汉字。
3. 简介为 1 到 2 句话，不超过 100 个汉字。
4. 不要输出解释、前缀或 Markdown 代码块。

【输出格式】
只返回以下结构的合法 JSON 对象：
{"title": "播客标题", "introduction": "节目简介"}
"""


TAG_FEW_SHOT_INPUT = """主持人：为什么保持稳定睡眠很重要？
嘉宾：固定作息有助于白天保持精力，也能减少入睡困难。"""

TAG_FEW_SHOT_OUTPUT = '{"tags":["睡眠健康","固定作息","精力管理"]}'


def _clean_transcript(transcript: str) -> str:
    """清理输入，并尽早拒绝空文本。"""
    text = transcript.strip()
    if not text:
        raise ValueError("待处理的播客文字不能为空。")
    return text


def build_summary_messages(transcript: str) -> list[ChatCompletionMessageParam]:
    """构造摘要任务的 Zero-shot Prompt。"""
    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": _clean_transcript(transcript)},
    ]


def build_tag_messages(
    transcript: str,
    *,
    few_shot: bool = False,
) -> list[ChatCompletionMessageParam]:
    """构造标签 Prompt；可选择 Zero-shot 或 Few-shot。"""
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": TAG_SYSTEM_PROMPT}
    ]
    if few_shot:
        # 先给模型一个输入输出示例，帮助它稳定遵循 JSON 结构。
        messages.extend(
            [
                {"role": "user", "content": TAG_FEW_SHOT_INPUT},
                {"role": "assistant", "content": TAG_FEW_SHOT_OUTPUT},
            ]
        )
    messages.append({"role": "user", "content": _clean_transcript(transcript)})
    return messages


def build_title_intro_messages(
    transcript: str,
) -> list[ChatCompletionMessageParam]:
    """构造标题与简介任务的 Zero-shot Prompt。"""
    return [
        {"role": "system", "content": TITLE_INTRO_SYSTEM_PROMPT},
        {"role": "user", "content": _clean_transcript(transcript)},
    ]
