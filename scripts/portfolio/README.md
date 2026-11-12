# Podcast AI Assistant · 五分钟体验

用两段原创模拟播客体验带引用的问答、证据不足拒答和整期摘要。需要 Python 3.11 或 3.13。

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
python -m scripts.portfolio.demo
```

Windows 可用 `.venv\Scripts\activate` 激活环境。首次安装需要访问 Python 包仓库。默认演示无需 API Key，运行时没有供应商调用。

| 场景 | 预期结果 |
| --- | --- |
| `qa` | 罗勒花盆容量为 12 升，引用显示 0–26 秒、说话人和原文。 |
| `refusal` | 转写没有年度预算数据，返回证据不足且无引用。 |
| `summary` | 摘要覆盖登记和维修交接两个章节，分别附上原文和时间戳。 |

命令末尾加场景名可单独运行。离线模式使用预先写好的模型响应，实际执行转写解析、章节切块、BM25 查询或整集证据加载，以及应用端引用校验。它用来复现流程，项目的默认问答检索方案为 Vector Top-5。

真实模型模式可选。通过环境变量设置 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL_FAST` 和 `LLM_MODEL_STRONG`，然后运行 `python -m scripts.portfolio.demo --live`。这会把原创样例和问题发给配置的模型供应商，可能产生费用。实际回答随模型而异。

两份 `samples/*.json` 是为本演示编写的虚构转写，与演示代码一同按 MIT 提供。历史评测材料和原 Git 历史不在本演示包中。
