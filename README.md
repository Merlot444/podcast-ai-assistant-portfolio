# Podcast AI Assistant

一个面向长播客转写的 Python 助手。它把回答和摘要关联到带时间戳的转写片段，读者可以回查原文。证据不足时，演示会返回明确的拒答结果。

- **检索与引用** · 转写解析、分块、向量检索与 BM25，以及回答引用校验。
- **Agent 工具调用** · 用工具白名单和执行预算约束工具调用，提供可独立运行的 MCP 示例。
- **服务接口** · 用 FastAPI 提供检索和问答接口，并提供同源演示页面。

## 本地演示

需要 Python 3.11 或 3.13。从仓库根目录运行：

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
python -m scripts.portfolio.demo
```

演示用两段原创虚构播客转写，覆盖有引用的问答、证据不足拒答和整集摘要。默认使用预设模型响应，无需 API Key。转写解析、切块、检索和引用校验会实际运行。更多用法见[演示说明](scripts/portfolio/README.md)。

## 离线测试

```sh
python -m pip install '.[dev,api,mcp,experiment]'
python -m pytest -q
```

作品集附带可独立运行的测试。依赖历史评测材料和内部文档的用例留在原项目中。

## 代码入口

| 模块 | 位置 |
| --- | --- |
| 检索、应用与工具调用 | [app](app/) |
| FastAPI | [app/api](app/api/) |
| MCP 示例 | [experiments/mcp_minimal](experiments/mcp_minimal/) |
| 离线演示 | [scripts/portfolio](scripts/portfolio/) |

代码与原创演示样例按 [MIT](LICENSE) 提供。
