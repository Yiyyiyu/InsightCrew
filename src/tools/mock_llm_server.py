"""
Mock LLM Server — 模拟 OpenAI 兼容的 LLM API
监听 127.0.0.1:8001，返回结构化假数据供开发调试使用。
启动: python -m src.tools.mock_llm_server
"""

import json
import re
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

HOST = "127.0.0.1"
PORT = 8001


class MockLLMHandler(BaseHTTPRequestHandler):

    def _set_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "*")
        self.send_header("Access-Control-Allow-Headers", "*")

    def _json_response(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._set_cors()
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors()
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/v1/models":
            self._json_response({
                "object": "list",
                "data": [
                    {"id": "deepseek-flash", "object": "model"},
                    {"id": "deepseek-v4-pro", "object": "model"},
                    {"id": "glm-4v-plus", "object": "model"},
                    {"id": "gpt-4o-mini", "object": "model"},
                    {"id": "text-embedding-3-small", "object": "model"},
                ]
            })
        elif path == "/health":
            self._json_response({"status": "ok", "service": "mock-llm"})
        else:
            self._json_response({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if "/embeddings" in path:
            dim = 1536
            if "text-embedding-3-large" in str(body.get("model", "")):
                dim = 3072
            elif "embedding-3" in str(body.get("model", "")):
                dim = 2048
            self._json_response({
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1] * dim}],
                "model": body.get("model", "unknown"),
                "usage": {"prompt_tokens": 10, "total_tokens": 10},
            })
            return

        # --- 聊天补全（非流式）---
        model = body.get("model", "deepseek-flash")
        messages = body.get("messages", [])
        last_content = messages[-1]["content"] if messages else ""
        text = last_content if isinstance(last_content, str) else str(last_content)

        # 检测工具调用请求
        tools = body.get("tools", [])
        if tools and "get_weather" in str(tools):
            self._json_response({
                "id": "mock-call-001",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "mock-tool-001",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps({"city": "北京"}, ensure_ascii=False),
                            },
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
            })
            return

        # 检测 JSON 模式
        is_json = body.get("response_format", {}).get("type") == "json_object"

        # 根据关键词返回不同假数据
        content = self._mock_reply(text, model, is_json)
        self._json_response({
            "id": "mock-cmpl-001",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 50, "completion_tokens": 80, "total_tokens": 130},
        })

    def _mock_reply(self, text: str, model: str, is_json: bool) -> str:
        """根据输入关键词返回对应的假数据"""
        text_lower = text.lower()

        if is_json:
            return json.dumps({"result": "ok", "message": "模拟JSON响应"}, ensure_ascii=False)

        if "研究计划" in text or "plan" in text_lower:
            return """## 调研计划

1. **框架概述**：对比 CrewAI、LangGraph、AutoGen 的核心理念与架构
2. **协作模式**：分析三种框架的 Agent 协作机制（层级/图/对话）
3. **工具生态**：对比内置工具、第三方集成能力
4. **工程化**：评估部署、监控、错误处理等生产级能力
5. **社区与生态**：对比 GitHub Stars、贡献者、文档质量
6. **选型建议**：基于不同场景给出推荐"""

        if "检索" in text or "search" in text_lower:
            return """## 检索结果

### CrewAI
- GitHub: https://github.com/crewAIInc/crewAI (25k+ Stars)
- 核心理念：基于角色的多Agent协作框架，通过Crew组织Agent团队
- 最新版本：1.15.25

### LangGraph
- GitHub: https://github.com/langchain-ai/langgraph (8k+ Stars)
- 核心理念：基于图的Agent工作流编排，支持循环与条件分支
- 最新版本：0.2.x

### AutoGen
- GitHub: https://github.com/microsoft/autogen (35k+ Stars)
- 核心理念：多Agent对话框架，支持人机协作
- 最新版本：0.8.x"""

        if "对比" in text or "compare" in text_lower:
            return """## 对比矩阵

| 维度 | CrewAI | LangGraph | AutoGen |
|------|--------|-----------|---------|
| 架构模式 | 角色协作 | 有向图 | 对话式 |
| 学习曲线 | 低 | 中 | 中 |
| 灵活性 | 中 | 高 | 高 |
| 工具集成 | 丰富 | 灵活 | 中等 |
| 调试能力 | 中等 | 强 | 中等 |
| 社区活跃 | 高 | 高 | 高 |
| 生产就绪 | ✅ | ⚠️ | ⚠️ |

**关键发现**：三种框架不是竞争关系，而是面向不同场景的设计选择。"""

        if "质疑" in text or "critic" in text_lower:
            return """## 质疑与挑战

1. **CrewAI 的真实协作深度**：角色分工是否真正带来了超出单Agent的增益？需要定量对比实验证据。
2. **LangGraph 的工程复杂度**：图编排的灵活性是否以增加调试难度为代价？
3. **AutoGen 的对话效率**：多Agent对话是否存在token浪费问题？
4. **引用支撑不足**：以上对比结论需要更多生产环境的实际案例支撑。"""

        if "报告" in text or "write" in text_lower or "撰写" in text:
            return """## 多Agent框架选型调研报告

### 摘要
本报告对比了 CrewAI、LangGraph、AutoGen 三个主流多Agent框架，从架构设计、协作模式、工程化能力、社区生态等维度进行系统性评估。

### 背景与动机
随着 LLM 能力的快速提升，多Agent协作成为构建复杂AI应用的关键范式。选择合适的框架对项目成功至关重要。

### 对比分析
...（详细对比见对比矩阵部分）

### 选型建议
- **快速原型验证**：推荐 CrewAI
- **复杂工作流编排**：推荐 LangGraph
- **研究实验与对话式场景**：推荐 AutoGen

### 引用
- CrewAI 官方文档
- LangGraph GitHub 仓库
- AutoGen 研究论文"""

        # 默认回复
        return f"Mock LLM ({model}) 已收到您的输入。当前为模拟模式，返回预设内容以供开发调试。"


def main():
    server = HTTPServer((HOST, PORT), MockLLMHandler)
    print(f"🧪 Mock LLM Server 已启动: http://{HOST}:{PORT}")
    print(f"   Chat endpoint: POST {HOST}:{PORT}/v1/chat/completions")
    print(f"   Models endpoint: GET {HOST}:{PORT}/v1/models")
    print(f"   Embedding endpoint: POST {HOST}:{PORT}/v1/embeddings")
    print(f"   Health: GET {HOST}:{PORT}/health")
    print("按 Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nMock LLM Server 已停止")
        server.server_close()


if __name__ == "__main__":
    main()
