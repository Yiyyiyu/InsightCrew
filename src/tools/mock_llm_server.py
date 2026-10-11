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

# ── 从入参里抽取课题 / 子方向名（假数据不得写死具体课题）──
# plan 阶段提示词里有「课题：$topic」行；research 阶段有
# 「【检索任务 i/n — NAME】」。抽不到时返回通用措辞，绝不回落固定主题。
_TOPIC_RE = re.compile(r"^课题[：:]\s*(.+?)\s*$", re.M)
_RESEARCH_RE = re.compile(r"【(?:你的)?检索任务\s*\d+\s*/\s*\d+\s*—\s*(.+?)】")


def _topic_of(text: str, default: str = "该调研课题") -> str:
    m = _TOPIC_RE.search(text)
    return m.group(1).strip() if m else default


def _research_name(text: str) -> str:
    m = _RESEARCH_RE.search(text)
    return m.group(1).strip() if m else ""


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
                    {"id": "glm-4v-flash", "object": "model"},
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
        """根据输入关键词返回对应的假数据。

        ★ 分发必须按「**本阶段的指令**」而不是通用关键词。

        原因（实测踩过）：每个阶段的提示词都会**内嵌上一阶段的输出**。
        例如 critic 的提示词里带着 analysis 的全文，而 analysis 的假数据开头
        就是「## 检索结果」——若按通用关键词分发，「检索」分支会抢先命中，
        critic 阶段于是返回了检索结果而不是质疑。同理 write 会被「研究计划」
        抢先、rebuttal 会被「检索」抢先。

        因此这里用**只有该阶段指令才含有的措辞**做唯一标识，通用关键词
        只作为兜底（供其它工具用任意文本调用 mock 时使用）。

        ★ 假数据**不得写死任何具体课题**（P0 硬编码缺陷）：以前这里把每个
        阶段都填成 CrewAI / LangGraph / AutoGen 三个框架的演示内容，于是
        mock 模式下不管调研什么课题，产出都是那三个框架，掩盖了真实问题。
        现在正文一律从**入参里的课题**派生（plan 阶段抽「课题：」行，
        research 阶段抽【检索任务 … — NAME】里的 NAME），抽不到就用通用措辞。
        """
        text_lower = text.lower()

        if is_json:
            return json.dumps({"result": "ok", "message": "模拟JSON响应"}, ensure_ascii=False)

        topic = _topic_of(text)

        # ── 阶段唯一标识（按 STAGES 顺序，各标识只出现在该阶段的指令里）──
        # plan：planner.yaml "请制定一份结构化的多Agent协作调研计划"
        if "结构化的多Agent协作调研计划" in text:
            return f"""## 调研计划

**课题**：{topic}

### 1. 调研目标与核心问题
- 明确「{topic}」的选型边界与关键决策点
- 核心问题：在给定约束下，哪些候选对象最契合目标场景

### 2. 关键对比维度
- 能力覆盖、成熟度、集成成本、运维复杂度、社区活跃度

### 3. 中英文检索关键词
- `{topic}` / `{topic} comparison` / `{topic} benchmark`

### 4. 目标信息来源
- arXiv、官方文档、GitHub 仓库、技术博客

### 5. 预期报告结构大纲
- 背景与动机 → 对象概述 → 对比矩阵 → 差距分析 → 推荐方向 → 结论

### 6. 检索子方向

```json
{{"subdirections": [
  {{"name": "现状与能力", "desc": "检索 {topic} 各候选对象的能力边界与现状（中英文关键词）", "expected": "各对象的能力清单，含至少 3 条带引用标记的发现。"}},
  {{"name": "对比与代价", "desc": "检索 {topic} 在集成成本、性能与运维复杂度上的对比证据", "expected": "对比结论，含至少 3 条带引用标记的发现。"}}
]}}
```"""

        # research：researcher.yaml "【检索任务 $index/$total — $name】"
        if "【检索任务" in text or "【你的检索任务" in text:
            name = _research_name(text) or topic
            return f"""## 检索结果：{name}

### 发现 1
围绕「{name}」检索到若干一手材料，官方文档与仓库 README 给出了能力边界说明。
- 来源：https://example.org/docs
- 要点：能力范围、限制条件与适用场景

### 发现 2
社区讨论与评测文章提供了实践反馈，可作为定性参考。
- 来源：https://example.org/benchmark
- 要点：与相邻方案的取舍

### 发现 3
arXiv 预印本给出了方法论层面的分析（若检索到相关论文）。
- 来源：https://arxiv.org/abs/0000.00000
- 要点：评价维度与结论边界

> 注：以上为 mock 模式下的占位内容，真实运行时会由检索工具填入实际来源。"""

        # analysis：analyst.yaml "请完成以下分析："
        if "请完成以下分析" in text:
            return f"""## 对比矩阵

围绕「{topic}」，按计划确定的维度整理如下（mock 占位）：

| 维度 | 候选 A | 候选 B |
|------|--------|--------|
| 能力覆盖 | 中 | 高 |
| 成熟度 | 高 | 中 |
| 集成成本 | 低 | 中 |
| 运维复杂度 | 低 | 高 |

**关键发现**：候选对象的差异主要体现在集成成本与运维复杂度上，需结合约束取舍。"""

        # critic：critic.yaml "请以红队身份，从以下角度提出质疑"
        if "请以红队身份" in text:
            return f"""## 质疑与挑战

1. **证据强度**：关于「{topic}」的对比结论目前多为定性描述，缺少定量基准。
2. **适用边界**：结论未说明在何种规模/约束下成立，存在过度外推风险。
3. **时效性**：候选对象的版本迭代较快，结论可能随版本失效。
4. **引用支撑不足**：部分论断没有落到具体来源，需要补充引用。"""

        # rebuttal：analyst.yaml rebuttal 块 "请**逐条回应**每一条质疑"
        if "逐条回应" in text:
            return f"""## 对比分析（修订版）

（此处为修订后的对比矩阵与结论，已在下方「质疑回应」中逐条表态。）

## 质疑回应

质疑 1 → **接受**。定量基准确实缺失，已将相关结论标注为定性判断。
质疑 2 → **部分接受**。已补充结论成立的规模与约束前提。
质疑 3 → **接受**。已在报告中标注版本与检索时间，提示时效风险。
质疑 4 → **暂无证据支撑**。当前检索范围内没有对应来源，已明确标注缺乏证据。"""

        # write：writer.yaml "请撰写一份完整的中文技术调研报告"
        if "请撰写一份完整的中文技术调研报告" in text:
            return f"""## {topic} 调研报告

### 摘要
本报告围绕「{topic}」，从能力覆盖、成熟度、集成成本、运维复杂度与社区活跃度等维度进行系统性梳理与对比。

### 背景与动机
随着技术选型空间扩大，如何在本课题的约束下做出可辩护的选择，是本次调研要回答的核心问题。

### 对象概述
按调研计划确定的对象逐一介绍（mock 模式下为占位内容）。

### 对比分析
详见对比矩阵部分。

### 差距与缺口分析
现有材料以定性描述为主，缺少统一的定量基准。

### 推荐方向
结合约束条件给出分场景建议。

### 结论与展望
选型结论应随版本演进定期复核。

> 注：以上为 mock 模式下的占位内容，真实运行时会由各阶段产出替换。"""

        # review：reviewer.yaml "请从以下维度审核报告质量"
        if "请从以下维度审核报告质量" in text:
            return """## 审核报告

**总体评价**：有条件通过

| 维度 | 评分 |
|------|------|
| 引用完整性 | 3/5 |
| 逻辑连贯性 | 4/5 |
| 结构完整性 | 5/5 |
| 语言质量 | 4/5 |
| 结论支撑 | 3/5 |

**问题清单**：
1. 部分结论缺少引用支撑
2. 推荐结论未说明适用边界

**修改建议**：补充可回溯引用，并明确结论成立的前提。"""

        # ── 通用关键词兜底（供任意文本调用 mock 时使用）──
        if "研究计划" in text or "plan" in text_lower:
            return f"""## 调研计划

**课题**：{topic}

1. 调研目标与核心问题
2. 关键对比维度
3. 中英文检索关键词
4. 目标信息来源
5. 预期报告结构大纲
6. 检索子方向

```json
{{"subdirections": [
  {{"name": "现状与能力", "desc": "检索 {topic} 的现状", "expected": "含引用标记的能力清单。"}},
  {{"name": "对比与代价", "desc": "检索 {topic} 的对比证据", "expected": "含引用标记的对比结论。"}}
]}}
```"""

        if "检索" in text or "search" in text_lower:
            name = _research_name(text) or topic
            return f"""## 检索结果：{name}

- 来源：https://example.org/docs
- 要点：围绕「{name}」的能力范围与限制

> 注：以上为 mock 模式下的占位内容。"""

        if "对比" in text or "compare" in text_lower:
            return f"""## 对比矩阵

围绕「{topic}」按计划维度整理的占位对比表。

**关键发现**：差异主要落在集成成本与运维复杂度上。"""

        if "质疑" in text or "critic" in text_lower:
            return f"""## 质疑与挑战

1. **证据强度**：关于「{topic}」的结论缺少定量基准。
2. **适用边界**：未说明结论在何种约束下成立。
3. **引用支撑不足**：部分论断未落到具体来源。"""

        if "报告" in text or "write" in text_lower or "撰写" in text:
            return f"""## {topic} 调研报告

### 摘要
本报告围绕「{topic}」展开系统性梳理与对比。

### 背景与动机
说明本课题的选型背景与决策目标。

### 对比分析
详见对比矩阵部分。

### 结论与展望
结论应随版本演进定期复核。

> 注：以上为 mock 模式下的占位内容。"""

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
