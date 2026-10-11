"""检索子方向的派生 —— 从 Planner 产出的调研计划里解析出来。

**为什么需要这个模块（P0 缺陷）**：在此之前，检索子方向被写死在
`src/prompts/researcher.yaml` 的 `subdirections` 里（CrewAI / LangGraph /
AutoGen 三个固定框架）。这等于把「多 Agent 协作调研」这个产品做成了
**一个只能调研那三个框架的演示**：

  * 用户新建任意课题（比如「RAG 评估方法」「向量数据库选型」），
    research 阶段仍会去检索那三个框架，报告自然驴唇不对马嘴；
  * 更糟的是它**不报错**——mock 与真实模型都会"成功"返回三框架内容，
    只有人读到报告才会发现，属于静默错误。

修法：让 Planner 在计划末尾输出一个 ``json`` 代码块给出 subdirections，
本模块负责解析；解析失败时**按计划正文降级派生**，而不是回落到硬编码主题。

解析优先级（越靠前越可信）：
  1. 最后一个 ```json 代码块里的 ``{"subdirections": [...]}``
  2. 全文里任一 ``{"subdirections": ...}`` 片段（LLM 常忘记包代码块）
  3. 从计划正文的二级/三级标题派生（每个标题当一个方向）
  4. 兜底：单个方向，覆盖整个课题（保证流程能跑，绝不回落到固定主题）
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

# 单个子方向的硬上限：再多也不会更快（受 max_parallel_researchers 信号量约束），
# 只会让提示词更长、每方向分到的注意力更少。
MAX_SUBDIRECTIONS = 4
MIN_SUBDIRECTIONS = 1

# ```json ... ``` （允许 json 后有空格、允许大写）
_FENCE_RE = re.compile(r"```[ \t]*json[ \t]*\r?\n(.*?)```", re.S | re.I)
# 任何看起来像 {"subdirections": [...]} 的片段（非贪婪到第一个匹配的 ]}）
_BARE_RE = re.compile(r'\{\s*"subdirections"\s*:\s*\[.*?\]\s*\}', re.S)
# Markdown 标题（## / ###），用于降级派生
_HEADING_RE = re.compile(r"^#{2,3}\s+(.+?)\s*$", re.M)

# 标题里的噪音词：这些标题不是检索方向（是计划的固定小节）
_NOT_A_DIRECTION = (
    "调研目标", "核心问题", "检索策略", "检索关键词", "目标信息来源",
    "预期产出", "预期报告", "报告结构", "输出格式", "子方向", "subdirection",
    "信息来源", "关键词", "大纲", "结论", "建议", "参考",
)


def _clean_name(raw: Any, fallback: str) -> str:
    """把 LLM 给的 name 洗成一个短标签（用于角色名与事件流展示）。"""
    text = str(raw or "").strip()
    # 去掉 markdown 强调与列表符号
    text = text.strip("*_`#-— \t")
    # 去掉可能带的「框架」「方向」后缀噪音（保留原文，只截断过长）
    if not text:
        return fallback
    return text[:24]


def _clean_desc(raw: Any) -> str:
    text = str(raw or "").strip()
    return text or "围绕本课题的公开资料进行检索与取证"


def _normalize(items: Any) -> list[dict]:
    """把 LLM 返回的 subdirections 列表规范化成内部结构。

    容忍 LLM 的常见变体：字符串列表、缺 expected、name 过长、多余键。
    """
    if isinstance(items, dict):
        # 有的模型会返回 {"0": {...}, "1": {...}}
        items = list(items.values())
    if not isinstance(items, list):
        return []

    out: list[dict] = []
    for i, item in enumerate(items, start=1):
        if isinstance(item, str):
            name = _clean_name(item, f"方向{i}")
            desc = _clean_desc(item)
        elif isinstance(item, dict):
            name = _clean_name(
                item.get("name") or item.get("title") or item.get("direction"),
                f"方向{i}",
            )
            desc = _clean_desc(
                item.get("desc") or item.get("description") or item.get("query")
                or item.get("keywords")
            )
        else:
            continue

        expected = ""
        if isinstance(item, dict):
            expected = str(item.get("expected") or item.get("expected_output") or "").strip()
        if not expected:
            expected = f"{name} 方向的检索摘要，含至少 3 条带引用标记的发现。"

        out.append({
            "name": name,
            "desc": desc,
            "expected": expected,
            "async": bool(item.get("async", False)) if isinstance(item, dict) else False,
        })
        if len(out) >= MAX_SUBDIRECTIONS:
            break
    return out


def _try_json(blob: str) -> Optional[list[dict]]:
    """解析一段 JSON 文本，取出 subdirections。失败返回 None。"""
    try:
        data = json.loads(blob)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    items = data.get("subdirections")
    if items is None:
        return None
    parsed = _normalize(items)
    return parsed or None


def _from_headings(plan: str) -> list[dict]:
    """降级派生：把计划正文的标题当作检索方向。

    Planner 的第 2 项就是「关键对比维度（至少5个维度）」，所以正文标题里
    天然含有可用的方向信息。这里只取「不像固定小节」的标题。
    """
    out: list[dict] = []
    for raw in _HEADING_RE.findall(plan or ""):
        title = raw.strip().strip("*_` ")
        if not title or len(title) > 40:
            continue
        if any(noise in title for noise in _NOT_A_DIRECTION):
            continue
        name = _clean_name(title, f"方向{len(out) + 1}")
        if any(x["name"] == name for x in out):
            continue
        out.append({
            "name": name,
            "desc": f"围绕「{title}」检索公开资料，收集中英文来源并提取可引用证据",
            "expected": f"{title} 的检索摘要，含至少 3 条带引用标记的发现。",
            "async": False,
        })
        if len(out) >= MAX_SUBDIRECTIONS:
            break
    return out


def derive_subdirections(plan: str, topic: str = "") -> list[dict]:
    """从调研计划里派生检索子方向。

    Args:
        plan: Planner 阶段产出的计划全文（Markdown）。
        topic: 课题名，仅在完全无法派生时用于生成兜底方向。

    Returns:
        至少 1 个、至多 ``MAX_SUBDIRECTIONS`` 个 ``{name, desc, expected, async}``。
        **永不返回空**，也**永不返回固定主题**——最坏情况是单方向覆盖全课题。
    """
    text = plan or ""

    # 1) 代码块里的 json
    for block in _FENCE_RE.findall(text):
        parsed = _try_json(block.strip())
        if parsed:
            return parsed

    # 2) 裸的 {"subdirections": [...]}
    for m in _BARE_RE.findall(text):
        parsed = _try_json(m)
        if parsed:
            return parsed

    # 3) 从正文标题派生
    from_headings = _from_headings(text)
    if from_headings:
        return from_headings

    # 4) 兜底：单方向覆盖整个课题
    label = (topic or "本课题").strip()[:24]
    return [{
        "name": label or "综合检索",
        "desc": (
            f"围绕课题「{topic or '（未命名）'}」检索公开资料"
            "（arXiv、GitHub、官方文档、技术博客），收集中英文来源并提取可引用证据"
        ),
        "expected": f"{label} 的检索摘要，含至少 3 条带引用标记的发现。",
        "async": False,
    }]


def subdirection_names(plan: str, topic: str = "") -> list[str]:
    """派生子方向并只取名称（供编排层打印/统计）。"""
    return [s["name"] for s in derive_subdirections(plan, topic)]


__all__ = [
    "MAX_SUBDIRECTIONS",
    "derive_subdirections",
    "subdirection_names",
]
