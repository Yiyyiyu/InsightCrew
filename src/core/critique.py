"""质疑回应率 —— Critic 的每条质疑是否被 Analyst 显式回应。

从 `orchestrator.py` 拆分而来（M1/M2 遗留缺陷第 7 项，纯搬移零行为变更）。

★ 这个指标是「多 Agent 协作」区别于「多 Agent 顺序调用」的证据：
   多 Agent 最常见的失败模式是互相附和（sycophancy），所以 Critic 提出的
   每条质疑都必须能在 Analyst 的回应段落里找到对应。
"""

import re



# Critic 的质疑条目：Markdown 有序列表（`1. xxx`）或无序列表（`- xxx`）。
_CRITIQUE_ITEM_RE = re.compile(r"^\s*(?:\d+[.、)]|[-*+])\s+\S", re.MULTILINE)
# 回应里对第 N 条质疑的显式表态：`质疑 3` / `质疑3` / `质疑 03` / `3.`（回应小节内）
_REBUTTAL_SECTION_RE = re.compile(r"##+\s*质疑回应\s*(.*)$", re.DOTALL)


def _critique_resolution_rate(rebuttal_output: str, critic_output: str) -> float:
    """计算「质疑被显式回应」的比例，范围 [0, 1]。

    这是 PRD G7 的 `critique_resolution_rate` 指标。口径刻意保守：

    - 分母 = Critic 输出里可识别的质疑条目数（有序/无序列表项）
    - 分子 = 回应小节里能对号入座表态的条目数
      （命中「质疑 N」或回应小节内以 `N.` 开头的行）

    只在 `## 质疑回应` 小节内找回应，避免把正文里偶发的「质疑」二字算成回应。
    Critic 没产出可识别条目时返回 0.0 —— 宁可记 0 也不编一个好看的数。
    """
    total = len(_CRITIQUE_ITEM_RE.findall(critic_output or ""))
    if total == 0:
        return 0.0

    section_match = _REBUTTAL_SECTION_RE.search(rebuttal_output or "")
    if not section_match:
        return 0.0
    section = section_match.group(1)

    answered = 0
    for i in range(1, total + 1):
        if re.search(rf"质疑\s*0?{i}\b", section) or re.search(
            rf"^\s*{i}\s*[.、)]", section, re.MULTILINE
        ):
            answered += 1
    return round(answered / total, 4)
