# -*- coding: utf-8 -*-
"""超时归因 —— 区分「本层墙钟超时」与「内层 Crew 超时」。

**为什么需要单独一个模块**：C1 真实预演（任务 990170 / 990172）暴露出
`research_direction_failed` 的 `reason` **谎报**：日志写「超时（>240s）」，
可那个方向实际只跑了 134.8s / 143.0s，外层墙钟根本没触发。

根因（已读源码实证）：
  1. Python 3.11 起 `asyncio.TimeoutError is TimeoutError` → `True`（实测），
     所以 `except asyncio.TimeoutError` **同时**捕获本层 `wait_for` 的墙钟超时
     与 crewai 内部的 `max_execution_time` 超时，**类型上不可区分**。
  2. crewai 的 `kickoff_async` 实为 `asyncio.to_thread(self.kickoff)`
     （crewai/crew.py:1187），走同步 `execute_task` → `_execute_with_timeout`
     （crewai/agent/core.py:1045-1059），其 `with ThreadPoolExecutor()`
     退出时 `shutdown(wait=True)` 会阻塞到工作线程跑完，**超时既非硬中断、
     抛出时刻也远晚于声明上限**。
  3. 唯一可靠的指纹是**异常消息**：本层 `asyncio.wait_for` 抛出的
     `TimeoutError` 消息为空（实测 `str=''`、`args=()`）；crewai 内部超时
     消息非空，含 `execution timed out after`。

把这段判断抽成纯函数，是为了让回归测试能锁住它 —— 写在 `except` 块里
的逻辑没法被单元测试覆盖，而「日志如实归因」正是这次要修的东西。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeoutAttribution:
    """一次超时事件的归因结果。

    Attributes:
        internal: True = 内层（Crew/Agent 级）超时；False = 本层墙钟超时。
        detail:   内层超时的原始消息（本层墙钟超时时为空串）。
        reason:   写进事件 payload 的 `reason` 文案（前端直接渲染）。
    """

    internal: bool
    detail: str
    reason: str


def classify_timeout(
    exc: BaseException,
    limit_seconds: float,
    *,
    max_reason_detail: int = 180,
) -> TimeoutAttribution:
    """按异常消息区分内层/外层超时，并生成如实的 `reason` 文案。

    Args:
        exc: 被捕获的 `TimeoutError`（或 `asyncio.TimeoutError`，3.11 起同类）。
        limit_seconds: 本层墙钟上限，仅用于文案。
        max_reason_detail: 内层消息截断长度，防止超长 prompt 回显撑爆事件表。

    Returns:
        TimeoutAttribution。**宁可如实说"内部超时"，也不把内层超时谎报成本层墙钟。**
    """
    detail = str(exc).strip()
    if detail:
        # 内层（crewai Agent 级 max_execution_time）超时：消息非空。
        return TimeoutAttribution(
            internal=True,
            detail=detail,
            reason=f"内部超时: {detail[:max_reason_detail]}",
        )
    # 本层 asyncio.wait_for 墙钟超时：消息为空（实测 str=''、args=()）。
    return TimeoutAttribution(
        internal=False,
        detail="",
        reason=f"超时（>{limit_seconds:g}s）",
    )
