"""运行上下文 —— 让工具层能知道"当前在为哪个任务干活"。

## 为什么需要

工具（`src/tools/*.py`）运行在 CrewAI 的调用链里，签名由 LLM 决定
（`read_image(image_path=...)`），**没有地方塞 task_id**。但工具内部
（如多模态读图）会真实产生 LLM 费用，必须记进 `llm_calls` 并累加到
`tasks.total_cost_cny`，否则成本台账漏账、预算熔断失灵。

用 ContextVar 而不是全局变量：research 阶段是 asyncio 并发多方向，
全局变量会被并发任务互相覆盖；ContextVar 按任务上下文隔离。

## 注意

ContextVar 依赖执行上下文传播。CrewAI 若把工具调用丢进**不复制的**
线程池，值会读不到 —— 此时 `current_task_id()` 返回 0，记账降级为
"只记日志不落库"，不会写脏数据（见 `cost.record_usage_sync`）。
"""

from __future__ import annotations

from contextvars import ContextVar

_current_task_id: ContextVar[int] = ContextVar("insightcrew_task_id", default=0)

# 任务级模型分配表 `{role: ModelOption}`（M2.5：用户逐角色自选模型）。
# 由 execute_stage 在构造 Crew 之前写入，使 `_resolve_agent_config`
# （唯一构造 CrewAI LLM 的地方）能读到本任务每个角色用哪个模型。
# 值为 **已解析的 ModelOption 对象**（不是 id 字符串）—— 构造 LLM 需要
# model_id / api_key / api_base / 生成参数，解析一次即可，避免反复查库。
# 用 ContextVar 而非参数透传：Crew 工厂 → _resolve_agent_config 之间隔着
# create_plan_crew/create_analysis_crew 等七层调用，逐层加形参噪音大且易漏；
# research 阶段 asyncio 并发多方向，全局变量会被互相覆盖。
_current_model_plan: ContextVar[dict] = ContextVar("insightcrew_model_plan", default=None)

# 「本任务是否已经显式解析过模型分配表」。
# ★ 存在的唯一理由：区分「没有任务上下文」（开发/测试，应回落内置默认目录）
#   与「有任务上下文但该用户一个可用模型都没有」（真实任务，必须报错而不是
#   偷偷给他内置模型 —— 否则未解锁用户就绕过了内置模型的解锁门禁）。
#   这两者在 `_current_model_plan` 上都表现为空，靠这个 flag 分辨。
_current_model_plan_explicit: ContextVar[bool] = ContextVar(
    "insightcrew_model_plan_explicit", default=False
)


def set_current_task_id(task_id: int):
    """设置当前任务 id，返回可用于重置的 token。"""
    return _current_task_id.set(int(task_id or 0))


def reset_current_task_id(token) -> None:
    """恢复到 set 之前的上下文（在 finally 里调用）。"""
    try:
        _current_task_id.reset(token)
    except (ValueError, LookupError):
        # 跨上下文 reset 会抛错；记账场景下忽略即可
        pass


def current_task_id() -> int:
    """当前任务 id；无上下文时为 0。"""
    return int(_current_task_id.get() or 0)


def set_current_model_plan(plan: dict):
    """设置当前任务的模型分配表 `{role: ModelOption}`，返回重置 token。

    传 None / 空 dict 表示"未指定"，此时 `_resolve_agent_config` 回落
    到内置默认目录（保证开发期与 mock 测试仍能跑通）。
    """
    return _current_model_plan.set(dict(plan) if plan else None)


def reset_current_model_plan(token) -> None:
    """恢复到 set 之前的上下文（在 finally 里调用）。"""
    try:
        _current_model_plan.reset(token)
    except (ValueError, LookupError):
        pass


def current_model_plan() -> dict:
    """当前任务的模型分配表；无上下文时返回 {}（表示用内置默认）。"""
    return _current_model_plan.get() or {}


def set_model_plan_explicit(flag: bool = True):
    """标记「本任务已显式解析过模型分配表」（见 `_current_model_plan_explicit`）。"""
    return _current_model_plan_explicit.set(bool(flag))


def reset_model_plan_explicit(token) -> None:
    try:
        _current_model_plan_explicit.reset(token)
    except (ValueError, LookupError):
        pass


def model_plan_explicit() -> bool:
    """是否处于「已显式解析」的任务上下文。"""
    return bool(_current_model_plan_explicit.get())


__all__ = [
    "set_current_task_id",
    "reset_current_task_id",
    "current_task_id",
    "set_current_model_plan",
    "reset_current_model_plan",
    "current_model_plan",
    "set_model_plan_explicit",
    "reset_model_plan_explicit",
    "model_plan_explicit",
]
