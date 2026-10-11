"""LLM 构造点 —— 按角色把用户选定的 ``ModelOption`` 变成 CrewAI ``LLM``。

## 这个模块现在做什么

只有一个职责：**给定 Agent 角色名，构造出该角色在本任务里要用的
``crewai.LLM`` 实例**。模型来自当前任务的 ``model_plan``
（用户在新建任务时逐角色选定，见 ``src/core/context.py``）。

## 为什么文件名还叫 llm_fallback

历史上这里实现的是「文本链路跨厂商降级」。M2.5（用户 m12784 拍板）
**移除了运行时降级**：这条流水线只有一条正常路径，模型不通属于异常情况，
应当在新建任务的 preflight 连通性预检里拦截，而不是执行到一半偷偷换模型
（换掉的模型用户没选过，账单与结果都不可解释）。

函数名 ``build_llm_with_fallback`` 保留是为了不动
``src/agents/__init__.py::_resolve_agent_config`` 这个唯一构造点；
M2.5-7 已把降级实现（``install_fallback`` / ``is_degradable`` /
``degradation_events`` / ``candidate_routes``）整体删除。

## 为什么不再用 CrewAI 内建的 fallbacks

（历史记录，勿改成"用 crewai 自带的"）
1. crewai 1.15 的 ``LLM`` **没有**可用的 ``fallbacks`` 参数——传了不报错，但会被
   塞进 ``additional_params`` 原样发给厂商 API：既不降级，还多送一个非法字段。
2. crewai 内建的 ``_wrap_call_method`` 只重试 429 类限流
   （``crewai/llms/retry.py`` 的 ``_RETRYABLE_ERROR_CODES``），不处理超时/5xx/
   鉴权失败，更不会换厂商。
"""

from __future__ import annotations

from typing import Any

from src.core.logger import logger


def _fallback_option(agent_name: str):
    """任务级 model_plan 缺失时的兜底：用内置默认目录的角色分配。

    仅在开发/测试（未建任务上下文）时走到，保证 mock 链路仍可运行。
    """
    from src.core.model_registry import (
        apply_role_overrides,
        builtin_default_plan,
        builtin_options,
        find_option,
    )

    opts = builtin_options()
    plan = builtin_default_plan()
    opt = find_option(opts, plan.get(agent_name, ""))
    if opt is None:
        opt = opts[0] if opts else None
    return apply_role_overrides(opt, agent_name) if opt is not None else None


def resolve_agent_option(agent_name: str):
    """解析当前任务里该 Agent 角色应使用的 ``ModelOption``。

    优先级：任务级 ``model_plan``（用户在新建任务时逐角色选定）>
    内置默认目录（**仅开发/测试**兜底）。返回 None 表示无可用模型。

    ★ 兜底只在「没有任务上下文」时生效（M2.5-5 的第 ③ 道强制点）。
    真实任务里 ``execute_stage`` 会置 ``model_plan_explicit``，此时目录为空
    说明该用户既没解锁内置、也没有自建模型 —— **必须返回 None** 让上层抛
    ``ServiceUnavailableError``，绝不能偷偷回落内置目录，否则未解锁用户就
    绕过了内置模型的解锁门禁（preflight 的 403 形同虚设）。
    """
    from src.core.context import current_model_plan, model_plan_explicit
    from src.core.model_registry import apply_role_overrides

    opt = current_model_plan().get(agent_name)
    if opt is not None:
        # per-Agent 生成参数覆盖（Writer 需要更长输出等）在此施加一次
        return apply_role_overrides(opt, agent_name)
    if model_plan_explicit():
        return None
    return _fallback_option(agent_name)


def _llm_kwargs(route, timeout: int) -> dict[str, Any]:
    """把 ``ModelRoute`` 翻成 ``crewai.LLM`` 的构造参数。

    ★ 注意 ``base_url`` 而不是 ``api_base``：crewai 的 LLM 工厂用的是前者。
    """
    kwargs: dict[str, Any] = {
        "model": route.model_id,
        "api_key": route.api_key,
        "temperature": route.temperature,
        "max_tokens": route.max_tokens,
        "timeout": timeout,
    }
    if route.api_base:
        kwargs["base_url"] = route.api_base
    return kwargs


def build_llm_with_fallback(agent_name: str) -> Any:
    """构造 Agent 的 CrewAI LLM 实例。

    ★ M2.5 起改为「按角色查 registry → 构造**单个** LLM」：模型来自当前任务
    的 ``model_plan``（用户逐角色选定），不再按档位名路由、不再挂降级链。
    函数名保留是为了不动 ``_resolve_agent_config`` 这个唯一构造点。
    """
    from crewai import LLM

    from src.core.config import settings
    from src.core.errors import ServiceUnavailableError
    from src.core.model_router import router as model_router

    timeout = settings.llm_timeout_sec
    option = resolve_agent_option(agent_name)
    if option is None:
        raise ServiceUnavailableError("model", f"角色 {agent_name} 没有可用模型")

    route = model_router.route_from_option(option, agent_name)
    logger.debug(
        f"[{agent_name}] 使用模型 {route.model_id}"
        f"（option={route.name or '-'}，max_tokens={route.max_tokens}）"
    )
    return LLM(**_llm_kwargs(route, timeout))
