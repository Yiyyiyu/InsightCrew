"""成本台账与预算熔断。

设计要点：
1. **用量来源**：CrewAI 每次 `kickoff` 后会把累计用量放在 `crew.usage_metrics`
   （`crewai/types/usage_metrics.py` 的 `UsageMetrics`，字段
   `prompt_tokens` / `completion_tokens` / `successful_requests`）。
   它是**单次 crew 的累计值**，所以直接读即可，不需要做差分。
2. **落库**：每次 crew 结束写一行 `llm_calls`，并把成本累加到
   `tasks.total_cost_cny`。这既是账单，也是熔断的判据。
3. **熔断**：累加后立刻与 `budget.per_task_limit_cny` 比较，超限抛
   `BudgetExceededError`，由 orchestrator 中止任务——不能让一个跑飞的任务
   把额度烧穿。
4. 单价是**配置数据**（`config/models.yaml` 的 `pricing`），不是代码常量：
   厂商调价时改配置即可，不用动代码。
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import update

from src.core.config import settings
from src.core.logger import logger
from src.core.model_router import router
from src.db.database import async_session
from src.db.models import LlmCall, Task


class BudgetExceededError(RuntimeError):
    """任务成本超过预算上限。"""


# 兜底上限：配置文件缺失时用 settings 里的值
_DEFAULT_LIMIT_CNY = 5.0


def _pricing_options() -> list:
    """本次记账可用的模型目录：当前任务的 model_plan + 内置目录。

    M2.5 起单价不再是 ``config/models.yaml`` 的档位价目表，而是**每个
    ``ModelOption`` 自带** ``price_in`` / ``price_out``（内置的在
    ``builtin_models.yaml``、自建的存 ``models`` 表）。用户没填单价的模型
    查到 (0, 0) —— 只记 token 不记钱，好过编一个假价格。
    """
    from src.core.context import current_model_plan
    from src.core.model_registry import builtin_options

    opts: list = [o for o in current_model_plan().values() if o is not None]
    seen = {o.id for o in opts}
    for o in builtin_options():
        if o.id not in seen:
            opts.append(o)
            seen.add(o.id)
    return opts


def _pricing_for_model(model_id: str) -> tuple[float, float, str]:
    """按 litellm 模型串查单价，返回 (输入单价, 输出单价, 配置键)。

    单价单位：元 / 百万 token。查不到时返回 (0, 0, model_id)——只记 token
    不记钱，好过编一个假价格。

    ★ 前缀无关匹配（实测踩坑）：CrewAI 的 ``LLM(model=...)`` 会把 provider
    前缀剥掉（配置 ``deepseek/deepseek-flash``，``llm.model`` 是
    ``deepseek-flash``）。``model_registry.pricing_for`` 已做归一匹配。
    """
    try:
        from src.core.model_registry import pricing_for

        in_price, out_price = pricing_for(model_id, _pricing_options())
        return float(in_price), float(out_price), model_id
    except Exception as e:
        logger.warning(f"查单价失败（{model_id}）: {e}")
        return 0.0, 0.0, model_id


def cost_of(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    """把 token 数折算成人民币成本。"""
    in_price, out_price, _ = _pricing_for_model(model_id)
    return (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000.0


def budget_limit_cny() -> float:
    """读取**全局**任务预算上限（配置优先，回落到 settings）。

    单任务覆盖见 `task_budget_limit()`——用户在新建任务时填了上限就用它，
    没填才用这里的全局值。
    """
    cfg = router.budget_config()
    try:
        return float(cfg.get("per_task_limit_cny", settings.task_budget_limit_cny))
    except (TypeError, ValueError):
        return _DEFAULT_LIMIT_CNY


async def task_budget_limit(task_id: int) -> float:
    """本任务的预算上限：任务自带值优先，否则回落全局配置。

    M1 遗留项 5：用户在新建任务时可填 `budget_limit`，需要真的生效，
    否则表单上那个输入框只是装饰。
    """
    if task_id:
        try:
            async with async_session() as session:
                task = await session.get(Task, task_id)
                if task is not None and task.budget_limit_cny:
                    value = float(task.budget_limit_cny)
                    if value > 0:
                        return value
        except Exception as e:
            logger.warning(f"[{task_id}] 读取任务预算上限失败，回落全局配置: {e}")
    return budget_limit_cny()


def on_exceed_action() -> str:
    """超限时的动作（abort / warn）。"""
    return str(router.budget_config().get("on_exceed", "abort")).lower()


async def get_task_cost(task_id: int) -> float:
    """读取任务已累计成本。"""
    async with async_session() as session:
        task = await session.get(Task, task_id)
        return float(task.total_cost_cny or 0.0) if task else 0.0


async def record_usage(
    task_id: int,
    model_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    stage: str = "",
) -> float:
    """记录一次 crew 的用量：写 llm_calls + 累加 tasks.total_cost_cny。

    返回本次成本（元）。数据库异常不抛出，避免把记账问题升级成任务失败。

    ★ 累加必须用原子 SQL（`total_cost_cny = total_cost_cny + cost`）：
    research 阶段是 asyncio.gather 并发多方向，各方向同时记账，
    读-改-写会丢更新（后写的覆盖先写的），成本被少算 → 熔断失灵。
    """
    cost = cost_of(model_id, prompt_tokens, completion_tokens)
    try:
        async with async_session() as session:
            session.add(LlmCall(
                task_id=task_id,
                model=model_id[:64],
                prompt_tokens=int(prompt_tokens or 0),
                completion_tokens=int(completion_tokens or 0),
                cost_cny=cost,
            ))
            await session.execute(
                update(Task)
                .where(Task.id == task_id)
                .values(total_cost_cny=Task.total_cost_cny + cost)
            )
            await session.commit()
    except Exception as e:
        logger.warning(f"[{task_id}] 成本记账失败（{stage or model_id}）: {e}")
    return cost


async def record_crew_usage(task_id: int, crew, stage: str = "") -> float:
    """从 crew.usage_metrics 取用量并记账。

    CrewAI 把用量挂在 crew 上（每个 Agent 的 LLM 累计值之和），
    没有用量信息时记 0，不报错。
    """
    prompt_tokens = 0
    completion_tokens = 0
    model_id = "unknown"

    metrics = getattr(crew, "usage_metrics", None)
    if metrics is not None:
        prompt_tokens = int(getattr(metrics, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(metrics, "completion_tokens", 0) or 0)

    # 模型名用于查单价。★ M2.5 起优先按**角色**从当前任务的 model_plan 取
    #   （用户逐角色选定），因为 CrewAI 的 LLM 会把 provider 前缀剥掉、
    #   且同一个模型串可能对应不同 provider 的不同单价。取不到再回落
    #   agents[0].llm.model（开发/测试无任务上下文时）。
    model_id = _crew_model_id(crew)

    cost = await record_usage(task_id, model_id, prompt_tokens, completion_tokens, stage)
    logger.info(
        f"[{task_id}] 用量记账 stage={stage or '-'} model={model_id} "
        f"in={prompt_tokens} out={completion_tokens} cost=¥{cost:.4f}"
    )
    return cost


def _crew_model_id(crew) -> str:
    """记账用的模型名。

    ★ 为什么要绕一圈（实测踩坑）：CrewAI 的 ``LLM(model=...)`` 会**剥掉
    provider 前缀**（配置 ``deepseek/deepseek-flash``，``llm.model`` 读到
    ``deepseek-flash``）。裸名可能对应多个来源（内置 vs 用户自建，单价不同），
    所以先拿当前任务 ``model_plan`` 里的候选按裸名对齐、恢复**完整**模型串，
    取不到才退回 LLM 串。价格匹配本身也做了前缀归一（双保险）。
    """
    agents = getattr(crew, "agents", None) or []
    if not agents:
        return "unknown"

    llm = getattr(agents[0], "llm", None)
    raw = str(getattr(llm, "model", "") or "")
    bare = raw.split("/", 1)[1] if "/" in raw else raw

    if bare:
        try:
            from src.core.context import current_model_plan

            for opt in current_model_plan().values():
                mid = str(getattr(opt, "model_id", "") or "")
                if not mid:
                    continue
                mid_bare = mid.split("/", 1)[1] if "/" in mid else mid
                if mid_bare == bare:
                    return mid
        except Exception:
            pass

    return raw or "unknown"


async def enforce_budget(task_id: int) -> None:
    """检查任务是否超预算；超限按配置动作抛 BudgetExceededError。

    在每次记账之后、以及每个阶段开始前调用：前者防止单次大额调用把额度
    烧穿，后者防止已经超限的任务继续往下跑。

    ★ 上限取本任务自带值（`task_budget_limit`），没有才用全局配置——
    用户填的预算上限必须真的生效。
    """
    limit = await task_budget_limit(task_id)
    if limit <= 0:
        return  # 0 或负数表示不限制
    spent = await get_task_cost(task_id)
    if spent <= limit:
        return

    msg = f"[{task_id}] 成本 ¥{spent:.4f} 已超预算上限 ¥{limit:.2f}"
    if on_exceed_action() == "warn":
        logger.warning(msg + "（配置为 warn，继续执行）")
        return
    logger.error(msg + "，中止任务")
    raise BudgetExceededError(msg)


def record_usage_sync(
    task_id: int,
    model_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    stage: str = "",
) -> float:
    """**同步**记账：供工具层（运行在 CrewAI 工作线程、无法 await）调用。

    与 `record_usage` 写同一张表、同样做原子累加（`total_cost_cny =
    total_cost_cny + cost`），只是用同步引擎——工具层在事件循环之外，
    用 asyncio.run 会被"已有事件循环在跑"挡住。

    task_id 为 0（工具不在任务上下文中）时**不落库**：宁可漏记一笔，
    也不要把成本挂到不存在的任务上污染账目。
    """
    if not task_id:
        logger.info(f"[成本] 工具调用无任务上下文，跳过记账 stage={stage or model_id} "
                    f"in={prompt_tokens} out={completion_tokens}")
        return 0.0

    cost = cost_of(model_id, prompt_tokens, completion_tokens)
    try:
        from sqlalchemy import update as _update
        from sqlalchemy.orm import Session

        from src.core.ledger import get_sync_engine
        from src.db.models import LlmCall as _LlmCall
        from src.db.models import Task as _Task

        with Session(get_sync_engine()) as s:
            s.add(_LlmCall(
                task_id=task_id,
                model=model_id[:64],
                prompt_tokens=int(prompt_tokens or 0),
                completion_tokens=int(completion_tokens or 0),
                cost_cny=cost,
            ))
            s.execute(
                _update(_Task)
                .where(_Task.id == task_id)
                .values(total_cost_cny=_Task.total_cost_cny + cost)
            )
            s.commit()
        logger.info(
            f"[{task_id}] 工具用量记账 stage={stage or '-'} model={model_id} "
            f"in={prompt_tokens} out={completion_tokens} cost=¥{cost:.4f}"
        )
    except Exception as e:
        # 记账问题不应升级成工具失败（工具失败会让 Agent 重试、浪费更多额度）
        logger.warning(f"[{task_id}] 工具记账失败（{stage or model_id}）: {e}")
    return cost


def budget_exceeded_sync(task_id: int) -> bool:
    """**同步**检查是否已超预算（供工具层用），超限返回 True。

    上限优先取任务自带值（与 `task_budget_limit` 同语义），没有才用全局配置。
    """
    if not task_id:
        return False
    try:
        from sqlalchemy.orm import Session

        from src.core.ledger import get_sync_engine
        from src.db.models import Task as _Task

        with Session(get_sync_engine()) as s:
            row = s.get(_Task, task_id)
            spent = float(row.total_cost_cny or 0.0) if row else 0.0
            own = float(row.budget_limit_cny or 0.0) if row else 0.0
        limit = own if own > 0 else budget_limit_cny()
        if limit <= 0:
            return False
        return spent > limit
    except Exception as e:
        logger.warning(f"[{task_id}] 预算检查失败: {e}")
        return False
