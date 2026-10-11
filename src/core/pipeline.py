"""流水线生命周期入口 —— 创建任务、查询状态、停滞任务回收。

从 `orchestrator.py` 拆分而来（M1/M2 遗留缺陷第 7 项，纯搬移零行为变更）。

★ `start_pipeline` 先做依赖体检（`_ensure_executable`）再落库，避免
   worker/Redis 不可用时留下永远不会被执行的僵尸任务行。
"""

from datetime import datetime, timedelta
from typing import Optional

from src.core.config import settings
from src.core.errors import ServiceUnavailableError
from src.core.logger import logger
from src.db.database import async_session
from src.db.models import Task, HumanReview

from src.core.events import _push_event
from src.core.queue import _enqueue_stage, _ensure_executable, _mark_exec_path
from src.core.gates import _update_task_status
from src.core.artifacts import _load_intermediate


# ── 工作流编排 ──

def _serialize_model_plan(model_plan: dict | None) -> str | None:
    """把 ``{role: option_id}`` 序列化成落库用的 JSON 文本。

    只保留已知角色键、只接受非空字符串值；非法输入返回 None（等价于
    "未指定"，执行期回落内置默认），绝不让脏数据进库。
    """
    if not model_plan or not isinstance(model_plan, dict):
        return None

    from src.core.model_registry import PLAN_ROLES

    clean: dict[str, str] = {}
    for role in PLAN_ROLES:
        value = model_plan.get(role)
        if isinstance(value, str) and value.strip():
            clean[role] = value.strip()
    if not clean:
        return None

    import json

    return json.dumps(clean, ensure_ascii=False)


async def start_pipeline(
    topic: str,
    user_id: int = 0,
    dimensions: str = "",
    constraints: str = "",
    budget_limit: float | None = None,
    model_plan: dict | None = None,
) -> int:
    """
    启动一个新的调研流水线。

    返回 task_id，后续所有操作通过 task_id 引用。

    `dimensions` / `constraints` / `budget_limit` 是用户在新建任务时填的
    可选参数（M1 遗留项 5），全部落库随任务携带：worker 侧执行阶段时读回来
    用，不靠 API 进程内存传递（跨进程）。

    `model_plan`（M2.5，用户 m12784 拍板）是 ``{角色: option_id}`` 的映射，
    7 个键 = 6 个 Agent 角色（Planner/Researcher/Analyst/Critic/Writer/
    Reviewer）+ ``vision``（读图，只在 research 阶段的工具层用）。落库存 JSON，
    worker 侧按**任务归属用户**的目录把它解析回具体的 ``ModelOption``。
    为 None 时留空，执行期回落到内置默认目录（开发/测试路径）。

    执行路径唯一：arq 入队 → worker 执行。服务不可用时抛
    ServiceUnavailableError（API 层转 503），**不落任务行**——
    避免留下一堆永远不会被执行、只能靠人工清理的僵尸任务。
    """
    # 先体检：Redis / worker / 队列池任一不可用就抛，此时还没有任何副作用
    await _ensure_executable()

    limit = None
    if budget_limit is not None:
        try:
            limit = float(budget_limit)
        except (TypeError, ValueError):
            limit = None
        if limit is not None and limit <= 0:
            limit = None  # 0 / 负数表示不限制，等价于没填

    plan_json = _serialize_model_plan(model_plan)

    async with async_session() as session:
        task = Task(
            user_id=user_id,
            topic=topic,
            status="running",
            current_stage="plan",
            dimensions=(dimensions or "").strip() or None,
            constraints=(constraints or "").strip() or None,
            model_plan=plan_json,
            budget_limit_cny=limit,
        )
        session.add(task)
        await session.commit()
        await session.refresh(task)
        task_id = task.id
        logger.info(
            f"[{task_id}] 新调研任务创建: topic={topic} "
            f"model_plan={plan_json or '(内置默认)'} "
            f"budget={limit if limit is not None else '(全局默认)'} "
            f"dimensions={'有' if dimensions else '无'} "
            f"constraints={'有' if constraints else '无'}"
        )

    # 入队失败（重试后仍失败）会把任务标 failed，不留下永久 running 的僵尸
    try:
        await _enqueue_stage(task_id, "plan")
        await _mark_exec_path(task_id, "arq")
    except ServiceUnavailableError:
        await _update_task_status(task_id, "failed")
        await _push_event(task_id, "stage_error", {
            "stage": "plan",
            "error": "任务执行器不可用，任务未能投递",
            "reason": "worker_unavailable",
        })
        raise

    return task_id

# ── 停滞任务回收（兜底监督，不是降级）──
#
# 入队前的健康门控能拦住"worker 从一开始就不在"的情况，但拦不住
# "检查通过之后 worker 恰好崩溃"。这类任务会永远停在 running，
# 没有任何错误、没有任何超时——正是我们要消灭的静默失败。
#
# 回收任务只做一件事：**判定失败并写明原因**，绝不接管执行。

# 停滞判定阈值：3 倍阶段超时（默认 300s → 900s）
STALE_FACTOR = 3


async def reap_stale_tasks(ctx=None) -> int:
    """扫出停滞任务并标记失败，返回处理条数。

    判定条件（三个必须同时成立）：
      1. status == "running"
      2. gate_status != "pending"  —— **等待人工确认不算停滞**
      3. updated_at 早于 now - stage_timeout_sec*3

    第 2 条是必须的：任务停在闸门上等人类点确认时，状态同样是 running
    且 updated_at 不再变化。若不排除，一个人思考超过 15 分钟任务就会被
    误判为失败——这会把"等待审批"这个正常状态当成故障。

    updated_at 在每次阶段完成/状态变更时刷新，因此"长时间没更新"
    等价于"没有任何阶段在推进"。
    """
    from sqlalchemy import select

    threshold_sec = settings.stage_timeout_sec * STALE_FACTOR
    cutoff = datetime.utcnow() - timedelta(seconds=threshold_sec)
    reaped: list[int] = []

    async with async_session() as session:
        rows = await session.execute(
            select(Task).where(
                Task.status == "running",
                # 等人工确认的任务不算停滞（见 docstring 第 2 条）
                Task.gate_status != "pending",
                Task.updated_at < cutoff,
            )
        )
        for task in rows.scalars().all():
            task.status = "failed"
            task.updated_at = datetime.utcnow()
            reaped.append(task.id)
        if reaped:
            await session.commit()

    for task_id in reaped:
        await _push_event(task_id, "stage_error", {
            "stage": "unknown",
            "error": f"任务停滞超过 {threshold_sec} 秒无进展，已判定失败",
            "reason": "stale_timeout",
        })
        logger.warning(f"[{task_id}] 停滞超 {threshold_sec}s，已标记 failed(stale_timeout)")

    return len(reaped)

# ── 任务状态查询 ──

async def get_task_status(task_id: int) -> Optional[dict]:
    """获取任务当前状态（用于 API 返回）"""
    async with async_session() as session:
        task = await session.get(Task, task_id)
        if not task:
            return None

        from sqlalchemy import select
        pending = await session.execute(
            select(HumanReview).where(
                HumanReview.task_id == task_id,
                HumanReview.status == "pending",
            ).limit(1)
        )
        pending_gate = pending.scalar_one_or_none()

    # 中间产物：任务详情页右侧面板要展示每个阶段的结果
    plan_output = await _load_intermediate(task_id, "plan_output")
    analysis_output = await _load_intermediate(task_id, "analysis_output")
    critic_output = await _load_intermediate(task_id, "critic_output")
    rebuttal_output = await _load_intermediate(task_id, "rebuttal_output")
    report_draft = await _load_intermediate(task_id, "report_draft")
    review_output = await _load_intermediate(task_id, "review_output")

    # 成本与预算上限：前端顶部要显示「已花费 ¥x.xx / 上限 ¥5.00」
    # ★ 用 task_budget_limit（任务自带值优先）：否则用户填了上限却在页面上
    #   看到全局默认值，以为没生效。
    from src.core.cost import task_budget_limit

    return {
        "task_id": task.id,
        "topic": task.topic,
        "status": task.status,
        "current_stage": task.current_stage,
        "gate_status": task.gate_status,
        "pending_gate": pending_gate.gate_name if pending_gate else None,
        "plan_output": plan_output,
        "analysis_output": analysis_output,
        "critic_output": critic_output,
        "rebuttal_output": rebuttal_output,
        "report_draft": report_draft,
        "review_output": review_output,
        "exec_path": task.exec_path or "unknown",
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "total_cost_cny": float(task.total_cost_cny or 0.0),
        "budget_limit_cny": await task_budget_limit(task_id),
        # 任务级参数（M1 遗留项 5）：前端详情页要回显用户填的维度/约束
        "dimensions": task.dimensions or "",
        "constraints": task.constraints or "",
        # 模型分配表（M2.5）：回显用户逐角色选定的模型，前端据此渲染
        "model_plan": _deserialize_model_plan(task.model_plan),
    }


def _deserialize_model_plan(raw: str | None) -> dict:
    """把落库的 model_plan JSON 还原成 ``{role: option_id}``；坏数据返回 {}。"""
    if not raw:
        return {}
    try:
        import json

        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}
