"""
工作流编排器 — 管理多 Agent 调研的全生命周期

职责:
1. 6 阶段流水线：Plan → Research → Analysis → Critic → Write → Review
2. 每阶段结束后 HITL 闸门（人工确认/驳回/回退）
3. SSE 事件推送
4. 中间产物持久化（数据库）
5. 预算熔断检查
"""

import json
import asyncio
from datetime import datetime
from typing import Optional

from src.core.logger import logger
from src.db.database import async_session
from src.db.models import Task, TaskEvent, HumanReview, Report, LlmCall

# SSE 事件队列: task_id -> list[asyncio.Queue]
_sse_queues: dict[int, list[asyncio.Queue]] = {}


# ── SSE 辅助 ──

def _push_event(task_id: int, event_type: str, data: dict):
    """向指定 task_id 的所有 SSE 连接推送事件"""
    queues = _sse_queues.get(task_id, [])
    payload = {"type": event_type, "data": data}
    stale = []
    for q in queues:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            stale.append(q)
    for q in stale:
        q.put_nowait({"type": "overflow", "data": {"dropped": True}})


def subscribe(task_id: int) -> asyncio.Queue:
    """为 SSE 订阅创建一个队列"""
    q = asyncio.Queue(maxsize=128)
    _sse_queues.setdefault(task_id, []).append(q)
    return q


def unsubscribe(task_id: int, q: asyncio.Queue):
    """取消 SSE 订阅"""
    queues = _sse_queues.get(task_id, [])
    if q in queues:
        queues.remove(q)


# ── 阶段定义 ──

STAGES = ["plan", "research", "analysis", "critic", "write", "review"]

STAGE_LABELS = {
    "plan": "调研规划",
    "research": "检索取证",
    "analysis": "对比分析",
    "critic": "红队质疑",
    "write": "报告撰写",
    "review": "质量审核",
}


# ── 工作流编排 ──

async def start_pipeline(topic: str, user_id: int = 0) -> int:
    """
    启动一个新的调研流水线。

    返回 task_id，后续所有操作通过 task_id 引用。
    """
    async with async_session() as session:
        task = Task(
            user_id=user_id,
            topic=topic,
            status="running",
            current_stage="plan",
        )
        session.add(task)
        await session.commit()
        await session.refresh(task)
        task_id = task.id
        logger.info(f"[{task_id}] 新调研任务创建: topic={topic}")

    # 异步启动第一阶段
    asyncio.create_task(_run_stage(task_id, "plan"))

    return task_id


async def _load_intermediate(task_id: int, key: str) -> Optional[str]:
    """异步查询中间产物"""
    from sqlalchemy import select
    event_type = f"intermediate_{key}"
    async with async_session() as session:
        result = await session.execute(
            select(TaskEvent).where(
                TaskEvent.task_id == task_id,
                TaskEvent.event_type == event_type,
            ).order_by(TaskEvent.id.desc()).limit(1)
        )
        row = result.scalar_one_or_none()
        if row and row.payload:
            data = row.payload if isinstance(row.payload, dict) else json.loads(row.payload)
            return data.get("value")
    return None


async def _run_stage(task_id: int, stage: str):
    """
    执行一个阶段。阶段结束后自动创建 HITL 闸门，等待用户确认。
    """
    from src.agents import (
        create_plan_crew,
        create_research_crew,
        create_analysis_crew,
        create_critic_crew,
        create_write_crew,
        create_review_crew,
    )

    logger.info(f"[{task_id}] 开始阶段: {stage}")

    # 获取 task 上下文与中间产物
    async with async_session() as session:
        task = await session.get(Task, task_id)
        if task is None or task.status == "aborted":
            logger.warning(f"[{task_id}] 任务已中止，跳过阶段 {stage}")
            return
        topic = task.topic

    plan = await _load_intermediate(task_id, "plan_output")
    research = await _load_intermediate(task_id, "research_output")
    analysis = await _load_intermediate(task_id, "analysis_output")
    critic_output = await _load_intermediate(task_id, "critic_output")

    _push_event(task_id, "stage_start", {"stage": stage, "label": STAGE_LABELS.get(stage, stage)})

    try:
        if stage == "plan":
            crew = create_plan_crew(topic)
            result = await _run_crew_safe(crew)
            output = result if isinstance(result, str) else str(result)
            await _save_intermediate(task_id, "plan_output", output)

        elif stage == "research":
            if not plan:
                raise ValueError("缺少 plan_output，无法执行 research 阶段")
            crew = create_research_crew(plan)
            result = await _run_crew_safe(crew)
            # research 返回的是多个任务的输出拼接
            output = _flatten_crew_output(result)
            await _save_intermediate(task_id, "research_output", output)

        elif stage == "analysis":
            if not research:
                raise ValueError("缺少 research_output，无法执行 analysis 阶段")
            crew = create_analysis_crew(research)
            result = await _run_crew_safe(crew)
            output = result if isinstance(result, str) else str(result)
            await _save_intermediate(task_id, "analysis_output", output)

        elif stage == "critic":
            if not analysis:
                raise ValueError("缺少 analysis_output，无法执行 critic 阶段")
            crew = create_critic_crew(analysis)
            result = await _run_crew_safe(crew)
            output = result if isinstance(result, str) else str(result)
            await _save_intermediate(task_id, "critic_output", output)

        elif stage == "write":
            if not all([plan, analysis, critic_output]):
                raise ValueError("缺少中间产物，无法执行 write 阶段")
            crew = create_write_crew(plan, analysis, critic_output)
            result = await _run_crew_safe(crew)
            output = result if isinstance(result, str) else str(result)
            await _save_intermediate(task_id, "report_draft", output)

        elif stage == "review":
            report_draft = await _load_intermediate(task_id, "report_draft")
            if not report_draft:
                raise ValueError("缺少 report_draft，无法执行 review 阶段")
            crew = create_review_crew(report_draft)
            result = await _run_crew_safe(crew)
            output = result if isinstance(result, str) else str(result)
            await _save_intermediate(task_id, "review_output", output)

        _push_event(task_id, "stage_done", {"stage": stage})

    except Exception as e:
        logger.error(f"[{task_id}] 阶段 {stage} 执行失败: {e}")
        _push_event(task_id, "stage_error", {"stage": stage, "error": str(e)})
        await _update_task_status(task_id, "failed")
        return

    # 阶段完成 → 创建 HITL 闸门
    gate_name = f"gate_{stage}"
    async with async_session() as session:
        task = await session.get(Task, task_id)
        if task:
            task.current_stage = stage
            task.gate_status = "pending"
            session.add(HumanReview(
                task_id=task_id,
                gate_name=gate_name,
                status="pending",
            ))
            await session.commit()

    _push_event(task_id, "gate_pending", {
        "stage": stage,
        "label": STAGE_LABELS.get(stage, stage),
        "gate_name": gate_name,
    })
    logger.info(f"[{task_id}] 阶段 {stage} 完成，等待人工确认 (gate: {gate_name})")


async def approve_gate(task_id: int, gate_name: str, comment: str = "") -> bool:
    """
    用户通过某个闸门，触发下一阶段。
    如果当前已是最后一个阶段，标记任务完成。
    """
    async with async_session() as session:
        task = await session.get(Task, task_id)
        if not task:
            logger.warning(f"[{task_id}] 任务不存在")
            return False

        # 更新审核记录
        review = await _find_pending_review(session, task_id, gate_name)
        if review:
            review.status = "approved"
            review.comment = comment or "已确认"
            review.resolved_at = datetime.utcnow()

        stage = task.current_stage or "plan"
        task.gate_status = "approved"

        # 记录事件
        session.add(TaskEvent(
            task_id=task_id,
            event_type="gate_approved",
            payload={"gate": gate_name, "comment": comment},
        ))
        await session.commit()

    _push_event(task_id, "gate_approved", {"gate": gate_name})

    # 确定下一阶段
    try:
        idx = STAGES.index(stage)
    except ValueError:
        idx = -1

    if idx >= len(STAGES) - 1:
        # 所有阶段完成，标记完成
        await _finish_pipeline(task_id)
    else:
        next_stage = STAGES[idx + 1]
        await _update_task_status(task_id, "running")
        asyncio.create_task(_run_stage(task_id, next_stage))

    return True


async def reject_gate(task_id: int, gate_name: str, comment: str = "") -> bool:
    """
    用户驳回闸门，可以回退到上一阶段或重新执行当前阶段。
    默认回退到上一阶段。
    """
    async with async_session() as session:
        task = await session.get(Task, task_id)
        if not task:
            return False

        review = await _find_pending_review(session, task_id, gate_name)
        if review:
            review.status = "rejected"
            review.comment = comment or "需修改"
            review.resolved_at = datetime.utcnow()

        stage = task.current_stage or "plan"
        task.gate_status = "rejected"

        session.add(TaskEvent(
            task_id=task_id,
            event_type="gate_rejected",
            payload={"gate": gate_name, "comment": comment},
        ))
        await session.commit()

    _push_event(task_id, "gate_rejected", {"gate": gate_name, "comment": comment})

    # 回退到上一阶段
    try:
        idx = STAGES.index(stage)
    except ValueError:
        idx = 1  # 默认回退到 plan

    prev_stage = STAGES[max(0, idx - 1)]
    await _update_task_status(task_id, "running")
    asyncio.create_task(_run_stage(task_id, prev_stage))
    return True


async def _finish_pipeline(task_id: int):
    """标记任务完成，生成最终报告记录"""
    report_draft = await _load_intermediate(task_id, "report_draft")
    review_output = await _load_intermediate(task_id, "review_output")

    async with async_session() as session:
        task = await session.get(Task, task_id)
        if not task:
            return

        report = Report(
            task_id=task_id,
            content_md=report_draft or "",
            bibtex="",  # TODO: 引用台账生成 BibTeX
        )
        session.add(report)
        await session.flush()
        await session.refresh(report)

        task.status = "completed"
        task.result_report_id = report.id
        task.updated_at = datetime.utcnow()
        task.gate_status = "done"

        session.add(TaskEvent(
            task_id=task_id,
            event_type="pipeline_complete",
            payload={"report_id": report.id},
        ))
        await session.commit()

    _push_event(task_id, "pipeline_complete", {"report_id": report.id})
    logger.info(f"[{task_id}] 调研流水线完成")


async def abort_pipeline(task_id: int):
    """中止任务"""
    await _update_task_status(task_id, "aborted")
    _push_event(task_id, "pipeline_aborted", {})


# ── 内部辅助 ──

async def _run_crew_safe(crew) -> str:
    """安全运行 Crew.kickoff()，捕获异常"""
    try:
        result = await crew.kickoff_async()
        return result
    except Exception as e:
        logger.error(f"Crew 执行失败: {e}")
        raise


def _flatten_crew_output(result) -> str:
    """将 Crew 的多个任务输出合并为一个字符串"""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return "\n\n---\n\n".join(str(r) for r in result)
    return str(result)


async def _save_intermediate(task_id: int, key: str, value: str):
    """保存中间产物到 TaskEvent（JSON payload）"""
    async with async_session() as session:
        session.add(TaskEvent(
            task_id=task_id,
            event_type=f"intermediate_{key}",
            payload={"value": value},
        ))
        await session.commit()


async def _update_task_status(task_id: int, status: str):
    """更新任务状态"""
    async with async_session() as session:
        task = await session.get(Task, task_id)
        if task:
            task.status = status
            task.updated_at = datetime.utcnow()
            await session.commit()


async def _find_pending_review(session, task_id: int, gate_name: str) -> Optional[HumanReview]:
    """查找待审核记录"""
    from sqlalchemy import select
    result = await session.execute(
        select(HumanReview).where(
            HumanReview.task_id == task_id,
            HumanReview.gate_name == gate_name,
            HumanReview.status == "pending",
        ).limit(1)
    )
    return result.scalar_one_or_none()


# ── 任务状态查询 ──

async def get_task_status(task_id: int) -> Optional[dict]:
    """获取任务当前状态（用于 API 返回）"""
    async with async_session() as session:
        task = await session.get(Task, task_id)
        if not task:
            return None

        # 获取待审核的闸门
        from sqlalchemy import select
        pending = await session.execute(
            select(HumanReview).where(
                HumanReview.task_id == task_id,
                HumanReview.status == "pending",
            ).limit(1)
        )
        pending_gate = pending.scalar_one_or_none()

    # 在 session 外查询中间产物
    report_draft = await _load_intermediate(task_id, "report_draft")
    review_output = await _load_intermediate(task_id, "review_output")

    return {
        "task_id": task.id,
        "topic": task.topic,
        "status": task.status,
        "current_stage": task.current_stage,
        "gate_status": task.gate_status,
        "pending_gate": pending_gate.gate_name if pending_gate else None,
        "report_draft": report_draft,
        "review_output": review_output,
        "created_at": task.created_at.isoformat() if task.created_at else None,
    }
