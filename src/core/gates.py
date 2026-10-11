"""HITL 闸门与任务生命周期 —— 人工确认/驳回/回退、收尾、状态更新。

从 `orchestrator.py` 拆分而来（M1/M2 遗留缺陷第 7 项，纯搬移零行为变更）。

设计：每阶段结束写一条 `human_reviews(gate_name, status=pending)` 并置
`tasks.gate_status="pending"`，任务**不占用 worker**（用户确认后重新入队）。
"""

from datetime import datetime
from typing import Optional

from src.core.errors import ServiceUnavailableError
from src.core.logger import logger
from src.db.database import async_session
from src.db.models import Task, HumanReview, Report

from src.core.events import _push_event
from src.core.queue import (
    STAGES,
    STAGE_LABELS,
    _enqueue_stage,
    _ensure_executable,
    _mark_exec_path,
)
from src.core.artifacts import _load_intermediate



async def _create_gate(task_id: int, stage: str):
    """阶段执行完成后创建 HITL 闸门（由 worker 内的阶段执行器调用）"""
    gate_name = f"gate_{stage}"
    async with async_session() as session:
        task = await session.get(Task, task_id)
        if task:
            task.gate_status = "pending"
            session.add(HumanReview(
                task_id=task_id,
                gate_name=gate_name,
                status="pending",
            ))
            await session.commit()

    await _push_event(task_id, "gate_pending", {
        "stage": stage,
        "label": STAGE_LABELS.get(stage, stage),
        "gate_name": gate_name,
    })
    logger.info(f"[{task_id}] 阶段 {stage} 完成，等待人工确认 (gate: {gate_name})")


async def _run_stage(task_id: int, stage: str):
    """【已移除】进程内执行阶段的降级路径。

    正常路径是 arq 入队 → worker 执行（见 `_enqueue_stage`）。
    降级到 API 进程执行会把阶段挂在 uvicorn 生命周期上：`--reload`
    开发模式下每次改代码都重启，阶段凭空消失，任务永久 running。
    保留此符号只为让残留调用立刻显式失败，而不是悄悄跑起来。
    """
    raise ServiceUnavailableError(
        "worker",
        "_run_stage 降级路径已移除：任务必须由 arq worker 执行",
    )

async def approve_gate(task_id: int, gate_name: str, comment: str = "") -> bool:
    """
    用户通过某个闸门，触发下一阶段。
    如果当前已是最后一个阶段，标记任务完成。

    ★ 顺序很重要：先只读地取阶段 → 体检 → 才落闸门状态 → 才入队。
    这样 worker 不在时抛 ServiceUnavailableError，闸门**保持 pending**，
    运维修好服务后用户直接再点一次即可；而不是闸门被改成 approved
    却没有任何后续执行（旧的降级写法就会留下这种半成品状态）。
    """
    # 1. 只读取任务与阶段（无副作用）
    async with async_session() as session:
        task = await session.get(Task, task_id)
        if not task:
            logger.warning(f"[{task_id}] 任务不存在")
            return False
        stage = task.current_stage or "plan"

    try:
        idx = STAGES.index(stage)
    except ValueError:
        idx = -1
    is_last = idx >= len(STAGES) - 1

    # 2. 非终态才需要投递下一阶段 → 投递前先体检（失败即抛，无副作用）
    if not is_last:
        await _ensure_executable()

    # 3. 体检通过才落闸门状态
    async with async_session() as session:
        task = await session.get(Task, task_id)
        review = await _find_pending_review(session, task_id, gate_name)
        if review:
            review.status = "approved"
            review.comment = comment or "已确认"
            review.resolved_at = datetime.utcnow()
        task.gate_status = "approved"
        await session.commit()

    # 事件由 _push_event 统一落库 + 双通道推送（含 event id 供断线续传）
    await _push_event(task_id, "gate_approved", {"gate": gate_name, "comment": comment})

    if is_last:
        await _finish_pipeline(task_id)
    else:
        next_stage = STAGES[idx + 1]
        await _update_task_status(task_id, "running")
        try:
            await _enqueue_stage(task_id, next_stage)
            await _mark_exec_path(task_id, "arq")
        except ServiceUnavailableError:
            # 入队仍失败（重试已用尽）→ 回滚闸门，避免"已确认但无人执行"
            await _reopen_gate(task_id, gate_name)
            raise

    return True


async def _reopen_gate(task_id: int, gate_name: str) -> None:
    """把闸门退回 pending（入队失败时的回滚，保证不留半成品状态）"""
    async with async_session() as session:
        task = await session.get(Task, task_id)
        if task:
            task.gate_status = "pending"
        review = await _find_pending_review(session, task_id, gate_name)
        if review is None:
            # _find_pending_review 只找 pending，这里直接按 gate_name 找最近一条
            from sqlalchemy import select
            row = await session.execute(
                select(HumanReview)
                .where(HumanReview.task_id == task_id, HumanReview.gate_name == gate_name)
                .order_by(HumanReview.id.desc())
                .limit(1)
            )
            review = row.scalar_one_or_none()
        if review:
            review.status = "pending"
            review.resolved_at = None
        await session.commit()
    logger.warning(f"[{task_id}] 入队失败，闸门 {gate_name} 已退回 pending")


async def reject_gate(task_id: int, gate_name: str, comment: str = "") -> bool:
    """
    用户驳回闸门，默认回退到上一阶段。

    与 approve_gate 同样遵循"先体检、后副作用"的顺序：worker 不可用时
    抛 ServiceUnavailableError 且闸门保持 pending，可重试。
    """
    # 1. 只读取任务与阶段
    async with async_session() as session:
        task = await session.get(Task, task_id)
        if not task:
            return False
        stage = task.current_stage or "plan"

    try:
        idx = STAGES.index(stage)
    except ValueError:
        idx = 1
    prev_stage = STAGES[max(0, idx - 1)]

    # 2. 体检（失败即抛，此时闸门尚未被改动）
    await _ensure_executable()

    # 3. 落闸门状态
    async with async_session() as session:
        review = await _find_pending_review(session, task_id, gate_name)
        if review:
            review.status = "rejected"
            review.comment = comment or "需修改"
            review.resolved_at = datetime.utcnow()
        task = await session.get(Task, task_id)
        task.gate_status = "rejected"
        await session.commit()

    await _push_event(task_id, "gate_rejected", {"gate": gate_name, "comment": comment})

    await _update_task_status(task_id, "running")
    try:
        await _enqueue_stage(task_id, prev_stage)
        await _mark_exec_path(task_id, "arq")
    except ServiceUnavailableError:
        await _reopen_gate(task_id, gate_name)
        raise
    return True

async def _finish_pipeline(task_id: int):
    """标记任务完成，生成最终报告记录"""
    from src.core.bibtex import build_bibtex

    report_draft = await _load_intermediate(task_id, "report_draft")

    # BibTeX 从引用台账真实生成（M1 遗留项 6）：生成失败不能让整条流水线
    # 失败——报告正文已经写好，参考文献留空只是导出不便，比丢掉整份报告轻。
    # 生成逻辑在 src/core/bibtex.py，这里只负责调用与失败降级。
    # 没有可引用来源时返回空串，前端会显示「（无）」。
    try:
        bibtex = await build_bibtex(task_id, report_draft or "")
    except Exception as exc:
        logger.warning(f"[{task_id}] BibTeX 生成失败，留空: {exc}")
        bibtex = ""

    async with async_session() as session:
        task = await session.get(Task, task_id)
        if not task:
            return

        report = Report(
            task_id=task_id,
            content_md=report_draft or "",
            bibtex=bibtex,
        )
        session.add(report)
        await session.flush()
        await session.refresh(report)

        task.status = "completed"
        task.result_report_id = report.id
        task.updated_at = datetime.utcnow()
        task.gate_status = "done"
        await session.commit()
        report_id = report.id

    await _push_event(task_id, "pipeline_complete", {"report_id": report_id})
    logger.info(f"[{task_id}] 调研流水线完成")


async def abort_pipeline(task_id: int):
    """中止任务"""
    await _update_task_status(task_id, "aborted")
    await _push_event(task_id, "pipeline_aborted", {})

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
