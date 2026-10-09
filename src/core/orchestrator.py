"""
工作流编排器 — 管理多 Agent 调研的全生命周期

职责:
1. 6 阶段流水线：Plan → Research → Analysis → Critic → Write → Review
2. 每阶段结束后 HITL 闸门（人工确认/驳回/回退）
3. SSE 事件推送（内存队列 + Redis Pub/Sub 双通道）
4. 中间产物持久化（数据库）
5. arq 任务队列 / fallback 异步执行
6. 预算熔断检查
"""

import json
import asyncio
import time
from datetime import datetime
from typing import Optional

import redis.asyncio as aioredis

from src.core.config import settings
from src.core.logger import logger
from src.db.database import async_session
from src.db.models import Task, TaskEvent, HumanReview, Report

# ── SSE 事件队列（内存降级通道）──
_sse_queues: dict[int, list[asyncio.Queue]] = {}

# ── Redis 连接缓存 ──
_redis_pool: Optional[aioredis.Redis] = None


async def _get_redis() -> Optional[aioredis.Redis]:
    """获取 Redis 连接（惰性初始化），连接失败返回 None"""
    global _redis_pool
    if _redis_pool is not None:
        return _redis_pool
    try:
        r = aioredis.from_url(settings.redis_url, decode_responses=True, max_connections=4)
        await r.ping()
        _redis_pool = r
        logger.info("Redis 连接成功")
        return r
    except Exception as e:
        logger.warning(f"Redis 不可用 ({e})，SSE 降级为单进程模式")
        return None


# ── SSE 辅助（内存队列 + Redis Pub/Sub 双通道）──

def _push_event(task_id: int, event_type: str, data: dict):
    """向指定 task_id 推送 SSE 事件（内存队列 + Redis Pub/Sub 双通道）"""
    payload = {"type": event_type, "data": data}

    # 1. 内存队列（降级通道）
    queues = _sse_queues.get(task_id, [])
    stale = []
    for q in queues:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            stale.append(q)
    for q in stale:
        q.put_nowait({"type": "overflow", "data": {"dropped": True}})

    # 2. Redis Pub/Sub（跨进程通道）
    asyncio.create_task(_push_redis_event(task_id, payload))


async def _push_redis_event(task_id: int, payload: dict):
    """通过 Redis Pub/Sub 发布事件（供 arq Worker 跨进程使用）"""
    redis = await _get_redis()
    if redis:
        try:
            await redis.publish(
                f"task:{task_id}:events",
                json.dumps(payload, ensure_ascii=False),
            )
        except Exception:
            pass  # Redis 异常不影响主流程


def subscribe(task_id: int) -> asyncio.Queue:
    """为 SSE 订阅创建一个内存队列"""
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


# ── arq 入队辅助 ──

_arq_pool = None


async def _get_arq_pool():
    """获取或初始化 arq 连接池（全局缓存，避免泄漏）"""
    global _arq_pool
    if _arq_pool is not None:
        return _arq_pool  # 信任缓存，失败靠重试机制恢复
    from arq import create_pool
    from arq.connections import RedisSettings
    try:
        pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
        _arq_pool = pool
        logger.info("arq 连接池已初始化")
        return pool
    except Exception as e:
        logger.warning(f"arq 连接池初始化失败: {e}")
        return None


async def _enqueue_stage(task_id: int, stage: str) -> bool:
    """尝试一次 arq 入队，成功返回 True，失败返回 False。"""
    redis = await _get_redis()
    if not redis:
        return False
    pool = await _get_arq_pool()
    if not pool:
        return False
    try:
        await pool.enqueue_job("run_pipeline_stage", task_id, stage)
        logger.info(f"[{task_id}] 入队 arq 阶段: {stage}")
        return True
    except Exception as e:
        logger.warning(f"[{task_id}] arq 入队失败: {e}")
        return False


async def _enqueue_stage_with_retry(task_id: int, stage: str, origin: str = "") -> bool:
    """带重试的入队 + fallback 记录。

    origin: "main"(主进程) 或 "worker"(Worker 进程)，用于标记 exec_path。
    重试 3 次，间隔 1 秒。
    失败后设置 exec_path 并返回 False，由调用方决定 fallback。
    """
    for attempt in range(1, 4):
        ok = await _enqueue_stage(task_id, stage)
        if ok:
            # 成功时记录 exec_path="arq"
            async with async_session() as session:
                t = await session.get(Task, task_id)
                if t:
                    t.exec_path = "arq"
                    await session.commit()
            return True
        if attempt < 3:
            logger.info(f"[{task_id}] 第 {attempt} 次重试入队 {stage}（1 秒后）")
            await asyncio.sleep(1)

    # 三次全失败 → 标记 fallback，返回 False
    async with async_session() as session:
        t = await session.get(Task, task_id)
        if t:
            t.exec_path = f"fallback({origin or 'unknown'})"
            await session.commit()
    logger.warning(f"[{task_id}] arq 入队失败（重试 3 次），标记 fallback({origin})")
    return False


# ── 工作流编排 ──

async def start_pipeline(topic: str, user_id: int = 0) -> int:
    """
    启动一个新的调研流水线。

    返回 task_id，后续所有操作通过 task_id 引用。
    arq 可用时通过 arq Worker 执行，否则使用 asyncio.create_task fallback。
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

    try:
        # 尝试 arq 入队（带重试+fallback 标记），失败则 fallback
        ok = await _enqueue_stage_with_retry(task_id, "plan", origin="main")
        if not ok:
            asyncio.create_task(_run_stage(task_id, "plan"))
    except Exception as e:
        logger.error(f"[{task_id}] start_pipeline 异常: {e}")
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


async def execute_stage(task_id: int, stage: str) -> dict:
    """
    arq Worker / fallback 共用入口：执行单个阶段并写入数据库。

    返回 {"stage": stage, "status": "completed"} 或抛出异常。
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

    # 获取 task 上下文与非本阶段的中间产物
    async with async_session() as session:
        task = await session.get(Task, task_id)
        if task is None or task.status == "aborted":
            raise RuntimeError(f"[{task_id}] 任务已中止")

    plan = await _load_intermediate(task_id, "plan_output")
    research = await _load_intermediate(task_id, "research_output")
    analysis = await _load_intermediate(task_id, "analysis_output")
    critic_output = await _load_intermediate(task_id, "critic_output")

    _push_event(task_id, "stage_start", {"stage": stage, "label": STAGE_LABELS.get(stage, stage)})

    try:
        if stage == "plan":
            crew = create_plan_crew(task.topic)
            result = await _run_crew_safe(crew)
            output = result if isinstance(result, str) else str(result)
            await _save_intermediate(task_id, "plan_output", output)

        elif stage == "research":
            if not plan:
                raise ValueError("缺少 plan_output，无法执行 research 阶段")
            # ★ 不走 CrewAI 自带的 async_execution：实测 3 任务共享一个 Crew
            # 时耗时远超单任务之和（单任务 18.8s，3 任务撞穿 300s 阶段超时），
            # 且单任务超时无法隔离。改为每个子方向一个独立 Crew，由本层
            # asyncio.gather 并发 + 各自硬超时，单个方向失败不拖垮其余方向。
            parts = await _run_research_fanout(task_id, plan)
            output = _flatten_crew_output([text for _, text in parts])
            await _save_intermediate(task_id, "research_output", output)
            await _save_intermediate(task_id, "research_parts", {
                name: {"chars": len(text)} for name, text in parts
            })
            _push_event(task_id, "research_fanout", {
                "stage": stage,
                "directions": [{"name": n, "chars": len(t)} for n, t in parts],
            })

            # ── 引用台账硬校验（PRD：不入库不引用）──
            # 工具层已把每个来源写入 sources/chunks，这里校验 LLM 抄写的标记
            # 是否真实存在，并把带标记的句子落成 claims + claim_evidence。
            from src.core.ledger import verify_citations, record_claims

            check = await verify_citations(output)
            claims = await record_claims(output, task_id)
            await _save_intermediate(task_id, "citation_check", {
                "total": check["total"],
                "valid": len(check["valid"]),
                "invalid": len(check["invalid"]),
                "rate": round(check["rate"], 4),
                "pass": check["pass"],
                "hallucinated": [d["marker"] for d in check["invalid"]][:20],
                "claims": claims["claims"],
                "evidence": claims["evidence"],
            })
            logger.info(
                f"[{task_id}] 引用校验: {len(check['valid'])}/{check['total']} 有效"
                f"（幻觉 {len(check['invalid'])}），"
                f"落账 claims={claims['claims']} evidence={claims['evidence']}"
            )
            _push_event(task_id, "citation_check", {
                "stage": stage, "total": check["total"],
                "valid": len(check["valid"]), "invalid": len(check["invalid"]),
                "pass": check["pass"],
            })

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

            # ── 终稿引用硬校验：Writer 可能新增或改坏引用标记 ──
            from src.core.ledger import verify_citations, record_claims

            check = await verify_citations(output)
            claims = await record_claims(output, task_id)
            await _save_intermediate(task_id, "citation_check_final", {
                "total": check["total"],
                "valid": len(check["valid"]),
                "invalid": len(check["invalid"]),
                "rate": round(check["rate"], 4),
                "pass": check["pass"],
                "hallucinated": [d["marker"] for d in check["invalid"]][:20],
                "claims": claims["claims"],
                "evidence": claims["evidence"],
            })
            logger.info(
                f"[{task_id}] 终稿引用校验: {len(check['valid'])}/{check['total']} 有效"
                f"（幻觉 {len(check['invalid'])}）"
            )
            _push_event(task_id, "citation_check", {
                "stage": stage, "total": check["total"],
                "valid": len(check["valid"]), "invalid": len(check["invalid"]),
                "pass": check["pass"],
            })

        elif stage == "review":
            report_draft = await _load_intermediate(task_id, "report_draft")
            if not report_draft:
                raise ValueError("缺少 report_draft，无法执行 review 阶段")
            crew = create_review_crew(report_draft)
            result = await _run_crew_safe(crew)
            output = result if isinstance(result, str) else str(result)
            await _save_intermediate(task_id, "review_output", output)

        # 更新 current_stage 字段（execute_stage 不经过 _run_stage）
        async with async_session() as session:
            t = await session.get(Task, task_id)
            if t:
                t.current_stage = stage
                t.updated_at = datetime.utcnow()
                await session.commit()

        _push_event(task_id, "stage_done", {"stage": stage})
        logger.info(f"[{task_id}] 阶段 {stage} 完成")
        return {"stage": stage, "status": "completed"}

    except Exception as e:
        logger.error(f"[{task_id}] 阶段 {stage} 执行失败: {e}")
        _push_event(task_id, "stage_error", {"stage": stage, "error": str(e)})
        raise


async def _create_gate(task_id: int, stage: str):
    """阶段执行完成后创建 HITL 闸门（arq Worker 和 fallback 共用）"""
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

    _push_event(task_id, "gate_pending", {
        "stage": stage,
        "label": STAGE_LABELS.get(stage, stage),
        "gate_name": gate_name,
    })
    logger.info(f"[{task_id}] 阶段 {stage} 完成，等待人工确认 (gate: {gate_name})")


async def _run_stage(task_id: int, stage: str):
    """
    执行一个阶段（含 HITL 闸门创建）。
    用于 fallback 模式（asyncio.create_task）。
    """
    try:
        await execute_stage(task_id, stage)
    except Exception:
        await _update_task_status(task_id, "failed")
        return

    await _create_gate(task_id, stage)


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

        review = await _find_pending_review(session, task_id, gate_name)
        if review:
            review.status = "approved"
            review.comment = comment or "已确认"
            review.resolved_at = datetime.utcnow()

        stage = task.current_stage or "plan"
        task.gate_status = "approved"

        session.add(TaskEvent(
            task_id=task_id,
            event_type="gate_approved",
            payload={"gate": gate_name, "comment": comment},
        ))
        await session.commit()

    _push_event(task_id, "gate_approved", {"gate": gate_name})

    try:
        idx = STAGES.index(stage)
    except ValueError:
        idx = -1

    if idx >= len(STAGES) - 1:
        await _finish_pipeline(task_id)
    else:
        next_stage = STAGES[idx + 1]
        await _update_task_status(task_id, "running")
        # 尝试 arq 入队（带重试+fallback 标记），失败则 fallback
        ok = await _enqueue_stage_with_retry(task_id, next_stage, origin="main")
        if not ok:
            asyncio.create_task(_run_stage(task_id, next_stage))

    return True


async def reject_gate(task_id: int, gate_name: str, comment: str = "") -> bool:
    """
    用户驳回闸门，默认回退到上一阶段。
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

    try:
        idx = STAGES.index(stage)
    except ValueError:
        idx = 1

    prev_stage = STAGES[max(0, idx - 1)]
    await _update_task_status(task_id, "running")
    ok = await _enqueue_stage_with_retry(task_id, prev_stage, origin="main")
    if not ok:
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
            bibtex="",
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
    """安全运行 Crew.kickoff_async()，带硬超时。

    超时是硬约束：单阶段墙钟超过 settings.stage_timeout_sec 后取消并抛
    TimeoutError，由 execute_stage 的 except 分支写 stage_error 事件、
    标记任务 failed，而不是让整个流水线永久挂起。
    """
    timeout = settings.stage_timeout_sec
    try:
        result = await asyncio.wait_for(crew.kickoff_async(), timeout=timeout)
        return result
    except asyncio.TimeoutError:
        logger.error(f"Crew 执行超时（>{timeout}s），已强制中断")
        raise TimeoutError(f"Crew 执行超过 {timeout} 秒未返回，已中断")
    except Exception as e:
        logger.error(f"Crew 执行失败: {e}")
        raise


async def _run_research_fanout(task_id: int, plan: str) -> list[tuple[str, str]]:
    """并发跑各检索子方向，每个方向一个独立 Crew + 独立硬超时。

    返回 [(方向名, 输出文本)]，**失败的子方向被剔除**（记日志 + 推事件），
    只要还有 ≥1 个方向成功就继续（部分成功优于整体失败）。
    全部失败才抛错。

    超时用 settings.stage_timeout_sec 的一半做单方向上限：并发下总墙钟
    约等于最慢的那个方向，留出余量给引用校验与落库。
    """
    from src.agents import create_single_research_crew, subdirection_names
    from src.agents import _resolve_agent_config

    names = subdirection_names()
    per_timeout = max(60, settings.stage_timeout_sec // 2)
    sem = asyncio.Semaphore(settings.max_parallel_researchers)

    async def run_one(idx: int, name: str) -> tuple[str, str]:
        async with sem:
            # 每个方向独立 LLM 实例，避免并发共享客户端状态
            crew = create_single_research_crew(
                llm=_resolve_agent_config("Researcher"), index=idx, plan=plan
            )
            t0 = time.monotonic()
            try:
                result = await asyncio.wait_for(crew.kickoff_async(), timeout=per_timeout)
            except asyncio.TimeoutError:
                logger.warning(f"[{task_id}] 子方向 {name} 超时（>{per_timeout}s），跳过")
                _push_event(task_id, "research_direction_failed",
                            {"direction": name, "reason": "timeout"})
                raise
            except Exception as e:
                logger.warning(f"[{task_id}] 子方向 {name} 失败: {e}")
                _push_event(task_id, "research_direction_failed",
                            {"direction": name, "reason": str(e)[:200]})
                raise
            dt = time.monotonic() - t0
            text = result if isinstance(result, str) else str(result)
            logger.info(f"[{task_id}] 子方向 {name} 完成: {len(text)} 字符 / {dt:.1f}s")
            _push_event(task_id, "research_direction_done",
                        {"direction": name, "chars": len(text), "seconds": round(dt, 1)})
            return name, text

    results = await asyncio.gather(
        *(run_one(i, n) for i, n in enumerate(names)),
        return_exceptions=True,
    )

    parts: list[tuple[str, str]] = []
    for name, r in zip(names, results):
        if isinstance(r, BaseException):
            continue
        parts.append(r)

    if not parts:
        raise RuntimeError(
            f"[{task_id}] research 阶段所有 {len(names)} 个子方向均失败"
        )
    if len(parts) < len(names):
        logger.warning(f"[{task_id}] research 仅 {len(parts)}/{len(names)} 个子方向成功")
    return parts


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

        from sqlalchemy import select
        pending = await session.execute(
            select(HumanReview).where(
                HumanReview.task_id == task_id,
                HumanReview.status == "pending",
            ).limit(1)
        )
        pending_gate = pending.scalar_one_or_none()

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
        "exec_path": task.exec_path or "unknown",
        "created_at": task.created_at.isoformat() if task.created_at else None,
    }
