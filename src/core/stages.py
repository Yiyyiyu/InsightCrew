"""单个流水线阶段的执行 —— 各阶段 Crew 编排与产物落库。

从 `orchestrator.py` 拆分而来（M1/M2 遗留缺陷第 7 项，纯搬移零行为变更）。

包含：
  * `execute_stage`        —— 按阶段名分派到对应 Crew 工厂
  * `_run_crew_safe`       —— 带超时的 Crew 执行 + 用量记账 + 预算熔断
  * `_run_research_fanout` —— research 阶段按子方向并行 fan-out
"""

import json
import time
import asyncio
from datetime import datetime
from typing import Optional

from src.core.config import settings
from src.core.cost import BudgetExceededError, enforce_budget, record_crew_usage
from src.core.logger import logger
from src.db.database import async_session
from src.db.models import Task

from src.core.artifacts import (
    _load_intermediate,
    _save_evaluation,
    _save_intermediate,
)
from src.core.critique import _critique_resolution_rate
from src.core.events import _push_event
from src.core.queue import STAGE_LABELS


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
        create_rebuttal_crew,
        create_write_crew,
        create_review_crew,
    )

    logger.info(f"[{task_id}] 开始阶段: {stage}")

    # 获取 task 上下文与非本阶段的中间产物
    async with async_session() as session:
        task = await session.get(Task, task_id)
        if task is None or task.status == "aborted":
            raise RuntimeError(f"[{task_id}] 任务已中止")
        # 任务级参数（M1 遗留项 5）：从库里读回来，不靠 API 进程内存传递
        topic = task.topic
        dimensions = task.dimensions or ""
        constraints = task.constraints or ""
        model_plan_raw = task.model_plan or ""
        user_id = int(task.user_id or 0)

    plan = await _load_intermediate(task_id, "plan_output")
    research = await _load_intermediate(task_id, "research_output")
    analysis = await _load_intermediate(task_id, "analysis_output")
    critic_output = await _load_intermediate(task_id, "critic_output")
    rebuttal_output = await _load_intermediate(task_id, "rebuttal_output")

    await _push_event(task_id, "stage_start", {"stage": stage, "label": STAGE_LABELS.get(stage, stage)})

    # ★ 模型分配表写进 ContextVar：`_resolve_agent_config`（唯一构造 CrewAI LLM
    #   的地方）按角色读它决定本阶段用哪个模型。必须在构造任何 Crew 之前设置；
    #   asyncio 子任务（research fan-out）会复制当前上下文，并发方向也能读到。
    from src.core.context import (
        reset_current_model_plan,
        reset_model_plan_explicit,
        set_current_model_plan,
        set_model_plan_explicit,
    )

    plan_token = set_current_model_plan(_resolve_task_model_plan(model_plan_raw, user_id))
    # ★ 「已显式解析」= 任务行里真的存了 model_plan。
    #   真实任务必然如此：`/research/start` 的三道门（解锁校验 → 补全 7 个角色
    #   → 连通性预检）全过才建 Task 行，所以生产环境不存在 model_plan 为空的任务。
    #   model_plan 为 NULL 只可能是直接调 `start_pipeline` 的开发/测试路径 ——
    #   那种情况下允许回落内置默认目录，否则测试与本地调试都跑不起来。
    explicit_token = set_model_plan_explicit(bool(model_plan_raw))

    try:
        # ── 预算前置门 ──
        # 已超限的任务不允许再进入下一个阶段：否则一个跑飞的任务会在后续阶段
        # 继续烧钱，熔断就失去意义。超限时抛 BudgetExceededError，由本函数的
        # except 分支写 stage_error 事件、上层标记任务 failed。
        # ★ 必须放在 try 内：放在 try 外时异常直接逃逸，stage_error 事件不写，
        #   前端只看到任务变 failed 却没有任何原因（实测抓到过）。
        await enforce_budget(task_id)

        if stage == "plan":
            crew = create_plan_crew(topic, dimensions=dimensions, constraints=constraints)
            result = await _run_crew_safe(crew, task_id=task_id, stage=stage)
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
            await _push_event(task_id, "research_fanout", {
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
            await _push_event(task_id, "citation_check", {
                "stage": stage, "total": check["total"],
                "valid": len(check["valid"]), "invalid": len(check["invalid"]),
                "pass": check["pass"],
            })

        elif stage == "analysis":
            if not research:
                raise ValueError("缺少 research_output，无法执行 analysis 阶段")
            crew = create_analysis_crew(research)
            result = await _run_crew_safe(crew, task_id=task_id, stage=stage)
            output = result if isinstance(result, str) else str(result)
            await _save_intermediate(task_id, "analysis_output", output)

        elif stage == "critic":
            if not analysis:
                raise ValueError("缺少 analysis_output，无法执行 critic 阶段")
            crew = create_critic_crew(analysis)
            result = await _run_crew_safe(crew, task_id=task_id, stage=stage)
            output = result if isinstance(result, str) else str(result)
            await _save_intermediate(task_id, "critic_output", output)

        elif stage == "rebuttal":
            # ★ 「质疑 → 回应」的落点：把 Critic 的质疑回灌给 Analyst 本人重做。
            #   此前质疑只被塞进 Writer 的提示词，由写手在正文里敷衍，等于没回应。
            if not analysis:
                raise ValueError("缺少 analysis_output，无法执行 rebuttal 阶段")
            if not critic_output:
                raise ValueError("缺少 critic_output，无法执行 rebuttal 阶段")
            crew = create_rebuttal_crew(analysis, critic_output)
            result = await _run_crew_safe(crew, task_id=task_id, stage=stage)
            output = result if isinstance(result, str) else str(result)
            await _save_intermediate(task_id, "rebuttal_output", output)

            # ── 质疑回应率：写 evaluations 表，供 M3 的 G7 评测聚合 ──
            rate = _critique_resolution_rate(output, critic_output)
            await _save_evaluation(task_id, "critique_resolution_rate", rate)
            await _push_event(task_id, "critique_resolution", {
                "stage": stage,
                "rate": rate,
            })

        elif stage == "write":
            if not all([plan, analysis, critic_output]):
                raise ValueError("缺少中间产物，无法执行 write 阶段")
            # 写手优先采用「回应质疑后的修订版分析」；rebuttal 阶段缺失时退回原分析。
            final_analysis = rebuttal_output or analysis
            crew = create_write_crew(plan, final_analysis, critic_output)
            result = await _run_crew_safe(crew, task_id=task_id, stage=stage)
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
            await _push_event(task_id, "citation_check", {
                "stage": stage, "total": check["total"],
                "valid": len(check["valid"]), "invalid": len(check["invalid"]),
                "pass": check["pass"],
            })

        elif stage == "review":
            report_draft = await _load_intermediate(task_id, "report_draft")
            if not report_draft:
                raise ValueError("缺少 report_draft，无法执行 review 阶段")
            crew = create_review_crew(report_draft)
            result = await _run_crew_safe(crew, task_id=task_id, stage=stage)
            output = result if isinstance(result, str) else str(result)
            await _save_intermediate(task_id, "review_output", output)

        # 更新 current_stage 字段（execute_stage 不经过 _run_stage）
        async with async_session() as session:
            t = await session.get(Task, task_id)
            if t:
                t.current_stage = stage
                t.updated_at = datetime.utcnow()
                await session.commit()

        await _push_event(task_id, "stage_done", {"stage": stage})
        logger.info(f"[{task_id}] 阶段 {stage} 完成")
        return {"stage": stage, "status": "completed"}

    except Exception as e:
        logger.error(f"[{task_id}] 阶段 {stage} 执行失败: {e}")
        await _push_event(task_id, "stage_error", {"stage": stage, "error": str(e)})
        raise
    finally:
        # 恢复模型策略上下文（worker 里同一进程会连着跑多个阶段/任务）
        reset_model_plan_explicit(explicit_token)
        reset_current_model_plan(plan_token)

# ── 内部辅助 ──

def _resolve_task_model_plan(raw: str, user_id: int = 0) -> dict:
    """把任务落库的 ``model_plan`` JSON 解析成 ``{role: ModelOption}``。

    - 解析失败 / 为空 → 返回 {}，由 ``_resolve_agent_config`` 回落到内置默认；
    - 目录（内置按解锁过滤 + 自建）按**任务归属用户**组装：用户自建的模型
      只有他自己能用，绝不能拿到别人的 Key。
    - 同步会话：本函数由 worker 进程调用，不持有请求上下文，用同步 Session
      最省事且不会与外层 async Session 冲突。

    ★ 只解析不校验连通性：连通性已在新建任务的 preflight 阶段拦过
      （见 ``src/core/preflight.py``），执行期再探测等于重复付费。
    """
    if not raw:
        return {}
    try:
        import json

        from sqlalchemy.orm import Session

        from src.core.ledger import get_sync_engine
        from src.core.model_registry import builtin_options, catalog_sync, resolve_plan

        plan = json.loads(raw)
        if not isinstance(plan, dict):
            return {}
        if user_id:
            with Session(get_sync_engine()) as s:
                options = catalog_sync(s, user_id)
        else:
            # 无任务归属（开发/测试）时用内置目录兜底
            options = builtin_options()
        # ★ 只有「没有归属用户」才允许回落内置目录（开发/测试路径）。
        #   真实任务（user_id 非 0）的目录为空 = 该用户既没解锁内置、也没有
        #   任何自建模型 —— 此时**绝不能**给他内置模型，那等于绕过解锁门禁
        #   （M2.5-5 的第 ③ 道强制点）。返回 {} 让上层按无模型处理。
        if not options and not user_id:
            options = builtin_options()
        return {role: opt for role, opt in resolve_plan(plan, options).items() if opt is not None}
    except Exception as e:
        logger.warning(f"解析任务模型分配表失败，回落内置默认: {e}")
        return {}

async def _run_crew_safe(crew, task_id: int = 0, stage: str = "") -> str:
    """安全运行 Crew.kickoff_async()，带硬超时 + 成本记账。

    超时是硬约束：单阶段墙钟超过 settings.stage_timeout_sec 后取消并抛
    TimeoutError，由 execute_stage 的 except 分支写 stage_error 事件、
    标记任务 failed，而不是让整个流水线永久挂起。

    记账：kickoff 返回后从 crew.usage_metrics 取 token 用量写入 llm_calls，
    并检查预算——超限抛 BudgetExceededError 由上层中止任务。
    """
    from src.core.context import reset_current_task_id, set_current_task_id
    from src.core.cost import enforce_budget, record_crew_usage

    timeout = settings.stage_timeout_sec
    # 把 task_id 放进 ContextVar：工具层（如多模态读图）签名由 LLM 决定，
    # 没有地方传 task_id，但它会真实产生 LLM 费用，必须能归属到任务上记账。
    token = set_current_task_id(task_id)
    try:
        try:
            result = await asyncio.wait_for(crew.kickoff_async(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(f"Crew 执行超时（>{timeout}s），已强制中断")
            raise TimeoutError(f"Crew 执行超过 {timeout} 秒未返回，已中断")
        except Exception as e:
            logger.error(f"Crew 执行失败: {e}")
            raise
    finally:
        reset_current_task_id(token)

    # 记账 + 熔断检查（记账失败不影响主流程，由 record_crew_usage 内部兜底）
    if task_id:
        await record_crew_usage(task_id, crew, stage)
        await enforce_budget(task_id)
    return result


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
            # 与 _run_crew_safe 同理：让工具层（parse_pdf / read_image）能归属记账。
            # 每个并发方向各自 set，ContextVar 按协程上下文隔离，不会互相覆盖。
            from src.core.context import reset_current_task_id, set_current_task_id

            token = set_current_task_id(task_id)
            try:
                result = await asyncio.wait_for(crew.kickoff_async(), timeout=per_timeout)
            except asyncio.TimeoutError:
                logger.warning(f"[{task_id}] 子方向 {name} 超时（>{per_timeout}s），跳过")
                await _push_event(task_id, "research_direction_failed",
                            {"direction": name, "reason": "timeout"})
                raise
            except Exception as e:
                logger.warning(f"[{task_id}] 子方向 {name} 失败: {e}")
                await _push_event(task_id, "research_direction_failed",
                            {"direction": name, "reason": str(e)[:200]})
                raise
            finally:
                reset_current_task_id(token)
            dt = time.monotonic() - t0
            text = result if isinstance(result, str) else str(result)

            # 记账：每个子方向各记一笔（并发下用原子累加，见 cost.record_usage）
            from src.core.cost import enforce_budget, record_crew_usage

            await record_crew_usage(task_id, crew, f"research:{name}")
            try:
                await enforce_budget(task_id)
            except BudgetExceededError:
                # 并发场景下预算刚被其他方向突破：本方向已跑完，结果仍返回，
                # 但让 execute_stage 的后续检查把它拦住（不在这里吞掉）。
                logger.warning(f"[{task_id}] 子方向 {name} 完成后预算已超限")

            logger.info(f"[{task_id}] 子方向 {name} 完成: {len(text)} 字符 / {dt:.1f}s")
            await _push_event(task_id, "research_direction_done",
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
