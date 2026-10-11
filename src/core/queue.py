"""arq 任务队列 —— 阶段常量、入队、worker 健康门控。

从 `orchestrator.py` 拆分而来（M1/M2 遗留缺陷第 7 项，纯搬移零行为变更）。

设计：**唯一执行路径是 arq worker**。Redis 或 worker 不可用即抛
`ServiceUnavailableError`，不做进程内降级 —— 降级只是调试期的临时替代，
不是分支方案（用户 m08865 拍板）。入队保留 3 次重试，只吸收连接抖动
（用户 m08903）。
"""

import asyncio
from typing import Optional

from src.core.config import settings
from src.core.errors import ServiceUnavailableError
from src.core.logger import logger
from src.db.database import async_session
from src.db.models import Task

from src.core.events import _get_redis


# ── 阶段定义 ──

STAGES = ["plan", "research", "analysis", "critic", "rebuttal", "write", "review"]

# ── 人工闸门阶段（闸门收敛，用户 m18246/m18259 拍板）──
#
# 判据：**这个节点上人类能不能做出机器做不了的判断，且错了代价多大。**
#
#   * plan     ✅ 唯一能改「问题本身」的点：方向错，后面全白跑。评审成本最低
#                （几百字），杠杆最高。
#   * research ✅ 唯一能改「证据基础」的点：产品承诺就是可追溯。评审成本低
#                （来源清单 + 关键 chunk）。
#   * review   ✅ 交付物验收点：批准即完成。打回时回退到 write。
#
# 被砍掉的四个阶段不是不重要，而是「机器能自己判断」：
#   * analysis —— 对**已批准**证据的推导；实质问题由 critic→rebuttal 自动闭环。
#   * critic   —— 产物本就是给 rebuttal 消费的，人工在此挑质疑会让自动回应
#                  机制失去意义。
#   * rebuttal —— 内部修正步骤，产物直接进 write。
#   * write    —— 验收挪到 review 之后做（能看到「报告 + 审核意见」，信息更全）。
#
# 结果：人工交互 7 次 → 3 次，砍掉的是中间加工，留下的是只有用户能拍板的
# 三件事：**方向对不对、证据够不够、交付物收不收**。
#
# 这一份是**默认路径**。接口留宽（`approve_gate`/`reject_gate` 的 `gate_name`
# 本就是入参，机制通用；`/research/{id}/pause` 可把任意阶段边界临时变成闸门），
# 默认路径收窄。
GATE_STAGES = ("plan", "research", "review")

STAGE_LABELS = {
    "plan": "调研规划",
    "research": "检索取证",
    "analysis": "对比分析",
    "critic": "红队质疑",
    "rebuttal": "质疑回应",
    "write": "报告撰写",
    "review": "质量审核",
}


# ── arq 入队辅助（唯一执行路径）──
#
# ★ 设计原则：这条流水线只有一条正常路径 —— arq 入队 → worker 执行。
#
#   历史缺陷：入队函数曾返回 bool，`enqueue_job` 不抛异常就 return True，
#   而 True 只证明"事件进了 Redis 列表"，**不证明有 worker 在取**。
#   于是 worker 缺席时任务永久停在 running：前端一直转圈、无错误、无超时、
#   不自愈。降级到 `asyncio.create_task(_run_stage(...))` 更糟——阶段跑在
#   API 进程里，uvicorn 一重启（开发模式 --reload 每次改代码都重启）阶段
#   就凭空消失，任务照样永久 running，只是换了个更难查的成因。
#
#   现在的做法：入队前先做执行前置检查（Redis / worker 健康键 / 队列池），
#   任一缺失即抛 ServiceUnavailableError，由 API 层转 503 让调用方看见。
#   失败必须是响亮的、可见的、可诊断的。

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


async def _worker_alive() -> bool:
    """检查是否有活着的 arq worker。

    原理：worker 每 ``health_check_interval`` 秒用 psetex 写一个健康键
    （`arq:queue:health-check`），TTL = interval + 1 秒。
    `async_check_health` 读该键：返回 0 = 健康，1 = 键不存在或已过期。
    注意必须用协程版 `async_check_health`——同步版 `check_health` 内部
    `asyncio.run`，在事件循环里调用会抛
    `RuntimeError: asyncio.run() cannot be called from a running event loop`。
    """
    from arq.worker import async_check_health
    from arq.connections import RedisSettings
    try:
        rc = await async_check_health(RedisSettings.from_dsn(settings.redis_url))
        return rc == 0
    except Exception as e:
        logger.warning(f"worker 健康检查失败: {e}")
        return False


async def _ensure_executable() -> None:
    """执行前置检查：三个依赖任一缺失，立即抛服务异常。

    调用方必须**在产生副作用之前**调用它（例如在改闸门状态、落任务行之前），
    这样服务不可用时不会留下半成品状态——用户修好服务后可以直接重试。
    """
    if not await _get_redis():
        raise ServiceUnavailableError("redis", "Redis 连接不可用，无法投递任务")
    if not await _worker_alive():
        raise ServiceUnavailableError(
            "worker", "任务执行器（arq worker）未启动，无法处理任务"
        )
    if not await _get_arq_pool():
        raise ServiceUnavailableError("queue", "arq 任务队列连接池不可用")


async def _enqueue_stage(task_id: int, stage: str) -> None:
    """唯一入队入口：把某阶段投给 worker。失败即抛，绝不静默返回。

    带 3 次重试（间隔 1 秒）只针对 ``enqueue_job`` 本身的瞬时抖动
    （连接波动、Redis 短暂忙），不改变"依赖不可用就是异常"的结论。

    ★ ``_job_id`` 去重与"打回重跑"的冲突（实测缺陷，已修）：

    ``_job_id=f"{task_id}:{stage}"`` 用来防止同一 (任务, 阶段) 被重复入队
    （曾实测 task 11 被启动两次）。但 arq 的 ``enqueue_job`` 在
    ``arq:job:<id>`` **或** ``arq:result:<id>`` 任一存在时，会**静默返回 None**
    （不抛异常，见 ``arq/connections.py::ArqRedis.enqueue_job`` 里的
    ``if await pipe.exists(job_key, result_key_prefix + job_id): return None``）。
    而 ``WorkerSettings.keep_result = 3600`` 让结果键存活一小时，于是：

      * ``reject_gate`` 打回后重投同一阶段 → 结果键还在 → 返回 None →
        **"打回重跑"静默失效，任务永远停在闸门**；
      * 测试库重建后 task id 从 1 重新开始，旧结果键会让新任务永远 running。

    所以入队前先看 ``Job.status()``：
      * ``queued`` / ``deferred`` / ``in_progress`` → 确有同 id 的 job 在飞，
        按幂等处理，直接返回（这正是去重想要的效果）；
      * ``complete`` → 陈旧结果键挡路，先删掉它再入队；
      * ``not_found`` → 干净，直接入队。

    最后仍要检查 ``enqueue_job`` 的返回值：并发下可能被别人抢先入队，
    ``None`` 此时等价于"已入队"，视为成功。
    """
    await _ensure_executable()
    pool = await _get_arq_pool()
    job_id = f"{task_id}:{stage}"

    from arq.jobs import Job, JobStatus, result_key_prefix

    status = await Job(job_id, pool).status()
    if status in (JobStatus.queued, JobStatus.deferred, JobStatus.in_progress):
        logger.info(f"[{task_id}] 阶段 {stage} 已在队列/执行中，跳过重复入队")
        return
    if status == JobStatus.complete:
        # 陈旧的 keep_result 结果键会挡住入队（arq 返回 None 而非报错）
        await pool.delete(result_key_prefix + job_id)
        logger.info(f"[{task_id}] 清理阶段 {stage} 的陈旧结果键，准备重投")

    last_err: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            created = await pool.enqueue_job(
                "run_pipeline_stage",
                task_id,
                stage,
                _job_id=job_id,
            )
            if created is None:
                # 并发下被抢先入队：等价于已入队，不是失败
                logger.info(f"[{task_id}] 阶段 {stage} 已被并发入队，跳过")
                return
            logger.info(f"[{task_id}] 入队 arq 阶段: {stage}")
            return
        except Exception as e:
            last_err = e
            if attempt < 3:
                logger.info(f"[{task_id}] 第 {attempt} 次入队 {stage} 失败，1 秒后重试: {e}")
                await asyncio.sleep(1)

    logger.error(f"[{task_id}] arq 入队失败（重试 3 次）: {last_err}")
    raise ServiceUnavailableError("queue", f"任务入队失败: {last_err}")


async def _mark_exec_path(task_id: int, path: str) -> None:
    """记录任务的实际执行路径。现在只会有 "arq" 一种正常值。"""
    async with async_session() as session:
        t = await session.get(Task, task_id)
        if t:
            t.exec_path = path
            await session.commit()
