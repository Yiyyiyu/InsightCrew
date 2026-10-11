"""arq Worker 配置与任务函数"""

import logging

from arq import cron
from arq.connections import RedisSettings

from src.core.config import settings

# 让 arq Worker 本身输出 INFO 日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("arq").setLevel(logging.INFO)

REDIS_SETTINGS = RedisSettings.from_dsn(settings.redis_url)


async def run_pipeline_stage(ctx, task_id: int, stage: str):
    """arq 任务：执行单个调研阶段（Plan / Research / Analysis / Critic / Write / Review）

    执行完成后自动创建 HITL 闸门等待人工审批。

    ★ 失败必须落状态：execute_stage 抛错（阶段超时、预算熔断
    BudgetExceededError、LLM 报错等）时，这里要把任务标成 failed，
    否则任务会永远停在 running，前端一直转圈、闸门也不会出现。
    """
    from src.core.orchestrator import execute_stage, _create_gate, _update_task_status

    try:
        await execute_stage(task_id, stage)
    except Exception as e:
        logging.getLogger("src.core.worker").error(
            f"[{task_id}] 阶段 {stage} 执行失败，标记任务 failed: {e}"
        )
        await _update_task_status(task_id, "failed")
        # 不 re-raise：arq 会重试失败的 job，而阶段失败（尤其预算熔断）
        # 重试只会重复烧钱。状态已落库、事件已推送，交给人工处置。
        return

    await _create_gate(task_id, stage)


async def reap_stale_tasks(ctx):
    """每 60 秒扫一次停滞任务并标记失败（兜底监督，不接管执行）。

    入队前的健康门控拦不住"检查通过后 worker 恰好崩溃"，
    这类任务会永久停在 running 且毫无报错，由本任务兜底判定。
    """
    from src.core.orchestrator import reap_stale_tasks as _reap
    n = await _reap()
    if n:
        logging.getLogger("src.core.worker").warning(f"回收停滞任务 {n} 个")


class WorkerSettings:
    """arq Worker 配置类

    ★ 注意参数名必须与 arq ``Worker.__init__`` 一致：
    arq 用 ``inspect.signature(Worker).parameters`` 过滤本类的 __dict__
    （见 arq/worker.py:887-890），**名字不对会被静默丢弃**。
    曾经的 ``timeout = ...`` 就是这样失效的——arq 的参数名是 ``job_timeout``，
    于是阶段超时保护实际用的是 arq 默认值 300s，与 stage_timeout_sec=300
    撞车，设计预留的 60s 余量消失。
    """
    functions = [run_pipeline_stage]
    redis_settings = REDIS_SETTINGS
    keep_result = 3600          # 结果保留 1 小时
    keep_result_forever = False

    # 单个 job 的墙钟上限 = 阶段超时 + 60s 余量。
    # 阶段自身已有 asyncio.wait_for 硬超时，这里留出余量以便 orchestrator
    # 有机会写 stage_error 事件、标记 failed。
    job_timeout = settings.stage_timeout_sec + 60

    # 并发上限：不设的话 arq 默认 max_jobs=10，实测曾一次同时启动 10 个阶段
    # （旧队列积压时），4C4G 上会直接把内存打爆。
    max_jobs = settings.worker_concurrency

    # 健康键写入间隔（秒）。健康键 TTL = interval + 1，因此 30 秒后
    # worker 崩溃就能被 async_check_health 检测到（默认 3600 意味着
    # worker 崩了整整一小时外部还认为它健康）。
    health_check_interval = 30

    poll_delay = 0.5            # 轮询间隔

    # 停滞任务回收：每 60 秒（second=0 表示每分钟的第 0 秒）
    cron_jobs = [cron(reap_stale_tasks, second=0)]


if __name__ == "__main__":
    """python -m src.core.worker"""
    from arq import run_worker
    run_worker(WorkerSettings)
