"""arq Worker 配置与任务函数"""

import logging

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
    """
    from src.core.orchestrator import execute_stage, _create_gate
    await execute_stage(task_id, stage)
    await _create_gate(task_id, stage)


class WorkerSettings:
    """arq Worker 配置类"""
    functions = [run_pipeline_stage]
    redis_settings = REDIS_SETTINGS
    keep_result = 3600          # 结果保留 1 小时
    keep_result_forever = False
    # arq 层超时 = 阶段超时 + 60s 余量（阶段自身已有 asyncio.wait_for 硬超时，
    # 这里留出余量以便 orchestrator 有机会写 stage_error 事件、标记 failed）
    timeout = settings.stage_timeout_sec + 60
    poll_delay = 0.5            # 轮询间隔


if __name__ == "__main__":
    """python -m src.core.worker"""
    from arq import run_worker
    run_worker(WorkerSettings)
