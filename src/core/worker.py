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
    timeout = 600               # 单任务最长 10 分钟
    poll_delay = 0.5            # 轮询间隔


if __name__ == "__main__":
    """python -m src.core.worker"""
    from arq import run_worker
    run_worker(WorkerSettings)
