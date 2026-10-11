"""中间产物与评测指标的落库 —— TaskEvent(JSON payload) 读写。

从 `orchestrator.py` 拆分而来（M1/M2 遗留缺陷第 7 项，纯搬移零行为变更）。

中间产物不建独立表，而是以 `intermediate_<key>` 事件类型存进 `task_events`
的 JSON payload：流水线各阶段写、后续阶段与报告页读。SSE 回放会过滤掉
`intermediate_%`（体积大且属内部数据）。
"""

import json
from typing import Optional

from src.core.logger import logger
from src.db.database import async_session
from src.db.models import TaskEvent, Evaluation


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
async def _save_intermediate(task_id: int, key: str, value: str):
    """保存中间产物到 TaskEvent（JSON payload）"""
    async with async_session() as session:
        session.add(TaskEvent(
            task_id=task_id,
            event_type=f"intermediate_{key}",
            payload={"value": value},
        ))
        await session.commit()


async def _save_evaluation(task_id: int, metric: str, value: float):
    """写一行评测指标到 evaluations 表（供 M3 的 G7 同构/异构对比聚合）。

    与成本记账同理：评测失败不应拖垮流水线，因此只 warning 不抛。
    """
    try:
        async with async_session() as session:
            session.add(Evaluation(task_id=task_id, metric=metric[:64], value=float(value)))
            await session.commit()
    except Exception as e:  # noqa: BLE001 — 评测指标缺失不是致命错误
        logger.warning(f"[{task_id}] 评测指标写入失败（{metric}）: {e}")
