"""SSE 事件推送 —— 落库 + 内存队列 + Redis Pub/Sub 三通道。

从 `orchestrator.py` 拆分而来（M1/M2 遗留缺陷第 7 项，纯搬移零行为变更）。

职责：
1. 事件落库 `task_events` 并取回自增 id（它同时是 SSE 的 Last-Event-ID）
2. 同进程内存队列（单进程调试通道）
3. Redis Pub/Sub（arq worker 进程 → Web 进程的唯一通路）
4. 断线重连的增量回放（`_replay_events`）与直播循环（`event_stream`）

★ `_redis_pool` 与 `_redis_sub_pool` 必须分开：SSE 每个在线客户端长期独占
   一条订阅连接，共用一个池会被占满，之后 `Too many connections` 被 except
   吞掉 → 静默退回内存通道 → 跨进程事件全丢。
"""

import json
import asyncio
from typing import Optional

import redis.asyncio as aioredis

from src.core.config import settings
from src.core.logger import logger
from src.db.database import async_session
from src.db.models import Task, TaskEvent


# ── SSE 事件队列（内存降级通道）──
_sse_queues: dict[int, list[asyncio.Queue]] = {}

# ── Redis 连接缓存 ──
# 两个独立客户端：
#   _redis_pool      —— 命令/发布（短连接，用后归还）
#   _redis_sub_pool  —— SSE 订阅（每个在线客户端长期独占一条连接）
# 必须分开：曾经共用一个 max_connections=4 的池，几个 SSE 客户端就把池占满，
# 之后 `Too many connections` 被 except 吞掉 → 静默退回内存通道 → 跨进程事件全丢。
_redis_pool: Optional[aioredis.Redis] = None
_redis_sub_pool: Optional[aioredis.Redis] = None


async def _get_redis() -> Optional[aioredis.Redis]:
    """命令/发布用 Redis 客户端（惰性初始化），连接失败返回 None"""
    global _redis_pool
    if _redis_pool is not None:
        return _redis_pool
    try:
        r = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            max_connections=settings.redis_max_connections,
        )
        await r.ping()
        _redis_pool = r
        logger.info("Redis 连接成功")
        return r
    except Exception as e:
        logger.warning(f"Redis 不可用 ({e})，SSE 降级为单进程模式")
        return None


async def _get_sub_redis() -> Optional[aioredis.Redis]:
    """SSE 订阅专用 Redis 客户端（独立连接池，容量远大于命令池）"""
    global _redis_sub_pool
    if _redis_sub_pool is not None:
        return _redis_sub_pool
    try:
        r = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            max_connections=settings.redis_sub_max_connections,
        )
        await r.ping()
        _redis_sub_pool = r
        logger.info(f"Redis 订阅池就绪（上限 {settings.redis_sub_max_connections}）")
        return r
    except Exception as e:
        logger.warning(f"Redis 订阅池不可用 ({e})，SSE 退回内存通道")
        return None


# ── SSE 事件通道（落库 + 内存队列 + Redis Pub/Sub 三通道）──
#
# 设计要点（修复跨进程断链）：
#   1. 每个事件先写 task_events 落库并拿到自增 id —— 这个 id 同时是 SSE 的
#      Last-Event-ID，断线重连时用它做增量回放（replay）。
#   2. 内存队列只对本进程有效（fallback 模式）。
#   3. Redis Pub/Sub 是 arq Worker（独立进程）→ Web 进程的唯一通路；
#      没有它，Worker 发的事件永远到不了浏览器。
# 三通道携带同一个 event id，因此无论走哪条路径，客户端看到的顺序一致。

def _event_channel(task_id: int) -> str:
    return f"task:{task_id}:events"


async def _persist_event(task_id: int, event_type: str, data: dict) -> Optional[int]:
    """把事件写入 task_events，返回自增 id（失败返回 None，不影响主流程）"""
    try:
        async with async_session() as session:
            row = TaskEvent(task_id=task_id, event_type=event_type, payload=data)
            session.add(row)
            await session.commit()
            return row.id
    except Exception as e:
        logger.warning(f"[{task_id}] 事件落库失败 ({event_type}): {e}")
        return None


async def _push_event(task_id: int, event_type: str, data: dict) -> Optional[int]:
    """推送 SSE 事件：落库 → 内存队列 → Redis Pub/Sub，返回事件 id"""
    event_id = await _persist_event(task_id, event_type, data)
    payload = {"type": event_type, "data": data, "id": event_id}

    # 1. 内存队列（同进程降级通道）
    for q in list(_sse_queues.get(task_id, [])):
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            try:
                q.get_nowait()
                q.put_nowait({"type": "overflow", "data": {"dropped": True}, "id": event_id})
            except Exception:
                pass

    # 2. Redis Pub/Sub（跨进程通道，arq Worker → Web）
    await _push_redis_event(task_id, payload)
    return event_id


async def _push_redis_event(task_id: int, payload: dict):
    """通过 Redis Pub/Sub 发布事件（供 arq Worker 跨进程使用）"""
    redis = await _get_redis()
    if redis:
        try:
            await redis.publish(_event_channel(task_id), json.dumps(payload, ensure_ascii=False))
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


# SSE 流的真正终态。
# ★ 不能把 stage_error 放进来：一个阶段失败后用户完全可能 reject/approve 把它
#   救回来，若 stage_error 关闭了事件流，浏览器此后收不到任何后续事件，
#   页面永远停在"出错"状态而任务其实还在推进。
TERMINAL_EVENTS = ("pipeline_complete", "pipeline_aborted")


async def _replay_events(task_id: int, last_event_id: int) -> list[dict]:
    """回放 last_event_id 之后落库的事件（断线续传）"""
    from sqlalchemy import select
    async with async_session() as session:
        rows = await session.execute(
            select(TaskEvent)
            .where(
                TaskEvent.task_id == task_id,
                TaskEvent.id > last_event_id,
                # 中间产物（intermediate_*）体量大且是内部数据，不进 SSE
                ~TaskEvent.event_type.like("intermediate_%"),
            )
            .order_by(TaskEvent.id)
        )
        return [
            {"type": r.event_type, "data": r.payload or {}, "id": r.id}
            for r in rows.scalars().all()
        ]


async def _task_is_terminal(task_id: int) -> bool:
    async with async_session() as session:
        task = await session.get(Task, task_id)
        if task is None:
            return True
        return task.status in ("completed", "aborted", "failed")


async def event_stream(task_id: int, last_event_id: int = 0):
    """SSE 事件生成器：先回放落库事件，再转直播（Redis 优先，内存兜底）。

    这是跨进程修复的核心：arq Worker 与 Web 是两个进程，Worker 的
    `_push_event` 写库 + publish，本生成器在 Web 进程读库回放 + subscribe，
    因此 Worker 的事件可以到达浏览器。
    """
    memory_q = subscribe(task_id)
    pubsub = None
    seen_ids: set[int] = set()

    try:
        # ── 阶段 0：先订阅 Redis，再回放 ──
        # 顺序不能颠倒：先回放后订阅，两者之间产生的事件会丢。先订阅后回放，
        # 重叠部分靠 seen_ids 去重，保证 at-least-once。
        redis = await _get_sub_redis()
        if redis:
            try:
                pubsub = redis.pubsub()
                await pubsub.subscribe(_event_channel(task_id))
                logger.info(f"[{task_id}] SSE 已订阅 Redis 频道 {_event_channel(task_id)}")
            except Exception as e:
                logger.warning(f"[{task_id}] SSE 订阅 Redis 失败，退回内存通道: {e}")
                if pubsub is not None:
                    try:
                        await pubsub.aclose()
                    except Exception:
                        pass
                pubsub = None

        # ── 阶段 1：回放（补齐断线期间或订阅前产生的事件）──
        for ev in await _replay_events(task_id, last_event_id):
            if ev["id"] is not None:
                seen_ids.add(ev["id"])
            yield {"event": ev["type"], "data": json.dumps(ev["data"], ensure_ascii=False), "id": str(ev["id"])}
            if ev["type"] in TERMINAL_EVENTS:
                return

        # ── 阶段 2：直播 ──
        while True:
            payload = None

            # 2a. Redis 通道（跨进程）
            if pubsub is not None:
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if msg and msg.get("data"):
                    try:
                        payload = json.loads(msg["data"])
                    except Exception:
                        payload = None

            # 2b. 内存通道（同进程兜底）
            if payload is None:
                try:
                    payload = await asyncio.wait_for(memory_q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    payload = None

            if payload is None:
                # 空闲：任务已进入终态且再无事件 → 收尾
                if await _task_is_terminal(task_id):
                    return
                continue

            ev_id = payload.get("id")
            # 同一事件可能同时走 Redis 与内存，用 id 去重
            if ev_id is not None:
                if ev_id in seen_ids:
                    continue
                seen_ids.add(ev_id)

            yield {
                "event": payload["type"],
                "data": json.dumps(payload["data"], ensure_ascii=False),
                "id": str(ev_id) if ev_id is not None else None,
            }

            if payload["type"] in TERMINAL_EVENTS:
                return
    finally:
        unsubscribe(task_id, memory_q)
        if pubsub is not None:
            try:
                await pubsub.unsubscribe(_event_channel(task_id))
                await pubsub.aclose()
            except Exception:
                pass
