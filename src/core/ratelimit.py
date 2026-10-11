"""进程内滑动窗口限流。

## 为什么不用 slowapi

设计规划原写"加 slowapi"，实际改成自研，理由：

1. **部署形态是单容器**（C2 计划里 api 只有一个实例，4C4G 也没有横向扩展
   的余量），slowapi 的默认内存后端与自研效果等价，引入依赖不换来任何能力。
2. **需要 IP + user 双维度组合键**，slowapi 的 `key_func` 只能返回单键，
   要组合还得自己写函数，省下的代码有限。
3. 项目一贯取向是少依赖（自研 agent loop、不引 LangGraph）。

代价：多进程部署时计数不共享。当前不适用；若将来 api 多副本，
把 `_HITS` 换成 Redis `INCR + EXPIRE` 即可，接口不变。

## 语义

- 滑动窗口（记录每次命中时间戳），不是固定窗口——固定窗口在窗口边界
  可以瞬间打满两倍配额。
- 超限抛 `RateLimitExceeded`，由 API 层转 429 并带 `Retry-After`。
- 失败计数与成功计数共用一套（登录/解锁场景防的是"猜密码"，
  成功请求同样算消耗）。
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Deque, Optional

from src.core.logger import logger

# key -> 命中时间戳队列（单调时钟，不受系统时间调整影响）
_HITS: dict[str, Deque[float]] = defaultdict(deque)

# 防止 key 无限增长（每个 IP 一个键，公网扫描器会造出大量键）：
# 单键清空后仍在字典里，定期做一次粗粒度清理。
_LAST_SWEEP = 0.0
_SWEEP_INTERVAL_SEC = 300.0


class RateLimitExceeded(RuntimeError):
    """请求超过配额。"""

    def __init__(self, key: str, limit: int, window_sec: int, retry_after: int) -> None:
        super().__init__(f"请求过于频繁（{limit} 次 / {window_sec}s）")
        self.key = key
        self.limit = limit
        self.window_sec = window_sec
        self.retry_after = retry_after


def _sweep(now: float, window_sec: int) -> None:
    """粗粒度清理空键，避免长期运行后字典膨胀。"""
    global _LAST_SWEEP
    if now - _LAST_SWEEP < _SWEEP_INTERVAL_SEC:
        return
    _LAST_SWEEP = now
    stale = [
        k for k, dq in _HITS.items()
        if not dq or (now - dq[-1]) > max(window_sec, 3600)
    ]
    for k in stale:
        _HITS.pop(k, None)


def check_rate_limit(
    key: str,
    *,
    limit: int,
    window_sec: int,
    cost: int = 1,
) -> None:
    """记录一次命中；超限抛 `RateLimitExceeded`。

    `key` 由调用方组合（如 `f"unlock:{ip}:{user_id}"`）——
    本模块不关心维度含义，只按字符串计数。
    """
    now = time.monotonic()
    _sweep(now, window_sec)

    dq = _HITS[key]
    cutoff = now - window_sec
    while dq and dq[0] <= cutoff:
        dq.popleft()

    if len(dq) + cost > limit:
        # 最早一次命中滑出窗口时即可重试
        retry_after = max(1, int(dq[0] + window_sec - now) + 1) if dq else window_sec
        logger.warning(f"限流触发 key={key} hits={len(dq)} limit={limit}/{window_sec}s")
        raise RateLimitExceeded(key, limit, window_sec, retry_after)

    for _ in range(cost):
        dq.append(now)


def reset_rate_limit(key: Optional[str] = None) -> None:
    """清空计数。`key=None` 清全部（供测试用）。"""
    if key is None:
        _HITS.clear()
    else:
        _HITS.pop(key, None)


def client_ip(request) -> str:
    """取客户端 IP：优先反代头，回退 socket 对端。

    ★ 反代头**默认不信**（``settings.trust_proxy=False``）：单容器直连时
    任何客户端都能自带 `X-Forwarded-For: 1.2.3.4` 伪造来源，限流形同虚设。
    C2 部署到 Nginx 之后，在 `.env` 里设 `TRUST_PROXY=1` 才启用反代头
    （此时 socket 对端是 Nginx 容器 127.0.0.1，不看 XFF 就所有人共用一个
    计数器，会互相误伤）。取 XFF 第一段——那是真实客户端。
    """
    from src.core.config import settings

    if settings.trust_proxy:
        xff = (request.headers.get("x-forwarded-for") or "").strip()
        if xff:
            first = xff.split(",")[0].strip()
            if first:
                return first
        real_ip = (request.headers.get("x-real-ip") or "").strip()
        if real_ip:
            return real_ip
    client = getattr(request, "client", None)
    return getattr(client, "host", "") or "unknown"


__all__ = [
    "RateLimitExceeded",
    "check_rate_limit",
    "reset_rate_limit",
    "client_ip",
]
