"""API 依赖项：JWT 鉴权（Cookie + Bearer 双通道）+ 任务归属校验"""

from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.core.auth import ACCESS_COOKIE, decode_access_token
from src.db.database import async_session
from src.db.models import Task

# auto_error=False：没有 Authorization 头时不直接 401，
# 以便继续尝试 Cookie（浏览器页面与 EventSource 走这条路径）。
bearer_scheme = HTTPBearer(auto_error=False)


def _token_from_request(request: Request) -> Optional[str]:
    """按 Bearer 头 → Cookie 的顺序取 token。

    浏览器无法给 `<a href>` 跳转和 `EventSource` 设置请求头，只有 Cookie 会
    自动携带，所以 Cookie 不是可选项而是前端能工作的前提；Bearer 保留给
    API 客户端与 curl 验证。
    """
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
        if token:
            return token
    cookie = request.cookies.get(ACCESS_COOKIE)
    return cookie or None


def get_current_user_optional(request: Request) -> Optional[dict]:
    """解析当前用户，未登录返回 None（供页面路由决定是否跳登录页）。"""
    token = _token_from_request(request)
    if not token:
        return None
    return decode_access_token(token)


def get_current_user(request: Request) -> dict:
    """从 Bearer 头或 Cookie 解析当前用户信息，鉴权失败抛 401"""
    payload = get_current_user_optional(request)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效或过期的 Token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload  # 包含 user_id / username


async def get_owned_task(
    task_id: int,
    current_user: dict = Depends(get_current_user),
) -> dict:
    """校验任务归属：不存在、或不属于当前用户，一律 404。

    两个设计要点：
    1. **不区分"不存在"与"别人的"**：若对越权返回 403，攻击者可用 403/404
       的差异枚举出哪些 task_id 真实存在。统一 404 不泄漏任务存在性。
    2. 归属校验放在依赖里，而不是散落在每个 endpoint：漏写一个 endpoint
       就是一个越权漏洞，而依赖是路由声明的强制项（PRD 要求"越权返回 404"）。
    """
    async with async_session() as session:
        task = await session.get(Task, task_id)

    if task is None or task.user_id != current_user.get("user_id"):
        raise HTTPException(status_code=404, detail="任务不存在")

    return {"task": task, "user": current_user}
