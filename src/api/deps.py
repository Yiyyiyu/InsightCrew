"""API 依赖项：JWT 鉴权（Cookie + Bearer 双通道）+ 任务归属校验"""

from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.core.auth import ACCESS_COOKIE, decode_access_token
from src.db.database import async_session
from src.db.models import Task, User

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


async def get_current_user_optional(request: Request) -> Optional[dict]:
    """解析当前用户，未登录返回 None（供页面路由决定是否跳登录页）。

    ★ 会查库比对 `token_version`：改密码时该值 +1，此前签发的所有 token
      立刻失效（否则 JWT 一旦发出就无法撤回，改密对旧会话毫无作用）。
      「令牌无效」与「没带令牌」在这里统一表现为 None，调用方按未登录处理。
    """
    token = _token_from_request(request)
    if not token:
        return None
    payload = decode_access_token(token)
    if payload is None:
        return None
    return await _verify_token_version(payload)


async def _verify_token_version(payload: dict) -> Optional[dict]:
    """比对 JWT 载荷里的 `token_version` 与库中当前值。

    老 token 没有该字段 → 按 0 处理；老库补列也是默认 0，两边一致。
    用户已不存在（被删号）同样返回 None。

    ★ 顺手把 `email` 塞进返回的 dict：这里本来就要查一次库，多取一个字段
      不产生额外往返。页面模板据此判断「要不要提示补填邮箱」，否则每个页面
      都得再查一次用户表。注意 `email` **不在 JWT 里**（JWT 里放邮箱会让
      改邮箱后旧令牌仍带旧值），而是本次查库的实时值 —— 加 `db_` 前缀标明
      它与载荷字段来源不同。

    ★ `db_email_verified` 同样来自本次查库（用户 m26742 引入）。模板用它区分
      「已绑定且已验证」与「填了但没验过」：后者不能用于找回密码，也不该显示
      成正常的已绑定状态。判断口径与 `PUT /profile/email` 里的
      `old_verified` 保持一致 —— **有邮箱且 email_verified_at 非空**，
      两处若不一致会出现「页面说已验证、后端却按未验证处理」。
    """
    user_id = payload.get("user_id")
    if not user_id:
        return None
    async with async_session() as session:
        user = await session.get(User, user_id)
    if user is None:
        return None
    if int(payload.get("token_version", 0) or 0) != int(user.token_version or 0):
        return None
    return {
        **payload,
        "db_email": user.email or "",
        "db_email_verified": bool(user.email and user.email_verified_at),
    }


async def get_current_user(request: Request) -> dict:
    """从 Bearer 头或 Cookie 解析当前用户信息，鉴权失败抛 401"""
    payload = await get_current_user_optional(request)
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
