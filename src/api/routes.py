"""API 路由注册"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel
from sqlalchemy import select

from src.api.deps import get_current_user, get_current_user_optional, get_owned_task
from src.core.auth import (
    ACCESS_COOKIE,
    hash_password,
    verify_password,
    create_access_token,
    create_reset_token,
    hash_reset_token,
    parse_reset_token,
)
from src.core.captcha import new_captcha, verify_captcha
from src.core.config import settings
from src.core import emailcode, mailer
from src.core.errors import (
    BuiltinLockedError,
    ModelUnreachableError,
    ServiceUnavailableError,
)
from src.core.logger import logger
from src.core.orchestrator import (
    start_pipeline,
    get_task_status,
    approve_gate,
    reject_gate,
    abort_pipeline,
    pause_pipeline,
    delete_task,
    event_stream,
    _push_event,
)
from src.core.ratelimit import RateLimitExceeded, check_rate_limit, client_ip
from src.db.database import async_session
from src.db.models import PasswordReset, Report, Task, User
from sse_starlette.sse import EventSourceResponse

router = APIRouter()


# ---- 接口限流（C4-1）----
# 上线前必须有：公网暴露后，注册/登录接口可被暴力枚举，建任务接口
# 可被刷爆 LLM 账单。按 IP + user 双维度限制。
_LIMITS: dict[str, tuple[int, int]] = {
    "register": (5, 60),        # 5 次 / 分钟
    "login": (5, 60),           # 5 次 / 分钟
    "research_start": (10, 3600),  # 10 次 / 小时
    "forgot": (5, 300),         # 5 次 / 5 分钟（找回密码申请）
    "reset": (10, 300),         # 10 次 / 5 分钟（提交新密码）
    "password_change": (5, 300),  # 5 次 / 5 分钟（已登录改密，防爆破旧密码）
    # 邮箱验证码（用户 m26742）：发码接口会被用来刷别人邮箱，必须限。
    # 另有 60 秒/邮箱 的冷却（emailcode.cooldown_hit），两层互补：
    # IP 限流挡不住「换 IP 轰炸同一个邮箱」，冷却挡不住「一个 IP 刷很多邮箱」。
    "email_code": (10, 300),    # 10 次 / 5 分钟
    "email_verify": (20, 300),  # 20 次 / 5 分钟（提交验证码）
}


def _rate_limit(request: Request, action: str, user_id: int = 0) -> None:
    """按 IP + user 双维度限流；超限抛 429（带 Retry-After）。

    ``user_id`` 为 0 时只按 IP 限（注册/登录尚未有身份）。
    生产恒开；开发模式由 ``RATE_LIMIT_ENABLED`` 控制（见 ``rate_limit_on``）。
    """
    from src.core.config import rate_limit_on

    if not rate_limit_on():
        return
    limit, window = _LIMITS.get(action, (60, 60))
    key = f"{action}:{client_ip(request)}:{user_id or 0}"
    try:
        check_rate_limit(key, limit=limit, window_sec=window)
    except RateLimitExceeded as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "code": "rate_limited",
                "message": f"操作过于频繁，请 {e.retry_after} 秒后重试",
                "retry_after": e.retry_after,
            },
            headers={"Retry-After": str(e.retry_after)},
        )


# ---- Pydantic 请求/响应模型 ----


class RegisterRequest(BaseModel):
    username: str
    password: str
    # 独立注册页（用户 m25143 第 2 项）：邮箱可选，老客户端不传也不受影响。
    email: str = ""
    # ★ 邮箱验证码（用户 m26742）：填了邮箱就必须带码，且码校验通过才建账号。
    #   不填邮箱时该字段被忽略（老客户端不传也不受影响）。
    code: str = ""


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ResearchRequest(BaseModel):
    topic: str
    # ── 任务级参数（M1 遗留项 5：新建任务不再只能填主题）──
    # 全部可选，留空走原行为，老客户端不受影响。
    dimensions: str = ""          # 评估维度，如「性能, 生态, 学习曲线」
    constraints: str = ""         # 技术约束，如「Python 3.11、不引入商业 API」
    budget_limit: float | None = None    # 本任务预算上限（元）；None/<=0 用全局配置
    # ── 模型分配表（M2.5，用户 m12784 拍板）──
    # {角色: option_id}，角色 = Planner/Researcher/Analyst/Critic/Writer/
    # Reviewer/vision（读图）。option_id 形如 "builtin:deepseek-flash" /
    # "user:7"。留空/缺角色 → 执行期回落内置默认。连通性在落库前 preflight
    # 预检（M2.5-5），不通则 400 且不建任务。
    model_plan: dict[str, str] | None = None


class GateRequest(BaseModel):
    gate_name: str
    comment: str = ""


# ── 找回密码（用户 m26093）──


class ForgotRequest(BaseModel):
    """申请重置：邮箱或用户名 + 图形验证码。"""

    account: str = ""          # 邮箱或用户名，二者都能定位账号
    captcha_id: str = ""
    captcha: str = ""


class ResetRequest(BaseModel):
    """提交新密码：一次性令牌 + 新密码（前端已做二次确认，服务端只收一个）。"""

    token: str = ""
    password: str = ""


# ── 个人信息（用户 m26104）──
class ProfileUpdateRequest(BaseModel):
    """补填/修改邮箱。

    邮箱在注册时可选，未填的用户收不到重置邮件（找回密码只能靠管理员），
    所以要给一个补填入口。只收 email 一个字段：用户名是登录凭据，改名会
    让既有会话与台账的归属表述变复杂，本轮不做。

    ★ 验证码（用户 m26742 拍板「新旧邮箱都要验证」）：
      * 首次绑定（当前无邮箱）→ 只需 ``code``（发往新邮箱）。
      * 更换邮箱（当前已有邮箱）→ 需要 ``code``（新邮箱）+ ``old_code``（旧邮箱）。
        旧邮箱那一枚证明「操作人控制着账号原有邮箱」，防的是「会话被劫持后
        把邮箱换成攻击者的」—— 只有新邮箱验证的话，攻击者拿到会话就能接管
        账号的找回入口。
    """

    email: str = ""
    code: str = ""
    old_code: str = ""


class PasswordChangeRequest(BaseModel):
    """已登录状态下的改密：必须验旧密码 **且** 验证已绑定邮箱。

    ★ 与「找回密码」的区别：那条路用户已经证明了对邮箱的控制权（拿到令牌），
      这条路用户只有会话，必须再验一次旧密码 —— 否则会话被劫持就等于账号被夺。

    ★ 用户 m26742 追加：还要一枚发往**已绑定且已验证**邮箱的验证码。
      于是改密需要三样东西同时成立：会话语权 + 知道旧密码 + 控制绑定邮箱。
      未绑定邮箱的账号必须先完成绑定与验证才能改密（用户明确要求「现有未绑定
      的账号也得先验证邮箱才能改密码」）。
    """

    old_password: str = ""
    new_password: str = ""
    code: str = ""


# ── 中间产物编辑（用户 m18267：web 端要能改内容）──
# 键 = `_load_intermediate` 的 key，也是前端 `<pre id="ic-p-*">` 的 data 来源。
# 值与阶段一一对应，前端据此判断「这一段属于哪个阶段」。
ARTIFACT_KEYS = {
    "plan_output": "plan",
    "analysis_output": "analysis",
    "critic_output": "critic",
    "rebuttal_output": "rebuttal",
    "report_draft": "write",
    "review_output": "review",
}


class ArtifactRequest(BaseModel):
    value: str
    comment: str = ""


class TaskStatusResponse(BaseModel):
    task_id: int
    topic: str
    status: str
    current_stage: str
    gate_status: str
    pending_gate: str | None = None
    plan_output: str | None = None
    analysis_output: str | None = None
    critic_output: str | None = None
    rebuttal_output: str | None = None
    report_draft: str | None = None
    review_output: str | None = None
    exec_path: str = "unknown"
    created_at: str | None = None
    total_cost_cny: float = 0.0
    budget_limit_cny: float = 0.0
    dimensions: str = ""
    constraints: str = ""
    model_plan: dict[str, str] = {}
    pause_requested: bool = False


class TaskSummary(BaseModel):
    task_id: int
    topic: str
    status: str
    current_stage: str
    gate_status: str
    total_cost_cny: float = 0.0
    created_at: str | None = None
    updated_at: str | None = None


class ReportResponse(BaseModel):
    task_id: int
    topic: str
    report_id: int
    content_md: str
    bibtex: str | None = None
    created_at: str | None = None


# ---- 鉴权路由 ----


@router.post("/auth/register", summary="用户注册")
async def register(req: RegisterRequest, request: Request):
    """注册。填了邮箱就必须带上验证码，**码校验通过才建账号**（用户 m26742 拍板）。

    ★ 校验顺序：格式 → 用户名唯一 → 邮箱唯一 → 验证码。
      验证码放在最后是因为前面几步都可能失败，而验证码一旦校验通过就会
      被消费掉 —— 让用户「明明填对了码，却因为用户名被别人抢了要重新收信」
      是没必要的（`verify_code(consume=False)` 正是为此）。

    ★ 为什么这里也用 `consume=False`：真正作废的时机是**账号落库成功之后**。
      若校验通过后进程崩在 commit 前，用户既没建成账号、码也没了。
    """
    _rate_limit(request, "register")
    if len(req.username.strip()) < 2 or len(req.password.strip()) < 4:
        raise HTTPException(status_code=400, detail="用户名至少2位，密码至少4位")
    # 邮箱可选：填了才校验格式与唯一性。格式用最宽松的「有 @ 且两侧非空」，
    # 严格 RFC 校验既复杂又拦不住真正写错的地址（只有发信才知道），
    # 前端还做了同样的即时提示，这里兜住绕过前端的调用。
    email = (req.email or "").strip().lower()
    if email:
        local, _, domain = email.partition("@")
        if not local or not domain or "." not in domain or " " in email:
            raise HTTPException(status_code=400, detail="邮箱格式不正确")
        if not (req.code or "").strip():
            raise HTTPException(status_code=400, detail={"code": "code_required",
                                                          "message": "请先获取邮箱验证码"})
    async with async_session() as session:
        result = await session.execute(select(User).where(User.username == req.username))
        if result.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="用户名已存在")
        if email:
            dup = await session.execute(select(User).where(User.email == email))
            if dup.scalar_one_or_none():
                raise HTTPException(status_code=409, detail="邮箱已被注册")

    if email:
        ok, _reason = await emailcode.verify_code(email, "register", req.code, consume=False)
        if not ok:
            # 不区分「码错」与「码过期」—— 区分会让攻击者知道猜测落在哪个阶段
            raise HTTPException(status_code=400, detail={"code": "code_invalid",
                                                          "message": "验证码错误或已过期，请重新获取"})

    async with async_session() as session:
        user = User(
            username=req.username,
            email=email or None,
            hashed_password=hash_password(req.password),
            # 邮箱验证通过才落这一列；不填邮箱则为 NULL（未验证状态）
            email_verified_at=datetime.utcnow() if email else None,
        )
        session.add(user)
        await session.commit()

    if email:
        # 账号已落库，此时才真正作废验证码
        await emailcode.consume_code(email, "register")
    return {"msg": "注册成功"}


@router.post("/auth/login", summary="用户登录", response_model=LoginResponse)
async def login(req: LoginRequest, response: Response, request: Request):
    _rate_limit(request, "login")
    async with async_session() as session:
        result = await session.execute(select(User).where(User.username == req.username))
        user = result.scalar_one_or_none()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_access_token(
        user_id=user.id, username=user.username, token_version=user.token_version or 0
    )

    # 同时下发 HttpOnly Cookie：浏览器页面跳转与 EventSource 无法设置
    # Authorization 头，只能靠 Cookie 自动携带。httponly 防 XSS 窃取；
    # secure 在本地 http 下必须关掉，否则 Cookie 根本不会被写入。
    response.set_cookie(
        key=ACCESS_COOKIE,
        value=token,
        max_age=settings.jwt_access_ttl_min * 60,
        httponly=True,
        samesite="lax",
        secure=False,
    )
    return LoginResponse(access_token=token)


@router.post("/auth/logout", summary="退出登录")
async def logout(response: Response):
    """清除登录 Cookie。前端页面靠这个退出。"""
    response.delete_cookie(ACCESS_COOKIE)
    return {"msg": "已退出登录"}


# ---- 找回密码（用户 m26093）----

#: 无论账号是否存在，都回这一句 —— 防枚举（否则接口变成「这个邮箱注册了吗」的查询器）
#: ★ 有效期从配置读，别写死：改 TTL 时文案会跟着变，否则用户看到的时间与实际不符。
_FORGOT_UNIFIED_MSG = (
    "如果该账号已绑定邮箱，我们已发送重置链接，请查收"
    f"（{settings.password_reset_ttl_min} 分钟内有效）"
)


async def _mail_cooldown_hit(account_key: str) -> bool:
    """同一邮箱 60 秒内是否已发过。

    与 IP 限流互补：IP 限流挡不住「换 IP 轰炸同一个邮箱」（把别人邮箱刷爆
    会让对方收到一堆邮件）。Redis 不可用时放行 —— 宁可少一层保护，也不要
    因为缓存故障把找回密码整体打死。
    """
    import hashlib

    from src.core.events import _get_redis  # 复用命令连接池

    r = await _get_redis()
    if r is None:
        return False
    key = "mailcooldown:" + hashlib.sha256(account_key.encode("utf-8")).hexdigest()[:32]
    try:
        # SET NX EX：只有键不存在时才写成功 → 写成功=本次是这一分钟内的第一封
        ok = await r.set(key, "1", ex=60, nx=True)
        return not bool(ok)
    except Exception:
        return False


@router.get("/auth/captcha", summary="获取图形验证码")
async def captcha():
    """返回 ``{captcha_id, image}``；`image` 是可直接塞进 `<img src>` 的 data URL。"""
    cid, data_url = await new_captcha()
    return {"captcha_id": cid, "image": data_url}


# ---- 邮箱验证码（用户 m26742）----

#: 发码时「收件邮箱从哪来」。这张表是本模块的安全核心，不是配置：
#:
#:   * ``register`` / ``bind`` —— 邮箱由请求体指定。这两个用途的本质就是
#:     「证明某个我还没拥有的邮箱归我」，收件人必须是待证明的那个地址。
#:   * ``verify`` / ``password`` —— 邮箱**一律取库里已绑定且已验证的那个**，
#:     请求体里的 email 被忽略。它们是「证明操作人控制着账号原有邮箱」，
#:     若允许调用方指定收件人，攻击者拿到会话后就能把码发到自己的邮箱，
#:     再用它去授权换绑/改密 —— 验证环节形同虚设。
#:     ★ 这是一处**必须写死在服务端**的判断，前端传什么都不影响。
_CODE_TO_FROM_BODY = ("register", "bind")


class EmailCodeRequest(BaseModel):
    """申请邮箱验证码。"""

    email: str = ""
    purpose: str = ""


@router.post("/auth/email-code", summary="发送邮箱验证码")
async def send_email_code_api(
    req: EmailCodeRequest,
    request: Request,
    current_user: dict | None = Depends(get_current_user_optional),
):
    """给指定邮箱发一枚验证码。

    ★ 四条用途的鉴权不同（见 `_CODE_TO_FROM_BODY` 上方说明）：
      ``register`` / ``bind`` 允许匿名（注册时还没有账号），
      ``verify`` / ``password`` 必须已登录，且收件地址只能是自己绑定的邮箱。

    ★ 返回体**不区分**「邮箱已注册」以外的细节，但注册用途必须明确告知冲突 ——
      否则用户收不到信也不知道为什么（`/auth/register` 本来也会回 409，
      这里提前一步告知不新增泄漏面）。
    """
    purpose = (req.purpose or "").strip()
    if purpose not in emailcode.PURPOSES:
        raise HTTPException(status_code=400, detail="验证码用途不合法")

    need_auth = purpose not in _CODE_TO_FROM_BODY
    if need_auth and not current_user:
        raise HTTPException(
            status_code=401,
            detail="登录后才能执行该操作",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id = (current_user or {}).get("user_id")
    _rate_limit(request, "email_code", user_id or 0)

    # ── 决定收件地址 ──
    if purpose in _CODE_TO_FROM_BODY:
        email = _normalize_email(req.email)
        if not email:
            raise HTTPException(status_code=400, detail="请填写邮箱")
    else:
        # 收件地址只能来自库：忽略请求体里的 email，杜绝「把码发到攻击者邮箱」
        async with async_session() as session:
            user = await session.get(User, user_id)
        if not user or not user.email:
            raise HTTPException(status_code=400, detail={"code": "no_email",
                                                          "message": "请先绑定邮箱后再操作"})
        if not user.email_verified_at:
            raise HTTPException(status_code=400, detail={"code": "email_unverified",
                                                          "message": "当前邮箱尚未验证，请先完成验证"})
        email = user.email

    # ── 业务前置校验（不校验的话用户会白等一封永远用不上的信）──
    async with async_session() as session:
        if purpose == "register":
            dup = (await session.execute(select(User).where(User.email == email))).scalar_one_or_none()
            if dup:
                raise HTTPException(status_code=409, detail="邮箱已被注册")
        elif purpose == "bind":
            dup = (await session.execute(
                select(User).where(User.email == email, User.id != user_id)
            )).scalar_one_or_none()
            if dup:
                raise HTTPException(status_code=409, detail="邮箱已被其它账号使用")
        if user_id:
            user = await session.get(User, user_id)
        else:
            user = None

    # ── 冷却：同一 (邮箱, 用途) 60 秒内只发一封 ──
    if await emailcode.cooldown_hit(email, purpose):
        raise HTTPException(status_code=429, detail={
            "code": "cooldown",
            "message": f"验证码已发送，请 {settings.email_code_cooldown_sec} 秒后再试",
        })

    code = await emailcode.issue_code(email, purpose, user_id=user_id)
    if code is None:
        raise HTTPException(status_code=400, detail="验证码用途不合法")

    sent = await mailer.send_email_code(
        to=email, code=code, ttl_min=settings.email_code_ttl_min,
        purpose=purpose, username=(user.username if user else ""),
    )
    if not sent:
        # 与找回密码同样的取舍：不把「发信失败」暴露成 5xx（那会让用户以为
        # 请求本身没成功而反复重试，反而把冷却撞满），但日志里必须留痕。
        # 本机没配 RESEND_API_KEY 时降级会把验证码打进日志，联调仍可走通。
        logger.warning(f"[emailcode] 验证码邮件未真实发出 → {email}（purpose={purpose}）")

    return {
        "msg": "验证码已发送，请查收邮件",
        "email": _mask_email(email),
        "ttl_min": settings.email_code_ttl_min,
        "cooldown_sec": settings.email_code_cooldown_sec,
    }


def _mask_email(email: str) -> str:
    """回显时打码：``ab***@example.com``。

    前端要显示「已发送到 xxx」让用户确认没写错，但完整地址会出现在
    浏览器历史、截图、以及共享屏幕里 —— 打码后仍够用户判断是不是自己那个。
    """
    local, at, domain = (email or "").partition("@")
    if not at:
        return email or ""
    keep = local[:2] if len(local) > 2 else local[:1]
    return f"{keep}{'*' * max(1, len(local) - len(keep))}@{domain}"



@router.post("/auth/forgot", summary="申请重置密码")
async def forgot(req: ForgotRequest, request: Request):
    """发重置邮件。

    ★ 顺序很关键：**先校验验证码，再做账号判定**。
      验证码错必须真实报错（否则用户不知道要刷新验证码，只会反复点提交）；
      而账号是否存在**绝不能**从响应里看出来。两者顺序颠倒就会泄漏。
    """
    _rate_limit(request, "forgot")

    if not await verify_captcha(req.captcha_id, req.captcha):
        raise HTTPException(status_code=400, detail={"code": "captcha_invalid",
                                                     "message": "验证码错误或已过期，请重新获取"})

    account = (req.account or "").strip()
    if not account:
        raise HTTPException(status_code=400, detail="请填写邮箱或用户名")

    async with async_session() as session:
        # 邮箱不区分大小写（注册时已 lower）；用户名保持原样匹配
        stmt = select(User)
        if "@" in account:
            user = (await session.execute(
                stmt.where(User.email == account.lower())
            )).scalar_one_or_none()
        else:
            user = (await session.execute(
                stmt.where(User.username == account)
            )).scalar_one_or_none()

        # 找不到、或找到了但没绑邮箱 → 都不发信，但回同一句话
        if not user or not user.email:
            logger.info(f"[forgot] 账号未命中或无邮箱：{account!r}")
            return {"msg": _FORGOT_UNIFIED_MSG}

        if await _mail_cooldown_hit(user.email):
            # 冷却期内不再发信，但响应与成功完全一致
            logger.info(f"[forgot] 邮件冷却中，跳过发送：{user.email}")
            return {"msg": _FORGOT_UNIFIED_MSG}

        token, expires = create_reset_token(user.id)
        session.add(PasswordReset(
            user_id=user.id,
            token_hash=hash_reset_token(token),
            expires_at=expires,
        ))
        await session.commit()
        email, username, uid = user.email, user.username, user.id

    reset_url = f"{settings.app_base_url.rstrip('/')}/reset?token={token}"
    sent = await mailer.send_password_reset(
        to=email,
        reset_url=reset_url,
        ttl_min=settings.password_reset_ttl_min,
        username=username,
    )
    if not sent:
        # 不把失败暴露给调用方（会泄漏账号存在），但日志里要留痕
        logger.warning(f"[forgot] 邮件未真实发出（user={uid}），链接已落库，仍可手工取用")
    return {"msg": _FORGOT_UNIFIED_MSG}


@router.get("/auth/reset/verify", summary="校验重置令牌是否可用")
async def reset_verify(token: str = ""):
    """重置页加载时先问一句令牌还有效吗，无效就直接显示「链接已失效」。"""
    uid = parse_reset_token(token)
    if not uid:
        return {"valid": False, "reason": "expired"}
    async with async_session() as session:
        row = (await session.execute(
            select(PasswordReset).where(PasswordReset.token_hash == hash_reset_token(token))
        )).scalar_one_or_none()
    if not row:
        return {"valid": False, "reason": "not_found"}
    if row.used_at is not None:
        return {"valid": False, "reason": "used"}
    return {"valid": True, "reason": ""}


@router.post("/auth/reset", summary="用重置令牌设置新密码")
async def reset_password(req: ResetRequest, request: Request):
    """改密 + 作废所有旧会话。"""
    _rate_limit(request, "reset")

    if len((req.password or "").strip()) < 8:
        raise HTTPException(status_code=400, detail="密码至少 8 位")

    uid = parse_reset_token(req.token)
    if not uid:
        raise HTTPException(status_code=400, detail={"code": "reset_invalid",
                                                     "message": "重置链接无效或已过期，请重新申请"})

    digest = hash_reset_token(req.token)
    async with async_session() as session:
        row = (await session.execute(
            select(PasswordReset).where(PasswordReset.token_hash == digest)
        )).scalar_one_or_none()
        if not row or row.used_at is not None:
            raise HTTPException(status_code=400, detail={"code": "reset_invalid",
                                                         "message": "重置链接已失效，请重新申请"})
        user = await session.get(User, row.user_id)
        if not user:
            raise HTTPException(status_code=400, detail={"code": "reset_invalid",
                                                         "message": "重置链接已失效，请重新申请"})

        user.hashed_password = hash_password(req.password)
        # ★ 版本号 +1：此前签发的所有 JWT 立刻失效（改密后旧会话必须踢下线）
        user.token_version = int(user.token_version or 0) + 1
        row.used_at = datetime.utcnow()
        # 同一用户的其它未用令牌一并作废：用户可能点了几次「发送重置邮件」
        for other in (await session.execute(
            select(PasswordReset).where(
                PasswordReset.user_id == user.id,
                PasswordReset.used_at.is_(None),
            )
        )).scalars().all():
            other.used_at = datetime.utcnow()
        await session.commit()
        logger.info(f"[reset] 用户 {user.username}(id={user.id}) 已重置密码，token_version → {user.token_version}")

    return {"msg": "密码已重置，请用新密码登录"}


# ---- 个人信息（用户 m26104）----


def _normalize_email(raw: str) -> str:
    """邮箱归一化 + 格式校验，返回空串表示未填。

    与注册接口同一口径：最宽松的「有 @ 且两侧非空、域名含点、无空格」。
    严格 RFC 校验既复杂又拦不住真正写错的地址（只有发信才知道）。
    """
    email = (raw or "").strip().lower()
    if not email:
        return ""
    local, _, domain = email.partition("@")
    if not local or not domain or "." not in domain or " " in email:
        raise HTTPException(status_code=400, detail="邮箱格式不正确")
    return email


@router.get("/profile/info", summary="获取当前用户信息")
async def get_profile(current_user: dict = Depends(get_current_user)):
    """返回当前登录用户的基本信息。

    ★ 路径是 `/profile/info` 而不是 `/profile`：页面路由 `GET /profile`
      由 `src/web/routes.py` 提供，而 API router 在 `src/main.py` 里**先于**
      web router 注册，同路径同方法会命中先注册的那个 —— 页面会被 API 的
      JSON 顶掉（实测返回 94 字节 JSON）。加一段路径即可共存。

    ★ 不返回 `hashed_password`。这里刻意用一个显式的字段白名单而不是
      `user.__dict__` —— 后者在给模型加字段时会把新字段（比如未来的手机号、
      内部标记）无声地泄漏出去，而且没人会注意到。
    """
    async with async_session() as session:
        user = await session.get(User, current_user["user_id"])
        if not user:
            raise HTTPException(status_code=404, detail="用户不存在")
        return {
            "user_id": user.id,
            "username": user.username,
            "email": user.email or "",
            # 前端据这两个字段决定：显示「已验证」徽标、还是显示「去验证」按钮，
            # 以及改密区是「可提交」还是「先绑定邮箱」。
            "email_verified": bool(user.email and user.email_verified_at),
            "has_email": bool(user.email),
            "created_at": user.created_at.isoformat() if user.created_at else None,
        }


@router.put("/profile/email", summary="绑定或更换邮箱（需邮箱验证码）")
async def update_email(
    req: ProfileUpdateRequest,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """设置/更换邮箱。**必须凭验证码**（用户 m26742）。

    流程按「当前有没有已验证邮箱」分两条：

      * **首次绑定**（库中无邮箱，或有邮箱但从未验证过）：
        只验 ``code``（发往新邮箱）。没有旧邮箱可发，硬要旧码就是死锁。
      * **更换**（已有**已验证**邮箱）：``old_code`` 验旧邮箱 + ``code`` 验新邮箱。

    ★ 为什么「有邮箱但没验证过」按首次绑定处理：那种状态下旧邮箱的价值为零
      ——它既没被证明归用户所有，也不是找回密码的可用凭据。要求用一枚发往
      未验证邮箱的码来授权，只会把用户锁死，不增加任何安全性。

    ★ 清空邮箱（email 传空）**不需要**验证码：清空是「放弃找回能力」，
      不产生新的攻击面；而绑定一个攻击者邮箱才是要防的。若清空也要码，
      用户丢邮箱时反而无法自救。
    """
    _rate_limit(request, "email_code", current_user["user_id"])
    email = _normalize_email(req.email)

    async with async_session() as session:
        user = await session.get(User, current_user["user_id"])
        if not user:
            raise HTTPException(status_code=404, detail="用户不存在")
        old_email = user.email or ""
        old_verified = bool(old_email and user.email_verified_at)
        if email:
            dup = (await session.execute(
                select(User).where(User.email == email, User.id != user.id)
            )).scalar_one_or_none()
            if dup:
                raise HTTPException(status_code=409, detail="邮箱已被注册")
        uid = user.id

    if not email:
        # 清空：连带清掉验证时间，否则「无邮箱但已验证」是个自相矛盾的状态
        async with async_session() as session:
            user = await session.get(User, uid)
            user.email = None
            user.email_verified_at = None
            await session.commit()
            logger.info(f"[profile] 用户 {user.username}(id={uid}) 清空邮箱")
        return {"msg": "邮箱已清空", "email": "", "email_verified": False}

    if email == old_email:
        raise HTTPException(status_code=400, detail="新邮箱与当前邮箱相同")

    if old_verified:
        # 更换：先验旧邮箱（证明是本人），再验新邮箱（证明新地址归本人）
        if not (req.old_code or "").strip():
            raise HTTPException(status_code=400, detail={"code": "old_code_required",
                                                          "message": "请先获取当前邮箱的验证码"})
        ok_old, _ = await emailcode.verify_code(old_email, "verify", req.old_code, consume=False)
        if not ok_old:
            raise HTTPException(status_code=400, detail={"code": "old_code_invalid",
                                                          "message": "当前邮箱验证码错误或已过期"})

    if not (req.code or "").strip():
        raise HTTPException(status_code=400, detail={"code": "code_required",
                                                      "message": "请先获取新邮箱的验证码"})
    # 首次绑定与更换的新邮箱都走 bind 用途（「证明这个我还没拥有的地址归我」）
    ok_new, _ = await emailcode.verify_code(email, "bind", req.code, consume=False)
    if not ok_new:
        raise HTTPException(status_code=400, detail={"code": "code_invalid",
                                                      "message": "新邮箱验证码错误或已过期"})

    async with async_session() as session:
        user = await session.get(User, uid)
        if not user:
            raise HTTPException(status_code=404, detail="用户不存在")
        # ★ 落库前再查一次唯一性：上面那次查库到这次 commit 之间可能被别人抢注，
        #   而 user.email 是 unique 列，撞上会抛 IntegrityError（变成 500）。
        #   这里显式判一次，把 500 变成明确的 409。
        dup = (await session.execute(
            select(User).where(User.email == email, User.id != uid)
        )).scalar_one_or_none()
        if dup:
            raise HTTPException(status_code=409, detail="邮箱已被注册")
        user.email = email
        user.email_verified_at = datetime.utcnow()
        await session.commit()
        username = user.username

    # 账号已落库，此时才真正作废两枚码
    await emailcode.consume_code(email, "bind")
    if old_verified:
        await emailcode.consume_code(old_email, "verify")
    logger.info(f"[profile] 用户 {username}(id={uid}) 邮箱 → {email}"
                f"（{'更换' if old_verified else '首次绑定'}，已验证）")
    return {"msg": "邮箱已保存并验证", "email": email, "email_verified": True}


@router.post("/profile/password", summary="修改密码（需验旧密码 + 邮箱验证码）")
async def change_password(
    req: PasswordChangeRequest,
    request: Request,
    response: Response,
    current_user: dict = Depends(get_current_user),
):
    """已登录状态改密：验邮箱验证码 → 验旧密码 → 写新密码 → 作废其它会话 → 重签当前会话。

    ★ 三要素同时成立才放行（用户 m26742）：**会话语权 + 知道旧密码 + 控制绑定邮箱**。
      只有前两者时，会话被劫持（XSS / 公共电脑没退出 / token 泄漏）就等于账号被夺
      —— 攻击者本来就不需要旧密码如果他能从别处读到的话；多一道邮箱验证码，
      把「拿到会话」与「拿到邮箱」两个条件拆开。

    ★ 未绑定/未验证邮箱一律拒（400 `email_unverified`），不做「跳过验证」的兜底：
      用户明确要求「现有未绑定的账号也得先验证邮箱才能改密码」。给兜底就等于
      给攻击者留了一条绕开邮箱的路，那道验证也就白加了。

    ★ 为什么改完还要重签 token：改密会让 `token_version` +1，所有旧 JWT
      （包括当前这一个）立刻失效。若不重签，用户改完密码下一次点击就被踢到
      登录页，看起来像「改密把我登出了」的 bug。这里顺手把当前会话续上。
    """
    _rate_limit(request, "password_change", current_user["user_id"])
    if len((req.new_password or "").strip()) < 8:
        raise HTTPException(status_code=400, detail="新密码至少 8 位")

    async with async_session() as session:
        user = await session.get(User, current_user["user_id"])
        if not user:
            raise HTTPException(status_code=404, detail="用户不存在")
        bound_email = user.email or ""
        email_verified = bool(bound_email and user.email_verified_at)
        username = user.username

    if not email_verified:
        raise HTTPException(status_code=400, detail={
            "code": "email_unverified",
            "message": "修改密码前请先绑定并验证邮箱",
        })

    # 邮箱验证码：先验后改。失败一律回同一句话，不区分「码错」与「码过期」。
    if not (req.code or "").strip():
        raise HTTPException(status_code=400, detail={"code": "code_required",
                                                      "message": "请先获取并填写邮箱验证码"})
    ok, _reason = await emailcode.verify_code(bound_email, "password", req.code, consume=False)
    if not ok:
        raise HTTPException(status_code=400, detail={"code": "code_invalid",
                                                      "message": "邮箱验证码错误或已过期"})

    async with async_session() as session:
        user = await session.get(User, current_user["user_id"])
        if not user:
            raise HTTPException(status_code=404, detail="用户不存在")
        if not verify_password(req.old_password or "", user.hashed_password):
            raise HTTPException(status_code=400, detail="原密码不正确")
        if req.new_password == req.old_password:
            raise HTTPException(status_code=400, detail="新密码不能与原密码相同")

        user.hashed_password = hash_password(req.new_password)
        user.token_version = int(user.token_version or 0) + 1
        await session.commit()
        new_version = user.token_version
        logger.info(f"[profile] 用户 {username}(id={user.id}) 已改密，"
                    f"token_version → {new_version}")

    # 密码已落库，此时才作废验证码（与注册同一取舍：业务成功才消费）
    await emailcode.consume_code(bound_email, "password")

    token = create_access_token(
        user_id=current_user["user_id"], username=username, token_version=new_version
    )
    response.set_cookie(
        key=ACCESS_COOKIE,
        value=token,
        max_age=settings.jwt_access_ttl_min * 60,
        httponly=True,
        samesite="lax",
        secure=False,
    )
    return {"msg": "密码已修改", "access_token": token}


# ---- 调研任务路由（需登录）----


@router.post("/research/start", status_code=202, summary="启动调研任务")
async def start_research(
    req: ResearchRequest,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """提交调研任务。

    返回 202 Accepted：任务已受理并投递给执行器，但尚未开始执行
    （执行是异步的，进度通过 SSE 推送）。

    ## 落库前的三道门（M2.5-5）

    1. **解析计划**：把 ``model_plan`` 里每个角色指向的 ``option_id``
       解析成具体模型。缺失/指向不存在的 id 回落该用户目录里的第一个模型。
    2. **预检连通性**：并发探测本次涉及的全部模型（GET /models，失败再发
       `max_tokens=1` 的 ping）。任一**必需档**不通 → 400
       `{"code":"model_unreachable","failures":[...]}`。
    3. **内置解锁门禁**：计划里引用了 ``builtin:*`` 但当前用户没解锁 →
       403；一个可用模型都没有 → 403。

    ★ 三道门全过才建 Task 行 —— 与 Phase A「不留僵尸任务行」同一原则：
      失败必须发生在产生副作用之前。

    ## 其他状态码

    - 503：Redis / worker / 队列池任一不可用（我们这边的服务故障，
      不是用户配置问题）。这条流水线只有一条正常路径（arq 入队 →
      worker 执行），依赖不在就是服务故障，必须让调用方立刻看见。

    `dimensions` / `constraints` / `budget_limit` / `model_plan` 可选，
    会随任务落库并在 worker 执行各阶段时生效（M1 遗留项 5 / M2.5）。
    """
    uid = current_user["user_id"]
    _rate_limit(request, "research_start", uid)

    # ── 门 1+2+3：解锁校验 → 解析计划 → 连通性预检 ──
    # 必须在 start_pipeline 之前完成，因为它会建 Task 行（产生副作用）。
    try:
        resolved_plan = await _preflight_models(req.model_plan, uid)
    except BuiltinLockedError as e:
        logger.info(f"用户 {current_user['username']} 预检被拒: {e}")
        raise HTTPException(status_code=403, detail=e.as_dict())
    except ModelUnreachableError as e:
        logger.warning(f"用户 {current_user['username']} 模型预检失败: {e.failures}")
        raise HTTPException(status_code=400, detail=e.as_dict())

    try:
        task_id = await start_pipeline(
            req.topic,
            user_id=uid,
            dimensions=req.dimensions,
            constraints=req.constraints,
            budget_limit=req.budget_limit,
            model_plan=resolved_plan,
        )
    except ServiceUnavailableError as e:
        logger.error(f"用户 {current_user['username']} 启动任务失败: {e}")
        raise HTTPException(status_code=503, detail=e.as_dict())
    logger.info(f"用户 {current_user['username']} 启动了任务 {task_id}")
    return {"task_id": task_id, "status": "started"}


async def _preflight_models(requested: dict | None, user_id: int) -> dict:
    """三道门：解锁校验 → 解析计划 → 连通性预检。返回补全后的 plan。

    抛 ``BuiltinLockedError``（→403）或 ``ModelUnreachableError``（→400）。

    ★ 顺序有讲究：**解锁校验必须在 ``build_plan_for_user`` 之前**。
    后者会把「指向不存在模型的角色」回落到目录里的第一个模型，而
    「引用了 ``builtin:*`` 但目录里没有」正是未解锁的判定依据 —— 先补全
    就等于把证据抹掉了，403 永远发不出去。
    """
    from src.core.errors import BuiltinLockedError, ModelUnreachableError
    from src.core.model_registry import PLAN_ROLES, catalog_async, find_option
    from src.core.preflight import (
        all_required_ok,
        build_plan_for_user,
        format_failures,
        preflight,
    )

    async with async_session() as session:
        options = await catalog_async(session, user_id)

    # ── 门 3：目录为空 = 既没解锁内置、也没有任何自建模型 ──
    if not options:
        raise BuiltinLockedError(list(PLAN_ROLES), reason="no_model")

    # ── 门 3：用户显式引用了内置模型，但他没解锁 ──
    # 注意只看 requested（用户原样提交的），不能看补全后的 plan。
    locked_roles = [
        role
        for role, oid in (requested or {}).items()
        if str(oid).startswith("builtin:") and find_option(options, str(oid)) is None
    ]
    if locked_roles:
        raise BuiltinLockedError(locked_roles, reason="locked")

    # ── 门 1：把计划补全成 7 个角色都有值 ──
    plan = build_plan_for_user(requested, options)

    # ── 门 2：连通性预检（并发探 GET /models，失败再 ping）──
    results = await preflight(plan, options)
    if not all_required_ok(results):
        raise ModelUnreachableError(format_failures(results))

    return plan


@router.get("/research/tasks", summary="任务列表（当前用户）", response_model=list[TaskSummary])
async def list_tasks(current_user: dict = Depends(get_current_user)):
    """列出当前用户的任务。

    必须按 user_id 过滤：这是用户隔离在"列表"这一层的体现，
    漏掉就等于把所有人的任务都列出来了。
    """
    async with async_session() as session:
        rows = (await session.execute(
            select(Task)
            .where(Task.user_id == current_user["user_id"])
            .order_by(Task.id.desc())
        )).scalars().all()
    return [
        TaskSummary(
            task_id=t.id,
            topic=t.topic,
            status=t.status,
            current_stage=t.current_stage,
            gate_status=t.gate_status,
            total_cost_cny=float(t.total_cost_cny or 0.0),
            created_at=t.created_at.isoformat() if t.created_at else None,
            updated_at=t.updated_at.isoformat() if t.updated_at else None,
        )
        for t in rows
    ]


@router.get("/research/{task_id}/status", response_model=TaskStatusResponse)
async def research_status(
    ctx: dict = Depends(get_owned_task),
):
    status = await get_task_status(ctx["task"].id)
    if not status:
        raise HTTPException(status_code=404, detail="任务不存在")
    return status


@router.get("/research/{task_id}/report", summary="获取最终报告", response_model=ReportResponse)
async def get_report(ctx: dict = Depends(get_owned_task)):
    """返回最终报告（Markdown + BibTeX）。

    任务未完成时返回 409 而不是 404：任务确实存在、只是还没有报告，
    两者语义不同，前端需要区分（前者该提示"还没跑完"，后者该提示"不存在"）。
    """
    task = ctx["task"]
    async with async_session() as session:
        report = (await session.execute(
            select(Report).where(Report.task_id == task.id).order_by(Report.id.desc()).limit(1)
        )).scalar_one_or_none()

    if report is None:
        raise HTTPException(status_code=409, detail="任务尚未生成报告（可能仍在运行或已失败）")

    return ReportResponse(
        task_id=task.id,
        topic=task.topic,
        report_id=report.id,
        content_md=report.content_md or "",
        bibtex=report.bibtex,
        created_at=report.created_at.isoformat() if report.created_at else None,
    )


@router.post("/research/{task_id}/approve", summary="通过审核闸门")
async def approve(
    req: GateRequest,
    ctx: dict = Depends(get_owned_task),
):
    task_id = ctx["task"].id
    try:
        ok = await approve_gate(task_id, req.gate_name, req.comment)
    except ServiceUnavailableError as e:
        # 闸门本身已由 _reopen_gate 退回 pending，任务状态没被破坏，
        # 用户可在执行器恢复后重新点一次。
        raise HTTPException(status_code=503, detail=e.as_dict())
    if not ok:
        raise HTTPException(status_code=400, detail="无法通过闸门")
    logger.info(f"用户 {ctx['user']['username']} 通过了闸门 {req.gate_name}")
    return {"status": "approved"}


@router.post("/research/{task_id}/reject", summary="驳回审核闸门")
async def reject(
    req: GateRequest,
    ctx: dict = Depends(get_owned_task),
):
    task_id = ctx["task"].id
    try:
        ok = await reject_gate(task_id, req.gate_name, req.comment)
    except ServiceUnavailableError as e:
        raise HTTPException(status_code=503, detail=e.as_dict())
    if not ok:
        raise HTTPException(status_code=400, detail="无法驳回闸门")
    logger.info(f"用户 {ctx['user']['username']} 驳回了闸门 {req.gate_name}，备注: {req.comment}")
    return {"status": "rejected"}


@router.put("/research/{task_id}/artifact/{key}", summary="编辑中间产物")
async def edit_artifact(
    key: str,
    req: ArtifactRequest,
    ctx: dict = Depends(get_owned_task),
):
    """人工修正某个中间产物（用户 m18267：前端要能改内容，不只是看）。

    为什么只能在闸门处改：闸门之外阶段正在跑，此时改产物要么被下一步
    覆盖、要么让下游拿到自相矛盾的前后文。闸门处是**唯一的静默点** ——
    任务停着、下一步还没入队，改完 approve 就直接生效。

    实现：写一条新的 `intermediate_<key>` 事件（`_save_intermediate` 本就
    取 `id.desc()` 最新一条），因此**不改历史、可追溯**：旧值仍在事件流里。
    """
    task = ctx["task"]
    task_id = task.id

    if key not in ARTIFACT_KEYS:
        raise HTTPException(status_code=400, detail=f"未知产物：{key}")
    if task.status != "running" or task.gate_status != "pending":
        raise HTTPException(
            status_code=409,
            detail="只能在闸门等待确认时编辑中间产物（阶段运行中编辑会被覆盖）",
        )

    from src.core.artifacts import _load_intermediate, _save_intermediate

    old = await _load_intermediate(task_id, key)
    if old is None:
        raise HTTPException(status_code=404, detail=f"产物 {key} 尚未生成")

    await _save_intermediate(task_id, key, req.value)
    await _push_event(task_id, "artifact_edited", {
        "key": key,
        "stage": ARTIFACT_KEYS[key],
        "comment": req.comment,
        "old_len": len(old),
        "new_len": len(req.value),
    })
    logger.info(
        f"用户 {ctx['user']['username']} 编辑了任务 {task_id} 的产物 {key}"
        f"（{len(old)} → {len(req.value)} 字符）"
    )
    return {"status": "saved", "key": key, "length": len(req.value)}


@router.post("/research/{task_id}/pause", summary="请求在下一个阶段边界暂停")
async def pause(
    ctx: dict = Depends(get_owned_task),
):
    """把「下一个阶段边界」临时变成一道闸门（用户 m18259：保留及时修正的接口）。

    不是立刻打断正在跑的阶段 —— 那样当前阶段的 token 白花、中间产物半成品。
    请求在边界停下，当前阶段成果完整保留，下一个阶段等人放行。
    默认路径仍是 3 道闸门（plan / research / review），本接口把接口留宽。
    """
    task_id = ctx["task"].id
    ok = await pause_pipeline(task_id)
    if not ok:
        raise HTTPException(status_code=400, detail="任务已结束，无法暂停")
    logger.info(f"用户 {ctx['user']['username']} 请求暂停任务 {task_id}")
    return {"status": "pause_requested"}


@router.post("/research/{task_id}/abort", summary="中止任务")
async def abort(
    ctx: dict = Depends(get_owned_task),
):
    """中止任务（用户 m25143 第 7 项：只能触发一次）。

    任务已是终态时返回 409 而不是再中止一遍 —— 否则重复点击会在时间线上
    又推一条「任务已中止」事件。前端同时把按钮置灰，但那只是体验，服务端
    这里才是真正的边界。
    """
    task_id = ctx["task"].id
    ok = await abort_pipeline(task_id)
    if not ok:
        raise HTTPException(status_code=409, detail="任务已结束，无需再次中止")
    logger.info(f"用户 {ctx['user']['username']} 中止了任务 {task_id}")
    return {"status": "aborted"}


@router.delete("/research/{task_id}", summary="删除任务及其全部记录")
async def remove_task(
    ctx: dict = Depends(get_owned_task),
):
    """删除任务及其全部关联记录（用户 m25143 第 8 项）。

    不可恢复：论断、证据、事件、报告、评审记录、用量记录一并删除。
    归属校验由 get_owned_task 完成（非本人任务 404），delete_task 再校验一次。
    """
    task_id = ctx["task"].id
    ok = await delete_task(task_id, ctx["user"]["user_id"])
    if not ok:
        raise HTTPException(status_code=404, detail="任务不存在")
    logger.info(f"用户 {ctx['user']['username']} 删除了任务 {task_id}")
    return {"status": "deleted", "task_id": task_id}


@router.get("/research/{task_id}/events", summary="SSE 事件流")
async def task_events(
    request: Request,
    ctx: dict = Depends(get_owned_task),
):
    """SSE 事件流（跨进程安全 + 断线续传）。

    - 事件源：orchestrator.event_stream()，先订阅 Redis 再回放 task_events 落库
      事件，因此 arq Worker（独立进程）推送的事件也能到达浏览器。
    - 断线续传：客户端断线重连时浏览器自动带 `Last-Event-ID` 头，服务端从该
      id 之后增量回放；也支持 `?last_event_id=` 查询参数（便于 curl 验证）。
    """
    task_id = ctx["task"].id
    last_id = 0
    header_last = request.headers.get("last-event-id")
    query_last = request.query_params.get("last_event_id")
    for raw in (header_last, query_last):
        if raw:
            try:
                last_id = max(last_id, int(raw))
            except ValueError:
                pass

    return EventSourceResponse(event_stream(task_id, last_id))
