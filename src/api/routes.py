"""API 路由注册"""

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel
from sqlalchemy import select

from src.api.deps import get_current_user, get_owned_task
from src.core.auth import (
    ACCESS_COOKIE,
    hash_password,
    verify_password,
    create_access_token,
)
from src.core.config import settings
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
    event_stream,
    _push_event,
)
from src.core.ratelimit import RateLimitExceeded, check_rate_limit, client_ip
from src.db.database import async_session
from src.db.models import Report, Task, User
from sse_starlette.sse import EventSourceResponse

router = APIRouter()


# ---- 接口限流（C4-1）----
# 上线前必须有：公网暴露后，注册/登录接口可被暴力枚举，建任务接口
# 可被刷爆 LLM 账单。按 IP + user 双维度限制。
_LIMITS: dict[str, tuple[int, int]] = {
    "register": (5, 60),        # 5 次 / 分钟
    "login": (5, 60),           # 5 次 / 分钟
    "research_start": (10, 3600),  # 10 次 / 小时
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
    _rate_limit(request, "register")
    if len(req.username.strip()) < 2 or len(req.password.strip()) < 4:
        raise HTTPException(status_code=400, detail="用户名至少2位，密码至少4位")
    async with async_session() as session:
        result = await session.execute(select(User).where(User.username == req.username))
        if result.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="用户名已存在")
        user = User(
            username=req.username,
            hashed_password=hash_password(req.password),
        )
        session.add(user)
        await session.commit()
    return {"msg": "注册成功"}


@router.post("/auth/login", summary="用户登录", response_model=LoginResponse)
async def login(req: LoginRequest, response: Response, request: Request):
    _rate_limit(request, "login")
    async with async_session() as session:
        result = await session.execute(select(User).where(User.username == req.username))
        user = result.scalar_one_or_none()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_access_token(user_id=user.id, username=user.username)

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
    task_id = ctx["task"].id
    await abort_pipeline(task_id)
    logger.info(f"用户 {ctx['user']['username']} 中止了任务 {task_id}")
    return {"status": "aborted"}


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
