"""API 路由注册"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select

from src.api.deps import get_current_user
from src.core.auth import hash_password, verify_password, create_access_token
from src.core.logger import logger
from src.core.orchestrator import (
    start_pipeline,
    get_task_status,
    approve_gate,
    reject_gate,
    abort_pipeline,
    subscribe,
    unsubscribe,
)
from src.db.database import async_session
from src.db.models import User
from sse_starlette.sse import EventSourceResponse

router = APIRouter()


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


class GateRequest(BaseModel):
    gate_name: str
    comment: str = ""


class TaskStatusResponse(BaseModel):
    task_id: int
    topic: str
    status: str
    current_stage: str
    gate_status: str
    pending_gate: str | None = None
    report_draft: str | None = None
    review_output: str | None = None
    exec_path: str = "unknown"
    created_at: str | None = None


# ---- 鉴权路由 ----


@router.post("/auth/register", summary="用户注册")
async def register(req: RegisterRequest):
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
async def login(req: LoginRequest):
    async with async_session() as session:
        result = await session.execute(select(User).where(User.username == req.username))
        user = result.scalar_one_or_none()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_access_token(user_id=user.id, username=user.username)
    return LoginResponse(access_token=token)


# ---- 调研任务路由（需登录）----


@router.post("/research/start", summary="启动调研任务")
async def start_research(
    req: ResearchRequest,
    current_user: dict = Depends(get_current_user),
):
    task_id = await start_pipeline(req.topic)
    logger.info(f"用户 {current_user['username']} 启动了任务 {task_id}")
    return {"task_id": task_id, "status": "started"}


@router.get("/research/{task_id}/status", response_model=TaskStatusResponse)
async def research_status(
    task_id: int,
    current_user: dict = Depends(get_current_user),
):
    status = await get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="任务不存在")
    return status


@router.post("/research/{task_id}/approve", summary="通过审核闸门")
async def approve(
    task_id: int,
    req: GateRequest,
    current_user: dict = Depends(get_current_user),
):
    ok = await approve_gate(task_id, req.gate_name, req.comment)
    if not ok:
        raise HTTPException(status_code=400, detail="无法通过闸门")
    logger.info(f"用户 {current_user['username']} 通过了闸门 {req.gate_name}")
    return {"status": "approved"}


@router.post("/research/{task_id}/reject", summary="驳回审核闸门")
async def reject(
    task_id: int,
    req: GateRequest,
    current_user: dict = Depends(get_current_user),
):
    ok = await reject_gate(task_id, req.gate_name, req.comment)
    if not ok:
        raise HTTPException(status_code=400, detail="无法驳回闸门")
    logger.info(f"用户 {current_user['username']} 驳回了闸门 {req.gate_name}，备注: {req.comment}")
    return {"status": "rejected"}


@router.post("/research/{task_id}/abort", summary="中止任务")
async def abort(
    task_id: int,
    current_user: dict = Depends(get_current_user),
):
    await abort_pipeline(task_id)
    logger.info(f"用户 {current_user['username']} 中止了任务 {task_id}")
    return {"status": "aborted"}


@router.get("/research/{task_id}/events", summary="SSE 事件流")
async def task_events(
    task_id: int,
    current_user: dict = Depends(get_current_user),
):
    queue = subscribe(task_id)
    async def event_generator():
        try:
            while True:
                payload = await queue.get()
                yield {"event": payload["type"], "data": payload["data"]}
                if payload["type"] in ("pipeline_complete", "pipeline_aborted"):
                    break
        finally:
            unsubscribe(task_id, queue)
    return EventSourceResponse(event_generator())
