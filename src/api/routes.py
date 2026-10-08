"""API 路由注册"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from src.core.orchestrator import (
    start_pipeline,
    get_task_status,
    approve_gate,
    reject_gate,
    abort_pipeline,
    subscribe,
    unsubscribe,
)
from src.core.logger import logger
from sse_starlette.sse import EventSourceResponse
import asyncio

router = APIRouter()


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
    created_at: str | None = None


@router.get("/ping")
async def ping():
    return {"ping": "pong"}


@router.post("/research/start", summary="启动调研任务")
async def start_research(req: ResearchRequest):
    task_id = await start_pipeline(req.topic)
    return {"task_id": task_id, "status": "started"}


@router.get("/research/{task_id}/status", response_model=TaskStatusResponse)
async def research_status(task_id: int):
    status = await get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="任务不存在")
    return status


@router.post("/research/{task_id}/approve", summary="通过审核闸门")
async def approve(task_id: int, req: GateRequest):
    ok = await approve_gate(task_id, req.gate_name, req.comment)
    if not ok:
        raise HTTPException(status_code=400, detail="无法通过闸门")
    return {"status": "approved"}


@router.post("/research/{task_id}/reject", summary="驳回审核闸门")
async def reject(task_id: int, req: GateRequest):
    ok = await reject_gate(task_id, req.gate_name, req.comment)
    if not ok:
        raise HTTPException(status_code=400, detail="无法驳回闸门")
    return {"status": "rejected"}


@router.post("/research/{task_id}/abort", summary="中止任务")
async def abort(task_id: int):
    await abort_pipeline(task_id)
    return {"status": "aborted"}


@router.get("/research/{task_id}/events", summary="SSE 事件流")
async def task_events(task_id: int):
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
