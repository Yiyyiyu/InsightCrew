"""服务端渲染页面路由（Jinja2 模板 + Cookie 鉴权）。

与 `src/api/routes.py` 的 JSON API 分开：
- `/api/*`  返回 JSON，供脚本与前端 fetch 调用
- 本模块    返回 HTML 页面，供浏览器直接打开

页面鉴权用 Cookie（`ACCESS_COOKIE`）：`<a href>` 跳转无法设置请求头，
所以页面路由必须能从未登录 → 重定向到 /login。
"""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, tuple_

from src.api.deps import get_current_user_optional
from src.core.auth import ACCESS_COOKIE
from src.core.ledger import CITATION_RE
from src.core.logger import logger
from src.db.database import async_session
from src.db.models import Claim, ClaimEvidence, Chunk, Report, Source, Task

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=_TEMPLATE_DIR)
# tojson 默认 ensure_ascii=True，会把中文全部转成 \uXXXX 转义，
# 报告正文动辄上万字，页面体积会膨胀数倍。关掉转义（浏览器解析无差异）。
templates.env.policies["json.dumps_kwargs"] = {"ensure_ascii": False, "sort_keys": True}

router = APIRouter()

STAGE_LABELS = {
    "plan": "规划",
    "research": "调研",
    "analysis": "分析",
    "critic": "质疑",
    "rebuttal": "回应",
    "write": "撰写",
    "review": "评审",
}


def _ctx(request: Request, user: dict | None, **kw) -> dict:
    base = {"request": request, "user": user, "stage_labels": STAGE_LABELS}
    base.update(kw)
    return base


def _login_redirect() -> RedirectResponse:
    """未登录 → 跳登录页。303 保证浏览器把后续请求改成 GET。"""
    return RedirectResponse(url="/login", status_code=303)


@router.get("/login", response_class=HTMLResponse, include_in_schema=False)
async def login_page(request: Request):
    user = get_current_user_optional(request)
    if user:
        return RedirectResponse(url="/tasks", status_code=303)
    return templates.TemplateResponse(request, "login.html", _ctx(request, None))


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home(request: Request):
    user = get_current_user_optional(request)
    return RedirectResponse(url="/tasks" if user else "/login", status_code=303)


@router.get("/tasks", response_class=HTMLResponse, include_in_schema=False)
async def tasks_page(request: Request):
    user = get_current_user_optional(request)
    if not user:
        return _login_redirect()

    async with async_session() as session:
        rows = (await session.execute(
            select(Task)
            .where(Task.user_id == user["user_id"])
            .order_by(Task.id.desc())
        )).scalars().all()
        tasks = [
            {
                "id": t.id,
                "topic": t.topic,
                "status": t.status,
                "current_stage": t.current_stage,
                "gate_status": t.gate_status,
                "total_cost_cny": float(t.total_cost_cny or 0.0),
                "created_at": t.created_at.strftime("%Y-%m-%d %H:%M") if t.created_at else "",
            }
            for t in rows
        ]

    return templates.TemplateResponse(request, "tasks.html", _ctx(request, user, tasks=tasks))


@router.get("/tasks/{task_id}", response_class=HTMLResponse, include_in_schema=False)
async def task_detail_page(request: Request, task_id: int):
    user = get_current_user_optional(request)
    if not user:
        return _login_redirect()

    async with async_session() as session:
        task = await session.get(Task, task_id)
        if task is None or task.user_id != user["user_id"]:
            # 与 API 一致：不存在与越权都返回 404 页面，不泄漏存在性
            return templates.TemplateResponse(
                request, "error.html",
                _ctx(request, user, code=404, message="任务不存在"),
                status_code=404,
            )
        task_info = {
            "id": task.id,
            "topic": task.topic,
            "status": task.status,
            "current_stage": task.current_stage,
            "gate_status": task.gate_status,
            "total_cost_cny": float(task.total_cost_cny or 0.0),
            "created_at": task.created_at.strftime("%Y-%m-%d %H:%M") if task.created_at else "",
        }

    return templates.TemplateResponse(
        request, "task_detail.html", _ctx(request, user, task=task_info)
    )


@router.get("/tasks/{task_id}/report", response_class=HTMLResponse, include_in_schema=False)
async def report_page(request: Request, task_id: int):
    user = get_current_user_optional(request)
    if not user:
        return _login_redirect()

    async with async_session() as session:
        task = await session.get(Task, task_id)
        if task is None or task.user_id != user["user_id"]:
            return templates.TemplateResponse(
                request, "error.html",
                _ctx(request, user, code=404, message="任务不存在"),
                status_code=404,
            )
        report = (await session.execute(
            select(Report).where(Report.task_id == task_id).order_by(Report.id.desc()).limit(1)
        )).scalar_one_or_none()

    if report is None:
        # 任务存在但还没报告：给出可操作的提示，而不是 404
        return templates.TemplateResponse(
            request, "error.html",
            _ctx(request, user, code=409,
                 message="任务尚未生成报告（可能仍在运行或已失败）",
                 back_url=f"/tasks/{task_id}"),
            status_code=409,
        )

    content_md = report.content_md or ""

    # 引用角标 → 证据抽屉。
    # 直接按标记里的 (source_id, seq) 查 chunk/source：标记本身就编码了这两者，
    # 不需要绕经 ClaimEvidence —— 否则报告里引用了某个 chunk、但该 chunk 恰好
    # 没被任何 claim 引用时，抽屉就打不开（点角标没反应）。
    markers = sorted(set(CITATION_RE.findall(content_md)))
    evidence_map: dict[str, dict] = {}
    if markers:
        pairs = {(int(sid), int(seq)) for sid, seq in markers}
        async with async_session() as session:
            rows = (await session.execute(
                select(Chunk, Source)
                .join(Source, Chunk.source_id == Source.id)
                .where(tuple_(Chunk.source_id, Chunk.seq).in_(pairs))
                .order_by(Chunk.id)
            )).all()
        for chunk, source in rows:
            key = f"[S{source.id}#{chunk.seq}]"
            # 老库可能有重复 (source_id, seq)（并发写入遗留）；按 chunk.id 升序，
            # 首次命中即胜出，与 verify_citations / record_claims 的取值一致。
            if key in evidence_map:
                continue
            evidence_map[key] = {
                "url": source.url,
                "title": source.title,
                "page_num": chunk.page_num,
                "fetched_at": source.fetched_at.strftime("%Y-%m-%d %H:%M")
                if source.fetched_at else "",
                "snippet": (chunk.content or "")[:600],
            }

    return templates.TemplateResponse(
        request, "report.html",
        _ctx(request, user,
             task={"id": task.id, "topic": task.topic},
             content_md=content_md,
             bibtex=report.bibtex or "",
             evidence_map=evidence_map),
    )


@router.get("/tasks/{task_id}/evidence", response_class=HTMLResponse, include_in_schema=False)
async def evidence_page(request: Request, task_id: int):
    """证据台账：论断 ↔ chunk ↔ 来源 URL 的对应关系（可追溯性的展示面）。"""
    user = get_current_user_optional(request)
    if not user:
        return _login_redirect()

    async with async_session() as session:
        task = await session.get(Task, task_id)
        if task is None or task.user_id != user["user_id"]:
            return templates.TemplateResponse(
                request, "error.html",
                _ctx(request, user, code=404, message="任务不存在"),
                status_code=404,
            )

        # 论断 + 其证据链（claim → evidence → chunk → source）
        # ★ 必须按 task_id 过滤：claims 表是跨任务共享的，不过滤会把其他
        #   用户任务的论断展示出来（用户隔离失效）。
        claims = (await session.execute(
            select(Claim)
            .where(Claim.task_id == task_id)
            .order_by(Claim.id.desc())
            .limit(300)
        )).scalars().all()

        claim_ids = [c.id for c in claims]
        evidence_rows = []
        if claim_ids:
            evidence_rows = (await session.execute(
                select(ClaimEvidence, Chunk, Source)
                .join(Chunk, ClaimEvidence.chunk_id == Chunk.id)
                .join(Source, ClaimEvidence.source_id == Source.id)
                .where(ClaimEvidence.claim_id.in_(claim_ids))
            )).all()

    by_claim: dict[int, list[dict]] = {}
    for ev, chunk, source in evidence_rows:
        by_claim.setdefault(ev.claim_id, []).append({
            "marker": f"[S{source.id}#{chunk.seq}]",
            "url": source.url,
            "title": source.title,
            "page_num": chunk.page_num,
            "snippet": (chunk.content or "")[:300],
        })

    items = [{
        "id": c.id,
        "text": c.text,
        "level": c.evidence_level or "C",
        "evidence": by_claim.get(c.id, []),
    } for c in claims]

    return templates.TemplateResponse(
        request, "evidence.html",
        _ctx(request, user, task={"id": task.id, "topic": task.topic}, claims=items),
    )
