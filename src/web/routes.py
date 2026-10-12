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
from sqlalchemy import and_, func, or_, select, tuple_

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

# 分组时的显示顺序：按流水线推进顺序，而不是字母序 ——
# 字母序会把 write 排在 plan 前面，分组头读起来就乱。
STAGE_ORDER = ["plan", "research", "analysis", "critic", "rebuttal", "write", "review"]

STATUS_LABELS = {
    "pending": "待开始",
    "running": "运行中",
    "completed": "已完成",
    "failed": "失败",
    "aborted": "已中止",
}
# 状态分组顺序：需要人关注的排前面（运行中/待开始），已结束的排后面
STATUS_ORDER = ["running", "pending", "completed", "failed", "aborted"]


def _ctx(request: Request, user: dict | None, **kw) -> dict:
    base = {"request": request, "user": user, "stage_labels": STAGE_LABELS}
    base.update(kw)
    return base


def _login_redirect() -> RedirectResponse:
    """未登录 → 跳登录页。303 保证浏览器把后续请求改成 GET。"""
    return RedirectResponse(url="/login", status_code=303)


@router.get("/login", response_class=HTMLResponse, include_in_schema=False)
async def login_page(request: Request):
    user = await get_current_user_optional(request)
    if user:
        return RedirectResponse(url="/tasks", status_code=303)
    return templates.TemplateResponse(request, "login.html", _ctx(request, None))


@router.get("/register", response_class=HTMLResponse, include_in_schema=False)
async def register_page(request: Request):
    """独立注册页（用户 m25143 第 2 项）。

    原先是登录页里用 JS 切换「登录/注册」两种模式，注册项挤在登录表单里，
    既放不下字段（二次确认、邮箱、强度提示），也说不清当前处于哪个模式。
    """
    user = await get_current_user_optional(request)
    if user:
        return RedirectResponse(url="/tasks", status_code=303)
    return templates.TemplateResponse(request, "register.html", _ctx(request, None))


@router.get("/forgot", response_class=HTMLResponse, include_in_schema=False)
async def forgot_page(request: Request):
    """忘记密码：填邮箱/用户名 + 图形验证码，申请重置链接。"""
    user = await get_current_user_optional(request)
    if user:
        return RedirectResponse(url="/tasks", status_code=303)
    return templates.TemplateResponse(request, "forgot.html", _ctx(request, None))


@router.get("/reset", response_class=HTMLResponse, include_in_schema=False)
async def reset_page(request: Request, token: str = ""):
    """重置密码页。

    ★ 令牌**不在这里校验** —— 页面先渲染出来，再由前端调
      `/auth/reset/verify` 问「这个令牌还能用吗」。原因是校验要查库（async），
      而这里直接渲染更简单，且失效态在页面上提示比返回一个 400 页面友好。
      真正的校验在提交时还会再做一次（前端校验只是体验，不是安全边界）。
    """
    return templates.TemplateResponse(
        request, "reset.html", _ctx(request, None, token=token)
    )


@router.get("/profile", response_class=HTMLResponse, include_in_schema=False)
async def profile_page(request: Request):
    """个人信息页（用户 m26104）：补填邮箱、修改密码。"""
    user = await get_current_user_optional(request)
    if not user:
        return _login_redirect()
    return templates.TemplateResponse(request, "profile.html", _ctx(request, user))


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home(request: Request):
    user = await get_current_user_optional(request)
    return RedirectResponse(url="/tasks" if user else "/login", status_code=303)


def _int_param(raw, default: int) -> int:
    """URL 查询参数转 int，非法值回落到默认值（书签/手改 URL 不该报 422）。"""
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


@router.get("/tasks", response_class=HTMLResponse, include_in_schema=False)
async def tasks_page(request: Request, page: str = "1", page_size: str = "20",
                     q: str = "", group: str = "none", sort: str = "created_desc"):
    """任务列表：检索 + 分组 + 排序 + 分页（用户 m25143 第 3 项）。

    全部在服务端做：筛选/排序参数进 URL，刷新、分享、前进后退都保持状态，
    也避免前端拿全量数据自己过滤（任务多了以后那份数据会很大）。
    非法参数一律回落到默认值，不报 422 —— 书签和手改 URL 都该能用。
    """
    user = await get_current_user_optional(request)
    if not user:
        return _login_redirect()

    # 分页参数兜底：非法/越界一律夹到合法范围，不报错（书签、手改 URL 都能用）
    page = _int_param(page, 1)
    page_size = max(5, min(_int_param(page_size, 20), 100))
    page = max(1, page)
    where = Task.user_id == user["user_id"]

    # ── 关键词：同时匹配主题与任务号（用户常直接搜 990176 这种 ID）──
    kw = (q or "").strip()
    if kw:
        conds = [Task.topic.ilike(f"%{kw}%")]
        if kw.isdigit():
            conds.append(Task.id == int(kw))
        where = and_(where, or_(*conds))

    # ── 排序白名单：只认这几个值，其余回落默认 ──
    _SORTS = {
        "created_desc": Task.id.desc(),
        "created_asc": Task.id.asc(),
        "cost_desc": Task.total_cost_cny.desc(),
        "cost_asc": Task.total_cost_cny.asc(),
    }
    sort = sort if sort in _SORTS else "created_desc"

    # ── 分组白名单 ──
    group = group if group in ("none", "stage", "status") else "none"

    async with async_session() as session:
        total = (await session.execute(
            select(func.count()).select_from(Task).where(where)
        )).scalar_one()
        total_pages = max(1, (total + page_size - 1) // page_size)
        # 请求页超出末页时（如删任务后书签失效）回落到末页，避免空白列表
        page = min(page, total_pages)
        rows = (await session.execute(
            select(Task)
            .where(where)
            .order_by(_SORTS[sort])
            .limit(page_size)
            .offset((page - 1) * page_size)
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

    # ── 分组：把当前页的行按阶段或状态切成有序的组 ──
    # 顺序不是字母序，而是流程顺序（规划→调研→…→评审），这样分组头读起来
    # 就是「任务在流水线上的位置」，字母序会把 write 排在 plan 前面。
    groups: list[dict] = []
    if group != "none" and tasks:
        key_fn = (lambda t: t["current_stage"]) if group == "stage" else (lambda t: t["status"])
        order = STAGE_ORDER if group == "stage" else STATUS_ORDER
        labels = STAGE_LABELS if group == "stage" else STATUS_LABELS
        buckets: dict[str, list] = {}
        for t in tasks:
            buckets.setdefault(key_fn(t), []).append(t)
        for k in order:
            if k in buckets:
                groups.append({"key": k, "label": labels.get(k, k), "items": buckets.pop(k)})
        # 白名单之外的值（数据里出现的意外状态）也要显示，不能悄悄吞掉
        for k in sorted(buckets):
            groups.append({"key": k, "label": labels.get(k, k), "items": buckets[k]})

    # 分组头也要能点：保持当前筛选条件，只换分组方式
    def _qs(**over) -> str:
        base = {"page": 1, "page_size": page_size, "q": kw,
                "group": group, "sort": sort}
        base.update(over)
        return "&".join(f"{k}={v}" for k, v in base.items() if v not in ("", None))

    return templates.TemplateResponse(request, "tasks.html", _ctx(
        request, user, tasks=tasks, groups=groups,
        page=page, page_size=page_size, total=total, total_pages=total_pages,
        q=kw, group=group, sort=sort, qs=_qs,
    ))


@router.get("/tasks/{task_id}", response_class=HTMLResponse, include_in_schema=False)
async def task_detail_page(request: Request, task_id: int):
    user = await get_current_user_optional(request)
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
    user = await get_current_user_optional(request)
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


async def _load_report(request: Request, task_id: int, user: dict):
    """报告导出的公共前置：鉴权 + 任务归属 + 报告存在性。

    成功返回 (task, report)；失败返回 (错误响应, None)，调用方直接 `return err`。
    """
    async with async_session() as session:
        task = await session.get(Task, task_id)
        if task is None or task.user_id != user["user_id"]:
            return templates.TemplateResponse(
                request, "error.html",
                _ctx(request, user, code=404, message="任务不存在"),
                status_code=404,
            ), None
        report = (await session.execute(
            select(Report).where(Report.task_id == task_id).order_by(Report.id.desc()).limit(1)
        )).scalar_one_or_none()

    if report is None:
        return templates.TemplateResponse(
            request, "error.html",
            _ctx(request, user, code=409,
                 message="任务尚未生成报告（可能仍在运行或已失败）",
                 back_url=f"/tasks/{task_id}"),
            status_code=409,
        ), None
    return task, report


@router.get("/tasks/{task_id}/report.md", include_in_schema=False)
async def report_md(task_id: int, request: Request):
    """服务端 Markdown 导出（正文 + 参考文献章节，与 PDF 同源同序）。"""
    from fastapi.responses import Response

    from src.core.export import render_markdown

    user = await get_current_user_optional(request)
    if not user:
        return _login_redirect()

    task, report = await _load_report(request, task_id, user)
    if report is None:
        return task

    md = render_markdown(
        topic=task.topic,
        content_md=report.content_md or "",
        bibtex=report.bibtex or "",
    )
    filename = f"report-task{task_id}.md"
    return Response(
        content=md,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


@router.get("/tasks/{task_id}/report.pdf", include_in_schema=False)
async def report_pdf(task_id: int, request: Request):
    """服务端 PDF 导出（含参考文献）。鉴权与 404 语义与 report_page 完全一致。"""
    from fastapi.responses import Response

    from src.core.export import render_pdf

    user = await get_current_user_optional(request)
    if not user:
        return _login_redirect()

    task, report = await _load_report(request, task_id, user)
    if report is None:
        return task

    generated_at = (report.created_at.strftime("%Y-%m-%d %H:%M")
                    if getattr(report, "created_at", None) else "")
    pdf = render_pdf(
        topic=task.topic,
        content_md=report.content_md or "",
        bibtex=report.bibtex or "",
        task_id=task.id,
        generated_at=generated_at,
    )
    filename = f"report-task{task_id}.pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


@router.get("/tasks/{task_id}/evidence", response_class=HTMLResponse, include_in_schema=False)
async def evidence_page(request: Request, task_id: int):
    """证据台账：论断 ↔ chunk ↔ 来源 URL 的对应关系（可追溯性的展示面）。"""
    user = await get_current_user_optional(request)
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
