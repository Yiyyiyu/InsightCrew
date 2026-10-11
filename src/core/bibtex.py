"""BibTeX 生成器 —— 从引用台账真实导出参考文献

设计
----
报告正文里的引用标记是 `[S{source_id}#{chunk_seq}]`（见 `src/core/ledger.py`）。
本模块据此生成 BibTeX，并让**标记能直接对应到条目**：

* 条目键名 = `S{source_id}`（例如 `S12`）→ `[S12#3]` 与 `[S12#7]` 都指向 `@misc{S12, ...}`
* 每条目的 `note` 字段列出该来源被引用的**全部标记**，便于人工核对

收录范围（并集，保证正文里的标记一定有对应条目）：
1. 报告正文 `content_md` 中出现的全部 `[S{id}#{seq}]` 标记；
2. 该任务 `claims` → `claim_evidence` 关联到的全部 `source_id`。

`Source` 表按 URL 全局共享、**没有 task_id**，所以归属只能经
claims / 报告正文反查——这也是上面两条来源必须并集的原因。

同步与异步两套 API：`build_bibtex_sync` 供工具层/测试调用，
`build_bibtex` 供 orchestrator 在流水线收尾时调用。
"""

from __future__ import annotations

import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.core.ledger import extract_markers, get_sync_engine
from src.db.models import Claim, ClaimEvidence, Source

# BibTeX 特殊字符转义（反斜杠必须最先处理，用正则一次性替换避免二次转义）
_BIBTEX_ESCAPE = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_ESCAPE_RE = re.compile(r"[\\{}$&#_%~^]")


def _esc(text: str) -> str:
    """转义 BibTeX 特殊字符。"""
    return _ESCAPE_RE.sub(lambda m: _BIBTEX_ESCAPE[m.group()], text or "")


def bibtex_key(source_id: int) -> str:
    """来源 id → BibTeX 条目键名。`[S12#3]` → `S12`。"""
    return f"S{source_id}"


def _entry(source: Source, markers: list[str]) -> str:
    """把一行 Source 渲染成一条 @misc 条目。"""
    title = (source.title or "").strip() or (source.url or "").strip() or f"来源 {source.id}"
    fields: list[tuple[str, str]] = [("title", f"{{{_esc(title)}}}")]

    url = (source.url or "").strip()
    if url:
        fields.append(("howpublished", f"{{\\url{{{url}}}}}"))

    year = source.fetched_at.year if source.fetched_at else None
    if year:
        fields.append(("year", f"{{{year}}}"))

    # 来源类型落进 note：便于人工判断证据等级（pdf/web → 全文验证，*_snippet → 摘要级）
    note_parts = [f"来源类型：{source.source_type or '未知'}"]
    if markers:
        note_parts.append("引用标记：" + ", ".join(markers))
    fields.append(("note", "{" + _esc("；".join(note_parts)) + "}"))

    body = ",\n".join(f"  {k:<12} = {v}" for k, v in fields)
    return f"@misc{{{bibtex_key(source.id)},\n{body},\n}}"


def render_bibtex(sources: list[Source], marker_map: dict[int, list[str]]) -> str:
    """把来源列表渲染成完整 .bib 文本（按 source_id 升序，输出稳定）。"""
    if not sources:
        return ""
    blocks = [_entry(s, marker_map.get(s.id, [])) for s in sorted(sources, key=lambda x: x.id)]
    return "\n\n".join(blocks) + "\n"


def _collect(session: Session, task_id: int, report_md: str) -> tuple[list[Source], dict[int, list[str]]]:
    """收集该任务涉及的 source 行与"source_id → 引用标记列表"映射。"""
    # ① 报告正文里的标记
    marker_map: dict[int, list[str]] = {}
    for sid, seq in extract_markers(report_md or ""):
        marker_map.setdefault(sid, []).append(f"[S{sid}#{seq}]")

    # ② claims → claim_evidence 关联的 source（含正文没写标记、但论断已入库的来源）
    ev_source_ids = session.execute(
        select(ClaimEvidence.source_id)
        .join(Claim, Claim.id == ClaimEvidence.claim_id)
        .where(Claim.task_id == task_id)
        .distinct()
    ).scalars().all()
    for sid in ev_source_ids:
        marker_map.setdefault(sid, [])

    if not marker_map:
        return [], {}

    sources = session.execute(
        select(Source).where(Source.id.in_(list(marker_map.keys())))
    ).scalars().all()
    return list(sources), marker_map


def build_bibtex_sync(task_id: int, report_md: str = "") -> str:
    """同步生成 BibTeX（测试/工具层用）。无来源时返回空串。"""
    with Session(get_sync_engine()) as s:
        sources, marker_map = _collect(s, task_id, report_md)
        return render_bibtex(sources, marker_map)


async def build_bibtex(task_id: int, report_md: str = "") -> str:
    """异步生成 BibTeX（orchestrator 收尾时调用）。无来源时返回空串。"""
    from src.db.database import async_session

    async with async_session() as s:
        marker_map: dict[int, list[str]] = {}
        for sid, seq in extract_markers(report_md or ""):
            marker_map.setdefault(sid, []).append(f"[S{sid}#{seq}]")

        rows = (await s.execute(
            select(ClaimEvidence.source_id)
            .join(Claim, Claim.id == ClaimEvidence.claim_id)
            .where(Claim.task_id == task_id)
            .distinct()
        )).scalars().all()
        for sid in rows:
            marker_map.setdefault(sid, [])

        if not marker_map:
            return ""

        sources = (await s.execute(
            select(Source).where(Source.id.in_(list(marker_map.keys())))
        )).scalars().all()
        return render_bibtex(list(sources), marker_map)


__all__ = [
    "bibtex_key",
    "render_bibtex",
    "build_bibtex_sync",
    "build_bibtex",
]
