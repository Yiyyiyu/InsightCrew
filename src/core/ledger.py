"""引用台账（Citation Ledger）—— 检索结果落库 + 引用硬校验

PRD 硬约束：「所有引用必须回溯到真实检索记录（硬校验，不入库不引用）」。

设计
----
1. 工具层（search_web / search_arxiv / fetch_webpage）在返回文本**之前**，
   先把来源写入 sources、分块写入 chunks，并在返回文本里带上**真实存在**的
   `[S{source_id}#{chunk_seq}]` 标记。
   因此 LLM 只能"抄写"这些标记，它输出的引用天然可回溯到 DB 行。
2. 校验器 verify_citations() 扫描产物中的全部标记并逐个查库，
   查不到的即为**幻觉引用**（hallucinated citation），必须暴露出来。
3. record_claims() 把带引用标记的句子落成 claims + claim_evidence，
   证据分级：chunk 来自整页抓取 → A（全文验证）；来自搜索摘要 → B（摘要级）。

同步 API 供工具层（运行在 CrewAI 工作线程中）调用；
异步 API 供 orchestrator / 校验器调用。
"""

from __future__ import annotations

import re
from typing import Sequence

from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session

from src.core.config import settings
from src.db.models import Chunk, Source

# 引用标记：`[S12#3]` → source_id=12, chunk_seq=3
CITATION_RE = re.compile(r"\[S(\d+)#(\d+)\]")

# 句子切分（中文标点优先，兼容英文与换行）
_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;\n])")

# 来源类型 → 证据等级
_LEVEL_BY_TYPE = {
    "web": "A",             # 整页抓取 → 全文验证
    "search_snippet": "B",  # 搜索摘要 → 摘要级
    "arxiv_snippet": "B",
    "pdf": "A",
}


# --------------------------------------------------------------------- 同步引擎
_sync_engine = None


def _sync_url() -> str:
    url = settings.database_url
    if "+aiosqlite" in url:
        return url.replace("+aiosqlite", "")
    if "+asyncpg" in url:
        return url.replace("+asyncpg", "+psycopg")
    return url


def get_sync_engine():
    """工具层使用的同步引擎（与 async 引擎指向同一个 SQLite 文件）。"""
    global _sync_engine
    if _sync_engine is None:
        _sync_engine = create_engine(_sync_url(), echo=False, future=True)

        @event.listens_for(_sync_engine, "connect")
        def _set_wal(dbapi_conn, _rec):  # noqa: ANN001
            # WAL 让工具线程写入与 Web 进程读取可以并发，避免 database is locked
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA busy_timeout=5000")
            cur.close()

    return _sync_engine


# ------------------------------------------------------------------- 文本分块
def chunk_text(text: str, size: int = 800) -> list[str]:
    """按段落聚合切块，尽量保持语义完整。"""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    out: list[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= size:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            if buf:
                out.append(buf)
            while len(p) > size:
                out.append(p[:size])
                p = p[size:]
            buf = p
    if buf:
        out.append(buf)
    return out or [(text or "")[:size]]


# ------------------------------------------------------------------- 写入台账
def get_or_create_source(url: str, title: str = "", source_type: str = "web",
                         raw_text: str | None = None) -> int:
    """按 URL 复用来源行，返回 source_id。"""
    url = (url or "")[:1024]
    with Session(get_sync_engine()) as s:
        row = s.execute(select(Source).where(Source.url == url)).scalar_one_or_none()
        if row is not None:
            if title and not row.title:
                row.title = title[:256]
            if raw_text and not row.raw_text:
                row.raw_text = raw_text
            s.commit()
            return row.id
        src = Source(url=url, title=(title or "")[:256],
                     source_type=source_type, raw_text=raw_text)
        s.add(src)
        s.commit()
        s.refresh(src)
        return src.id


def add_source(url: str, title: str, source_type: str, texts: Sequence[str],
               raw_text: str | None = None,
               page_nums: Sequence[int] | None = None) -> tuple[int, list[tuple[int, int]]]:
    """写入一个来源及其分块。

    返回 ``(source_id, [(chunk_id, chunk_seq), ...])``，
    chunk_seq 即引用标记 `[S{source_id}#{chunk_seq}]` 中的第二个数字。
    """
    sid = get_or_create_source(url, title, source_type, raw_text)
    with Session(get_sync_engine()) as s:
        existing = s.execute(
            select(Chunk.seq).where(Chunk.source_id == sid)
        ).scalars().all()
        base = (max(existing) + 1) if existing else 1
        out: list[tuple[int, int]] = []
        for i, t in enumerate(texts):
            c = Chunk(source_id=sid, seq=base + i, content=t,
                      page_num=(page_nums[i] if page_nums else None))
            s.add(c)
            s.flush()
            out.append((c.id, c.seq))
        s.commit()
    return sid, out


# ------------------------------------------------------------------- 校验引用
def extract_markers(text: str) -> list[tuple[int, int]]:
    """抽取文本中所有引用标记，去重且保持出现顺序。"""
    seen: list[tuple[int, int]] = []
    for m in CITATION_RE.finditer(text or ""):
        pair = (int(m.group(1)), int(m.group(2)))
        if pair not in seen:
            seen.append(pair)
    return seen


async def verify_citations(text: str) -> dict:
    """硬校验：每个 [S{id}#{seq}] 必须在 chunks 表里真实存在。"""
    from src.db.database import async_session

    markers = extract_markers(text)
    if not markers:
        return {"total": 0, "valid": [], "invalid": [], "rate": 0.0, "pass": True}

    valid: list[dict] = []
    invalid: list[dict] = []
    async with async_session() as s:
        for sid, seq in markers:
            row = (await s.execute(
                select(Chunk).where(Chunk.source_id == sid, Chunk.seq == seq)
            )).scalar_one_or_none()
            if row is not None:
                valid.append({"marker": f"[S{sid}#{seq}]", "chunk_id": row.id,
                              "source_id": sid, "seq": seq})
            else:
                invalid.append({"marker": f"[S{sid}#{seq}]", "source_id": sid, "seq": seq})

    return {
        "total": len(markers),
        "valid": valid,
        "invalid": invalid,
        "rate": len(valid) / len(markers),
        "pass": not invalid,
    }


async def record_claims(text: str, task_id: int | None = None) -> dict:
    """把带引用标记的句子落成 claims + claim_evidence。"""
    from src.db.database import async_session
    from src.db.models import Claim, ClaimEvidence

    stats = {"claims": 0, "evidence": 0, "orphan_markers": 0}
    sentences = [s.strip() for s in _SENT_SPLIT.split(text or "") if s.strip()]

    async with async_session() as s:
        for sent in sentences:
            markers = extract_markers(sent)
            if not markers:
                continue
            rows: list[tuple[Chunk, Source | None]] = []
            for sid, seq in markers:
                ch = (await s.execute(
                    select(Chunk).where(Chunk.source_id == sid, Chunk.seq == seq)
                )).scalar_one_or_none()
                if ch is None:
                    stats["orphan_markers"] += 1
                    continue
                rows.append((ch, await s.get(Source, ch.source_id)))
            if not rows:
                continue
            level = "A" if any(
                src is not None and _LEVEL_BY_TYPE.get(src.source_type or "") == "A"
                for _, src in rows
            ) else "B"
            claim = Claim(text=sent[:4000], evidence_level=level)
            s.add(claim)
            await s.flush()
            stats["claims"] += 1
            for ch, _src in rows:
                s.add(ClaimEvidence(claim_id=claim.id, chunk_id=ch.id,
                                    source_id=ch.source_id))
                stats["evidence"] += 1
        await s.commit()

    return stats


__all__ = [
    "CITATION_RE",
    "get_sync_engine",
    "chunk_text",
    "get_or_create_source",
    "add_source",
    "extract_markers",
    "verify_citations",
    "record_claims",
]
