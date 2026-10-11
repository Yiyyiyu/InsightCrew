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
import threading
from typing import Sequence

from sqlalchemy import create_engine, event, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.core.config import settings
from src.core.logger import logger
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


# ------------------------------------------------------------------- 写入串行化
# 工具层跑在 CrewAI 的工作线程里，research 阶段会 asyncio.gather 并发多个方向；
# 多个方向完全可能同时抓到同一个 URL（官网/文档页），于是并发进入
# add_source()。read（查已有 chunk）→ compute（算 base seq）→ write 若不加锁，
# 两个线程会读到同一个 base，各自写入 seq 相同的 chunk。
#
# 锁解决**进程内**竞态；跨进程（多个 arq worker）由 chunks 的
# (source_id, seq) 唯一索引兜底，写入撞索引后走 IntegrityError 重试。
_LEDGER_LOCK = threading.RLock()


def _unique_conflict_retry(fn, attempts: int = 5):
    """撞唯一索引时重试：说明有并发写者已经完成了同样的工作，重读即可复用。"""
    last: Exception | None = None
    for _ in range(attempts):
        try:
            return fn()
        except IntegrityError as exc:  # noqa: PERF203
            last = exc
            continue
    raise last if last else RuntimeError("unreachable")


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
    """按 URL 复用来源行，返回 source_id。

    并发下两个线程可能同时查不到、同时插入 → 撞 `sources.url` 唯一索引。
    撞了就重查一次（另一个线程已插好），而不是抛错。
    """
    url = (url or "")[:1024]
    with _LEDGER_LOCK, Session(get_sync_engine()) as s:
        row = s.execute(select(Source).where(Source.url == url)).scalars().first()
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
        try:
            s.commit()
        except IntegrityError:
            # 并发写者抢先插入同一 url：回滚后重查，复用它的行。
            s.rollback()
            row = s.execute(select(Source).where(Source.url == url)).scalars().first()
            if row is None:
                raise
            return row.id
        s.refresh(src)
        return src.id


def add_source(url: str, title: str, source_type: str, texts: Sequence[str],
               raw_text: str | None = None,
               page_nums: Sequence[int] | None = None) -> tuple[int, list[tuple[int, int]]]:
    """写入一个来源及其分块。

    返回 ``(source_id, [(chunk_id, chunk_seq), ...])``，
    chunk_seq 即引用标记 `[S{source_id}#{chunk_seq}]` 中的第二个数字。

    幂等：若同一 URL 已存在内容完全相同的分块，直接复用原有标记，不重复追加。
    否则（内容确实变了）追加，不覆盖历史——旧 chunk 可能已被 claim_evidence 引用，
    删掉会让已有论断的证据链断裂。

    ★ 并发安全：research 阶段多方向并行，可能同时抓到同一 URL。整个
    「读已有 → 算 base → 写」必须在进程内锁里完成；跨进程由 chunks 的
    (source_id, seq) 唯一索引兜底，撞索引即重读复用（见 _unique_conflict_retry）。
    """
    sid = get_or_create_source(url, title, source_type, raw_text)

    def _write() -> tuple[int, list[tuple[int, int]]]:
        with _LEDGER_LOCK, Session(get_sync_engine()) as s:
            existing = s.execute(
                select(Chunk).where(Chunk.source_id == sid).order_by(Chunk.seq)
            ).scalars().all()

            # 幂等判定：内容一致即视为"同一份材料"，复用原标记，不重复追加。
            # 页码只在旧记录缺失时补齐（例如先用 fetch_webpage 抓了正文、
            # 再用 parse_pdf 解析出页码），不因页码差异而重复入库。
            if existing and len(existing) == len(texts) \
                    and [c.content for c in existing] == list(texts):
                if page_nums:
                    for c, pn in zip(existing, page_nums):
                        if c.page_num is None and pn is not None:
                            c.page_num = pn
                    s.commit()
                return sid, [(c.id, c.seq) for c in existing]

            base = (max((c.seq for c in existing), default=0) + 1) if existing else 1
            out: list[tuple[int, int]] = []
            for i, t in enumerate(texts):
                c = Chunk(source_id=sid, seq=base + i, content=t,
                          page_num=(page_nums[i] if page_nums else None))
                s.add(c)
                s.flush()
                out.append((c.id, c.seq))
            s.commit()
            return sid, out

    try:
        return _unique_conflict_retry(_write)
    except IntegrityError:
        # 跨进程并发：唯一索引已把重复写入挡掉。重读一次现有分块——
        # 若与本次内容一致就复用；否则说明 base 计算被别的写者推进了，
        # 再重试一轮拿到新的 base。
        with _LEDGER_LOCK, Session(get_sync_engine()) as s:
            existing = s.execute(
                select(Chunk).where(Chunk.source_id == sid).order_by(Chunk.seq)
            ).scalars().all()
            if existing and len(existing) == len(texts) \
                    and [c.content for c in existing] == list(texts):
                return sid, [(c.id, c.seq) for c in existing]
        return _unique_conflict_retry(_write)


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
            # 用 scalars().first() 而非 scalar_one_or_none()：老库可能存在重复行
            # （并发写入的历史遗留），此时抛 MultipleResultsFound 会让整个校验崩掉，
            # 而不是报告"这条标记是否可回溯"。取第一行即可完成校验语义。
            row = (await s.execute(
                select(Chunk).where(Chunk.source_id == sid, Chunk.seq == seq)
                .order_by(Chunk.id)
            )).scalars().first()
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
    """把带引用标记的句子落成 claims + claim_evidence。

    `task_id` 必须落库：论断是按任务归属的，否则证据台账只能全表查询，
    会把其他用户的论断展示出来（用户隔离失效）。
    """
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
                # 同 verify_citations：老库重复行不应让 record_claims 崩掉。
                ch = (await s.execute(
                    select(Chunk).where(Chunk.source_id == sid, Chunk.seq == seq)
                    .order_by(Chunk.id)
                )).scalars().first()
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
            claim = Claim(text=sent[:4000], evidence_level=level, task_id=task_id)
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
