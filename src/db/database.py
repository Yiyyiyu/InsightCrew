"""数据库引擎与会话管理

★ M1/M2 遗留项 9：schema 的唯一权威是 **Alembic**（`migrations/versions/`）。
   `init_db()` 不再用 `create_all` 建表，而是：

   - **空库** → `alembic upgrade head`（基线迁移建出全部 14 张表 + 唯一索引）
   - **老库**（有表、无 `alembic_version`）→ 先补列/合重复行（见下），再
     `alembic stamp head` 认账，此后只走迁移
   - **已认账的库** → `alembic upgrade head`（应用新增迁移）

   `_migrate` / `_dedupe_ledger` 保留但**只在给老库认账前执行一次**：
   它们修的是 Alembic 基线出现之前的库（缺列、并发写出的重复行），
   属于历史数据修复，不适合写进「从零建库」的基线迁移里。
"""

import asyncio
from pathlib import Path

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from src.core.config import settings
from src.core.logger import logger

engine = create_async_engine(settings.database_url, echo=False)
async_session = async_sessionmaker(engine, expire_on_commit=False)

# 轻量迁移：`create_all` 只建缺失的表，**不会**给已存在的表加列。
# 仅在「老库认账」路径执行一次，把 Alembic 基线之前的历史库补到基线状态。
# 格式：(表名, 列名, 列定义 DDL)
_MIGRATIONS: list[tuple[str, str, str]] = [
    ("claims", "task_id", "INTEGER REFERENCES tasks(id)"),
    # M1 遗留项 5：任务级参数落库（老库补齐，新库由基线迁移建好）
    ("tasks", "dimensions", "TEXT"),
    ("tasks", "constraints", "TEXT"),
    ("tasks", "budget_limit_cny", "FLOAT"),
    # M2.5：每个阶段的模型计划（JSON 文本）
    ("tasks", "model_plan", "TEXT"),
    # 闸门收敛：用户请求在下一个阶段边界暂停（1 = 已请求）
    ("tasks", "pause_requested", "INTEGER DEFAULT 0"),
    # 独立注册页：邮箱可选（唯一但允许 NULL，老账号 demo 不受影响）
    ("users", "email", "TEXT"),
    # 找回密码：改密后作废旧 JWT（老库补列，默认 0）
    ("users", "token_version", "INTEGER NOT NULL DEFAULT 0"),
    # 邮箱验证（用户 m26742）：非空才表示邮箱真的归用户所有。
    # 老库补列后为 NULL —— 既有用户需要重新验证一次邮箱。
    ("users", "email_verified_at", "DATETIME"),
]

# 已废弃、需要从老库删掉的列（M2.5-7 清理）。
# 格式：(表名, 列名)。SQLite 3.35+ 支持 `ALTER TABLE ... DROP COLUMN`，
# 本机 SQLite 远高于该版本。列不存在时跳过（幂等）。
_DROPPED_COLUMNS: list[tuple[str, str]] = [
    # `model_strategy`（同构/异构开关）已被 `model_plan`（用户逐角色选模型）取代。
    ("tasks", "model_strategy"),
]


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _sync_db_url() -> str:
    """Alembic 用的是同步驱动，把 async 驱动后缀换掉。"""
    return settings.database_url.replace("+aiosqlite", "").replace("+asyncpg", "+psycopg2")


def _run_alembic(action: str) -> None:
    """在独立线程里执行 alembic 命令（它是同步的，会阻塞事件循环）。

    action 取 "upgrade" 或 "stamp"。
    """
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(_project_root() / "alembic.ini"))
    cfg.set_main_option("script_location", str(_project_root() / "migrations"))
    cfg.set_main_option("sqlalchemy.url", _sync_db_url())
    # 程序内调用不重配 logging（否则会污染应用日志）
    cfg.attributes["configure_logger"] = False
    # 显式把 URL 传给 env.py，避免它在另一个线程里 import settings 时
    # 拿到不一致的值（env.py 的 _resolve_url 优先读这个）
    cfg.attributes["db_url"] = _sync_db_url()

    if action == "upgrade":
        command.upgrade(cfg, "head")
    else:
        command.stamp(cfg, "head")


async def _schema_state(conn) -> str:
    """返回 "empty"（无业务表）/ "legacy"（有表但无 alembic_version）/ "managed"。"""
    from sqlalchemy import inspect

    names = set(await conn.run_sync(lambda c: inspect(c).get_table_names()))
    if "alembic_version" in names:
        return "managed"
    if not (names - {"sqlite_sequence", "alembic_version"}):
        return "empty"
    return "legacy"


async def _migrate(conn) -> None:
    """补齐老库缺失的列 + 删除已废弃的列（幂等，可重复执行）。"""
    for table, column, ddl in _MIGRATIONS:
        cols = (await conn.execute(text(f"PRAGMA table_info({table})"))).fetchall()
        if not cols:
            continue  # 表还不存在，基线迁移会建好
        names = {row[1] for row in cols}
        if column in names:
            continue
        logger.info(f"[migrate] {table} 缺少列 {column}，自动补齐")
        await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))

    for table, column in _DROPPED_COLUMNS:
        cols = (await conn.execute(text(f"PRAGMA table_info({table})"))).fetchall()
        if not cols:
            continue
        names = {row[1] for row in cols}
        if column not in names:
            continue
        logger.info(f"[migrate] {table} 删除已废弃列 {column}")
        await conn.execute(text(f"ALTER TABLE {table} DROP COLUMN {column}"))


async def _ensure_indexes(conn) -> None:
    """给老库补建缺失的索引（`stamp head` 只写版本号，不会建任何东西）。

    ★ 为什么需要它：老库的某些列是 `ALTER TABLE ADD COLUMN` 加上去的
      （如 `claims.task_id`），`create_all` 不会给**已存在**的表补索引，
      于是 `ix_claims_task_id` 一直缺着。`stamp head` 之后 Alembic 认为
      基线已应用，也不会再补 —— 必须在这里一次性补齐。

    索引清单直接取自 ORM 元数据，新增索引无需改这里。
    """
    from src.db.models import Base

    for table in Base.metadata.sorted_tables:
        for index in table.indexes:
            cols = ", ".join(c.name for c in index.columns)
            unique = "UNIQUE " if index.unique else ""
            ddl = (f"CREATE {unique}INDEX IF NOT EXISTS {index.name} "
                   f"ON {table.name} ({cols})")
            try:
                await conn.execute(text(ddl))
            except Exception as exc:  # 唯一索引可能因残留重复数据建不上
                logger.warning(f"[migrate] 索引 {index.name} 建失败（跳过）：{exc}")


async def _dedupe_ledger(conn) -> None:
    """合并历史重复行，为 (source_id, seq) / sources.url 唯一索引清场。

    背景：`add_source` 早期在并发（research 多方向 fan-out 同时抓到同一 URL）下
    会各写一份 chunk，导致同一个 `[S{id}#{seq}]` 指向多个 chunk，可追溯性失效。
    代码侧已加锁 + 唯一索引，但**老库里已有的重复数据必须先合并**，否则索引建不上。

    两类的安全性不同：
    - **(source_id, seq) 合并是安全的**：引用标记编码的是 source_id+seq，
      合并只改 `claim_evidence.chunk_id`（内部主键），报告/论断里的标记文本不变。
    - **sources.url 合并会改 source_id**：若某个待删的 source_id 已经出现在
      已生成报告的标记文本里（`[S22#3]`），删掉它会让那条引用永久失效。
      因此这类 source **只在没有任何报告/论断引用它时才合并**；被引用的保留，
      此时 sources.url 唯一索引建不上（记 warning），但 chunks 索引照建。
    """
    import re as _re

    # 已落盘的标记文本里出现过哪些 source_id —— 这些不能删
    referenced_sids: set[int] = set()
    for (md,) in (await conn.execute(text("SELECT content_md FROM reports"))).fetchall():
        referenced_sids |= {int(x) for x in _re.findall(r"\[S(\d+)#", md or "")}
    for (t,) in (await conn.execute(text("SELECT text FROM claims"))).fetchall():
        referenced_sids |= {int(x) for x in _re.findall(r"\[S(\d+)#", t or "")}

    # ── 步骤 1：合并重复的 sources.url ──────────────────────────────
    dup_urls = (await conn.execute(text(
        "SELECT url, GROUP_CONCAT(id) FROM sources "
        "WHERE url IS NOT NULL AND url != '' "
        "GROUP BY url HAVING COUNT(*) > 1"
    ))).fetchall()
    skipped_urls = 0
    for url, ids_csv in dup_urls:
        ids = sorted(int(x) for x in str(ids_csv).split(","))
        # 优先保留被报告/论断引用过的那个 id；都没被引用则保留最小 id
        keep_pool = [i for i in ids if i in referenced_sids]
        keep = min(keep_pool) if keep_pool else ids[0]
        dups = [i for i in ids if i != keep]
        unsafe = [i for i in dups if i in referenced_sids]
        if unsafe:
            # 被引用的重复 source 不能删，否则旧报告里的标记会失效
            logger.warning(
                f"[migrate] sources.url 重复但 S{unsafe} 已被报告引用，跳过合并：{url[:80]}")
            skipped_urls += 1
            continue
        await conn.execute(
            text("UPDATE chunks SET source_id=:keep WHERE source_id IN :dups").bindparams(
                bindparam("dups", expanding=True)),
            {"keep": keep, "dups": dups})
        await conn.execute(
            text("UPDATE claim_evidence SET source_id=:keep WHERE source_id IN :dups").bindparams(
                bindparam("dups", expanding=True)),
            {"keep": keep, "dups": dups})
        await conn.execute(
            text("DELETE FROM sources WHERE id IN :dups").bindparams(
                bindparam("dups", expanding=True)),
            {"dups": dups})
        logger.warning(f"[migrate] sources.url 重复：合并 {dups} → {keep} ({url[:80]})")

    # ── 步骤 2：合并重复的 (source_id, seq) ────────────────────────
    dup_chunks = (await conn.execute(text(
        "SELECT source_id, seq, GROUP_CONCAT(id) FROM chunks "
        "GROUP BY source_id, seq HAVING COUNT(*) > 1"
    ))).fetchall()
    merged = 0
    for source_id, seq, ids_csv in dup_chunks:
        ids = sorted(int(x) for x in str(ids_csv).split(","))
        referenced = {r[0] for r in (await conn.execute(
            text("SELECT DISTINCT chunk_id FROM claim_evidence WHERE chunk_id IN :ids").bindparams(
                bindparam("ids", expanding=True)),
            {"ids": ids})).fetchall()}
        # 优先保留被论断引用的那一行，其次最小 id（确定性）
        keep = min(referenced) if referenced else ids[0]
        dups = [i for i in ids if i != keep]
        if not dups:
            continue
        await conn.execute(
            text("UPDATE claim_evidence SET chunk_id=:keep WHERE chunk_id IN :dups").bindparams(
                bindparam("dups", expanding=True)),
            {"keep": keep, "dups": dups})
        await conn.execute(
            text("DELETE FROM chunks WHERE id IN :dups").bindparams(
                bindparam("dups", expanding=True)),
            {"dups": dups})
        merged += len(dups)
    if dup_chunks:
        logger.warning(f"[migrate] chunks 重复 (source_id, seq)：合并 {len(dup_chunks)} 组，删除 {merged} 行")

    # ── 步骤 3：索引 ────────────────────────────────────────────────
    # 索引统一由 `_ensure_indexes` 在去重之后创建（见 init_db 顺序）。
    # 注意：若上面跳过了被引用的重复 url，`ix_sources_url` 这个唯一索引
    # 会建失败，由 `_ensure_indexes` 捕获并记 warning —— 不阻塞启动。
    if skipped_urls:
        logger.warning(f"[migrate] {skipped_urls} 组重复 url 因被引用未合并")


async def init_db():
    """建库/升级 schema + 修复历史数据（幂等）。

    流程（schema 权威 = Alembic，见模块 docstring）：
      1. 判断库状态：empty / legacy / managed
      2. legacy 库：先 `_migrate`（补列、删废弃列）+ `_dedupe_ledger`
         （合并并发写出的重复行）+ `_ensure_indexes`（补建缺失索引），
         然后 `stamp head` 认账
      3. `upgrade head` —— 空库建全表；已认账库应用新迁移
    """
    state = None
    async with engine.begin() as conn:
        state = await _schema_state(conn)
        if state == "legacy":
            logger.info("[db] 检测到 Alembic 之前的老库，先修复历史数据再认账")
            await _migrate(conn)
            await _dedupe_ledger(conn)
            await _ensure_indexes(conn)

    # alembic 是同步 API，丢到线程里跑，别阻塞事件循环
    if state == "legacy":
        await asyncio.to_thread(_run_alembic, "stamp")
    await asyncio.to_thread(_run_alembic, "upgrade")
    logger.info(f"[db] schema 就绪（state={state}，alembic head）")


async def get_session():
    async with async_session() as session:
        yield session
