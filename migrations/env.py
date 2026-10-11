"""Alembic 迁移环境（M1/M2 遗留项 9：Alembic 从未 apply）。

数据库 URL 的解析顺序（三者必须指向同一个库，否则 CLI 与运行时各迁各的）：
1. 环境变量 `ALEMBIC_DB_URL` —— 测试用，可指向临时库
2. `src.core.config.settings.database_url` —— 去掉 `+aiosqlite` 得到同步 URL
3. `alembic.ini` 的 `sqlalchemy.url` —— 兜底

之所以要统一：`init_db()` 现在会在启动时自动 `upgrade head`（空库）或
`stamp head`（历史库），而开发者手动执行 `alembic upgrade head` 时若读的是
另一个 URL，就会出现「迁移显示已 apply 但运行时库还是旧 schema」的鬼故事。
"""

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
# ★ 程序内调用（init_db → alembic command）时用 attributes 关掉：
#   fileConfig 会重配 stdlib logging 的 root handler，在应用启动路径上
#   会覆盖/污染 loguru 之外的日志配置。
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
from src.db.models import Base
target_metadata = Base.metadata


def _resolve_url() -> str:
    """按上述优先级解析出同步数据库 URL。"""
    import os

    # 程序内调用（init_db → alembic command）会显式塞进来，最优先
    injected = config.attributes.get("db_url")
    if injected:
        return injected

    explicit = os.environ.get("ALEMBIC_DB_URL", "").strip()
    if explicit:
        return explicit

    try:
        from src.core.config import settings

        url = settings.database_url
    except Exception:  # 配置不可用时退回 ini
        return config.get_main_option("sqlalchemy.url")

    # asyncpg/aiosqlite 驱动在 alembic 的同步 engine 下不可用，换成同步驱动
    url = url.replace("+aiosqlite", "").replace("+asyncpg", "+psycopg2")
    return url


# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = _resolve_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,  # SQLite 的 ALTER 需要 batch 模式
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    section = config.get_section(config.config_ini_section, {}) or {}
    section["sqlalchemy.url"] = _resolve_url()
    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
