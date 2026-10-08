"""
InsightCrew FastAPI 入口

用法:
  python -m src.main                  # 仅启动 FastAPI
  python -m src.main --worker         # 同时启动 FastAPI + arq Worker
  python -m src.core.worker           # 仅启动 arq Worker
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

from src.api.routes import router
from src.core.config import settings
from src.core.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    logger.info("InsightCrew 启动中 ...")

    # 初始化数据库（建表）
    from src.db.database import init_db
    await init_db()
    logger.info("数据库表已就绪")

    # 尝试连接 Redis（不阻塞启动）
    try:
        import redis.asyncio as aioredis
        _r = aioredis.from_url(settings.redis_url, decode_responses=True, socket_connect_timeout=3)
        await _r.ping()
        await _r.aclose()
        logger.info("Redis 连接正常")
    except Exception as e:
        logger.warning(f"Redis 未就绪 ({e}) — arq 降级为 asyncio.create_task fallback")

    yield

    logger.info("InsightCrew 已关闭")


app = FastAPI(
    title="InsightCrew",
    description="多Agent协作技术调研与选型平台",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router)

# 模板
_template_dir = os.path.join(os.path.dirname(__file__), "web", "templates")
templates = Jinja2Templates(directory=_template_dir)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "insightcrew"}


@app.get("/")
async def index():
    """首页"""
    from fastapi.responses import HTMLResponse
    html = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<title>InsightCrew</title><link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/pico.css@2/css/pico.min.css">
</head><body><main class="container"><h1>InsightCrew</h1>
<p>多 Agent 协作技术调研与选型平台</p>
<p><a href="/docs" role="button">API 文档</a></p></main></body></html>"""
    return HTMLResponse(html)


async def _start_worker():
    """启动 arq Worker"""
    from arq import run_worker
    from src.core.worker import WorkerSettings
    logger.info("arq Worker 启动")
    # run_worker 是同步阻塞的，在 async 上下文中用 asyncio.to_thread 隔离
    await asyncio.to_thread(run_worker, WorkerSettings)


def run():
    """CLI 入口：python -m src.main [--worker]"""
    if "--worker" in sys.argv:
        async def _both():
            import uvicorn
            await asyncio.gather(
                uvicorn.Server(uvicorn.Config(app, host=settings.host, port=settings.port, log_level="info")).serve(),
                _start_worker(),
            )
        asyncio.run(_both())
    else:
        import uvicorn
        uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    run()
