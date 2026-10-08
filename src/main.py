"""InsightCrew FastAPI 入口"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

from src.api.routes import router
from src.core.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    logger.info("InsightCrew 启动中 ...")

    # 初始化数据库（建表）
    from src.db.database import init_db
    await init_db()
    logger.info("数据库表已就绪")

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

# 模板（用绝对路径避免工作目录问题）
import os
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
