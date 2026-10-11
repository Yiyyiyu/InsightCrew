"""
InsightCrew FastAPI 入口

用法:
  python -m src.main                  # 仅启动 FastAPI
  python -m src.main --worker         # 同时启动 FastAPI + arq Worker
  python -m src.core.worker           # 仅启动 arq Worker
"""

import asyncio
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.routes import router
from src.core.config import settings
from src.core.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    # C4-3：生产模式下密钥不合法直接拒绝启动（开发模式只告警）
    from src.core.config import enforce_secrets

    enforce_secrets()

    logger.info("InsightCrew 启动中 ...")

    # 初始化数据库（Alembic upgrade head / 老库自动认账）
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
        # 不阻止启动：API 仍可提供登录/列表等只读功能。
        # 但任务提交会被 _ensure_executable 拒绝（503），不会静默挂起。
        logger.error(
            f"Redis 不可用 ({e}) — 任务提交将返回 503；"
            f"这条流水线只有一条正常路径：arq 入队 → worker 执行"
        )

    yield

    logger.info("InsightCrew 已关闭")


app = FastAPI(
    title="InsightCrew",
    description="多Agent协作技术调研与选型平台",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS（C4-4）：默认不挂。
# 页面与 API 由同一个服务提供，天然同源，不需要 CORS。原先挂
# `allow_origins=["*"] + allow_credentials=True` 是无效组合——浏览器
# 会拒绝带凭证的通配来源，且会放大 CSRF 面。确实需要跨域时（如把
# 前端部署到别的域）在 .env 里显式列出来源，逗号分隔。
_cors_origins = [o.strip() for o in (settings.cors_origins or "").split(",") if o.strip()]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# 注册路由
app.include_router(router)

# 用户自定义模型（来源 / 模型 / 内置解锁），路径前缀 /models/*
from src.api.model_routes import router as model_router  # noqa: E402

app.include_router(model_router)

# 服务端渲染页面路由（HTML）。放在 JSON API 之后注册：
# 页面路由与 API 路径不重叠（/tasks/* 是页面，/research/* 是 API），
# 但 `/` 只在页面路由里定义，避免两处重复注册。
from src.web.routes import router as web_router  # noqa: E402

app.include_router(web_router)


# ══════════════════════════════════════════════════════════════
# 全局异常处理器（C4-4）
#
# 目标：任何失败都以统一的 JSON 结构返回，且**绝不把堆栈泄露给调用方**。
# 兼容性：保留 `detail` 键（原有前端与验证脚本都读它），只是额外补上
# `code` / `message`，不破坏既有契约。
# ══════════════════════════════════════════════════════════════

def _err(status_code: int, code: str, message: str, detail=None, headers=None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"code": code, "message": message, "detail": detail},
        headers=headers,
    )


@app.exception_handler(StarletteHTTPException)
async def _http_exception_handler(request: Request, exc: StarletteHTTPException):
    """把 FastAPI/Starlette 的 HTTPException 也套进统一信封。

    ★ `detail` 原样透传：路由里大量 `detail=e.as_dict()`（结构化错误，
    含 `code`/`failures`/`roles`），前端据此区分 `builtin_locked` 与
    `model_unreachable`，不能被抹平。

    ★ `exc.headers` 必须转发：限流 429 的 `Retry-After`、鉴权 401 的
    `WWW-Authenticate` 都挂在这里。**重建响应时不带上它们 = 悄悄丢掉
    协议要求的头**（实测踩过：429 的 Retry-After 消失）。
    """
    # detail 本身已是结构化字典时，用它里面的 code/message 更精确
    detail = exc.detail
    code = "http_error"
    message = str(detail)
    if isinstance(detail, dict):
        code = str(detail.get("code") or code)
        message = str(detail.get("message") or detail.get("reason") or message)
    return _err(exc.status_code, code, message, detail, headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request: Request, exc: RequestValidationError):
    """请求体/参数校验失败 → 422（结构化，便于前端定位字段）。"""
    return _err(422, "validation_error", "请求参数不合法", exc.errors())


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    """兜底：未捕获异常 → 500。

    ★ 只把泛化的中文提示返回给调用方；完整堆栈留在服务端日志里。
    `logger.exception` 会带上 traceback，便于排查。
    """
    logger.exception(f"未捕获异常 {request.method} {request.url.path}: {exc}")
    return _err(500, "internal_error", "服务器内部错误，请稍后重试或联系管理员", None)


@app.get("/health")
async def health():
    """存活探针：只回答"这个进程还活着吗"，不做依赖检查。

    ★ 不要拿它判断系统可用性：它永远返回 ok，即使 worker 根本没起。
    要判断能否提交任务请用 /healthz。
    """
    return {"status": "ok", "service": "insightcrew"}


@app.get("/healthz")
async def healthz(models: int = 0):
    """就绪探针：逐项检查任务执行所需的依赖。

    返回 200 表示"可以提交任务"；任一依赖不可用返回 503。
    检查项：数据库 / Redis / arq worker 心跳 / 队列积压深度。

    ``?models=1`` 时**额外**探测内置模型连通性（C4-6，FR-9.3）。
    默认关闭：探测会真实发起网络请求（虽不耗 token，但每次数秒），
    监控探针每 30 秒打一次没必要。模型探测结果只作诊断信息，
    **不参与 200/503 判定** —— 内置模型不通不代表服务不可用，
    用户完全可以用自己的 BYOK 模型跑任务。
    """
    from fastapi.responses import JSONResponse
    from sqlalchemy import text

    result: dict = {"service": "insightcrew", "checks": {}}
    healthy = True

    # 1) 数据库
    try:
        from src.db.database import async_session
        async with async_session() as session:
            await session.execute(text("SELECT 1"))
        result["checks"]["db"] = "ok"
    except Exception as e:
        result["checks"]["db"] = f"error: {e}"
        healthy = False

    # 2) Redis
    redis_ok = False
    _r = None
    try:
        import redis.asyncio as aioredis
        _r = aioredis.from_url(settings.redis_url, decode_responses=True, socket_connect_timeout=3)
        await _r.ping()
        result["checks"]["redis"] = "ok"
        redis_ok = True
    except Exception as e:
        result["checks"]["redis"] = f"error: {e}"
        healthy = False

    # 3) worker 心跳 + 队列深度
    if redis_ok:
        try:
            from arq.connections import RedisSettings
            from arq.worker import async_check_health
            rc = await async_check_health(RedisSettings.from_dsn(settings.redis_url))
            if rc == 0:
                result["checks"]["worker"] = "ok"
            else:
                result["checks"]["worker"] = "no health check sentinel found (worker 未运行?)"
                healthy = False
        except Exception as e:
            result["checks"]["worker"] = f"error: {e}"
            healthy = False

        try:
            result["checks"]["queue_depth"] = await _r.zcard("arq:queue")
        except Exception as e:
            result["checks"]["queue_depth"] = f"error: {e}"

    # 4) 模型连通性（可选，C4-6）
    if models:
        result["checks"]["models"] = await _probe_builtin_models()

    await _close_redis(_r)

    result["status"] = "ok" if healthy else "unavailable"
    return JSONResponse(status_code=200 if healthy else 503, content=result)


async def _probe_builtin_models() -> dict:
    """探测内置模型目录的连通性，返回 ``{模型: "ok" | "error: ..."}``。

    只探内置档（平台侧配置，与具体用户无关）；用户自建模型不进探针——
    那是用户自己的配置问题，且探针没有用户上下文。异常一律吞掉转成
    诊断字符串，绝不让 /healthz 因模型探测失败而报错。
    """
    try:
        from src.core.model_registry import builtin_options
        from src.core.preflight import build_plan_for_user, preflight

        opts = builtin_options()
        plan = build_plan_for_user(None, opts)
        results = await preflight(plan, opts)
        out: dict[str, str] = {}
        for r in results:
            out[r.model_id or r.option_id] = "ok" if r.ok else f"error: {r.error or r.status}"
        return out or {"(none)": "no builtin options"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


async def _close_redis(client):
    """尽力关闭临时 Redis 连接，失败不影响探针结果。"""
    if client is None:
        return
    try:
        await client.aclose()
    except Exception:
        pass


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
