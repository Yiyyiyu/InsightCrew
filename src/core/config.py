"""配置管理"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_env_into_environ() -> None:
    """把 .env 的值注入 os.environ（已存在的真实环境变量优先，不覆盖）。

    必要性：pydantic-settings 只把 .env 读进 Settings 对象，
    而 model_router / litellm / crewai 是通过 os.environ 取 API Key 的。
    不注入会导致真实模式下 api_key 解析为空 → "no-key-configured"。
    """
    env_path = _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.split("#")[0].strip()   # 去掉行尾注释
            if key and value:
                os.environ.setdefault(key, value)


_load_env_into_environ()


class Settings(BaseSettings):
    # ---- 项目路径 ----
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    config_dir: Path = project_root / "config"

    # ---- 运行环境（C4-3）----
    # "development"（默认）/ "production"。
    # 生产模式下启动时会校验密钥是否仍为示例值，是则拒绝启动——
    # 见 `validate_secrets()`。用显式环境变量而不是猜，避免开发机
    # 误配一个假 JWT_SECRET 就把自己锁在门外。
    app_env: str = "development"

    # ---- 数据库 ----
    database_url: str = "sqlite+aiosqlite:///./insightcrew.db"

    # ---- Redis ----
    redis_url: str = "redis://localhost:6379/0"
    # 命令/发布用连接池上限
    redis_max_connections: int = 16
    # SSE 订阅用连接池上限：每个在线 SSE 客户端会长期独占一个订阅连接，
    # 必须与命令池分开，否则几个客户端就把池耗尽（曾导致 Too many connections
    # 后静默退回内存通道，跨进程事件全部丢失）。
    redis_sub_max_connections: int = 64

    # ---- 鉴权 ----
    jwt_secret: str = "dev-secret-change-in-production"
    jwt_access_ttl_min: int = 30
    jwt_refresh_ttl_days: int = 7

    # ---- 模型 ----
    deepseek_api_key: str = ""
    zhipu_api_key: str = ""
    openai_api_key: str = ""
    openai_api_base: str = ""
    dashscope_api_key: str = ""

    # ---- M2.5 用户自定义模型 ----
    # 用户 API Key 的加密主密钥（Fernet，44 字节 base64）。
    # 生成：python -c "from cryptography.fernet import Fernet;print(Fernet.generate_key().decode())"
    # 留空时自动退化为进程内临时密钥（开发够用；重启后旧密文解不开，生产必须配）。
    fernet_key: str = ""
    # 内置 5 个模型的解锁密码。**不入仓库**（.env 里设）。
    # 所有用户含站长默认锁上，各自输入该密码解锁；未解锁看不到内置模型。
    builtin_unlock_password: str = ""
    # 解锁有效期（小时）
    builtin_unlock_ttl_hours: int = 12

    # ---- 运行 ----
    host: str = "0.0.0.0"
    port: int = 8000
    # 允许跨域的来源，逗号分隔（C4-4）。
    # 默认空 = 不挂 CORS 中间件。页面与 API 同源提供，本来就不需要 CORS；
    # 挂 `allow_origins=["*"] + allow_credentials=True` 是无效组合
    # （浏览器会拒绝带凭证的 `*`），且会放大 CSRF 面。
    cors_origins: str = ""
    # 是否信任 X-Forwarded-For / X-Real-IP 反代头（C4-1）。
    # 默认 False：直连部署时这些头可被客户端伪造，信了限流就失效。
    # 部署到 Nginx 反代之后设 TRUST_PROXY=1，否则所有请求的 IP 都是
    # Nginx 容器地址，限流会互相误伤。
    trust_proxy: bool = False
    # 是否启用接口限流（C4-1）。
    # 默认 True（生产恒为 True，不允许关）。本机开发时所有请求都来自
    # 127.0.0.1、共用一个计数器，跑 30 个验证脚本会互相误伤，故本地 `.env`
    # 里设 RATE_LIMIT_ENABLED=0 关掉。只影响限流，不影响其它安全检查。
    rate_limit_enabled: bool = True
    worker_concurrency: int = 3
    max_parallel_researchers: int = 3
    task_budget_limit_cny: float = 5.0

    # ---- 超时（硬约束，防止任务无限挂起）----
    # 单个阶段的墙钟上限：超时后阶段标记失败并写事件，不再阻塞后续流程
    stage_timeout_sec: int = 300
    # 单次 LLM 请求上限：传给 litellm，避免 HTTP 连接永久挂起
    llm_timeout_sec: int = 120

    # ---- 日志 ----
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()


#: 示例/占位密钥集合。生产模式下命中任一项即拒绝启动（C4-3）。
_WEAK_SECRETS = {
    "",
    "dev-secret-change-in-production",
    "change_me_to_random_64_chars",
    "change_me_builtin_password",
    "secret",
    "changeme",
}


def is_production() -> bool:
    """当前是否为生产环境（大小写与空白不敏感）。"""
    return (settings.app_env or "").strip().lower() in ("production", "prod")


def rate_limit_on() -> bool:
    """接口限流是否生效。

    ★ 生产恒为 True —— 限流是上线必需项，不允许被环境变量静默关掉。
    开发模式看 ``RATE_LIMIT_ENABLED``（默认 True，安全默认）。
    本机跑验证脚本时在 `.env` 里设 0，否则 30 个脚本共用一个 127.0.0.1
    计数器会互相误伤。
    """
    if is_production():
        return True
    return bool(settings.rate_limit_enabled)


def validate_secrets(*, strict: bool | None = None) -> list[str]:
    """校验关键密钥；返回问题清单（空表示通过）。

    仅在**生产模式**下做强制校验（``strict=None`` 时自动取 ``is_production()``）；
    开发模式一律放行——否则本机开发者会为了跑起来去改 `.env`，
    反而更容易把真实密钥写进错误的地方。

    校验项：
      * ``JWT_SECRET`` 不能是示例值/空值（否则任何人都能伪造 token）；
      * ``FERNET_KEY`` 若留空只告警不拦截——它会退化成从 JWT_SECRET
        派生的进程内临时密钥，重启后旧密文解不开（用户自建模型 Key 失效），
        但这不影响"能不能启动"。
    """
    strict = is_production() if strict is None else strict
    problems: list[str] = []

    if (settings.jwt_secret or "").strip() in _WEAK_SECRETS:
        problems.append(
            "JWT_SECRET 仍是示例值或为空 —— 任何人都能伪造登录令牌。"
            '生成命令：python -c "import secrets;print(secrets.token_urlsafe(64))"'
        )

    if strict and not (settings.fernet_key or "").strip():
        problems.append(
            "FERNET_KEY 未配置 —— 用户自建模型的 API Key 将用进程内临时密钥加密，"
            "进程重启后旧密文无法解密。"
            '生成命令：python -c "from cryptography.fernet import Fernet;print(Fernet.generate_key().decode())"'
        )

    return problems


def enforce_secrets() -> None:
    """生产模式下若有密钥问题则拒绝启动。

    由 ``src/main.py`` 的 ``lifespan`` 最早期调用；开发模式下只把问题
    打到日志（warning），不影响启动。
    """
    problems = validate_secrets()
    if not problems:
        return

    from src.core.logger import logger

    header = "生产模式密钥校验失败" if is_production() else "密钥校验告警（开发模式放行）"
    body = "\n  - ".join([""] + problems)
    if is_production():
        raise RuntimeError(f"{header}：{body}\n修复 .env 后重启。")
    logger.warning(f"{header}：{body}")
