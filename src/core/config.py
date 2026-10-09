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

    # ---- 数据库 ----
    database_url: str = "sqlite+aiosqlite:///./insightcrew.db"

    # ---- Redis ----
    redis_url: str = "redis://localhost:6379/0"

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

    # ---- 运行 ----
    host: str = "0.0.0.0"
    port: int = 8000
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
