"""配置管理"""

from pathlib import Path
from pydantic_settings import BaseSettings


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

    # ---- 日志 ----
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
