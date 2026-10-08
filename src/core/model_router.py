"""
模型路由层 — 根据配置选择模型并调用

职责：
1. 读取 config/models.yaml 路由规则
2. 按 Agent/模态/任务类型 选择模型
3. 创建 LiteLLM 兼容的调用参数
4. dev 模式下自动指向 Mock LLM (127.0.0.1:8001)
"""

import os
import yaml
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from src.core.logger import logger


@dataclass
class ModelRoute:
    """一次路由决策的结果"""
    model_id: str           # LiteLLM 模型串, 如 deepseek/deepseek-flash
    provider: str           # 厂商名, 如 deepseek
    api_key: str            # API Key
    api_base: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.3


class ModelRouter:
    """模型路由器 — 单例"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._config = self._load_config()
        self._use_mock = self._detect_mock_mode()
        logger.info(f"ModelRouter 初始化完成 (mock={'开启' if self._use_mock else '关闭'})")

    # ── 公共接口 ──

    def route(self, agent: str = "", modality: str = "text", task_kind: str = "") -> ModelRoute:
        """
        根据规则路由到合适的模型

        参数:
            agent:      Agent 角色名 (Planner / Researcher / Analyst / Critic / Writer / Reviewer)
            modality:   模态 (text / vision)
            task_kind:  任务类型 (normal / long_context)

        返回:
            ModelRoute 对象
        """
        model_name = self._resolve_model(agent, modality, task_kind)
        model_def = self._config["models"][model_name]

        # 解析 provider
        litellm = model_def["litellm"]
        provider = litellm.split("/")[0] if "/" in litellm else "openai"

        # 获取 API Key
        api_key_env = model_def.get("api_key_env", "")
        api_key = os.environ.get(api_key_env, "")

        # Mock 模式下覆盖 api_base
        api_base = model_def.get("api_base")
        if self._use_mock:
            api_base = "http://127.0.0.1:8001/v1"
            api_key = "mock-key"

        return ModelRoute(
            model_id=litellm,
            provider=provider,
            api_key=api_key or "no-key-configured",
            api_base=api_base,
            max_tokens=model_def.get("max_tokens", 4096),
            temperature=model_def.get("temperature", 0.3),
        )

    def create_completion_args(self, agent: str = "", modality: str = "text", task_kind: str = "") -> dict:
        """生成传给 LiteLLM completion() 的参数 dict"""
        route = self.route(agent, modality, task_kind)
        args = {
            "model": route.model_id,
            "api_key": route.api_key,
            "max_tokens": route.max_tokens,
            "temperature": route.temperature,
        }
        if route.api_base:
            args["api_base"] = route.api_base
        return args

    # ── 内部方法 ──

    def _load_config(self) -> dict:
        """加载 config/models.yaml"""
        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "models.yaml"
        if not config_path.exists():
            logger.warning(f"模型配置文件不存在: {config_path}，使用默认配置")
            return self._default_config()
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _default_config(self) -> dict:
        return {
            "strategy": "heterogeneous",
            "default_model": "main",
            "models": {
                "main": {"litellm": "deepseek/deepseek-flash", "max_tokens": 4096, "temperature": 0.3},
            },
            "routing": [],
            "fallback_chain": ["main"],
            "budget": {"per_task_limit_cny": 5.0, "on_exceed": "abort"},
            "overrides": {},
        }

    def _detect_mock_mode(self) -> bool:
        """检测是否应使用 Mock LLM"""
        # 显式环境变量优先
        mock_env = os.environ.get("MOCK_LLM", "").lower()
        if mock_env in ("1", "true", "yes"):
            return True
        if mock_env in ("0", "false", "no"):
            return False
        # 未设置时默认开启 mock（开发模式）
        return True

    def _resolve_model(self, agent: str, modality: str, task_kind: str) -> str:
        """按路由规则解析目标模型名称"""
        rules = self._config.get("routing", [])
        overrides = self._config.get("overrides", {})

        # 1. 模态优先 (vision → vision 模型)
        if modality == "vision":
            for rule in rules:
                if rule.get("when", {}).get("modality") == "vision":
                    return rule["use"]

        # 2. 按 Agent 角色
        if agent:
            for rule in rules:
                when_agent = rule.get("when", {}).get("agent", [])
                if isinstance(when_agent, list) and agent in when_agent:
                    return rule["use"]
                if isinstance(when_agent, str) and agent == when_agent:
                    return rule["use"]

        # 3. 按任务类型
        if task_kind:
            for rule in rules:
                if rule.get("when", {}).get("task_kind") == task_kind:
                    return rule["use"]

        # 4. 默认
        return self._config.get("default_model", "main")


# 全局单例
router = ModelRouter()


# ── 便捷函数 ──

def route_for(agent: str = "", modality: str = "text", task_kind: str = "") -> ModelRoute:
    """快捷路由"""
    return router.route(agent, modality, task_kind)


def completion_args_for(agent: str = "", modality: str = "text", task_kind: str = "") -> dict:
    """快捷生成 completion 参数"""
    return router.create_completion_args(agent, modality, task_kind)
