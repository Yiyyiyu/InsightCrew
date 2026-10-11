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

# 显式导入 config：API Key 是 config 模块在导入时通过 _load_env_into_environ()
# 注入 os.environ 的（pydantic-settings 只把 .env 读进 Settings 对象，而
# litellm/crewai 从 os.environ 取 Key）。若依赖"别的模块先 import 了 config"，
# 单独 import model_router 时 route.api_key 会静默变成 "no-key-configured"，
# 表现为真实调用时报 AuthenticationError。这里不依赖导入顺序。
from src.core import config as _config  # noqa: F401
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
    name: str = ""          # 档位键名, 如 vision / vision_fallback
    disable_thinking: bool = False   # 是否关闭思考模式（见下）
    # ★ disable_thinking 的必要性（实测踩坑）：智谱的思考型视觉模型
    #   （如 glm-4.6v-flash）会把 max_tokens 全部耗在 reasoning_tokens 上，
    #   导致 content 返回空串——表现为"调用成功但读不到内容"。
    #   传 extra_body={"thinking":{"type":"disabled"}} 可关掉，实测有效
    #   （输出从 64 token 降到 20，内容正常）。对非思考模型无害。


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

    def route(
        self,
        agent: str = "",
        modality: str = "text",
        task_kind: str = "",
    ) -> ModelRoute:
        """无任务上下文时的兜底路由（**开发/测试专用**）。

        ★ M2.5 起生产链路不再走这里：模型由用户在新建任务时逐角色选定
        （``model_plan`` → ``route_from_option``）。本方法只在「没有任务
        上下文」时被开发脚本与测试调用，返回**内置默认目录**里该角色的
        模型，与 ``llm_fallback._fallback_option`` 语义一致。

        ``task_kind`` 形参保留仅为兼容旧调用方签名，已无作用。
        """
        from src.core.model_registry import (
            apply_role_overrides,
            builtin_default_plan,
            builtin_options,
            find_option,
        )

        plan = builtin_default_plan()
        key = "vision" if modality == "vision" else agent
        opt = find_option(builtin_options(), plan.get(key, ""))
        if opt is None:
            # 极端情况（内置目录为空 / 角色缺失）：回落 default_model 档位
            return self._build_route(self._config.get("default_model", "main"), agent)
        return self.route_from_option(apply_role_overrides(opt, key), agent)

    def route_named(self, model_name: str, agent: str = "") -> ModelRoute:
        """按 models 里的键名直接取档位（不走用户 model_plan）。

        用途：显式指定备选档位，如读图失败后回落 ``vision_fallback``。
        键名不存在时回落到 default_model，避免调用方拿到 None。
        """
        models = self._config.get("models", {}) or {}
        if model_name not in models:
            logger.warning(f"未知模型档位 {model_name!r}，回落到默认档位")
            model_name = self._config.get("default_model", "main")
        return self._build_route(model_name, agent)

    def fallback_models(self, model_name: str) -> list[str]:
        """返回某档位失败后应依次尝试的降级档位（不含自身）。

        优先级：
        1. 该档位自己的 ``fallback:`` 字段（精确指定，如 vision → vision_fallback）
        2. 全局 ``fallback_chain`` 中排在其后的档位

        注意：不能把全局 fallback_chain 直接套到 vision 上——链上是
        [main, premium] 这类**文本**档位，读图失败后回落到文本模型毫无意义。
        """
        models = self._config.get("models", {}) or {}

        own = (models.get(model_name, {}) or {}).get("fallback")
        if own:
            return [str(own)] if str(own) in models else []

        chain = [str(m) for m in (self._config.get("fallback_chain", []) or [])]
        if model_name in chain:
            return [m for m in chain[chain.index(model_name) + 1:] if m in models]
        return []

    def _build_route(self, model_name: str, agent: str = "") -> ModelRoute:
        """按档位名构造 ModelRoute（含 per-Agent overrides）。"""
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

        # 应用 per-Agent overrides（max_tokens / temperature）
        # 注意：overrides 里的 api_key_env / litellm 不支持覆盖，只覆盖生成参数
        max_tokens = model_def.get("max_tokens", 4096)
        temperature = model_def.get("temperature", 0.3)
        override = self._config.get("overrides", {}).get(agent, {})
        if "max_tokens" in override:
            max_tokens = override["max_tokens"]
        if "temperature" in override:
            temperature = override["temperature"]

        return ModelRoute(
            model_id=litellm,
            provider=provider,
            api_key=api_key or "no-key-configured",
            api_base=api_base,
            max_tokens=max_tokens,
            temperature=temperature,
            name=model_name,
            disable_thinking=bool(model_def.get("disable_thinking", False)),
        )

    def route_from_option(self, option, role: str = "") -> ModelRoute:
        """把用户选定的 ``ModelOption``（内置或自建）转成 ``ModelRoute``。

        M2.5 起模型不再按 main/reasoner/premium 档位名选择，而是用户在新建
        任务时**逐角色选定一个 ModelOption**。本函数只做「数据结构转换 +
        mock 覆盖」，不查库、不做降级——降级已由
        preflight 连通性预检取代（见 ``src/core/preflight.py``）。

        ★ per-Agent 生成参数覆盖（Writer 需要更长输出等）由
          ``model_registry.apply_role_overrides`` 在调用本函数**之前**施加，
          这里不重复应用，避免温度/上限被叠加两次。
        """
        model_id = str(getattr(option, "model_id", "") or "")
        provider = getattr(option, "provider", "") or (
            model_id.split("/")[0] if "/" in model_id else "openai"
        )
        api_key = str(getattr(option, "api_key", "") or "")
        api_base = getattr(option, "api_base", None)

        # Mock 模式下所有调用都指向本机假服务，与档位无关
        if self._use_mock:
            api_base = "http://127.0.0.1:8001/v1"
            api_key = "mock-key"

        return ModelRoute(
            model_id=model_id,
            provider=provider,
            api_key=api_key or "no-key-configured",
            api_base=api_base,
            max_tokens=int(getattr(option, "max_tokens", 4096) or 4096),
            temperature=float(getattr(option, "temperature", 0.3) or 0.3),
            name=str(getattr(option, "id", "") or ""),
            disable_thinking=bool(getattr(option, "disable_thinking", False)),
        )

    # ── 内部方法 ──

    def pricing_table(self) -> dict:
        """计费单价表（元/百万 token），键为 models 里的模型名。"""
        return self._config.get("pricing", {}) or {}

    def pricing_key_for_model(self, model_id: str) -> str:
        """按 litellm 模型串（如 deepseek/deepseek-flash）反查 models 里的键名。

        查不到时原样返回，由调用方决定记 0 成本。

        ★ 必须做前缀归一（实测踩坑）：CrewAI 的 ``LLM(model=...)`` 会把
        provider 前缀剥掉——配置里是 ``deepseek/deepseek-flash``，而
        ``crew.agents[0].llm.model`` 是 ``deepseek-flash``。若只做全等比较，
        真实模式下每次 crew 记账都会查不到单价、静默记 ¥0
        （cost_of 查不到时按 0 处理），账单和熔断就全失效了。
        """
        target = (model_id or "").strip()
        if not target:
            return target

        def _bare(s: str) -> str:
            """去掉 provider 前缀：'zai/glm-4v-plus' -> 'glm-4v-plus'。"""
            s = (s or "").strip()
            return s.split("/", 1)[1] if "/" in s else s

        target_bare = _bare(target)
        for name, model_def in (self._config.get("models", {}) or {}).items():
            declared = str(model_def.get("litellm", "") or "")
            if declared == target:
                return name
            # 前缀无关匹配（deepseek/deepseek-flash == deepseek-flash）
            if declared and _bare(declared) == target_bare:
                return name
        return target

    def budget_config(self) -> dict:
        """预算配置（per_task_limit_cny / on_exceed）。"""
        return self._config.get("budget", {}) or {}

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
            "default_model": "main",
            "models": {
                "main": {"litellm": "deepseek/deepseek-flash", "max_tokens": 4096, "temperature": 0.3},
            },
            "fallback_chain": ["main"],
            "budget": {"per_task_limit_cny": 5.0, "on_exceed": "abort"},
            "pricing": {},
            "overrides": {},
        }

    def _detect_mock_mode(self) -> bool:
        """检测是否应使用 Mock LLM

        ★ 未设置时默认 **关闭**（打真实模型）。

        早期默认开启，理由是"开发方便"，但后果是：`.env` 里没写
        `MOCK_LLM` 的部署会把所有 LLM 调用发往本机 8001 的 mock 服务，
        而那个服务通常没起 —— 表现为 `Failed to connect to OpenAI API:
        Connection error`，排查时完全看不出是配置问题。
        生产语义必须是"用真实模型"，mock 是显式选择的开发模式。
        """
        # 显式环境变量优先
        mock_env = os.environ.get("MOCK_LLM", "").lower()
        if mock_env in ("1", "true", "yes"):
            return True
        if mock_env in ("0", "false", "no"):
            return False
        # 未设置时默认使用真实模型
        return False

# 全局单例
router = ModelRouter()
