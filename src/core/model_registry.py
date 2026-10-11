"""统一模型目录 —— 把「内置」与「用户自建」合成一张只读视图。

设计（用户 m12784 / m13063 / m13103 / m13118）：

    ┌─ 内置 5 个：config/builtin_models.yaml，**只读、无 API 可改**，
    │             用站长 .env 的 Key（api_key_env）。所有用户含站长
    │             默认锁上，必须各自输入密码解锁后才出现在目录里。
    └─ 用户自建：providers + models 两张表，Key 是 Fernet 密文。

    两边合成同一个 ``ModelOption``，供路由、前端、preflight、计价共用。

**id 用字符串前缀而不是裸整数**：内置没有数据库主键。前缀
（``builtin:deepseek-flash`` / ``user:7``）让 ``Task.model_plan`` 一个字段
同时容纳两种来源，且天然无歧义。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import yaml
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from src.core import config as _config  # noqa: F401  （注入 os.environ 的 Key）
from src.core.config import settings
from src.core.crypto import decrypt_key
from src.core.logger import logger
from src.db.models import BuiltinUnlock, Provider, UserModel

BUILTIN_PREFIX = "builtin:"
USER_PREFIX = "user:"

# 6 个 Agent 角色（顺序即前端表单顺序）+ 读图档
AGENT_ROLES = ["Planner", "Researcher", "Analyst", "Critic", "Writer", "Reviewer"]
VISION_ROLE = "vision"
PLAN_ROLES = AGENT_ROLES + [VISION_ROLE]

_BUILTIN_CATALOG: Optional[dict] = None


@dataclass
class ModelOption:
    """一个可被选中的模型（内置或用户自建）。"""

    id: str                     # "builtin:deepseek-flash" / "user:7"
    source: str                 # "builtin" | "user"
    display_name: str
    model_id: str               # 实际发给 API 的模型串（含 litellm 前缀）
    api_key: str                # 已解出的明文 Key（仅内存中短暂存在）
    api_base: Optional[str] = None
    supports_vision: bool = False
    max_tokens: int = 4096
    temperature: float = 0.3
    disable_thinking: bool = False
    price_in: float = 0.0       # ¥/百万 token；0 = 只记 token 不计费
    price_out: float = 0.0
    fallback: str = ""          # 仅内置：读图失败时回落的 option id
    provider_name: str = ""     # 仅自建：来源名（前端展示用）
    # 生成参数覆盖（按角色生效，如 Writer max_tokens=8000）
    extra: dict = field(default_factory=dict)

    @property
    def provider(self) -> str:
        """LiteLLM 的 provider 前缀，如 deepseek / zai / openai。"""
        return self.model_id.split("/", 1)[0] if "/" in self.model_id else "openai"


# ══════════════════════════════════════════════════════════════
# 内置目录（只读配置）
# ══════════════════════════════════════════════════════════════

def _builtin_path():
    return settings.config_dir / "builtin_models.yaml"


def load_builtin_catalog(force: bool = False) -> dict:
    """读取 config/builtin_models.yaml（进程内缓存）。

    返回 ``{"models": {...}, "default_plan": {...}, "overrides": {...}}``。
    文件缺失时返回空目录并告警——内置模型只是「可选」的，不该让整个应用起不来。
    """
    global _BUILTIN_CATALOG
    if _BUILTIN_CATALOG is not None and not force:
        return _BUILTIN_CATALOG

    path = _builtin_path()
    if not path.exists():
        logger.warning(f"内置模型配置不存在: {path}，内置模型将不可用")
        _BUILTIN_CATALOG = {"models": {}, "default_plan": {}, "overrides": {}}
        return _BUILTIN_CATALOG

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    _BUILTIN_CATALOG = {
        "models": raw.get("builtin_models", {}) or {},
        "default_plan": raw.get("default_plan", {}) or {},
        "overrides": raw.get("overrides", {}) or {},
    }
    return _BUILTIN_CATALOG


def _builtin_option(key: str, raw: dict) -> ModelOption:
    """把一个内置配置项变成 ModelOption（解析 Key 与 api_base）。"""
    # api_base：显式值优先，其次从环境变量取（如 OPENAI_API_BASE）
    api_base = raw.get("api_base") or None
    base_env = raw.get("api_base_env")
    if base_env:
        api_base = os.environ.get(base_env, "") or api_base

    api_key = os.environ.get(raw.get("api_key_env", ""), "") or ""
    if not api_key:
        logger.debug(f"内置模型 {key} 的 Key 未配置（env={raw.get('api_key_env')}）")

    return ModelOption(
        id=f"{BUILTIN_PREFIX}{key}",
        source="builtin",
        display_name=raw.get("display_name") or key,
        model_id=raw.get("litellm", key),
        api_key=api_key,
        api_base=api_base,
        supports_vision=bool(raw.get("supports_vision", False)),
        max_tokens=int(raw.get("max_tokens", 4096)),
        temperature=float(raw.get("temperature", 0.3)),
        disable_thinking=bool(raw.get("disable_thinking", False)),
        price_in=float(raw.get("price_in", 0.0) or 0.0),
        price_out=float(raw.get("price_out", 0.0) or 0.0),
        fallback=raw.get("fallback") or "",
    )


def builtin_options() -> list[ModelOption]:
    """全部内置模型（**不做解锁过滤**，仅供内部/测试使用）。"""
    cat = load_builtin_catalog()
    return [_builtin_option(k, v) for k, v in cat["models"].items()]


def builtin_default_plan() -> dict:
    """内置的默认角色分配（解锁后表单初始值），已补 ``builtin:`` 前缀。"""
    plan = load_builtin_catalog()["default_plan"] or {}
    out = {}
    for role, key in plan.items():
        out[role] = key if str(key).startswith((BUILTIN_PREFIX, USER_PREFIX)) else f"{BUILTIN_PREFIX}{key}"
    return out


def builtin_overrides() -> dict:
    return load_builtin_catalog()["overrides"] or {}


# ══════════════════════════════════════════════════════════════
# 用户自建目录（数据库）
# ══════════════════════════════════════════════════════════════

def _user_option(row: UserModel, provider: Provider) -> ModelOption:
    """把 models 行 + providers 行合成 ModelOption（解密 Key）。"""
    try:
        api_key = decrypt_key(provider.api_key_encrypted)
    except ValueError as exc:
        # 单条解密失败不能拖垮整个目录：标记成空 Key，由 preflight 明确报错
        logger.warning(f"来源 #{provider.id} 的 Key 解密失败：{exc}")
        api_key = ""

    return ModelOption(
        id=f"{USER_PREFIX}{row.id}",
        source="user",
        display_name=row.display_name or row.model_id,
        model_id=row.model_id,
        api_key=api_key,
        api_base=provider.api_base or None,
        supports_vision=bool(row.supports_vision),
        max_tokens=int(row.max_tokens or 4096),
        temperature=float(row.temperature if row.temperature is not None else 0.3),
        disable_thinking=bool(row.disable_thinking),
        price_in=float(row.price_in or 0.0),
        price_out=float(row.price_out or 0.0),
        provider_name=provider.name or "",
    )


def user_options_sync(session: Session, user_id: int) -> list[ModelOption]:
    """同步版：读某用户全部自建模型。"""
    if not user_id:
        return []
    rows = session.execute(
        select(UserModel, Provider)
        .join(Provider, UserModel.provider_id == Provider.id)
        .where(UserModel.user_id == user_id, Provider.user_id == user_id)
        .order_by(UserModel.id)
    ).all()
    return [_user_option(m, p) for m, p in rows]


async def user_options_async(session, user_id: int) -> list[ModelOption]:
    """异步版：读某用户全部自建模型。"""
    if not user_id:
        return []
    rows = (await session.execute(
        select(UserModel, Provider)
        .join(Provider, UserModel.provider_id == Provider.id)
        .where(UserModel.user_id == user_id, Provider.user_id == user_id)
        .order_by(UserModel.id)
    )).all()
    return [_user_option(m, p) for m, p in rows]


# ══════════════════════════════════════════════════════════════
# 解锁态（内置模型的门禁）
# ══════════════════════════════════════════════════════════════

def unlock_ttl() -> timedelta:
    return timedelta(hours=int(settings.builtin_unlock_ttl_hours or 12))


def _unlock_valid(row: Optional[BuiltinUnlock]) -> bool:
    return bool(row and row.expires_at and row.expires_at > datetime.utcnow())


def is_unlocked_sync(session: Session, user_id: int) -> bool:
    """同步版解锁检查。"""
    if not user_id:
        return False
    row = session.get(BuiltinUnlock, user_id)
    return _unlock_valid(row)


async def is_unlocked_async(session, user_id: int) -> bool:
    """异步版解锁检查。"""
    if not user_id:
        return False
    row = await session.get(BuiltinUnlock, user_id)
    return _unlock_valid(row)


def grant_unlock_sync(session: Session, user_id: int) -> datetime:
    """写入/刷新解锁态，返回过期时间。"""
    expires = datetime.utcnow() + unlock_ttl()
    row = session.get(BuiltinUnlock, user_id)
    if row is None:
        row = BuiltinUnlock(user_id=user_id, unlocked_at=datetime.utcnow(), expires_at=expires)
        session.add(row)
    else:
        row.unlocked_at = datetime.utcnow()
        row.expires_at = expires
    return expires


def revoke_unlock_sync(session: Session, user_id: int) -> None:
    """撤销解锁（幂等）。"""
    session.execute(delete(BuiltinUnlock).where(BuiltinUnlock.user_id == user_id))


async def grant_unlock_async(session, user_id: int) -> datetime:
    """异步版写入/刷新解锁态，返回过期时间。"""
    expires = datetime.utcnow() + unlock_ttl()
    row = await session.get(BuiltinUnlock, user_id)
    if row is None:
        row = BuiltinUnlock(user_id=user_id, unlocked_at=datetime.utcnow(), expires_at=expires)
        session.add(row)
    else:
        row.unlocked_at = datetime.utcnow()
        row.expires_at = expires
    return expires


async def revoke_unlock_async(session, user_id: int) -> None:
    """异步版撤销解锁（幂等）。"""
    await session.execute(delete(BuiltinUnlock).where(BuiltinUnlock.user_id == user_id))


# ══════════════════════════════════════════════════════════════
# 统一目录
# ══════════════════════════════════════════════════════════════

def _assemble(session_unlocked: bool, user_opts: list[ModelOption]) -> list[ModelOption]:
    """内置（按解锁过滤）+ 自建，合成一张目录。"""
    opts: list[ModelOption] = []
    if session_unlocked:
        opts.extend(builtin_options())
    opts.extend(user_opts)
    return opts


def catalog_sync(session: Session, user_id: int) -> list[ModelOption]:
    """同步版统一目录（未解锁用户看不到内置 5 个）。"""
    unlocked = is_unlocked_sync(session, user_id)
    return _assemble(unlocked, user_options_sync(session, user_id))


async def catalog_async(session, user_id: int) -> list[ModelOption]:
    """异步版统一目录（未解锁用户看不到内置 5 个）。"""
    unlocked = await is_unlocked_async(session, user_id)
    return _assemble(unlocked, await user_options_async(session, user_id))


def find_option(options: list[ModelOption], option_id: str) -> Optional[ModelOption]:
    """在目录里按 id 找一个。"""
    target = (option_id or "").strip()
    for opt in options:
        if opt.id == target:
            return opt
    return None


def apply_role_overrides(option: ModelOption, role: str) -> ModelOption:
    """把 per-Agent 覆盖（max_tokens/temperature）应用到该角色的副本上。

    覆盖表来自 ``builtin_models.yaml`` 的 ``overrides``（如 Writer 需要更长输出）。
    只覆盖生成参数，不动模型与 Key。
    """
    ov = builtin_overrides().get(role) or {}
    if not ov:
        return option
    from dataclasses import replace

    kwargs = {}
    if "max_tokens" in ov:
        kwargs["max_tokens"] = int(ov["max_tokens"])
    if "temperature" in ov:
        kwargs["temperature"] = float(ov["temperature"])
    return replace(option, **kwargs) if kwargs else option


def default_plan(user_id: int, options: list[ModelOption]) -> dict:
    """给出该用户的初始角色分配。

    - 解锁用户：用内置 ``default_plan``（6 角色 + 读图）；
    - 未解锁用户：没有内置可用，全部指向他自建的第一个模型（可能为空）。
    """
    has_builtin = any(o.source == "builtin" for o in options)
    if has_builtin:
        plan = builtin_default_plan()
        # 兜底：配置里提到的内置项若不在目录（例如被删），换成第一个可用模型
        first = options[0].id if options else ""
        for role in PLAN_ROLES:
            if not find_option(options, plan.get(role, "")):
                plan[role] = first
        return plan

    first = options[0].id if options else ""
    return {role: first for role in PLAN_ROLES}


def resolve_plan(plan: dict, options: list[ModelOption]) -> dict:
    """把 ``{role: option_id}`` 解析成 ``{role: ModelOption}``。

    - 缺失的角色 → 该用户目录里的第一个模型；
    - 指向不存在的 id → 同上（**必须真回落**，不能把 None 交给调用方）；
    解析不出来（目录为空）时该角色值为 None，由 preflight 报错。
    """
    fallback = options[0] if options else None
    out: dict = {}
    for role in PLAN_ROLES:
        oid = (plan or {}).get(role) or ""
        opt = find_option(options, oid) if oid else None
        out[role] = opt if opt is not None else fallback
    return out


def pricing_for(model_id: str, options: list[ModelOption]) -> tuple[float, float]:
    """按模型串反查 (price_in, price_out)。

    ★ 前缀无关匹配（实测踩坑）：CrewAI 的 ``LLM(model=...)`` 会把 provider
    前缀剥掉——配置里是 ``deepseek/deepseek-flash``，而
    ``crew.agents[0].llm.model`` 是 ``deepseek-flash``。只做全等比较会导致
    真实模式下每次记账都查不到单价、静默记 ¥0，账单与熔断全失效。
    """
    def _bare(s: str) -> str:
        s = (s or "").strip()
        return s.split("/", 1)[1] if "/" in s else s

    target = (model_id or "").strip()
    if not target:
        return (0.0, 0.0)
    target_bare = _bare(target)
    for opt in options:
        if opt.model_id == target or _bare(opt.model_id) == target_bare:
            return (opt.price_in, opt.price_out)
    return (0.0, 0.0)
