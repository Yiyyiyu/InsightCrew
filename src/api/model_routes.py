"""用户自定义模型的 API —— 来源（providers）、模型（models）、内置解锁。

## 为什么单独一个文件而不是塞进 `src/api/routes.py`

`routes.py` 已经承担鉴权 + 调研任务两类职责；模型配置是一块独立能力
（CRUD + 密钥处理 + 解锁门禁），拆开可以让两边都保持可读。注册时与
`routes.py` 同级挂到 app 上，路径前缀天然不冲突（`/models/*` vs
`/research/*`）。

## 三条不可退让的约束

1. **明文 Key 永不返回**。所有 GET 只给 `KEY_MASK_PLACEHOLDER`；更新时
   Key 字段留空表示"不改"，避免前端回显掩码被当成新 Key 存进去。
2. **`api_base` 必须过 SSRF 校验**。用户能填任意 URL，而服务端会拿它
   发请求——这正是 SSRF 的入口（见 `src/core/net.py`）。
3. **越权一律 404**。provider/model 的归属校验与任务一致：不区分
   "不存在"与"别人的"，避免用 404/403 差异枚举 id。
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import delete, select

from src.api.deps import get_current_user
from src.core import model_registry as registry
from src.core.config import settings
from src.core.crypto import KEY_MASK_PLACEHOLDER, encrypt_key
from src.core.errors import ServiceUnavailableError
from src.core.logger import logger
from src.core.net import UnsafeUrlError, assert_public_url
from src.core.ratelimit import RateLimitExceeded, check_rate_limit, client_ip
from src.db.database import async_session
from src.db.models import Provider, UserModel

router = APIRouter(prefix="/models", tags=["models"])

# 解锁接口限流：5 次/分钟（按 IP + user 双维度）
_UNLOCK_LIMIT = 5
_UNLOCK_WINDOW_SEC = 60


# ══════════════════════════════════════════════════════════════
# 请求 / 响应模型
# ══════════════════════════════════════════════════════════════

class ProviderCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    api_base: str = Field(..., min_length=1, max_length=512)
    api_key: str = Field(..., min_length=1)
    protocol: str = "openai"


class ProviderUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=64)
    api_base: Optional[str] = Field(None, min_length=1, max_length=512)
    # 留空 = 不修改已存的 Key（前端只拿得到掩码，回传掩码会覆盖真 Key）
    api_key: Optional[str] = None


class ProviderOut(BaseModel):
    id: int
    name: str
    api_base: str
    protocol: str
    api_key_mask: str
    model_count: int = 0
    created_at: Optional[datetime] = None


class ModelCreate(BaseModel):
    provider_id: int
    model_id: str = Field(..., min_length=1, max_length=128)
    display_name: str = ""
    max_tokens: int = Field(4096, ge=1, le=200_000)
    temperature: float = Field(0.3, ge=0.0, le=2.0)
    supports_vision: bool = False
    disable_thinking: bool = False
    # 单价：0 = 只统计 token 不计费（不填会导致账单全 ¥0、预算熔断失效）
    price_in: float = Field(0.0, ge=0.0)
    price_out: float = Field(0.0, ge=0.0)


class ModelUpdate(BaseModel):
    model_id: Optional[str] = Field(None, min_length=1, max_length=128)
    display_name: Optional[str] = None
    max_tokens: Optional[int] = Field(None, ge=1, le=200_000)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    supports_vision: Optional[bool] = None
    disable_thinking: Optional[bool] = None
    price_in: Optional[float] = Field(None, ge=0.0)
    price_out: Optional[float] = Field(None, ge=0.0)


class ModelOut(BaseModel):
    id: int
    provider_id: int
    provider_name: str = ""
    model_id: str
    display_name: str
    max_tokens: int
    temperature: float
    supports_vision: bool
    disable_thinking: bool
    price_in: float
    price_out: float
    created_at: Optional[datetime] = None


class UnlockRequest(BaseModel):
    password: str = Field(..., min_length=1)


class OptionOut(BaseModel):
    """统一目录里的一个可选项（内置或自建）。"""

    id: str
    source: str
    display_name: str
    model_id: str
    provider: str
    provider_name: str = ""
    supports_vision: bool = False
    price_in: float = 0.0
    price_out: float = 0.0


class CatalogOut(BaseModel):
    unlocked: bool
    unlock_expires_at: Optional[datetime] = None
    ttl_hours: int
    roles: list[str]
    options: list[OptionOut]
    default_plan: dict


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════

def _normalize_api_base(raw: str) -> str:
    """规范化并校验 api_base。

    - 去尾部斜杠（否则 `/v1/` + `/chat/completions` 会拼出双斜杠）
    - 过 SSRF 校验：仅 http/https、禁内网段。
      自建来源在公网上必须指向公开网关；本机 Ollama 这类内网地址
      会让服务器替用户去打内网，属于必须拦下的攻击面。
    """
    base = (raw or "").strip().rstrip("/")
    if not base:
        raise HTTPException(status_code=400, detail="API 地址不能为空")
    try:
        assert_public_url(base)
    except UnsafeUrlError as e:
        raise HTTPException(
            status_code=400,
            detail=f"API 地址不被允许：{e}。请填写公开可访问的 http(s) 网关地址。",
        )
    return base


def _rate_limit(request: Request, current_user: dict, action: str) -> None:
    """按 IP + user 双维度限流。"""
    ip = client_ip(request)
    uid = current_user.get("user_id") or 0
    key = f"{action}:{ip}:{uid}"
    try:
        check_rate_limit(key, limit=_UNLOCK_LIMIT, window_sec=_UNLOCK_WINDOW_SEC)
    except RateLimitExceeded as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e),
            headers={"Retry-After": str(e.retry_after)},
        )


async def _get_owned_provider(session, provider_id: int, user_id: int) -> Provider:
    """取当前用户名下的 provider；不存在或不属于他一律 404。"""
    row = await session.get(Provider, provider_id)
    if row is None or row.user_id != user_id:
        raise HTTPException(status_code=404, detail="来源不存在")
    return row


async def _get_owned_model(session, model_id: int, user_id: int) -> UserModel:
    row = await session.get(UserModel, model_id)
    if row is None or row.user_id != user_id:
        raise HTTPException(status_code=404, detail="模型不存在")
    return row


# ══════════════════════════════════════════════════════════════
# 统一目录 + 解锁
# ══════════════════════════════════════════════════════════════

@router.get("/catalog", response_model=CatalogOut, summary="当前用户可选的模型目录")
async def get_catalog(current_user: dict = Depends(get_current_user)):
    """返回当前用户能看到的全部模型（未解锁则不含内置）+ 默认角色分配。

    前端新建任务表单的 7 个下拉就用这份数据渲染。
    """
    uid = current_user["user_id"]
    async with async_session() as session:
        unlocked = await registry.is_unlocked_async(session, uid)
        options = await registry.catalog_async(session, uid)
        row = None
        if unlocked:
            from src.db.models import BuiltinUnlock
            row = await session.get(BuiltinUnlock, uid)

    plan = registry.default_plan(uid, options)
    return CatalogOut(
        unlocked=unlocked,
        unlock_expires_at=getattr(row, "expires_at", None),
        ttl_hours=int(settings.builtin_unlock_ttl_hours or 12),
        roles=registry.PLAN_ROLES,
        options=[
            OptionOut(
                id=o.id,
                source=o.source,
                display_name=o.display_name,
                model_id=o.model_id,
                provider=o.provider,
                provider_name=o.provider_name,
                supports_vision=o.supports_vision,
                price_in=o.price_in,
                price_out=o.price_out,
            )
            for o in options
        ],
        default_plan=plan,
    )


@router.post("/unlock", summary="用密码解锁内置模型（限流 5 次/分钟）")
async def unlock(
    req: UnlockRequest,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """校验密码并为本用户开通内置模型（TTL 默认 12h）。

    ★ 解锁态按 user_id 隔离：站长解锁不影响任何其他账号，
      新注册用户默认锁上。
    """
    import hmac

    _rate_limit(request, current_user, "unlock")

    expected = (settings.builtin_unlock_password or "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="服务端未配置 BUILTIN_UNLOCK_PASSWORD，内置模型不可解锁",
        )

    # 常数时间比较，防时序侧信道
    if not hmac.compare_digest(req.password.strip(), expected):
        logger.warning(
            f"内置解锁失败 user_id={current_user['user_id']} ip={client_ip(request)}"
        )
        raise HTTPException(status_code=401, detail="密码不正确")

    uid = current_user["user_id"]
    async with async_session() as session:
        expires = await registry.grant_unlock_async(session, uid)
        await session.commit()

    logger.info(f"内置模型已解锁 user_id={uid} 到期={expires.isoformat()}")
    return {
        "unlocked": True,
        "expires_at": expires,
        "ttl_hours": int(settings.builtin_unlock_ttl_hours or 12),
    }


@router.post("/lock", summary="撤销内置模型解锁")
async def lock(current_user: dict = Depends(get_current_user)):
    uid = current_user["user_id"]
    async with async_session() as session:
        await registry.revoke_unlock_async(session, uid)
        await session.commit()
    logger.info(f"内置模型已上锁 user_id={uid}")
    return {"unlocked": False}


# ══════════════════════════════════════════════════════════════
# providers CRUD
# ══════════════════════════════════════════════════════════════

@router.get("/providers", response_model=list[ProviderOut], summary="我的 API 来源列表")
async def list_providers(current_user: dict = Depends(get_current_user)):
    uid = current_user["user_id"]
    async with async_session() as session:
        rows = (await session.execute(
            select(Provider).where(Provider.user_id == uid).order_by(Provider.id)
        )).scalars().all()
        counts = (await session.execute(
            select(UserModel.provider_id).where(UserModel.user_id == uid)
        )).scalars().all()

    counter: dict[int, int] = {}
    for pid in counts:
        counter[pid] = counter.get(pid, 0) + 1

    return [
        ProviderOut(
            id=p.id,
            name=p.name,
            api_base=p.api_base,
            protocol=p.protocol or "openai",
            # ★ 只回掩码：明文 Key 从不离开服务端
            api_key_mask=KEY_MASK_PLACEHOLDER,
            model_count=counter.get(p.id, 0),
            created_at=p.created_at,
        )
        for p in rows
    ]


@router.post(
    "/providers",
    response_model=ProviderOut,
    status_code=status.HTTP_201_CREATED,
    summary="新增 API 来源",
)
async def create_provider(
    req: ProviderCreate,
    current_user: dict = Depends(get_current_user),
):
    uid = current_user["user_id"]
    api_base = _normalize_api_base(req.api_base)
    protocol = (req.protocol or "openai").strip().lower()
    if protocol != "openai":
        raise HTTPException(
            status_code=400,
            detail="暂只支持 openai 兼容协议（DeepSeek/智谱/OneAPI/Ollama 均属此类）",
        )

    async with async_session() as session:
        # 同用户下不允许重名：来源名是前端下拉里的展示项，重名会混淆
        dup = (await session.execute(
            select(Provider).where(Provider.user_id == uid, Provider.name == req.name.strip())
        )).scalar_one_or_none()
        if dup is not None:
            raise HTTPException(status_code=409, detail="同名来源已存在")

        row = Provider(
            user_id=uid,
            name=req.name.strip(),
            api_base=api_base,
            api_key_encrypted=encrypt_key(req.api_key.strip()),
            protocol=protocol,
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)

    logger.info(f"新增来源 user_id={uid} provider_id={row.id} base={api_base}")
    return ProviderOut(
        id=row.id,
        name=row.name,
        api_base=row.api_base,
        protocol=row.protocol,
        api_key_mask=KEY_MASK_PLACEHOLDER,
        model_count=0,
        created_at=row.created_at,
    )


@router.patch("/providers/{provider_id}", response_model=ProviderOut, summary="修改 API 来源")
async def update_provider(
    provider_id: int,
    req: ProviderUpdate,
    current_user: dict = Depends(get_current_user),
):
    uid = current_user["user_id"]
    async with async_session() as session:
        row = await _get_owned_provider(session, provider_id, uid)

        if req.name is not None:
            new_name = req.name.strip()
            dup = (await session.execute(
                select(Provider).where(
                    Provider.user_id == uid,
                    Provider.name == new_name,
                    Provider.id != provider_id,
                )
            )).scalar_one_or_none()
            if dup is not None:
                raise HTTPException(status_code=409, detail="同名来源已存在")
            row.name = new_name

        if req.api_base is not None:
            row.api_base = _normalize_api_base(req.api_base)

        # 留空 = 不改 Key（前端只能拿到掩码，回传掩码会毁掉真 Key）
        if req.api_key is not None and req.api_key.strip():
            row.api_key_encrypted = encrypt_key(req.api_key.strip())

        await session.commit()
        await session.refresh(row)
        count = len((await session.execute(
            select(UserModel.id).where(UserModel.provider_id == row.id)
        )).scalars().all())

    return ProviderOut(
        id=row.id,
        name=row.name,
        api_base=row.api_base,
        protocol=row.protocol or "openai",
        api_key_mask=KEY_MASK_PLACEHOLDER,
        model_count=count,
        created_at=row.created_at,
    )


@router.delete("/providers/{provider_id}", summary="删除 API 来源（连同其下模型）")
async def delete_provider(
    provider_id: int,
    current_user: dict = Depends(get_current_user),
):
    uid = current_user["user_id"]
    async with async_session() as session:
        row = await _get_owned_provider(session, provider_id, uid)
        # SQLite 默认不强制外键，必须显式级联删子行，否则留下孤儿模型
        await session.execute(delete(UserModel).where(UserModel.provider_id == row.id))
        await session.delete(row)
        await session.commit()

    logger.info(f"删除来源 user_id={uid} provider_id={provider_id}")
    return {"deleted": True, "provider_id": provider_id}


# ══════════════════════════════════════════════════════════════
# models CRUD
# ══════════════════════════════════════════════════════════════

@router.get("/items", response_model=list[ModelOut], summary="我的模型列表")
async def list_models(current_user: dict = Depends(get_current_user)):
    uid = current_user["user_id"]
    async with async_session() as session:
        rows = (await session.execute(
            select(UserModel, Provider)
            .join(Provider, UserModel.provider_id == Provider.id)
            .where(UserModel.user_id == uid, Provider.user_id == uid)
            .order_by(UserModel.id)
        )).all()

    return [
        ModelOut(
            id=m.id,
            provider_id=m.provider_id,
            provider_name=p.name,
            model_id=m.model_id,
            display_name=m.display_name or m.model_id,
            max_tokens=int(m.max_tokens or 4096),
            temperature=float(m.temperature if m.temperature is not None else 0.3),
            supports_vision=bool(m.supports_vision),
            disable_thinking=bool(m.disable_thinking),
            price_in=float(m.price_in or 0.0),
            price_out=float(m.price_out or 0.0),
            created_at=m.created_at,
        )
        for m, p in rows
    ]


@router.post(
    "/items",
    response_model=ModelOut,
    status_code=status.HTTP_201_CREATED,
    summary="在某个来源下新增模型",
)
async def create_model(
    req: ModelCreate,
    current_user: dict = Depends(get_current_user),
):
    uid = current_user["user_id"]
    async with async_session() as session:
        provider = await _get_owned_provider(session, req.provider_id, uid)

        row = UserModel(
            user_id=uid,
            provider_id=provider.id,
            model_id=req.model_id.strip(),
            display_name=(req.display_name or "").strip() or req.model_id.strip(),
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            supports_vision=1 if req.supports_vision else 0,
            disable_thinking=1 if req.disable_thinking else 0,
            price_in=req.price_in,
            price_out=req.price_out,
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)

    if req.price_in == 0.0 and req.price_out == 0.0:
        logger.warning(
            f"模型 {row.model_id} 单价为 0，该模型调用只记 token 不计费"
            "（账单与预算熔断对它不生效）"
        )

    logger.info(f"新增模型 user_id={uid} model_pk={row.id} model_id={row.model_id}")
    return ModelOut(
        id=row.id,
        provider_id=row.provider_id,
        provider_name=provider.name,
        model_id=row.model_id,
        display_name=row.display_name,
        max_tokens=int(row.max_tokens),
        temperature=float(row.temperature),
        supports_vision=bool(row.supports_vision),
        disable_thinking=bool(row.disable_thinking),
        price_in=float(row.price_in),
        price_out=float(row.price_out),
        created_at=row.created_at,
    )


@router.patch("/items/{model_pk}", response_model=ModelOut, summary="修改模型")
async def update_model(
    model_pk: int,
    req: ModelUpdate,
    current_user: dict = Depends(get_current_user),
):
    uid = current_user["user_id"]
    async with async_session() as session:
        row = await _get_owned_model(session, model_pk, uid)
        provider = await session.get(Provider, row.provider_id)

        for field_name in (
            "model_id", "display_name", "max_tokens", "temperature",
            "price_in", "price_out",
        ):
            value = getattr(req, field_name)
            if value is not None:
                setattr(row, field_name, value)
        if req.supports_vision is not None:
            row.supports_vision = 1 if req.supports_vision else 0
        if req.disable_thinking is not None:
            row.disable_thinking = 1 if req.disable_thinking else 0

        await session.commit()
        await session.refresh(row)

    return ModelOut(
        id=row.id,
        provider_id=row.provider_id,
        provider_name=provider.name if provider else "",
        model_id=row.model_id,
        display_name=row.display_name or row.model_id,
        max_tokens=int(row.max_tokens or 4096),
        temperature=float(row.temperature if row.temperature is not None else 0.3),
        supports_vision=bool(row.supports_vision),
        disable_thinking=bool(row.disable_thinking),
        price_in=float(row.price_in or 0.0),
        price_out=float(row.price_out or 0.0),
        created_at=row.created_at,
    )


@router.delete("/items/{model_pk}", summary="删除模型")
async def delete_model(
    model_pk: int,
    current_user: dict = Depends(get_current_user),
):
    uid = current_user["user_id"]
    async with async_session() as session:
        row = await _get_owned_model(session, model_pk, uid)
        await session.delete(row)
        await session.commit()

    logger.info(f"删除模型 user_id={uid} model_pk={model_pk}")
    return {"deleted": True, "model_pk": model_pk}


__all__ = ["router"]
