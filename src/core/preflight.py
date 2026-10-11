"""模型连通性预检（preflight）—— 新建任务前的最后一道门。

## 为什么需要它（M2.5-5）

M2.5 之后每个 Agent 角色用哪个模型由**用户自己选**。用户会填错 Key、把
`api_base` 写成内网地址、选一个他账号下不存在的模型名——这些错误如果在
`research` 阶段第 3 个方向才炸出来，用户已经等了几分钟，而且任务会以
`failed` 收场（钱也花了）。

所以 `POST /research/start` 的顺序改为：

    解析计划 → 预检连通性 → **全通才建 Task 行** → 入队 → 202

任一不通 → 400 `{"code":"model_unreachable","failures":[...]}`，**任务不落库**。
这与 Phase A「不留僵尸任务行」是同一条原则：失败要发生在产生副作用之前。

## 探测策略（两级，实测校准）

1. `GET {api_base}/models` —— 首选。不耗 token、不产生费用，且能顺带
   确认「这个模型名在不在对方的模型列表里」。
2. 失败则 `POST {api_base}/chat/completions`，`max_tokens=1`。
   少数网关不实现 `/models`（或需要额外权限），只发一个 ping 兜底。

**两级都实测过**（2026-10-10，三个真实厂商）：

    deepseek GET  https://api.deepseek.com/v1/models        200 0.15s
    deepseek POST https://api.deepseek.com/v1/chat/completions 200 0.38s
    zai      GET  https://api.z.ai/api/paas/v4/models        200 0.31s
    zai      POST https://api.z.ai/api/paas/v4/chat/completions 200 0.76s
    lino     GET  https://linoapi.com.cn/v1/models           200 1.18s
    lino     POST https://linoapi.com.cn/v1/chat/completions 200 1.71s

注意 LiteLLM 为 `deepseek/` 解析出的默认 base 是
`https://api.deepseek.com/beta`，而 **`/beta/models` 实测 404**。所以本模块
自己维护一张 base URL 表（见 `_DEFAULT_BASE`），只把 LiteLLM 当兜底——
不能盲信它给的默认值。

## 探测范围

- 本次 `model_plan` 涉及的全部角色（6 Agent + vision），按 `option.id` 去重。
  同一个模型被 3 个角色选中只探一次。
- **vision 的兜底档也探**：内置 GLM-4V Flash 配了 `fallback: glm-4v-plus`，
  主档限流时靠它救场。但它**不是**必须通的——只要主档通就算过，兜底档
  不通只记一条 warning（免费档 + 付费档同时挂的概率极低，且读图失败
  只会让 Agent 丢一张图，不会毁掉整个任务）。
- 并发探测，每个 3 秒超时（`_PROBE_TIMEOUT_SEC`）。串行探 7 个角色最坏
  21 秒，用户会以为网页卡死了。

## 与「运行时降级」的关系

M2.5 起文本链路的运行时降级已移除（用户 m08865 拍板：正常路径唯一，
异常就报错）。preflight 正是那个替代品——把"模型不可用"从**执行中**提前
到**提交时**，用户立刻拿到明确报错，而不是跑到一半偷偷换一个他没选过的模型。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Iterable, Optional

import httpx

from src.core.logger import logger
from src.core.model_registry import (
    PLAN_ROLES,
    ModelOption,
    builtin_options,
    default_plan,
    find_option,
    resolve_plan,
)

#: 单个探测请求的超时（秒）。三个真实厂商实测 0.15–1.71s，3 秒足够宽裕。
_PROBE_TIMEOUT_SEC = 3.0

#: 探测 `/models` 时，是否要求「模型名出现在返回列表里」才算通过。
#: 关掉的原因：网关对模型名的叫法五花八门（lino 返回 `gpt-4-1106-preview`
#: 而我们要 `gpt-4o-mini`；zai 返回 `glm-4.5` 而我们要 `glm-4v-flash`），
#: 严格比对会把**完全可用**的配置判死。所以 `/models` 能 200 就说明
#: 「地址可达 + Key 有效」，足够作为通过标准。
_STRICT_MODEL_LIST = False

#: 各 provider 的默认 API 根地址（**不含** `/models` 这类路径）。
#: 为什么不用 LiteLLM 的 `get_llm_provider()` 返回值：它给 deepseek 的是
#: `https://api.deepseek.com/beta`，而该路径下没有 `/models`（实测 404）。
_DEFAULT_BASE = {
    "deepseek": "https://api.deepseek.com/v1",
    "zai": "https://api.z.ai/api/paas/v4",
    "openai": "https://api.openai.com/v1",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "moonshot": "https://api.moonshot.cn/v1",
    "siliconflow": "https://api.siliconflow.cn/v1",
}

#: 探测用的 ping 消息。短到不产生可观测的费用（max_tokens=1）。
_PING_BODY_TEMPLATE = {
    "messages": [{"role": "user", "content": "ping"}],
    "max_tokens": 1,
    "temperature": 0,
}


@dataclass
class ProbeResult:
    """一个模型（按 option.id 去重后）的探测结果。"""

    option_id: str
    model_id: str
    ok: bool
    method: str = ""          # "models" | "chat" | ""
    status: Optional[int] = None
    error: str = ""
    required: bool = True     # 主档 True；vision 兜底档 False
    roles: list[str] = field(default_factory=list)


def _base_url(route) -> str:
    """确定探测要用的 API 根地址（**不含** `/models`）。"""
    if route.api_base:
        return route.api_base.rstrip("/")
    return _DEFAULT_BASE.get(route.provider, "").rstrip("/")


def _headers(route) -> dict:
    return {
        "Authorization": f"Bearer {route.api_key}",
        "Content-Type": "application/json",
    }


async def _probe_one(client: httpx.AsyncClient, option: ModelOption) -> ProbeResult:
    """探测单个模型：先 GET /models，失败再 POST chat/completions。

    ★ 用 `route_from_option` 先转成 `ModelRoute` 再探，而不是直接拿
    `ModelOption` —— mock 模式下该函数会把地址改写成 `127.0.0.1:8001`、
    Key 改成 `mock-key`，于是**预检探的地址与执行时真正请求的地址完全一致**。
    直接拿 ModelOption 探会去戳真实厂商，测试里必然失败。
    """
    from src.core.model_router import router as model_router

    route = model_router.route_from_option(option)
    base = _base_url(route)
    res = ProbeResult(
        option_id=option.id,
        model_id=route.model_id,
        ok=False,
    )

    if not base:
        res.error = f"无法确定 {route.provider} 的 API 地址，请显式填写 api_base"
        return res
    if not route.api_key or route.api_key == "no-key-configured":
        res.error = "缺少 API Key（内置模型请检查 .env，自建模型请检查来源配置）"
        return res

    # ── 第一级：GET /models ──
    try:
        r = await client.get(f"{base}/models", headers=_headers(route))
        if r.status_code == 200:
            res.ok = True
            res.method = "models"
            res.status = 200
            return res
        first_err = f"GET /models → {r.status_code} {_snippet(r)}"
        res.status = r.status_code
    except Exception as e:
        first_err = f"GET /models → {type(e).__name__}: {e}"
        res.error = first_err

    # ── 第二级：POST /chat/completions（max_tokens=1 的 ping）──
    body = dict(_PING_BODY_TEMPLATE)
    body["model"] = _wire_model_name(route)
    try:
        r = await client.post(
            f"{base}/chat/completions", headers=_headers(route), json=body
        )
        if r.status_code == 200:
            res.ok = True
            res.method = "chat"
            res.status = 200
            res.error = ""
            return res
        res.status = r.status_code
        res.error = f"{first_err}；POST /chat/completions → {r.status_code} {_snippet(r)}"
    except Exception as e:
        res.error = f"{first_err}；POST /chat/completions → {type(e).__name__}: {e}"

    return res


def _wire_model_name(route) -> str:
    """发给厂商 API 的模型名（**不带** LiteLLM 的 provider 前缀）。

    `deepseek/deepseek-flash` → `deepseek-flash`。
    """
    mid = route.model_id or ""
    return mid.split("/", 1)[1] if "/" in mid else mid


def _snippet(resp: httpx.Response, limit: int = 160) -> str:
    """从响应里取一小段可读文本，供报错展示。"""
    try:
        text = resp.text or ""
    except Exception:
        return ""
    text = " ".join(text.split())
    return text[:limit]


# ══════════════════════════════════════════════════════════════
# 计划解析与探测编排
# ══════════════════════════════════════════════════════════════

def resolve_probe_targets(
    plan: dict,
    options: list[ModelOption],
) -> tuple[dict[str, ModelOption], list[tuple[ModelOption, bool, list[str]]], list[str]]:
    """把用户的 ``{role: option_id}`` 解析成「要探的模型」+「缺失的角色」。

    返回 ``(resolved, targets, missing_roles)``：

    - ``resolved``：``{role: ModelOption}``，``resolve_plan`` 的结果；
    - ``targets``：``[(ModelOption, required, roles)]``，按 ``option.id`` 去重；
      ``required=False`` 的是 vision 兜底档（不通只记 warning）；
    - ``missing_roles``：目录为空时无法分配的角色（应触发 403）。
    """
    resolved = resolve_plan(plan, options)
    missing = [role for role in PLAN_ROLES if resolved.get(role) is None]

    # option.id → (option, required, [roles])
    merged: dict[str, tuple[ModelOption, bool, list[str]]] = {}
    order: list[str] = []
    for role in PLAN_ROLES:
        opt = resolved.get(role)
        if opt is None:
            continue
        if opt.id not in merged:
            merged[opt.id] = (opt, True, [role])
            order.append(opt.id)
        else:
            merged[opt.id][2].append(role)

        # vision 的兜底档也要探（主档限流时靠它），但不算致命
        fb = str(getattr(opt, "fallback", "") or "")
        if role == "vision" and fb:
            fb_key = fb if ":" in fb else f"builtin:{fb}"
            fb_opt = find_option(builtin_options(), fb_key)
            if fb_opt is not None and fb_opt.id not in merged:
                merged[fb_opt.id] = (fb_opt, False, [f"{role}(兜底)"])
                order.append(fb_opt.id)

    targets = [merged[oid] for oid in order]
    return resolved, targets, missing


async def preflight(
    plan: dict | None,
    options: list[ModelOption],
    *,
    missing_roles: list[str] | None = None,
) -> list[ProbeResult]:
    """并发探测本次计划涉及的全部模型。

    只返回探测结果（含失败项），**不在这里抛异常** —— 由调用方
    （`src/api/routes.py`）决定怎么映射 HTTP 状态码。这样本模块保持纯粹，
    测试也能直接断言结果。

    `missing_roles` 非空时直接返回空列表：调用方应先抛 403（没有可用模型，
    探什么都无意义）。
    """
    if missing_roles:
        return []
    if not plan:
        return []

    _, targets, missing = resolve_probe_targets(plan, options)
    if missing:
        return []
    if not targets:
        return []

    async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_SEC) as client:
        results = await asyncio.gather(
            *(_probe_one(client, opt) for opt, _, _ in targets),
            return_exceptions=True,
        )

    out: list[ProbeResult] = []
    for (opt, required, roles), r in zip(targets, results):
        if isinstance(r, BaseException):
            res = ProbeResult(
                option_id=opt.id,
                model_id=opt.model_id,
                ok=False,
                error=f"{type(r).__name__}: {r}",
            )
        else:
            res = r
        res.required = required
        res.roles = list(roles)
        out.append(res)
    return out


def build_plan_for_user(
    requested: dict | None,
    options: list[ModelOption],
) -> dict:
    """把用户提交的 plan 补全成「7 个角色都有值」的完整 plan。

    规则与执行期一致（见 `model_registry.resolve_plan`）：缺失/指向不存在的
    id 的角色，回落到目录里的第一个模型；目录为空时保留原值（由调用方
    报 403）。用 ``default_plan`` 给解锁用户一个更合理的初始值。

    ``options`` 已是「按当前用户过滤后」的目录（未解锁者看不到内置），
    所以本函数天然不会给未解锁用户分配内置模型。
    """
    if not options:
        return dict(requested or {})

    if not requested:
        base = default_plan(0, options)
    else:
        base = dict(requested)

    resolved = resolve_plan(base, options)
    out: dict[str, str] = {}
    for role in PLAN_ROLES:
        opt = resolved.get(role)
        out[role] = opt.id if opt is not None else ""
    return out


def format_failures(results: Iterable[ProbeResult]) -> list[dict]:
    """把失败项整理成 API 响应的 ``failures`` 数组（只含必需档）。"""
    failures: list[dict] = []
    for r in results:
        if r.ok or not r.required:
            continue
        failures.append(
            {
                "role": ", ".join(r.roles) if r.roles else "",
                "model": r.model_id,
                "option_id": r.option_id,
                "error": r.error or f"HTTP {r.status}",
            }
        )
    return failures


def all_required_ok(results: Iterable[ProbeResult]) -> bool:
    """必需档是否全部通过（兜底档不通过不算失败）。"""
    return all(r.ok or not r.required for r in results)


__all__ = [
    "ProbeResult",
    "preflight",
    "resolve_probe_targets",
    "build_plan_for_user",
    "format_failures",
    "all_required_ok",
]
