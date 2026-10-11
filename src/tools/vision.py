"""多模态读图工具 —— 让 Agent 能解读 PDF/网页里的图表与截图。

## 为什么必须独立成一个工具

`config/models.yaml` 里 `vision` 档位（`zai/glm-4v-flash`，**免费**）与路由规则
`when: {modality: vision}` 早已配好，`src/core/model_router.py` 也支持
按模态选模型，但**没有任何调用方** —— 整条视觉链路是断的。本模块补上。

## 为什么不用 Agent 的 LLM 直接读图

CrewAI 的 `Agent(llm=...)` 绑定的是文本模型；要让某个 Agent 具备读图能力，
要么给它配一个多模态 LLM（但那样它所有文本调用也走多模态模型，贵且慢），
要么把它做成**工具**：需要读图时 Agent 显式调用，走 `vision` 档位。

选后者：模态是"按需"的，不是"按 Agent"的。同一个 Researcher 平时用
deepseek-flash 做文本检索，遇到图才切 glm-4v-flash。

## 两档降级（为什么不能只配一个模型）

`vision` 用的是**免费**模型，免费额度的代价是：上下文只有 4K（大图会溢出）、
且随时可能限流（实测同为免费的 `glm-4.6v-flash` 连续 10 次全 429）。
所以配了 `fallback: vision_fallback`（付费的 `zai/glm-4v-plus`，16K 上下文），
本模块按「主档位 → fallback 档位」依次尝试；**返回空内容也视为失败**并继续
回落（思考型模型会把预算耗在 reasoning 上、content 为空）。

## 成本归属

工具签名由 LLM 决定（只有 `image_path`），没有地方传 task_id，但读图
是**真实花钱**的调用。用 `src/core/context.py` 的 ContextVar 取当前任务，
通过 `cost.record_usage_sync` 落 `llm_calls` 并累加 `tasks.total_cost_cny`。
取不到任务上下文时跳过记账（不写脏数据）。
"""

from __future__ import annotations

import base64
from pathlib import Path

from crewai.tools import tool

from src.core.context import current_task_id
from src.core.logger import logger

# 支持的图片扩展名 → MIME
_IMAGE_EXTS = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}

# 单图体积上限：base64 后约膨胀 1/3，超大图会让请求体爆掉且 token 暴涨
_MAX_IMAGE_BYTES = 6 * 1024 * 1024

_DEFAULT_PROMPT = (
    "请解读这张图片。若是图表，说明它的类型、坐标轴/图例含义、以及图中"
    "能读出的关键数值或趋势；若是截图或界面，说明它展示的内容。"
    "只描述你确实看到的内容，不要推测。"
)


def _resolve_image(image_path: str) -> tuple[Path, str] | None:
    """把输入解析成本地图片路径与 MIME；无法解析时返回 None。"""
    raw = (image_path or "").strip().strip('"').strip("'")
    if not raw:
        return None

    # 允许传入 data URL（有些来源直接给 base64）
    if raw.startswith("data:"):
        return None  # 由调用方走 _read_data_url

    p = Path(raw)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    if not p.is_file():
        return None
    ext = p.suffix.lower()
    if ext not in _IMAGE_EXTS:
        return None
    if p.stat().st_size > _MAX_IMAGE_BYTES:
        return None
    return p, _IMAGE_EXTS[ext]


def _data_url_from_file(p: Path, mime: str) -> str:
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _read_image_bytes(image_path: str) -> tuple[bytes, str] | None:
    """读取图片原始字节与 MIME（含 data URL 支持）。"""
    raw = (image_path or "").strip().strip('"').strip("'")
    if raw.startswith("data:"):
        try:
            head, _, payload = raw.partition(",")
            mime = head[5:].split(";")[0] or "image/png"
            return base64.b64decode(payload), mime
        except Exception:
            return None

    resolved = _resolve_image(raw)
    if resolved is None:
        return None
    p, mime = resolved
    return p.read_bytes(), mime


def _call_vision(route, prompt: str, data_url: str) -> tuple[str, object]:
    """调一次视觉模型，返回 (文本, usage)。失败时抛异常由调用方处理。"""
    from litellm import completion

    kwargs = {
        "model": route.model_id,
        "api_key": route.api_key,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
        "max_tokens": route.max_tokens,
        "temperature": getattr(route, "temperature", 0.1),
        "timeout": 90,
    }
    if route.api_base:
        kwargs["base_url"] = route.api_base
    # 关闭思考模式：思考型视觉模型会把预算耗在 reasoning_tokens 上，
    # 导致 content 返回空串（实测 glm-4.6v-flash 用 63/64 token 思考）。
    if getattr(route, "disable_thinking", False):
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    resp = completion(**kwargs)
    text = ""
    try:
        text = resp.choices[0].message.content or ""
    except Exception:
        text = ""
    return text, getattr(resp, "usage", None)


def _is_rate_limited(exc: Exception) -> bool:
    """判断异常是否为限流/过载（免费模型最常见）。"""
    s = f"{type(exc).__name__}: {exc}"
    return any(k in s for k in ("429", "RateLimit", "1305", "访问量过大", "overloaded"))


def _vision_candidates(router) -> list:
    """按顺序给出读图要尝试的模型档位（主档 + 兜底档）。

    优先用当前任务 ``model_plan`` 里用户选定的 ``vision`` 模型；该模型若在
    内置目录里声明了 ``fallback``（免费档 glm-4v-flash → 付费档
    glm-4v-plus），把它接在后面。**这是模态内的兜底，不是跨厂商降级**：
    读图发生在 research 阶段的工具层，失败会直接让 Agent 丢掉一张图，
    免费模型限流又很常见，所以这里保留两档尝试（M2.5 决策，见设计规划
    §7.2.5）。

    无任务上下文（开发/测试）时回落到内置默认目录的 vision 角色。
    """
    from src.core.context import current_model_plan, model_plan_explicit
    from src.core.model_registry import (
        apply_role_overrides,
        builtin_default_plan,
        builtin_options,
        find_option,
    )

    primary_opt = None
    plan_opt = current_model_plan().get("vision")
    if plan_opt is not None:
        primary_opt = apply_role_overrides(plan_opt, "vision")
    elif model_plan_explicit():
        # 真实任务但该用户没有可用模型（未解锁 + 无自建）：**不回落内置目录**，
        # 否则未解锁用户通过读图工具照样用上了内置模型（M2.5-5 第 ③ 道强制点）。
        logger.warning("本任务无可用读图模型，跳过读图")
        return []
    else:
        primary_opt = find_option(builtin_options(), builtin_default_plan().get("vision", ""))
        if primary_opt is not None:
            primary_opt = apply_role_overrides(primary_opt, "vision")

    if primary_opt is not None:
        primary = router.route_from_option(primary_opt)
    else:
        primary = router.route(modality="vision")

    candidates = [primary]
    # 兜底档：优先用**模型自身声明的 fallback**（内置档 glm-4v-flash 配了
    # glm-4v-plus）。★ 不能再用旧的 `router.fallback_models(primary.name)`：
    # M2.5 起 primary.name 是 "builtin:glm-4v-flash" 这种 option id，
    # 不再是 models.yaml 的档位键，那个查表函数会返回空。
    fb = str(getattr(primary_opt, "fallback", "") or "") if primary_opt is not None else ""
    if fb:
        fb_key = fb if ":" in fb else f"builtin:{fb}"
        opt = find_option(builtin_options(), fb_key)
        if opt is not None:
            candidates.append(router.route_from_option(apply_role_overrides(opt, "vision")))
    else:
        # 用户自建模型没有声明 fallback 时，退回 models.yaml 的档位链（若有）
        for name in router.fallback_models(primary.name):
            try:
                candidates.append(router.route_named(name))
            except Exception:
                continue

    # 去重（按 model_id + api_base）
    seen: set[tuple[str, str]] = set()
    out: list = []
    for r in candidates:
        key = (r.model_id, r.api_base or "")
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


@tool("read_image")
def read_image(image_path: str, question: str = "") -> str:
    """解读一张图片（图表/截图），返回中文描述。

    输入:
        image_path: 本地图片路径（PNG/JPG/WebP/GIF/BMP）
        question:   可选，你希望从图中得到什么（如"读出三个柱子的数值"）
    """
    got = _read_image_bytes(image_path)
    if got is None:
        return (f"无法读取图片：{image_path}。"
                "请确认路径存在且为 PNG/JPG/WebP/GIF/BMP，单张不超过 6MB。")
    data, mime = got

    b64 = base64.b64encode(data).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"
    prompt = (question or "").strip() or _DEFAULT_PROMPT

    # ── 路由到 vision 模型 ──
    # ★ M2.5 起读图模型由用户在新建任务时单独选定（model_plan 的 "vision"
    #   角色，不属于任何 Agent 阶段），不再硬编码档位名。取不到（开发/测试
    #   无任务上下文）才回落到内置默认目录。
    from src.core.model_router import router

    candidates = _vision_candidates(router)

    errors: list[str] = []
    text = ""
    used = candidates[0] if candidates else None

    if not candidates:
        return "读图不可用：本任务没有配置读图模型。请跳过该图，改用文字描述。"

    for idx, route in enumerate(candidates):
        try:
            text, usage = _call_vision(route, prompt, data_url)
        except Exception as e:  # noqa: BLE001
            kind = "限流" if _is_rate_limited(e) else "失败"
            errors.append(f"{route.name}({route.model_id}) {kind}: {type(e).__name__}")
            logger.warning(f"读图{kind}（{route.model_id}）: {type(e).__name__}: {e}")
            continue

        # 记账：工具层同步写 llm_calls + 累加 total_cost_cny
        # 注意必须在"判定成功"之前记账——失败的调用也真实消耗了 token。
        if usage is not None:
            from src.core.cost import record_usage_sync

            record_usage_sync(
                current_task_id(),
                route.model_id,
                int(getattr(usage, "prompt_tokens", 0) or 0),
                int(getattr(usage, "completion_tokens", 0) or 0),
                stage=f"vision:read_image{'#fallback' if idx else ''}",
            )

        if text.strip():
            used = route
            if idx:
                logger.info(f"读图回落到 {route.model_id} 成功")
            return text.strip()

        # 空内容：思考型模型把预算吃光的典型表现，换下一个档位
        errors.append(f"{route.name}({route.model_id}) 返回空内容")
        logger.warning(f"读图返回空内容（{route.model_id}），尝试下一档位")

    detail = "；".join(errors) if errors else "未知原因"
    return (f"读图失败（已尝试 {len(candidates)} 个模型：{detail}）。"
            "请勿重试同一张图，可改用文字描述或跳过该图。")


@tool("list_pdf_images")
def list_pdf_images(url: str, max_images: int = 4) -> str:
    """列出 PDF 中内嵌图片的本地路径，供随后用 read_image 解读。

    输入:
        url:        PDF 的完整 URL
        max_images: 最多导出多少张（默认 4，避免把上下文和成本撑爆）
    """
    from src.core.ledger import add_source
    from src.core.net import UnsafeUrlError, safe_urlopen

    url = (url or "").strip()
    if not url.lower().startswith(("http://", "https://")):
        return "只支持 http(s) URL。"

    try:
        # ★ 走 safe_urlopen 而不是裸 urlopen：Agent 能拿到任意 URL，
        #   不拦内网段等于把云元数据端点（169.254.169.254）暴露给它。
        with safe_urlopen(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        }, timeout=20) as r:
            data = r.read()
    except UnsafeUrlError as e:
        return f"拒绝下载该 URL（{e}）。"
    except Exception as e:
        return f"PDF 下载失败（{type(e).__name__}: {e}）。请勿重试。"

    try:
        import fitz
    except Exception as e:
        return f"缺少 PyMuPDF（{e}）。"

    out_dir = Path.cwd() / ".cache" / "pdf_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[int, Path]] = []
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        try:
            for pno, page in enumerate(doc):
                if len(rows) >= max_images:
                    break
                for info in page.get_images(full=True):
                    if len(rows) >= max_images:
                        break
                    xref = info[0]
                    # smask 是透明遮罩层，导出它会得到重复的半透明图
                    if len(info) > 1 and info[1]:
                        continue
                    try:
                        base = doc.extract_image(xref)
                    except Exception:
                        continue
                    ext = base.get("ext", "png")
                    w, h = int(base.get("width", 0)), int(base.get("height", 0))
                    # 过滤装饰性小图（logo/图标/公式碎片）
                    if w < 100 or h < 100:
                        continue
                    path = out_dir / f"pdf{pno + 1}_x{xref}.{ext}"
                    path.write_bytes(base["image"])
                    rows.append((pno + 1, path))
        finally:
            doc.close()
    except Exception as e:
        return f"PDF 图片抽取失败（{type(e).__name__}: {e}）。"

    if not rows:
        return f"PDF 中未找到可解读的图片（已过滤小于 100×100 的装饰性图）：{url}"

    # 图片本身也登记为来源：解读结果要能引用回溯
    sid, chunks = add_source(
        f"{url}#images", "PDF 内嵌图表", "pdf",
        [f"第 {p} 页图片: {path.name}" for p, path in rows],
        page_nums=[p for p, _ in rows],
    )

    lines = [f"来源 [S{sid}] 共 {len(rows)} 张图。用 read_image(image_path=...) 解读后，"
             "引用时使用对应的 [S{id}#n] 标记："]
    for (pno, path), (_, seq) in zip(rows, chunks):
        lines.append(f"[S{sid}#{seq}] (第 {pno} 页) {path}")
    return "\n\n".join(lines)


__all__ = ["read_image", "list_pdf_images"]
