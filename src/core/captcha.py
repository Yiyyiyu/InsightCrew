"""图形验证码 —— PIL 自绘，答案存 Redis，校验即销毁。

## 为什么自绘而不是用第三方库

本机与部署环境都装不了 `captcha` / `svglib` 这类库（国内 pip 源里要么没有，
要么拖一堆图形依赖），而 **Pillow 项目里本来就有**（PDF 导出在用）。
自绘一个 4 字符验证码不到 80 行，零新增依赖。

## 为什么不把答案塞进签名令牌

一开始想过「把答案签进 token 交给客户端，回来时校验签名」——那样**不需要
服务端存储**。但答案就在令牌里，base64 一解就能看到（签名只保证不被篡改，
不保证不可读）。所以答案必须留在服务端。

## 存储：Redis 为主，进程内字典兜底

  * Redis 可用时答案放 `captcha:<id>`，TTL 由 `captcha_ttl_sec` 控制。
  * Redis 不可用时退到进程内字典 —— **只保证单进程部署可用**（本项目就是
    单 API 容器）。若哪天要横向扩多个 API 实例，这个兜底会让「在 A 实例拿
    验证码、在 B 实例提交」失败，届时应改为直接报错而不是兜底。

## 一次性

校验用 `GETDEL`（Redis 6.2+，本项目是 7.4）：**读与删是一条原子命令**，
并发提交同一个验证码时只有一个能拿到答案，天然防重放。校验失败也照样销毁，
否则攻击者可以拿同一个验证码无限次试密码。
"""

from __future__ import annotations

import base64
import hmac
import io
import random
import secrets
import time

from src.core.config import settings
from src.core.logger import logger

#: 去掉形近字符：0/O、1/I/L。用户看不清就会反复刷新，体验比安全性损失更明显。
CAPTCHA_ALPHABET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"

#: 验证码字符数
CAPTCHA_LENGTH = 4

#: 图片尺寸
_WIDTH, _HEIGHT = 132, 44

#: Redis 不可用时的进程内兜底：{captcha_id: (答案, 过期时间戳)}
_MEM: dict[str, tuple[str, float]] = {}

#: 惰性初始化的 Redis 客户端（与 SSE 的连接池分开，避免互相挤占）
_redis = None
_redis_checked = False

#: 字体候选：Windows 与常见 Linux 发行版各备几个，都没有就用 Pillow 内置字体。
_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)


def _load_font(size: int):
    """找一个能用的字体，找不到就用 Pillow 内置的可缩放默认字体。"""
    from PIL import ImageFont

    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    try:
        # Pillow >= 10.1 的 load_default 支持 size，缩放后不会糊成一小团
        return ImageFont.load_default(size=size)
    except TypeError:  # 老版本 Pillow 不接受 size
        return ImageFont.load_default()


def _render(code: str) -> bytes:
    """把验证码画成 PNG 字节。"""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (_WIDTH, _HEIGHT), (248, 250, 252))
    draw = ImageDraw.Draw(img)

    # 干扰线：浅色、细，避免影响识别但足以让简单 OCR 失效
    for _ in range(5):
        draw.line(
            (
                random.randint(0, _WIDTH),
                random.randint(0, _HEIGHT),
                random.randint(0, _WIDTH),
                random.randint(0, _HEIGHT),
            ),
            fill=(random.randint(180, 215), random.randint(180, 215), random.randint(190, 225)),
            width=1,
        )

    font = _load_font(26)
    step = _WIDTH // (CAPTCHA_LENGTH + 1)
    for i, ch in enumerate(code):
        # 每个字符单独渲染后旋转，避免整行一条直线（那种图很容易被切分识别）
        tile = Image.new("RGBA", (step, _HEIGHT), (0, 0, 0, 0))
        tdraw = ImageDraw.Draw(tile)
        color = (
            random.randint(20, 70),
            random.randint(30, 80),
            random.randint(90, 150),
        )
        tdraw.text((step // 2, _HEIGHT // 2), ch, font=font, fill=color, anchor="mm")
        tile = tile.rotate(random.uniform(-22, 22), resample=Image.BICUBIC, expand=False)
        img.paste(tile, (step * i + step // 2 - 6, random.randint(-2, 2)), tile)

    # 噪点
    for _ in range(120):
        draw.point(
            (random.randint(0, _WIDTH - 1), random.randint(0, _HEIGHT - 1)),
            fill=(random.randint(150, 215), random.randint(150, 215), random.randint(160, 225)),
        )

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


async def _get_redis():
    """惰性连接 Redis；不可用返回 None（由调用方退到内存字典）。"""
    global _redis, _redis_checked
    if _redis is not None:
        return _redis
    if _redis_checked:
        return None
    _redis_checked = True
    try:
        import redis.asyncio as aioredis

        r = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            max_connections=4,
            socket_connect_timeout=3,
        )
        await r.ping()
        _redis = r
        return r
    except Exception as exc:
        logger.warning(f"[captcha] Redis 不可用（{exc}），验证码退回进程内存储")
        return None


def _ttl() -> int:
    return max(60, int(settings.captcha_ttl_sec or 300))


def _mem_sweep() -> None:
    """顺手清掉内存里的过期项，避免长时间运行后无限增长。"""
    now = time.time()
    for key in [k for k, (_, exp) in _MEM.items() if exp < now]:
        _MEM.pop(key, None)


async def new_captcha() -> tuple[str, str]:
    """生成一个验证码。

    返回 ``(captcha_id, data_url)``，`data_url` 可直接塞进 `<img src>`，
    前端不必再处理 base64 解码。
    """
    code = "".join(secrets.choice(CAPTCHA_ALPHABET) for _ in range(CAPTCHA_LENGTH))
    cid = secrets.token_urlsafe(16)
    png = _render(code)
    ttl = _ttl()

    r = await _get_redis()
    if r is not None:
        try:
            await r.set(f"captcha:{cid}", code, ex=ttl)
        except Exception as exc:
            logger.warning(f"[captcha] 写 Redis 失败（{exc}），改存进程内")
            _MEM[cid] = (code, time.time() + ttl)
    else:
        _mem_sweep()
        _MEM[cid] = (code, time.time() + ttl)

    b64 = base64.b64encode(png).decode("ascii")
    return cid, f"data:image/png;base64,{b64}"


async def verify_captcha(captcha_id: str, answer: str) -> bool:
    """校验并**立即销毁**验证码。大小写不敏感。

    ★ 无论成功失败都销毁：否则攻击者可以拿一个验证码无限次试密码，
      验证码就完全失去限流作用了。
    """
    cid = (captcha_id or "").strip()
    ans = (answer or "").strip().upper()
    if not cid or not ans:
        return False

    expected = None
    r = await _get_redis()
    if r is not None:
        try:
            # GETDEL：读与删原子完成，并发提交只有一个能拿到答案
            expected = await r.getdel(f"captcha:{cid}")
        except Exception:
            # getdel 在极老版本 Redis 上不存在，退回「读 + 删」
            try:
                expected = await r.get(f"captcha:{cid}")
                await r.delete(f"captcha:{cid}")
            except Exception as exc:
                logger.warning(f"[captcha] 读 Redis 失败：{exc}")

    if expected is None:
        got = _MEM.pop(cid, None)
        if got and got[1] >= time.time():
            expected = got[0]

    if not expected:
        return False
    return hmac.compare_digest(expected.upper(), ans)


__all__ = ["CAPTCHA_ALPHABET", "CAPTCHA_LENGTH", "new_captcha", "verify_captcha"]
