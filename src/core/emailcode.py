"""邮箱验证码 —— 生成 / 校验 / 限流。

## 用途（用户 m26742）

注册、绑定邮箱、更换邮箱、修改密码，**都要先证明邮箱归自己所有**。
这四处共用一套码：同一个 `email_codes` 表，靠 `purpose` 区分。

## 为什么存 HMAC 而不是明文

6 位数字只有 100 万种可能。若存裸 `sha256(code)`，拿到库的人本地枚举一秒
就能反查出全部有效码 —— 哈希在这里不提供任何保护。用 `HMAC(jwt_secret, code)`
把密钥绑进来，没有密钥就撞不出表。这和「重置令牌存 sha256」的区别在于：
重置令牌本身有 128 位随机量，裸哈希就够；验证码没有熵，必须带密钥。

## 尝试次数上限

`attempts` 记连续错误次数，达到 `email_code_max_attempts` 即作废。
没有这个上限，6 位码在 10 分钟有效期里可以被慢慢试出来。

## 一次性

校验成功立即写 `used_at`。同一个码第二次提交直接失败。
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta

from sqlalchemy import select, update

from src.core.config import settings
from src.core.logger import logger
from src.db.database import async_session
from src.db.models import EmailCode

#: 支持的用途。写死成白名单，避免调用方传错字符串后静默地不校验。
#:
#:   * ``register`` —— 注册新账号时验证邮箱
#:   * ``bind``     —— 绑定一个新邮箱（首次绑定，或换绑时的**新**邮箱）
#:   * ``verify``   —— 验证**当前已绑定**的邮箱，用于授权换绑
#:   * ``password`` —— 验证当前已绑定的邮箱，用于授权改密
#:
#: ★ ``verify`` 与 ``password`` 刻意分开：换绑和改密是两种不同的敏感操作，
#:   若共用一个用途，那么「为了换绑而申请的码」也能拿去改密 —— 用户以为
#:   自己只授权了一件事，实际授权了两件。
PURPOSES = ("register", "bind", "verify", "password")

#: 发码冷却用的 Redis 键前缀（与图形验证码分开命名，互不干扰）
_COOLDOWN_PREFIX = "emailcode:cd:"


def _signing_key() -> bytes:
    """从 jwt_secret 派生一把专用密钥。

    领域分离（domain separation）：直接用 jwt_secret 签名会让「验证码的 HMAC」
    与「JWT 的签名」共用同一把密钥，一旦某处出现签名伪造的弱点，另一处跟着沦陷。
    """
    secret = (settings.jwt_secret or "").encode("utf-8")
    return hmac.new(secret, b"email-code-v1", hashlib.sha256).digest()


def hash_code(code: str) -> str:
    """验证码 → 64 位十六进制摘要。"""
    return hmac.new(_signing_key(), (code or "").strip().encode("utf-8"),
                    hashlib.sha256).hexdigest()


def generate_code() -> str:
    """生成一个数字验证码。

    用 `secrets` 而不是 `random`：后者是可预测的 Mersenne Twister，
    观察几个输出就能推后续序列。
    """
    n = max(4, min(int(settings.email_code_length or 6), 10))
    # randbelow 无模偏差（randint 会有，虽然对本场景影响极小）
    return "".join(str(secrets.randbelow(10)) for _ in range(n))


def _ttl_min() -> int:
    return max(1, int(settings.email_code_ttl_min or 10))


async def issue_code(email: str, purpose: str, user_id: int | None = None) -> str | None:
    """作废旧码、写新码，返回**明文**验证码（由调用方去发信）。

    返回 None 表示用途非法 —— 静默生成一个永远不会被校验通过的码，
    比抛异常更安全（异常会变成 500，暴露内部状态）。
    """
    email = (email or "").strip().lower()
    if not email or purpose not in PURPOSES:
        logger.warning(f"[emailcode] 拒绝生成：email={email!r} purpose={purpose!r}")
        return None

    code = generate_code()
    now = datetime.utcnow()
    expires = now + timedelta(minutes=_ttl_min())

    async with async_session() as session:
        # 同一 (邮箱, 用途) 只保留一条有效：用户连点几次「发送验证码」时，
        # 必须只有最后那封邮件里的码能用 —— 否则旧邮件也能通过，
        # 而用户已经删掉了那封邮件，体验上就是「明明填对了却报错」。
        await session.execute(
            update(EmailCode)
            .where(
                EmailCode.email == email,
                EmailCode.purpose == purpose,
                EmailCode.used_at.is_(None),
            )
            .values(used_at=now)
        )
        session.add(EmailCode(
            email=email,
            user_id=user_id,
            purpose=purpose,
            code_hash=hash_code(code),
            expires_at=expires,
        ))
        await session.commit()

    logger.info(f"[emailcode] 已生成 {purpose} 验证码 → {email}"
                f"（{_ttl_min()} 分钟内有效）")
    return code


async def verify_code(email: str, purpose: str, code: str, *,
                      consume: bool = True) -> tuple[bool, str]:
    """校验验证码。返回 ``(是否通过, 失败原因)``。

    ``consume=False`` 时**校验通过也不作废**，留给调用方在真正完成业务动作
    （比如账号成功落库）后再调 `consume_code` 收尾。

    ★ 为什么需要这个开关：注册流程里「验证码通过」和「账号建成」之间还隔着
      一次用户名唯一性检查。若在这里就把码作废，那么用户仅仅是**用户名被别人
      抢了**，却要重新去收件箱要一次验证码 —— 明明邮箱已经证明过了。
      验证码证明的是「邮箱归你」，这个事实不会因为用户名冲突而失效。

    失败原因是给接口层用的**内部**字段，**不要**原样回给用户：区分
    「码不对」和「码过期了」会让攻击者知道自己的猜测落在哪个阶段。
    接口层统一显示「验证码错误或已过期」。
    """
    email = (email or "").strip().lower()
    code = (code or "").strip()
    if not email or not code or purpose not in PURPOSES:
        return False, "invalid_args"

    async with async_session() as session:
        row = (await session.execute(
            select(EmailCode)
            .where(
                EmailCode.email == email,
                EmailCode.purpose == purpose,
                EmailCode.used_at.is_(None),
            )
            .order_by(EmailCode.id.desc())
        )).scalars().first()

        if row is None:
            return False, "not_found"
        if row.expires_at and row.expires_at < datetime.utcnow():
            # 过期的码直接作废，避免它继续占用「同一用途只有一条」的名额
            row.used_at = datetime.utcnow()
            await session.commit()
            return False, "expired"

        max_attempts = max(1, int(settings.email_code_max_attempts or 5))
        if (row.attempts or 0) >= max_attempts:
            row.used_at = datetime.utcnow()
            await session.commit()
            return False, "too_many_attempts"

        if not hmac.compare_digest(row.code_hash, hash_code(code)):
            row.attempts = int(row.attempts or 0) + 1
            left = max_attempts - row.attempts
            if left <= 0:
                # 用完次数即作废：用户必须重新申请，攻击者必须从头再来
                row.used_at = datetime.utcnow()
            await session.commit()
            logger.info(f"[emailcode] {purpose} 验证码错误 → {email}（剩余 {max(0, left)} 次）")
            return False, "mismatch"

        # 成功。consume=False 时保留 used_at 为空，由调用方稍后收尾
        if consume:
            row.used_at = datetime.utcnow()
            await session.commit()
            logger.info(f"[emailcode] {purpose} 验证码校验通过 → {email}")
        else:
            # 记一次「已验证待消费」，但不作废 —— 后续 consume_code 才真正作废
            logger.info(f"[emailcode] {purpose} 验证码校验通过（待消费）→ {email}")
        return True, ""


async def consume_code(email: str, purpose: str) -> bool:
    """作废该 (邮箱, 用途) 下所有未用的码。

    与 `verify_code(consume=False)` 配对使用。返回是否真的作废了至少一条 ——
    调用方据此判断「校验通过之后是不是被别人抢先消费了」。
    """
    email = (email or "").strip().lower()
    if not email or purpose not in PURPOSES:
        return False
    async with async_session() as session:
        res = await session.execute(
            update(EmailCode)
            .where(
                EmailCode.email == email,
                EmailCode.purpose == purpose,
                EmailCode.used_at.is_(None),
            )
            .values(used_at=datetime.utcnow())
        )
        await session.commit()
        return bool(res.rowcount)


async def cooldown_hit(email: str, purpose: str) -> bool:
    """同一邮箱同一用途是否处于发码冷却期。

    与 IP 限流互补：IP 限流挡不住「换 IP 轰炸同一个邮箱」——把别人邮箱刷爆
    会让对方收到一堆邮件。Redis 不可用时放行（宁可少一层保护，也不要因为
    缓存故障把注册/改密整体打死）。
    """
    import hashlib as _hashlib

    from src.core.events import _get_redis  # 复用命令连接池

    r = await _get_redis()
    if r is None:
        return False
    raw = f"{email}|{purpose}".encode("utf-8")
    key = _COOLDOWN_PREFIX + _hashlib.sha256(raw).hexdigest()[:32]
    ttl = max(10, int(settings.email_code_cooldown_sec or 60))
    try:
        # SET NX EX：写成功 = 本次是冷却期内的第一封
        ok = await r.set(key, "1", ex=ttl, nx=True)
        return not bool(ok)
    except Exception:
        return False


__all__ = [
    "PURPOSES",
    "generate_code",
    "hash_code",
    "issue_code",
    "verify_code",
    "consume_code",
    "cooldown_hit",
]
