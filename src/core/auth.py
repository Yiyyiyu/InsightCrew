"""JWT 鉴权工具：密码哈希、Token 生成与验证"""

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt

from src.core.config import settings

# ---- 密码哈希 ----


def hash_password(password: str) -> str:
    """明文 → bcrypt 哈希"""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """验证明文 vs 哈希"""
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# ---- JWT ----
ALGORITHM = "HS256"

# 浏览器场景必须走 Cookie：页面跳转（<a href>）和 EventSource 都**无法**设置
# Authorization 头，只有 Cookie 会自动携带。API 客户端仍可用 Bearer 头。
ACCESS_COOKIE = "insightcrew_token"


def create_access_token(user_id: int, username: str, token_version: int = 0) -> str:
    """生成 JWT access_token（含 user_id + username + 令牌版本 + 过期时间）

    ``token_version`` 是「可作废令牌」的关键：JWT 无状态，服务端无法撤回一个
    已发出的 token；把用户当前的版本号写进载荷，校验时与库中值比对，
    改密码时把库里的版本号 +1，此前签发的所有 token 就全部失效。
    """
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_access_ttl_min)
    payload = {
        "user_id": user_id,
        "username": username,
        "token_version": int(token_version or 0),
        "exp": expire,
        "type": "access",
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict | None:
    """解码并验证 JWT，成功返回 payload，失败返回 None

    ★ 这里**只验签名与过期**，不查库比对 ``token_version`` —— 该函数被
      `_token_from_request` 在同步上下文里调用，加一次数据库往返会让每个
      请求都变慢。版本号比对放在 `get_current_user`（async，能开 session）。
    """
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


# ---- 密码重置令牌 ----
#
# 与 access token 的差别：这个 token 是**一次性**的，且必须落库校验。
#   * 自签（hmac-sha256，密钥从 JWT_SECRET 派生）而不是用 jwt 库：
#     我们只需要「不可伪造」，不需要 JWT 的互操作性；自己拼串更省一层解析。
#   * 数据库只存 sha256(token)。明文只存在于邮件里 —— 与「密码只存 bcrypt
#     哈希」同理：库泄露也拿不到可用的重置链接。
#   * 载荷带 user_id 与过期时间戳，签名覆盖两者，所以改不了归属也延不了期。

RESET_TOKEN_TTL_MIN_DEFAULT = 30


def _reset_signing_key() -> bytes:
    """从 JWT_SECRET 派生重置令牌的签名密钥（领域分离）。"""
    return hmac.new(
        (settings.jwt_secret or "").encode("utf-8"),
        b"password-reset-v1",
        hashlib.sha256,
    ).digest()


def hash_reset_token(token: str) -> str:
    """明文 token → 入库用的 sha256 十六进制摘要（64 字符）。"""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_reset_token(user_id: int, ttl_min: int | None = None) -> tuple[str, datetime]:
    """生成一次性重置 token，返回 (明文 token, 过期时间)。

    ★ 过期时间同时返回，调用方直接写库，避免两处各算一遍出现偏差。
    """
    ttl = int(ttl_min or settings.password_reset_ttl_min or RESET_TOKEN_TTL_MIN_DEFAULT)
    expires = datetime.utcnow() + timedelta(minutes=ttl)
    # 载荷：user_id + 过期 unix 秒 + 随机数（保证同一用户两次生成不重样）
    body = f"{int(user_id)}.{int(expires.timestamp())}.{secrets.token_urlsafe(16)}"
    sig = hmac.new(_reset_signing_key(), body.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{body}.{sig}", expires


def parse_reset_token(token: str) -> int | None:
    """校验签名与过期，返回 user_id；任何异常一律返回 None。

    ★ 只做「格式 + 签名 + 过期」三重检查；**是否已用过、是否被改密作废
      必须查库**（`password_resets.used_at` 与 `users.token_version`），
      那是调用方的事 —— 本函数保持纯函数，便于单测。
    """
    if not token or token.count(".") != 3:
        return None
    body, _, sig = token.rpartition(".")
    expected = hmac.new(_reset_signing_key(), body.encode("utf-8"), hashlib.sha256).hexdigest()
    # 常数时间比较：避免通过响应时间逐字节猜签名
    if not hmac.compare_digest(sig, expected):
        return None
    uid_s, exp_s, _nonce = body.split(".", 2)
    try:
        uid = int(uid_s)
        exp_ts = int(exp_s)
    except ValueError:
        return None
    if datetime.utcnow().timestamp() > exp_ts:
        return None
    return uid
