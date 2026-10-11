"""JWT 鉴权工具：密码哈希、Token 生成与验证"""

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


def create_access_token(user_id: int, username: str) -> str:
    """生成 JWT access_token（含 user_id + username + 过期时间）"""
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_access_ttl_min)
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": expire,
        "type": "access",
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict | None:
    """解码并验证 JWT，成功返回 payload，失败返回 None"""
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None
