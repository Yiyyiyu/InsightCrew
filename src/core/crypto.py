"""用户 API Key 的加解密与掩码。

设计原则：
    1. 明文只存在于「用户提交表单的那一刻」和「构造 LLM 请求的那一刻」，
       落库的永远是 Fernet 密文。
    2. 任何 GET 接口、日志、异常信息都不得出现明文 —— 对外一律走 mask_key()。
    3. 主密钥 FERNET_KEY 放在 .env（不入仓库）。未配置时退化为进程内临时
       密钥：开发能跑，但进程重启后旧密文解不开（生产必须配）。

为什么用 Fernet 而不是自己拼 AES：
    Fernet = AES-128-CBC + HMAC-SHA256 的成熟封装，自带随机 IV 与认证标签，
    密文可安全落库；cryptography 50.0.2 本机实测可用。
"""

import base64
import hashlib
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

from src.core.config import settings
from src.core.logger import logger


def _build_fernet() -> Fernet:
    """按配置构造 Fernet 实例。

    - FERNET_KEY 合法（44 字节 base64）→ 直接用；
    - 配置了但格式不对 → 报错拒绝启动，而不是悄悄换密钥
      （否则用户存过的 Key 会在某次重启后集体解不开，且毫无提示）；
    - 未配置 → 从进程内随机值派生一个，并 warn（开发模式）。
    """
    raw = (settings.fernet_key or "").strip()
    if raw:
        try:
            return Fernet(raw.encode("utf-8"))
        except (ValueError, TypeError) as exc:
            raise RuntimeError(
                "FERNET_KEY 格式不合法（应为 Fernet.generate_key() 产生的 44 字节 base64）。"
                f" 生成命令：python -c \"from cryptography.fernet import Fernet;"
                f"print(Fernet.generate_key().decode())\"。原始错误：{exc}"
            ) from exc

    # 未配置：用 settings.jwt_secret 派生，保证同进程内稳定（比纯随机更可预期），
    # 但仍然不是持久密钥 —— 改了 JWT_SECRET 或换了进程就解不开旧密文。
    digest = hashlib.sha256((settings.jwt_secret or "insightcrew-dev").encode("utf-8")).digest()
    derived = base64.urlsafe_b64encode(digest)
    logger.warning(
        "未配置 FERNET_KEY，已从 JWT_SECRET 派生临时密钥 —— "
        "进程重启或更换 JWT_SECRET 后，用户已存的 API Key 将无法解密。生产环境必须在 .env 配置 FERNET_KEY。"
    )
    return Fernet(derived)


_fernet: Optional[Fernet] = None


def get_fernet() -> Fernet:
    """惰性单例（不在 import 期构造，避免配置错误直接炸掉整个进程启动）。"""
    global _fernet
    if _fernet is None:
        _fernet = _build_fernet()
    return _fernet


def encrypt_key(plaintext: str) -> str:
    """明文 Key → 密文（base64 文本，可直接落 Text 列）。"""
    if not plaintext:
        raise ValueError("API Key 不能为空")
    return get_fernet().encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt_key(ciphertext: str) -> str:
    """密文 → 明文。仅限构造 LLM 请求时调用。

    解密失败（换了主密钥 / 数据被改）抛 ValueError，由调用方转成
    「该来源的 Key 无法解密，请重新填写」这类可读提示，不泄露密文本身。
    """
    if not ciphertext:
        raise ValueError("API Key 密文为空")
    try:
        return get_fernet().decrypt(ciphertext.encode("ascii")).decode("utf-8")
    except (InvalidToken, ValueError, TypeError) as exc:
        raise ValueError("API Key 解密失败（主密钥已变更或数据损坏），请重新填写该来源的 Key") from exc


def mask_key(plaintext_or_cipher: str) -> str:
    """把 Key 变成可安全展示的掩码：`sk-abc...wxyz` → `sk-a****wxyz`。

    只在能拿到明文时用。拿不到明文（只有密文）时不要调用本函数，
    直接返回固定占位符即可 —— 密文本身就不可展示。
    """
    if not plaintext_or_cipher:
        return ""
    s = plaintext_or_cipher
    if len(s) <= 8:
        return "*" * len(s)
    return f"{s[:4]}{'*' * 6}{s[-4:]}"


# 前端展示用户已存 Key 时的固定占位符（永不回传明文）
KEY_MASK_PLACEHOLDER = "••••••••"
