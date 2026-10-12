"""邮件发送 —— 走 Resend HTTP API。

## 为什么不用 smtplib / aiosmtplib

原计划是「个人邮箱 SMTP + 标准库 smtplib」，后按用户要求改用 **Resend**
（<https://resend.com>）。改用 HTTP API 反而更省：

  * 依赖不变：`httpx` 项目里已有（模型预检就在用），不必新增 `aiosmtplib`；
  * 不必自己拼 MIME、处理 STARTTLS / SSL 分支、以及各种被当垃圾邮件的坑；
  * 失败是明确的 JSON 错误码，而不是 SMTP 那一串状态码。

## 降级：没配 Key 时把邮件打进日志

本机开发时 `RESEND_API_KEY` 可能还没配，此时**不让整个找回密码流程崩掉**，
而是把收件人 / 主题 / 正文打到日志，并从正文里把重置链接提出来单独打一行
——调试时直接点那行链接即可，比去收件箱翻邮件快。

★ 降级只影响「能不能真的送出去」，不影响令牌生成与落库，所以流程本身
  仍可完整验证。**生产必须配好 Key**（否则用户收不到邮件，而接口仍回 200，
  是最难排查的那类静默失效）。
"""

from __future__ import annotations

import asyncio
import re

import httpx

from src.core.config import settings
from src.core.logger import logger

RESEND_ENDPOINT = "https://api.resend.com/emails"

#: 从正文里抠出第一个链接（降级模式下方便直接复制去浏览器打开）
_LINK_RE = re.compile(r"https?://[^\s\"'<>]+")


def is_configured() -> bool:
    """是否具备真实发信能力。"""
    return bool((settings.resend_api_key or "").strip())


def _plain_from_html(html: str) -> str:
    """给纯文本兜底：去掉标签、还原常见实体、压掉多余空行。

    Resend 建议同时提供 text 与 html（部分客户端只显示纯文本）。
    """
    text = re.sub(r"<br\s*/?>|</p>|</div>|</h[1-6]>", "\n", html, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    for entity, char in (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">")):
        text = text.replace(entity, char)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _build_reset_email(reset_url: str, ttl_min: int, username: str) -> tuple[str, str]:
    """返回 (html, text)。"""
    html = f"""\
<div style="font-family:-apple-system,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;
            max-width:560px;margin:0 auto;padding:32px 24px;color:#0f172a;line-height:1.65">
  <h2 style="margin:0 0 4px;font-size:20px;color:#1e40af">重置你的 InsightCrew 密码</h2>
  <p style="margin:0 0 24px;color:#475569;font-size:14px">多 Agent 协作技术调研与选型平台</p>

  <p style="margin:0 0 12px">你好 {username}：</p>
  <p style="margin:0 0 24px">
    我们收到了重置该账号密码的请求。点击下面的按钮设置新密码，
    链接 <b>{ttl_min} 分钟内有效</b>，且只能使用一次。
  </p>

  <p style="margin:0 0 28px">
    <a href="{reset_url}"
       style="display:inline-block;background:#d97706;color:#fff;text-decoration:none;
              padding:12px 26px;border-radius:8px;font-weight:600">设置新密码</a>
  </p>

  <p style="margin:0 0 8px;color:#475569;font-size:13px">
    按钮点不开时，把下面这行地址复制到浏览器：
  </p>
  <p style="margin:0 0 28px;padding:10px 12px;background:#f1f5f9;border-radius:6px;
            font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;
            word-break:break-all;color:#334155">{reset_url}</p>

  <hr style="border:none;border-top:1px solid #e2e8f0;margin:0 0 16px">
  <p style="margin:0;color:#94a3b8;font-size:12px">
    如果这不是你发起的，忽略本邮件即可 —— 你的密码不会被改动。
  </p>
</div>"""
    text = _plain_from_html(html)
    return html, text


async def send_password_reset(to: str, reset_url: str, ttl_min: int, username: str) -> bool:
    """发送重置密码邮件。返回是否**真的投递出去**（降级时返回 False）。

    ★ 不抛异常。发信失败不该让 `/auth/forgot` 报错——那会把「邮箱是否注册」
      从响应差异里泄漏出去（防枚举的要求），也会让用户以为请求本身没成功。
      失败只记日志，接口照常回统一文案。
    """
    html, text = _build_reset_email(reset_url, ttl_min, username)
    subject = "重置你的 InsightCrew 密码"
    return await _deliver(to, subject, html, text, log_hint="重置链接")


# ── 邮箱验证码（用户 m26742）─────────────────────────────────────────────


#: 各用途在邮件里怎么称呼，以及为什么需要这一步。写进正文，用户才知道
#: 这封邮件对应哪个操作 —— 否则收到一封「你的验证码是 123456」会以为是钓鱼。
_PURPOSE_COPY = {
    "register": ("注册账号", "你正在用这个邮箱注册 InsightCrew 账号。"),
    "bind": ("绑定邮箱", "你正在把该邮箱绑定到 InsightCrew 账号。"),
    "verify": ("更换邮箱", "你正在更换 InsightCrew 账号的邮箱，这是对当前邮箱的确认。"),
    "password": ("修改密码", "你正在修改 InsightCrew 账号的密码。"),
}


def _build_code_email(code: str, ttl_min: int, purpose: str, username: str = "") -> tuple[str, str]:
    """返回 (html, text)。"""
    action, why = _PURPOSE_COPY.get(purpose, ("验证身份", "你正在执行一项需要验证邮箱的操作。"))
    greet = f"你好 {username}：" if username else "你好："
    # 码拆成两组显示（123 456），比一串六位数字好读；复制时用户自己会去掉空格。
    shown = f"{code[:3]} {code[3:]}"
    html = f"""\
<div style="font-family:-apple-system,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;
            max-width:560px;margin:0 auto;padding:32px 24px;color:#0f172a;line-height:1.65">
  <h2 style="margin:0 0 4px;font-size:20px;color:#1e40af">InsightCrew 邮箱验证码</h2>
  <p style="margin:0 0 24px;color:#475569;font-size:14px">多 Agent 协作技术调研与选型平台</p>

  <p style="margin:0 0 12px">{greet}</p>
  <p style="margin:0 0 20px">{why}请把下面的验证码填回页面，<b>{ttl_min} 分钟内有效</b>。</p>

  <p style="margin:0 0 24px;padding:16px 0;background:#eff6ff;border:1px solid #bfdbfe;
            border-radius:8px;text-align:center;font-size:30px;font-weight:700;
            letter-spacing:.18em;color:#1e40af;font-family:ui-monospace,Menlo,Consolas,monospace">{shown}</p>

  <hr style="border:none;border-top:1px solid #e2e8f0;margin:0 0 16px">
  <p style="margin:0;color:#94a3b8;font-size:12px">
    如果这不是你发起的，忽略本邮件即可 —— 没有这个验证码，{action}不会生效。
  </p>
</div>"""
    return html, _plain_from_html(html)


async def send_email_code(to: str, code: str, ttl_min: int, purpose: str,
                          username: str = "") -> bool:
    """发送邮箱验证码。返回是否真的投递出去（降级时 False）。

    ★ 与 `send_password_reset` 一样**不抛异常**：调用方（注册 / 绑定邮箱 /
      改密码）需要把「发信失败」与「验证码错误」区分对待 —— 前者要让用户
      稍后重试，后者要提示重新输入。异常在这里被吞掉，由返回值表达。
    """
    action, _ = _PURPOSE_COPY.get(purpose, ("验证身份", ""))
    html, text = _build_code_email(code, ttl_min, purpose, username)
    subject = f"【InsightCrew】{action}验证码"
    return await _deliver(to, subject, html, text, log_hint="验证码")


async def _deliver(to: str, subject: str, html: str, text: str, log_hint: str = "") -> bool:
    """真正投递。未配置 Key 时降级为打日志，返回 False。"""
    if not is_configured():
        # 降级：把邮件内容打到日志，并单独打一行链接方便直接打开
        logger.warning(
            "[mailer] 未配置 RESEND_API_KEY，邮件未真实发送（降级为日志输出）\n"
            f"  to={to}\n  subject={subject}"
        )
        link = _LINK_RE.search(text)
        if link:
            logger.warning(f"[mailer] {log_hint or '链接'}：{link.group(0)}")
        elif log_hint == "验证码":
            # 验证码邮件里没有链接，降级时把码本身打出来，否则本机根本没法联调
            m = re.search(r"\b(\d{3}\s\d{3})\b", text)
            if m:
                logger.warning(f"[mailer] 验证码：{m.group(1).replace(' ', '')}")
        return False

    payload = {
        "from": settings.mail_from,
        "to": [to],
        "subject": subject,
        "html": html,
        "text": text,
    }
    headers = {
        "Authorization": f"Bearer {settings.resend_api_key.strip()}",
        "Content-Type": "application/json",
    }
    try:
        # httpx 是同步客户端，放进线程池以免阻塞事件循环
        def _post() -> httpx.Response:
            with httpx.Client(timeout=15) as client:
                return client.post(RESEND_ENDPOINT, json=payload, headers=headers)

        resp = await asyncio.to_thread(_post)
    except Exception as exc:  # 网络层异常
        logger.error(f"[mailer] 发信失败（网络异常）：{exc}")
        return False

    if resp.status_code >= 400:
        # 常见 403：发件域名未验证（Resend 只允许发往账号本人邮箱）
        logger.error(f"[mailer] 发信失败 {resp.status_code}：{resp.text[:300]}")
        return False

    logger.info(f"[mailer] 邮件已投递 → {to}（Resend id={_resend_id(resp)})")
    return True


def _resend_id(resp: httpx.Response) -> str:
    try:
        return str(resp.json().get("id", ""))
    except Exception:
        return ""


async def send_test_email(to: str) -> bool:
    """连通性自检用：发一封最小邮件。"""
    if not is_configured():
        logger.warning("[mailer] 未配置 RESEND_API_KEY，无法发送测试邮件")
        return False
    payload = {
        "from": settings.mail_from,
        "to": [to],
        "subject": "InsightCrew 邮件通道自检",
        "text": "如果你收到这封邮件，说明 Resend 发信通道已经打通。",
    }
    headers = {
        "Authorization": f"Bearer {settings.resend_api_key.strip()}",
        "Content-Type": "application/json",
    }
    try:
        def _post() -> httpx.Response:
            with httpx.Client(timeout=15) as client:
                return client.post(RESEND_ENDPOINT, json=payload, headers=headers)

        resp = await asyncio.to_thread(_post)
    except Exception as exc:
        logger.error(f"[mailer] 测试邮件发送失败：{exc}")
        return False
    if resp.status_code >= 400:
        logger.error(f"[mailer] 测试邮件失败 {resp.status_code}：{resp.text[:300]}")
        return False
    return True


__all__ = [
    "is_configured",
    "send_password_reset",
    "send_email_code",
    "send_test_email",
]
