"""出站 URL 安全校验（SSRF 防护）。

## 为什么必须有这个模块

Agent 会抓取"任何 URL"：`fetch_webpage` / `parse_pdf` 的入参来自检索结果，
而检索结果又可能来自用户提供的主题或人工投喂的链接。如果服务端直接
`urlopen(url)`，任何人都能让**我们的服务器**去请求内网地址：

    http://169.254.169.254/latest/meta-data/iam/security-credentials/
    http://127.0.0.1:6379/          （Redis）
    http://10.0.0.5:8080/admin      （内网后台）

云厂商的元数据服务（169.254.169.254）是经典目标：一条 GET 就能读到实例
的临时凭据。这属于"低概率但后果严重"的一类风险，PRD §10 与设计规划
C4-2 都要求收口。

## 两道防线

1. **静态校验**（`assert_public_url`）：只允许 http/https；解析主机名后
   拒绝环回、私网、链路本地、CGNAT、IPv6 唯一本地地址等。
2. **重定向校验**（`safe_urlopen`）：默认 `urlopen` 会自动跟随 302/301，
   攻击者可用"公网 URL → 302 到 169.254.169.254"绕过第一道防线。
   因此这里换成自定义 opener，**每一跳都重新校验**。

DNS rebinding（解析时是公网、连接时变内网）理论上仍可绕过，但需要攻击者
控制 DNS 且能精确控制 TTL 时序，属于加固范围之外的已知残余风险。
"""

from __future__ import annotations

import ipaddress
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from src.core.logger import logger

# 开发/测试逃生口：本机验证脚本（如 tools/verify_arxiv_pdf.py）会用
# 127.0.0.1 上的临时 HTTP 服务托管文件来测 PDF 解析路径。默认关闭；
# 只有显式设置 ALLOW_PRIVATE_FETCH=1 才放开。
# ★ 生产环境绝不能设这个变量——它会让 SSRF 防护整体失效。
_ALLOW_PRIVATE_ENV = "ALLOW_PRIVATE_FETCH"


def private_fetch_allowed() -> bool:
    """是否允许抓取内网地址（默认 False，仅本地测试用）。"""
    return (os.environ.get(_ALLOW_PRIVATE_ENV, "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

# 只允许这两种协议。file://、gopher://、ftp:// 等一律拒绝——
# 它们要么能读本地文件，要么能打非 HTTP 服务（gopher 是经典 SSRF 跳板）。
_ALLOWED_SCHEMES = ("http", "https")

# 明确拒绝的主机名（DNS 解析前先拦，避免 localhost 被解析成 127.0.0.1 之外的写法绕过）
_BLOCKED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "ip6-localhost",
    "ip6-loopback",
    "metadata",
    "metadata.google.internal",
}

# 主机名解析结果里，任何一个 IP 命中这些网段就拒绝。
# 网段说明：
#   0.0.0.0/8         "本网络"
#   10/8 172.16/12 192.168/16   RFC1918 私网
#   127/8             环回
#   169.254/16        链路本地（含 AWS/GCP/Azure 元数据 169.254.169.254）
#   100.64/10         CGNAT（RFC6598）
#   192.0.0/24        IETF 协议分配
#   192.0.2/24 198.51.100/24 203.0.113/24  文档用（不该出现在真实流量里）
#   198.18/15         基准测试
#   224/4             组播
#   240/4             保留
#   ::/128 ::1/128    IPv6 未指定 / 环回
#   fc00::/7          IPv6 唯一本地地址
#   fe80::/10         IPv6 链路本地
#   ::ffff:0:0/96     IPv4 映射地址（需按映射后的 IPv4 再判一次）
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
    ipaddress.ip_network("224.0.0.0/4"),
    ipaddress.ip_network("240.0.0.0/4"),
    ipaddress.ip_network("::/128"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


class UnsafeUrlError(ValueError):
    """URL 不满足出站安全策略（协议不允许 / 指向内网 / 无法解析）。"""


def _is_blocked_ip(ip: ipaddress._BaseAddress) -> bool:
    """判断单个 IP 是否落在禁止网段内。"""
    # IPv4-mapped IPv6（::ffff:10.0.0.1）要按内嵌的 IPv4 再判一次，
    # 否则会被当成"普通 IPv6 公网地址"放行。
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    for net in _BLOCKED_NETWORKS:
        if ip.version != net.version:
            continue
        if ip in net:
            return True
    return False


def _resolve_ips(hostname: str) -> list[str]:
    """解析主机名的全部 A/AAAA 记录。

    返回空列表表示解析失败——调用方应视为不安全并拒绝
    （解析不了就没法确认它是不是内网，宁可拒绝）。
    """
    try:
        infos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError, OSError) as exc:
        logger.debug(f"SSRF 校验：主机名 {hostname!r} 解析失败: {exc}")
        return []
    ips: list[str] = []
    for info in infos:
        addr = info[4][0]
        if addr not in ips:
            ips.append(addr)
    return ips


def assert_public_url(url: str, *, allow_private: bool = False) -> str:
    """校验 URL 是否允许服务端出站访问，通过则原样返回。

    `allow_private=True` 供**显式受信任**的场景使用（例如用户自建的
    本地 OneAPI/Ollama 网关探测）——默认关闭，普通抓取一律走严格模式。

    抛 `UnsafeUrlError`，消息面向使用者、不含内部实现细节。
    """
    raw = (url or "").strip()
    if not raw:
        raise UnsafeUrlError("URL 为空")

    # 补全无协议写法前先拒绝：urllib 对 "127.0.0.1:8000" 会误判 scheme，
    # 直接要求调用方给完整 URL，避免歧义。
    parts = urllib.parse.urlsplit(raw)
    scheme = (parts.scheme or "").lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise UnsafeUrlError(f"仅允许 http/https 协议，收到 {scheme or '(空)'!r}")

    hostname = parts.hostname
    if not hostname:
        raise UnsafeUrlError("URL 缺少主机名")

    host_lower = hostname.lower().rstrip(".")
    if host_lower in _BLOCKED_HOSTNAMES:
        raise UnsafeUrlError(f"拒绝访问内部主机名 {hostname!r}")

    if allow_private or private_fetch_allowed():
        return raw

    # 主机名本身写成 IP 字面量时不必走 DNS
    try:
        literal = ipaddress.ip_address(host_lower.strip("[]"))
    except ValueError:
        literal = None

    if literal is not None:
        if _is_blocked_ip(literal):
            raise UnsafeUrlError(f"拒绝访问内部地址 {hostname!r}")
        return raw

    ips = _resolve_ips(host_lower)
    if not ips:
        raise UnsafeUrlError(f"无法解析主机名 {hostname!r}")

    for addr in ips:
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if _is_blocked_ip(ip):
            raise UnsafeUrlError(
                f"主机名 {hostname!r} 解析到内部地址 {addr}，已拒绝"
            )

    return raw


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """重定向处理器：每一跳都重新做 SSRF 校验。

    默认的 `HTTPRedirectHandler` 会无脑跟随 Location 头，等于给了攻击者
    一个"公网入口 → 内网目标"的跳板。
    """

    def __init__(self, allow_private: bool = False) -> None:
        super().__init__()
        self._allow_private = allow_private

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        assert_public_url(newurl, allow_private=self._allow_private)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def safe_urlopen(
    url: str,
    *,
    headers: Optional[dict] = None,
    timeout: int = 12,
    allow_private: bool = False,
):
    """校验后发起请求，返回 urllib 响应对象（可作上下文管理器）。

    调用方需自行处理 `UnsafeUrlError`（安全拒绝）与 `urllib.error.URLError`
    （网络失败）——两者语义不同，提示文案也应不同。
    """
    assert_public_url(url, allow_private=allow_private)
    req = urllib.request.Request(url, headers=headers or {})
    opener = urllib.request.build_opener(
        _SafeRedirectHandler(allow_private=allow_private)
    )
    return opener.open(req, timeout=timeout)


__all__ = [
    "UnsafeUrlError",
    "assert_public_url",
    "safe_urlopen",
    "private_fetch_allowed",
]
