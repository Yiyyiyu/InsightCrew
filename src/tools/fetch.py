"""网页抓取工具

★ 引用台账：正文按段落切块写入 sources/chunks，
  返回文本中每个分块都带真实存在的 [S{source_id}#{chunk_seq}] 标记。

★ SSRF 防护：URL 来自 Agent（可能被检索结果或用户输入诱导），
  必须走 src.core.net 的出站校验，不能直接 urlopen——
  否则 `http://169.254.169.254/...` 这类地址会让服务器自己去读云凭据。
"""

from src.core.ledger import add_source, chunk_text
from src.core.net import UnsafeUrlError, safe_urlopen
from crewai.tools import tool

_UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}
_FETCH_TIMEOUT = 12   # 原 30s 太宽松：坏 URL 会白等 30s 并被 Agent 反复重试


@tool("fetch_webpage")
def fetch_webpage(url: str) -> str:
    """抓取网页正文内容。输入: 完整 URL。"""
    import trafilatura

    try:
        with safe_urlopen(url, headers=_UA, timeout=_FETCH_TIMEOUT) as r:
            html = r.read().decode("utf-8", "replace")
    except UnsafeUrlError as e:
        # 安全拒绝：明确告知不可重试，且不泄露"为什么被拒"的内部细节
        return (f"该 URL 不被允许抓取（{e}）。"
                "请勿重试同一 URL，改用其他公开来源。")
    except Exception as e:
        return (f"抓取失败（{type(e).__name__}: {e}）。"
                "请勿重试同一 URL，改用其他来源。")

    text = trafilatura.extract(html)
    if not text:
        return "无法提取正文内容"

    pieces = chunk_text(text, size=800)
    sid, chunks = add_source(url, "", "web", pieces, raw_text=text)

    lines = [f"来源 [S{sid}] 共 {len(chunks)} 个分块，引用时请使用对应的 [S{sid}#n] 标记："]
    for (_, seq), piece in zip(chunks, pieces):
        lines.append(f"[S{sid}#{seq}] {piece}")
    return "\n\n".join(lines)
