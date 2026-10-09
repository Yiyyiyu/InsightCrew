"""网页抓取工具

★ 引用台账：正文按段落切块写入 sources/chunks，
  返回文本中每个分块都带真实存在的 [S{source_id}#{chunk_seq}] 标记。
"""

from src.core.ledger import add_source, chunk_text
from crewai.tools import tool

_UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}
_FETCH_TIMEOUT = 12   # 原 30s 太宽松：坏 URL 会白等 30s 并被 Agent 反复重试


@tool("fetch_webpage")
def fetch_webpage(url: str) -> str:
    """抓取网页正文内容。输入: 完整 URL。"""
    import urllib.request, trafilatura

    req = urllib.request.Request(url, headers=_UA)
    try:
        with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT) as r:
            html = r.read().decode("utf-8", "replace")
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
