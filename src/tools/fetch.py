"""网页抓取工具"""

from crewai.tools import tool


@tool("fetch_webpage")
def fetch_webpage(url: str) -> str:
    """抓取网页正文内容。输入: 完整 URL。"""
    import urllib.request, trafilatura
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        html = r.read().decode("utf-8", "replace")
    text = trafilatura.extract(html)
    return text[:3000] if text else "无法提取正文内容"
