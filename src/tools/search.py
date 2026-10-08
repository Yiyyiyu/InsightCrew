"""搜索工具 — Bing HTML + arXiv API"""

from crewai.tools import tool


@tool("search_web")
def search_web(query: str) -> str:
    """搜索互联网获取技术资料，使用 Bing 搜索。输入: 搜索关键词。"""
    import urllib.request, urllib.parse, re, html
    q = urllib.parse.quote(query)
    url = f"https://www.bing.com/search?q={q}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    })
    with urllib.request.urlopen(req, timeout=25) as r:
        txt = r.read().decode("utf-8", "replace")
    results = []
    for blk in re.findall(r'<li class="b_algo".*?</li>', txt, re.S):
        m = re.search(r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', blk, re.S)
        if not m:
            continue
        u = m.group(1)
        t = html.unescape(re.sub(r"<[^>]+>", "", m.group(2))).strip()
        cap = re.search(r'<p[^>]*>(.*?)</p>', blk, re.S)
        c = html.unescape(re.sub(r"<[^>]+>", "", cap.group(1))).strip()[:200] if cap else ""
        results.append(f"- [{t}]({u})\n  {c}")
    return "\n\n".join(results[:8]) if results else "未搜索到结果"


@tool("search_arxiv")
def search_arxiv(query: str) -> str:
    """搜索 arXiv 学术论文。输入: 搜索关键词。"""
    import urllib.request, urllib.parse, re
    q = urllib.parse.quote(query)
    url = f"http://export.arxiv.org/api/query?search_query=all:{q}&max_results=8&sortBy=relevance&sortOrder=descending"
    with urllib.request.urlopen(url, timeout=30) as r:
        txt = r.read().decode("utf-8", "replace")
    results = []
    for entry in re.findall(r"<entry>(.*?)</entry>", txt, re.S):
        title_m = re.search(r"<title>(.*?)</title>", entry, re.S)
        id_m = re.search(r"<id>(.*?)</id>", entry)
        summary_m = re.search(r"<summary>(.*?)</summary>", entry, re.S)
        title = re.sub(r"\s+", " ", title_m.group(1)).strip() if title_m else ""
        url_id = id_m.group(1).strip() if id_m else ""
        summary = re.sub(r"\s+", " ", summary_m.group(1)).strip()[:200] if summary_m else ""
        results.append(f"- [{title}]({url_id})\n  {summary}")
    return "\n\n".join(results[:8]) if results else "arXiv 未搜索到结果"
