"""搜索工具 — Bing HTML + arXiv API

★ 引用台账：检索结果在返回前先写入 sources/chunks，
  返回文本带**真实存在**的 [S{source_id}#{chunk_seq}] 标记。
  （PRD 硬约束：不入库不引用）

★ 超时纪律：网络工具一律短超时，失败即快速返回并明确告知"不要重试"，
  避免 Agent 在坏端点上反复重试把整个阶段拖到超时。
"""

from src.core.ledger import add_source
from crewai.tools import tool

_UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}
_SEARCH_TIMEOUT = 8   # Bing 实测 0.6~0.8s，8s 足够
_ARXIV_TIMEOUT = 8


@tool("search_web")
def search_web(query: str) -> str:
    """搜索互联网获取技术资料，使用 Bing 搜索。输入: 搜索关键词。"""
    import urllib.request, urllib.parse, re, html

    q = urllib.parse.quote(query)
    url = f"https://www.bing.com/search?q={q}"
    req = urllib.request.Request(url, headers=_UA)
    try:
        with urllib.request.urlopen(req, timeout=_SEARCH_TIMEOUT) as r:
            txt = r.read().decode("utf-8", "replace")
    except Exception as e:
        return (f"搜索失败（{type(e).__name__}: {e}）。"
                "请勿重复相同查询，改用其他关键词或直接使用 fetch_webpage。")

    blocks = []
    for blk in re.findall(r'<li class="b_algo".*?</li>', txt, re.S):
        m = re.search(r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', blk, re.S)
        if not m:
            continue
        u = m.group(1)
        t = html.unescape(re.sub(r"<[^>]+>", "", m.group(2))).strip()
        cap = re.search(r'<p[^>]*>(.*?)</p>', blk, re.S)
        c = html.unescape(re.sub(r"<[^>]+>", "", cap.group(1))).strip()[:400] if cap else ""
        blocks.append((u, t, c))
        if len(blocks) >= 8:
            break

    if not blocks:
        return "未搜索到结果"

    lines = []
    for u, t, c in blocks:
        sid, chunks = add_source(u, t, "search_snippet", [c or t])
        _, seq = chunks[0]
        lines.append(f"[S{sid}#{seq}] {t}\nURL: {u}\n摘要: {c}")
    return "\n\n".join(lines)


@tool("search_arxiv")
def search_arxiv(query: str) -> str:
    """搜索 arXiv 学术论文。输入: 搜索关键词。"""
    import urllib.request, urllib.parse, re, time

    q = urllib.parse.quote(query)
    url = (f"http://export.arxiv.org/api/query?search_query=all:{q}"
           f"&max_results=8&sortBy=relevance&sortOrder=descending")
    last_err = None
    txt = ""
    for attempt in range(2):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA["User-Agent"]})
            with urllib.request.urlopen(req, timeout=_ARXIV_TIMEOUT) as r:
                txt = r.read().decode("utf-8", "replace")
            last_err = None
            break
        except Exception as e:
            last_err = e
            if attempt == 0:
                time.sleep(1.5)

    if last_err is not None:
        return (f"arXiv 检索暂不可用（{type(last_err).__name__}: {last_err}）。"
                "请勿重试，改用 search_web 获取资料。")

    results = []
    for entry in re.findall(r"<entry>(.*?)</entry>", txt, re.S):
        title_m = re.search(r"<title>(.*?)</title>", entry, re.S)
        id_m = re.search(r"<id>(.*?)</id>", entry)
        summary_m = re.search(r"<summary>(.*?)</summary>", entry, re.S)
        title = re.sub(r"\s+", " ", title_m.group(1)).strip() if title_m else ""
        url_id = id_m.group(1).strip() if id_m else ""
        summary = re.sub(r"\s+", " ", summary_m.group(1)).strip()[:400] if summary_m else ""
        if not url_id:
            continue
        results.append((url_id, title, summary))

    if not results:
        return "arXiv 未搜索到结果"

    lines = []
    for url_id, title, summary in results[:8]:
        sid, chunks = add_source(url_id, title, "arxiv_snippet", [summary or title])
        _, seq = chunks[0]
        lines.append(f"[S{sid}#{seq}] {title}\nURL: {url_id}\n摘要: {summary}")
    return "\n\n".join(lines)
