"""PDF 解析工具 —— 把 PDF（尤其 arXiv 论文）转成带**页码**的可引用分块。

## 为什么页码是硬要求

PRD 的引用分级里，`pdf` 映射为 **A 级（全文验证）**：论断要能回溯到
"哪一篇的第几页"。`Chunk.page_num` 列一直存在，`ledger.add_source()`
也早已支持 `page_nums=` 形参，只是此前没有任何调用方——本模块补上这个缺口。

## 为什么优先抓 arXiv LaTeX 源码

arXiv 上多数论文提供 LaTeX 源码包（`/e-print/<id>`）。LaTeX 里章节结构
是显式的（`\section{...}`），公式是源码而不是渲染后的乱码，对下游
Analyst/Critic 更可用。但源码包是 tar.gz、结构不固定，解析失败率高，
因此做成**优先尝试、失败回落 PDF**，而不是替代。

## 超时纪律

与 search/fetch 一致：网络操作一律短超时，失败立刻返回并明确告知
"不要重试"，避免 Agent 在坏 URL 上反复重试把阶段拖到超时。
"""

from __future__ import annotations

import re
from typing import Optional

from crewai.tools import tool

from src.core.ledger import add_source, chunk_text
from src.core.net import UnsafeUrlError

_UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}
_DOWNLOAD_TIMEOUT = 20   # PDF 比网页大，给 20s（网页是 12s）

# 单篇 PDF 的分块上限：论文动辄几十页，不设上限会把上下文撑爆，
# 且成本随 chunk 数线性上涨。60 块 × 800 字符 ≈ 4.8 万字符，够覆盖论文主体。
_MAX_CHUNKS = 60

# 每块最小字符数：低于此值的页面多是空白页/纯图表页，单独成块没有信息量。
_MIN_CHUNK_CHARS = 40

_ARXIV_ID_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf|e-print)/([0-9]{4}\.[0-9]{4,5}|[a-z\-]+/\d{7})",
    re.I,
)


# ───────────────────────────────────────────────────────────── 下载
def _download(url: str) -> Optional[bytes]:
    """下载二进制内容，失败返回 None（由调用方转成给 LLM 的提示文本）。

    ★ 走 `safe_urlopen` 做 SSRF 校验：`parse_pdf` 的 URL 同样来自 Agent，
      与 `fetch_webpage` 是同一条攻击面（C4-2 要求两个工具一起修）。
    """
    from src.core.net import UnsafeUrlError, safe_urlopen

    try:
        with safe_urlopen(url, headers=_UA, timeout=_DOWNLOAD_TIMEOUT) as r:
            return r.read()
    except UnsafeUrlError:
        # 安全拒绝与网络失败要区分：调用方据此给出不同提示
        raise
    except Exception:
        return None


# ───────────────────────────────────────────────────────────── PDF 解析
def _pdf_pages(data: bytes) -> list[tuple[int, str]]:
    """逐页抽取文本，返回 [(页码从 1 开始, 文本)]。

    ★ `page.get_text()` 的页码从 0 开始，写入台账时必须 +1——
    引用里说"第 3 页"指的是人类看到的第 3 页，不是索引 2。
    """
    import fitz  # PyMuPDF

    out: list[tuple[int, str]] = []
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        if doc.needs_pass:
            raise ValueError("PDF 已加密，需要密码")
        for i, page in enumerate(doc):
            try:
                txt = page.get_text() or ""
            except Exception:
                txt = ""
            txt = txt.strip()
            if txt:
                out.append((i + 1, txt))
    finally:
        doc.close()
    return out


def _chunks_with_pages(pages: list[tuple[int, str]]) -> tuple[list[str], list[int]]:
    """把逐页文本切成块，并保留每块对应的页码。

    切块策略：**按页切块**，而不是把所有页拼起来再切。
    理由：引用要能定位到具体页码；跨页拼接会让一个 chunk 横跨第 3、4 页，
    页码就没法填了。代价是块会比 800 字符小，可接受。

    页内文本过长时按 chunk_text 再切，切出的子块共用同一页码。
    """
    texts: list[str] = []
    nums: list[int] = []

    for page_no, raw in pages:
        body = re.sub(r"[ \t]+", " ", raw)
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        if len(body) < _MIN_CHUNK_CHARS:
            continue
        # 页内文本超长（常见于双栏论文）→ 再按段落聚合切分，页码保持
        if len(body) > 800:
            pieces = chunk_text(body, size=800)
        else:
            pieces = [body]
        for p in pieces:
            if not p.strip():
                continue
            texts.append(p.strip())
            nums.append(page_no)
            if len(texts) >= _MAX_CHUNKS:
                return texts, nums

    return texts, nums


# ───────────────────────────────────────────────────────────── arXiv 源码
def _arxiv_id(url: str) -> Optional[str]:
    m = _ARXIV_ID_RE.search(url or "")
    return m.group(1) if m else None


def _try_latex_source(arxiv_id: str) -> Optional[tuple[str, str]]:
    """尝试抓 arXiv 的 LaTeX 源码包，返回 (正文, 说明)；失败返回 None。

    只做**尽力而为**：源码包是 tar.gz，可能缺文件、可能不是 LaTeX、
    可能编码非 UTF-8。任何一步失败都静默回落 PDF，不影响主路径。
    """
    import io
    import tarfile

    data = _download(f"https://arxiv.org/e-print/{arxiv_id}")
    if not data:
        return None
    try:
        tf = tarfile.open(fileobj=io.BytesIO(data), mode="r:*")
    except Exception:
        return None

    parts: list[str] = []
    try:
        members = [m for m in tf.getmembers() if m.isfile()]
        # 主文件通常是最大的 .tex；按大小降序取前 3 个，避免把附录全塞进来
        tex = [m for m in members if m.name.lower().endswith(".tex")]
        tex.sort(key=lambda m: m.size, reverse=True)
        for m in tex[:3]:
            f = tf.extractfile(m)
            if f is None:
                continue
            raw = f.read()
            for enc in ("utf-8", "latin-1"):
                try:
                    parts.append(raw.decode(enc))
                    break
                except UnicodeDecodeError:
                    continue
    except Exception:
        return None
    finally:
        try:
            tf.close()
        except Exception:
            pass

    if not parts:
        return None
    text = "\n\n".join(parts)
    if len(text.strip()) < 200:
        return None
    return text, "LaTeX 源码"


# ───────────────────────────────────────────────────────────── 工具入口
@tool("parse_pdf")
def parse_pdf(url: str) -> str:
    """下载并解析 PDF 论文（支持 arXiv），按页提取正文并生成带页码的引用标记。

    输入: PDF 的完整 URL（如 https://arxiv.org/pdf/2312.12345）。
    输出: 带 [S{id}#{seq}] 标记的正文分块，标记可回溯到具体页码。
    """
    url = (url or "").strip()
    if not url:
        return "URL 为空。"
    if not url.lower().startswith(("http://", "https://")):
        return "只支持 http(s) URL。"

    title = ""

    # ── 路径 1：arXiv 源码（优先，结构更清晰）──
    aid = _arxiv_id(url)
    if aid:
        src = _try_latex_source(aid)
        if src:
            text, how = src
            pieces = chunk_text(text, size=800)
            # LaTeX 源码没有可靠页码（.tex 文件不映射到 PDF 页码），
            # 不伪造页码：page_nums 传 None，证据等级仍是 A（全文），
            # 但引用里不含页码，前端会显示"无页码"。
            sid, chunks = add_source(
                f"https://arxiv.org/abs/{aid}", f"arXiv:{aid} (LaTeX)", "pdf",
                pieces[:_MAX_CHUNKS], raw_text=text,
            )
            lines = [
                f"来源 [S{sid}]（arXiv {aid}，{how}，共 {len(chunks)} 个分块）。"
                "引用时使用对应的 [S{id}#n] 标记。注意：源码无 PDF 页码。",
            ]
            for (_, seq), piece in zip(chunks, pieces[:_MAX_CHUNKS]):
                lines.append(f"[S{sid}#{seq}] {piece}")
            return "\n\n".join(lines)

    # ── 路径 2：PDF 正文 ──
    try:
        data = _download(url)
    except UnsafeUrlError as e:
        return (f"该 URL 不被允许下载（{e}）。"
                "请勿重试同一 URL，改用其他公开来源。")
    if not data:
        return (f"PDF 下载失败（{url}）。请勿重试同一 URL，改用其他来源。")
    if not data[:5].startswith(b"%PDF"):
        return (f"该 URL 返回的不是 PDF（{url}）。"
                "若是 arXiv 页面，请改用 /pdf/ 或 /abs/ 形式的链接。")

    try:
        pages = _pdf_pages(data)
    except Exception as e:
        return (f"PDF 解析失败（{type(e).__name__}: {e}）。"
                "请勿重试，改用其他来源。")

    if not pages:
        return f"PDF 无可用文本（可能是扫描件，需要 OCR）：{url}"

    texts, nums = _chunks_with_pages(pages)
    if not texts:
        return f"PDF 解析后无有效文本：{url}"

    raw_text = "\n\n".join(f"[p{n}]\n{t}" for n, t in pages)
    sid, chunks = add_source(url, title, "pdf", texts, raw_text=raw_text,
                             page_nums=nums)

    lines = [
        f"来源 [S{sid}] 共 {len(chunks)} 个分块（PDF，共 {len(pages)} 页有文本）。"
        "每个分块标注了对应页码，引用时必须使用 [S{id}#n] 标记。",
    ]
    for (_, seq), piece, pno in zip(chunks, texts, nums):
        lines.append(f"[S{sid}#{seq}] (第 {pno} 页) {piece}")
    return "\n\n".join(lines)


__all__ = ["parse_pdf"]
