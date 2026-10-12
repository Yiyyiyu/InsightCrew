"""报告导出 —— Markdown → PDF（中文排版 + 参考文献）

设计
----
* **解析用 markdown-it-py 的 token 流**，不用正则拼 HTML/Markup。报告里有 189 行
  表格与嵌套列表，正则方案在 `**粗体**` 与 `|` 同时出现时会误判。
* **中文字体两级策略**：优先内嵌 TTF（Windows `msyh` / Linux Noto CJK），
  找不到则回落到 reportlab 内置的 Adobe CID 字体 `STSong-Light`（不内嵌、
  依赖阅读器自带字体，但无需任何字体文件即可工作）。
  `PDF_CJK_FONT` 环境变量可强制指定字体文件。
* **参考文献**：直接解析 `src/core/bibtex.py` 生成的 `.bib` 文本（本仓库自产的
  稳定格式），渲染成带编号的参考文献章节 —— 报告正文有 300+ 个 `[S{id}#{seq}]`
  标记，导出物必须自洽，否则用户拿到 PDF 也查不到出处。

只用 `render_pdf` 一个入口；调用方（`src/web/routes.py`）负责鉴权与取数。
"""

from __future__ import annotations

import io
import os
import re

from markdown_it import MarkdownIt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── 字体 ────────────────────────────────────────────────────────────────────
# 候选字体文件：Windows / Linux(Noto) / macOS。第一个能注册成功的胜出。
_REGULAR_CANDIDATES = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)
_BOLD_CANDIDATES = (
    "C:/Windows/Fonts/msyhbd.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
)
_FALLBACK_CID = "STSong-Light"

_FONTS: dict[str, str] = {}


def _register_fonts() -> tuple[str, str]:
    """注册中文字体，返回 (常规字体名, 粗体字体名)。进程内只做一次。"""
    if _FONTS:
        return _FONTS["regular"], _FONTS["bold"]

    from reportlab.lib.fonts import addMapping

    regular = ""
    forced = os.environ.get("PDF_CJK_FONT", "").strip()
    candidates = ([forced] if forced else []) + list(_REGULAR_CANDIDATES)
    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        try:
            pdfmetrics.registerFont(TTFont("ICReg", path, subfontIndex=0))
            regular = "ICReg"
            break
        except Exception:  # noqa: BLE001 —— 单个字体文件不可用不应中断导出
            continue

    bold = ""
    for path in _BOLD_CANDIDATES:
        if not os.path.exists(path):
            continue
        try:
            pdfmetrics.registerFont(TTFont("ICBold", path, subfontIndex=0))
            bold = "ICBold"
            break
        except Exception:  # noqa: BLE001
            continue

    if not regular:
        # 回落：reportlab 内置 Adobe CID 中文字体，不需要任何字体文件
        pdfmetrics.registerFont(UnicodeCIDFont(_FALLBACK_CID))
        regular = _FALLBACK_CID
    if not bold:
        # 没有粗体变体时映射到常规字体：宁可「粗体不粗」，也不能让 `<b>` 落到
        # Helvetica 导致中文变方框。
        bold = regular

    addMapping(regular, 0, 0, regular)
    addMapping(regular, 1, 0, bold)
    addMapping(regular, 0, 1, regular)
    addMapping(regular, 1, 1, bold)

    _FONTS["regular"] = regular
    _FONTS["bold"] = bold
    return regular, bold


def font_status() -> dict[str, str]:
    """当前使用的字体（供 /healthz 与验证脚本查询）。"""
    regular, bold = _register_fonts()
    return {"regular": regular, "bold": bold,
            "embedded": "yes" if regular != _FALLBACK_CID else "no"}


# ── Markdown → reportlab flowables ─────────────────────────────────────────
_MD = MarkdownIt("commonmark", {"html": False, "linkify": False}).enable("table")


def _esc(text: str) -> str:
    """转义 reportlab Paragraph 的 XML 保留字符。"""
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _inline(children: list) -> str:
    """把 inline token 的子节点渲染成 Paragraph 内联标记。"""
    out: list[str] = []
    link_href: list[str] = []
    for tok in children or []:
        t = tok.type
        if t == "text":
            out.append(_esc(tok.content))
        elif t == "code_inline":
            out.append(_esc(tok.content))
        elif t == "strong_open":
            out.append("<b>")
        elif t == "strong_close":
            out.append("</b>")
        elif t == "em_open":
            out.append("<i>")
        elif t == "em_close":
            out.append("</i>")
        elif t == "link_open":
            href = tok.attrGet("href") or ""
            link_href.append(href)
            out.append(f'<link href="{_esc(href)}">')
        elif t == "link_close":
            if link_href:
                link_href.pop()
            out.append("</link>")
        elif t in ("softbreak", "hardbreak"):
            out.append("<br/>" if t == "hardbreak" else " ")
        elif t == "image":
            # 报告正文不含图片（图注已由 vision 阶段转成文字）；占位不阻塞导出
            out.append(_esc(tok.content or ""))
        else:
            if tok.content:
                out.append(_esc(tok.content))
    return "".join(out)


def _styles() -> dict[str, ParagraphStyle]:
    regular, bold = _register_fonts()
    # `base` 只放所有样式共用的键；`textColor` 单独传给需要的样式，
    # 否则 `ParagraphStyle(**base, textColor=...)` 会因重复关键字报 TypeError。
    base = dict(fontName=regular, alignment=TA_LEFT)
    return {
        "h1": ParagraphStyle("h1", **base, textColor=colors.black,
                             fontSize=17, leading=24, spaceBefore=2, spaceAfter=10),
        "h2": ParagraphStyle("h2", **base, textColor=colors.black,
                             fontSize=14, leading=21, spaceBefore=14, spaceAfter=7),
        "h3": ParagraphStyle("h3", **base, textColor=colors.black,
                             fontSize=12, leading=18, spaceBefore=11, spaceAfter=5),
        "h4": ParagraphStyle("h4", **base, textColor=colors.black,
                             fontSize=11, leading=17, spaceBefore=9, spaceAfter=4),
        "body": ParagraphStyle("body", **base, textColor=colors.black,
                               fontSize=10, leading=16.5, spaceAfter=5),
        "quote": ParagraphStyle("quote", **base, textColor=colors.HexColor("#3d4a5c"),
                                fontSize=9.5, leading=15.5,
                                leftIndent=10, spaceAfter=5),
        "cell": ParagraphStyle("cell", **base, textColor=colors.black,
                               fontSize=8, leading=12),
        "cellhead": ParagraphStyle("cellhead", alignment=TA_LEFT, fontName=bold,
                                   textColor=colors.black, fontSize=8.5, leading=12.5),
        "ref": ParagraphStyle("ref", **base, textColor=colors.black,
                              fontSize=8.5, leading=13.5,
                              leftIndent=16, firstLineIndent=-16, spaceAfter=3),
        "foot": ParagraphStyle("foot", **base, textColor=colors.HexColor("#8a8a8a"),
                               fontSize=7.5, leading=10),
    }


def _table_width() -> float:
    return A4[0] - 36 * mm


def _md_tokens(md: str) -> list:
    return _MD.parse(md or "")


def markdown_to_flowables(md: str, st: dict[str, ParagraphStyle]) -> list:
    """把 Markdown 文本转成 reportlab flowable 列表。"""
    flow: list = []
    tokens = _md_tokens(md)
    i = 0
    n = len(tokens)

    def cell_par(text: str, head: bool) -> Paragraph:
        return Paragraph(text or "&nbsp;", st["cellhead"] if head else st["cell"])

    while i < n:
        tok = tokens[i]
        t = tok.type

        if t == "heading_open":
            level = tok.tag  # h1..h6
            inline = tokens[i + 1] if i + 1 < n else None
            text = _inline(inline.children if inline else [])
            flow.append(Paragraph(text, st.get(level, st["h4"])))
            i += 3
            continue

        if t == "paragraph_open":
            inline = tokens[i + 1] if i + 1 < n else None
            text = _inline(inline.children if inline else [])
            if text.strip():
                flow.append(Paragraph(text, st["body"]))
            i += 3
            continue

        if t == "blockquote_open":
            depth = 1
            j = i + 1
            parts: list[str] = []
            while j < n and depth > 0:
                if tokens[j].type == "blockquote_open":
                    depth += 1
                elif tokens[j].type == "blockquote_close":
                    depth -= 1
                    if depth == 0:
                        break
                elif tokens[j].type == "inline":
                    parts.append(_inline(tokens[j].children))
                j += 1
            for p in parts:
                if p.strip():
                    flow.append(Paragraph(p, st["quote"]))
            i = j + 1
            continue

        if t == "bullet_list_open" or t == "ordered_list_open":
            ordered = t == "ordered_list_open"
            depth = 1
            j = i + 1
            items: list[list] = []
            current: list = []
            while j < n:
                tt = tokens[j].type
                if tt in ("bullet_list_open", "ordered_list_open"):
                    depth += 1
                elif tt in ("bullet_list_close", "ordered_list_close"):
                    depth -= 1
                    if depth == 0:
                        break
                elif tt == "list_item_open":
                    current = []
                elif tt == "inline":
                    current.append(_inline(tokens[j].children))
                elif tt == "list_item_close" and current:
                    items.append(current)
                j += 1
            if items:
                flow.append(ListFlowable(
                    [ListItem(Paragraph(" ".join(it), st["body"]), leftIndent=14)
                     for it in items],
                    bulletType="1" if ordered else "bullet",
                    bulletFontName=st["body"].fontName,
                    bulletFontSize=9,
                    leftIndent=14,
                ))
            i = j + 1
            continue

        if t == "table_open":
            depth = 1
            j = i + 1
            rows: list[list[str]] = []
            head_rows = 0
            row: list[str] = []
            in_head = False
            while j < n:
                tt = tokens[j].type
                if tt == "table_open":
                    depth += 1
                elif tt == "table_close":
                    depth -= 1
                    if depth == 0:
                        break
                elif tt == "thead_open":
                    in_head = True
                elif tt == "thead_close":
                    in_head = False
                elif tt == "tr_open":
                    row = []
                elif tt == "inline":
                    row.append(_inline(tokens[j].children))
                elif tt == "tr_close":
                    if row:
                        rows.append(row)
                        if in_head:
                            head_rows += 1
                j += 1
            if rows:
                ncol = max(len(r) for r in rows)
                rows = [r + [""] * (ncol - len(r)) for r in rows]
                avail = _table_width()
                col_w = [avail / ncol] * ncol
                data = [[cell_par(c, ri < max(head_rows, 1)) for c in r]
                        for ri, r in enumerate(rows)]
                tbl = Table(data, colWidths=col_w, repeatRows=max(head_rows, 1))
                tbl.setStyle(TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c8ccd4")),
                    ("BACKGROUND", (0, 0), (-1, max(head_rows - 1, 0)),
                     colors.HexColor("#eef2f7")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]))
                flow.append(Spacer(1, 3))
                flow.append(tbl)
                flow.append(Spacer(1, 7))
            i = j + 1
            continue

        if t == "fence" or t == "code_block":
            text = (tok.content or "").rstrip("\n")
            if text.strip():
                para = Paragraph(_esc(text).replace("\n", "<br/>"), st["cell"])
                box = Table([[para]], colWidths=[_table_width()])
                box.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f6f8fa")),
                    ("BOX", (0, 0), (-1, -1), 0.35, colors.HexColor("#d8dee6")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]))
                flow.append(box)
                flow.append(Spacer(1, 6))
            i += 1
            continue

        if t == "hr":
            flow.append(Spacer(1, 3))
            flow.append(HRFlowable(width="100%", thickness=0.6,
                                   color=colors.HexColor("#d0d5dd")))
            flow.append(Spacer(1, 6))
            i += 1
            continue

        i += 1

    return flow


# ── BibTeX → 参考文献列表 ───────────────────────────────────────────────────
_BIB_ENTRY_RE = re.compile(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", re.S)
_BIB_FIELD_RE = re.compile(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*(?=\n\s*\w+\s*=|\Z)", re.S)
_BIB_URL_RE = re.compile(r"\\url\{([^}]*)\}")


def _bib_unescape(text: str) -> str:
    """还原 `src/core/bibtex.py::_esc` 写入的转义。"""
    return (text or "").replace("\\textbackslash{}", "\\").replace("\\&", "&") \
        .replace("\\%", "%").replace("\\$", "$").replace("\\#", "#") \
        .replace("\\_", "_").replace("\\textasciitilde{}", "~") \
        .replace("\\textasciicircum{}", "^").replace("\\{", "{").replace("\\}", "}")


def parse_bibtex(text: str) -> list[dict]:
    """解析本仓库生成的 .bib，返回 [{key,title,url,year,note}]。"""
    entries: list[dict] = []
    for kind, key, body in _BIB_ENTRY_RE.findall(text or ""):
        fields = {k.lower(): v for k, v in _BIB_FIELD_RE.findall(body)}
        url = ""
        if "howpublished" in fields:
            m = _BIB_URL_RE.search(fields["howpublished"])
            if m:
                url = m.group(1).strip()
        entries.append({
            "type": kind,
            "key": key.strip(),
            "title": _bib_unescape(fields.get("title", "")).strip(),
            "url": url,
            "year": _bib_unescape(fields.get("year", "")).strip(),
            "note": _bib_unescape(fields.get("note", "")).strip(),
        })
    return entries


def _ref_flowables(bibtex: str, st: dict[str, ParagraphStyle]) -> list:
    entries = parse_bibtex(bibtex)
    if not entries:
        return []
    flow: list = [PageBreak(), Paragraph("参考文献", st["h1"])]
    for e in entries:
        title = e["title"] or e["url"] or e["key"]
        tail = []
        if e["url"]:
            tail.append(f'<link href="{_esc(e["url"])}">{_esc(e["url"])}</link>')
        if e["year"]:
            tail.append(f'年份：{_esc(e["year"])}')
        if e["note"]:
            tail.append(_esc(e["note"]))
        body = f'<b>[{_esc(e["key"])}]</b> {_esc(title)}'
        if tail:
            body += '<br/>' + ' · '.join(tail)
        flow.append(Paragraph(body, st["ref"]))
    return flow


# ── 入口 ────────────────────────────────────────────────────────────────────
def render_markdown(*, topic: str, content_md: str, bibtex: str = "") -> str:
    """报告正文 + 参考文献章节的 Markdown 全文（导出用，与 PDF 同源同序）。

    正文原样保留；参考文献由 BibTeX 解析而来，条目格式与 PDF 一致：
        [key] title
        URL · 年份：xxxx · note
    """
    parts = [(content_md or "").rstrip()]
    entries = parse_bibtex(bibtex)
    if entries:
        lines = ["", "---", "", "## 参考文献", ""]
        for e in entries:
            title = e["title"] or e["url"] or e["key"]
            lines.append(f"- **[{e['key']}]** {title}")
            tail = []
            if e["url"]:
                tail.append(e["url"])
            if e["year"]:
                tail.append(f"年份：{e['year']}")
            if e["note"]:
                tail.append(e["note"])
            if tail:
                lines.append(f"  - {' · '.join(tail)}")
        parts.append("\n".join(lines))
    return "\n".join(p for p in parts if p) + "\n"


def render_pdf(*, topic: str, content_md: str, bibtex: str = "",
               task_id: int = 0, generated_at: str = "") -> bytes:
    """把报告渲染成 PDF 字节流（正文 + 参考文献 + 页脚）。"""
    st = _styles()
    regular = st["body"].fontName
    buf = io.BytesIO()

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(regular, 7.5)
        canvas.setFillColor(colors.HexColor("#8a8a8a"))
        left = f"InsightCrew · 任务 #{task_id}" if task_id else "InsightCrew"
        canvas.drawString(18 * mm, 11 * mm, left)
        canvas.drawRightString(A4[0] - 18 * mm, 11 * mm, f"第 {doc.page} 页")
        if generated_at:
            canvas.drawCentredString(A4[0] / 2, 11 * mm, generated_at)
        canvas.restoreState()

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=18 * mm,
        title=topic or "InsightCrew 报告",
        author="InsightCrew",
        subject="多 Agent 协作技术调研报告",
    )

    flow: list = []
    if topic:
        flow.append(Paragraph(_esc(topic), st["h1"]))
    flow.extend(markdown_to_flowables(content_md, st))
    flow.extend(_ref_flowables(bibtex, st))

    doc.build(flow, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


__all__ = ["render_pdf", "render_markdown", "parse_bibtex",
           "markdown_to_flowables", "font_status"]
