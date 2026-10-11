"""质疑回应率 —— Critic 的每条质疑是否被 Analyst 显式回应。

从 `orchestrator.py` 拆分而来（M1/M2 遗留缺陷第 7 项，纯搬移零行为变更）。

★ 这个指标是「多 Agent 协作」区别于「多 Agent 顺序调用」的证据：
   多 Agent 最常见的失败模式是互相附和（sycophancy），所以 Critic 提出的
   每条质疑都必须能在 Analyst 的回应段落里找到对应。

★ 分母口径（D6 修复，C1 真实预演暴露）：Critic 的输出格式**完全不受提示词约束**。
   `src/prompts/critic.yaml` 要求「质疑点必须写成 Markdown 有序列表」，但四次真实
   运行全部无视，依次产出过四种形态：
     990170  `## N.` 分类标题 + `### N.N` 子标题 + 表格数据行（条目 = 表格行）
     990171  行首 `1.`–`46.` 有序列表（条目 = 列表项）
     990172  行首 `1.`–`40.` 有序列表（条目 = 列表项）
     990173  `### N.` 编号三级标题 15 条 + 每条下 3–4 个子 bullet + 末尾汇总表
   旧实现用「列表项 ∪ 表格数据行」的**并集**计数，在 990173 上把 55 个子 bullet
   和 5 行汇总表一并计入 → 15 条真实质疑被数成 61 条 → 9/61 = 0.1475 的假低值。
   所以计数改为**按形态互斥的优先级**：编号标题 > 有序列表 > 无序列表 > 表格，
   命中高优先级形态后不再看低优先级形态，且只认顶层（缩进 ≤3 空格）列表项。
"""

import re


# 表格表头识别词：单元格**整格**等于其中之一即视为表头行（不是数据行）。
_TABLE_HEADER_WORDS = {
    "原文位置",
    "质疑点",
    "要求",
    "序号",
    "编号",
    "问题",
    "位置",
    "质疑",
    "原文",
    "条目",
    "类型",
    "整改要求",
    "具体问题",
}

# ── 形态 ①：编号标题（最高优先级）──
# `### 1. 【引用缺失】…` / `### 2、…` / `#### 3) …`
# 只认三级及以下标题：`## 1. 引用缺失与引用不充分`（990170）是**分类**标题而非条目，
# 条目是它下面的表格行；而 `### N.`（990173）才是条目本身。
# `(?!\d)` 排除 `### 1.1 绝对化…` 这种小数编号的子标题（990170 的形态）。
_NUMBERED_HEADING_RE = re.compile(
    r"^#{3,6}[ \t]*(\d+)[ \t]*[.、)](?!\d)[ \t]*\S", re.MULTILINE
)

# ── 形态 ②：顶层有序列表（`1. xxx` / `1、xxx` / `1) xxx`）──
# 捕获缩进宽度，由 `_count_top_level` 取最小缩进作为「顶层」，
# 从而把条目内部的嵌套编号（`1.` 下面再缩进写的 `1.`/`2.`）排除在外。
_ORDERED_ITEM_RE = re.compile(r"^( *)\d+[ \t]*[.、)][ \t]+\S", re.MULTILINE)

# ── 形态 ③：顶层无序列表（`- xxx` / `* xxx` / `+ xxx`）──
# 注意：这是**兜底**形态。990173 里 55 条 bullet 是「质疑内容」下的论据，
# 不是独立质疑 —— 因为该输出同时存在 `### N.` 编号标题，形态 ① 会先命中并直接返回。
_BULLET_ITEM_RE = re.compile(r"^( *)[-*+][ \t]+\S", re.MULTILINE)

# ── 形态 ④：Markdown 表格数据行（兜底）──
_TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$", re.MULTILINE)
_TABLE_SEP_CELL_RE = re.compile(r"^:?-{2,}:?$")

# 回应里对第 N 条质疑的显式表态：`质疑 3` / `质疑3` / `质疑 03` / `3.`（回应小节内）
_REBUTTAL_SECTION_RE = re.compile(r"##+\s*质疑回应\s*(.*)$", re.DOTALL)


def _count_top_level(text: str, pattern: re.Pattern[str]) -> int:
    """数 `pattern` 命中的**顶层**列表项条数。

    `pattern` 的第一个捕获组必须是缩进字符串。取所有命中里的最小缩进当作
    「顶层」，只数与该缩进等宽的项 —— 这样既排除了条目内部的嵌套列表
    （`1. 质疑` 下面缩进写的 `1. 论据`），也不会因为模型把整份列表统一
    缩进两格而整份漏数（固定阈值会踩这个坑）。
    """
    indents = [len(m) for m in pattern.findall(text)]
    if not indents:
        return 0
    top = min(indents)
    return sum(1 for n in indents if n == top)


def _count_table_rows(text: str) -> int:
    """数 Markdown 表格的数据行（排除表头行与 `|---|---|` 分隔行）。"""
    counted = 0
    for m in _TABLE_ROW_RE.finditer(text):
        cells = [c.strip() for c in m.group(1).split("|")]
        if len(cells) < 2:
            continue
        nonempty = [c for c in cells if c]
        # 分隔行：所有非空单元格都是 `---` / `:---:`
        if nonempty and all(_TABLE_SEP_CELL_RE.match(c) for c in nonempty):
            continue
        # 表头行
        if any(c in _TABLE_HEADER_WORDS for c in cells):
            continue
        # 数据行至少要有一个非空单元格
        if not nonempty:
            continue
        counted += 1
    return counted


def _count_critique_items(critic_output: str) -> int:
    """数 Critic 提出了多少条可识别的质疑。

    四种真实输出形态按**互斥优先级**判定，命中高优先级后不再看低优先级：

    1. 编号标题 `### N.`（条目本身就是标题）
    2. 顶层有序列表 `1. xxx`
    3. 顶层无序列表 `- xxx`
    4. 表格数据行（排除表头与分隔行）

    互斥是必须的：990173 同时含「15 条 `### N.` 条目 + 55 条子 bullet + 汇总表」，
    并集计数会把论据和汇总行都当成质疑（分母 61），只有取最高优先级形态才得 15。
    """
    text = critic_output or ""

    # ① 编号标题
    headings = _NUMBERED_HEADING_RE.findall(text)
    if headings:
        return len(headings)

    # ② 顶层有序列表
    ordered = _count_top_level(text, _ORDERED_ITEM_RE)
    if ordered:
        return ordered

    # ③ 顶层无序列表
    bullets = _count_top_level(text, _BULLET_ITEM_RE)
    if bullets:
        return bullets

    # ④ 表格数据行
    return _count_table_rows(text)


def _critique_resolution_rate(rebuttal_output: str, critic_output: str) -> float:
    """计算「质疑被显式回应」的比例，范围 [0, 1]。

    这是 PRD G7 的 `critique_resolution_rate` 指标。口径刻意保守：

    - 分母 = Critic 输出里可识别的质疑条目数
      （编号标题 / 有序列表 / 无序列表 / 表格数据行 —— 见
      `_count_critique_items`，按形态互斥取最高优先级）
    - 分子 = 回应小节里能对号入座表态的条目数
      （命中「质疑 N」或回应小节内以 `N.` 开头的行）

    只在 `## 质疑回应` 小节内找回应，避免把正文里偶发的「质疑」二字算成回应。
    Critic 没产出可识别条目时返回 0.0 —— 宁可记 0 也不编一个好看的数。
    """
    total = _count_critique_items(critic_output)
    if total == 0:
        return 0.0

    section_match = _REBUTTAL_SECTION_RE.search(rebuttal_output or "")
    if not section_match:
        return 0.0
    section = section_match.group(1)

    answered = 0
    for i in range(1, total + 1):
        if re.search(rf"质疑\s*0?{i}\b", section) or re.search(
            rf"^\s*{i}\s*[.、)]", section, re.MULTILINE
        ):
            answered += 1
    return round(answered / total, 4)
