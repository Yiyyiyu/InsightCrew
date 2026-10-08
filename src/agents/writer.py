"""Writer — 报告撰写员"""

from crewai import Agent, Task


def create_writer(llm: dict = None) -> Agent:
    return Agent(
        role="报告撰写员",
        goal=(
            "将 Analyst 的分析结果组织和撰写为结构严谨、论证清晰的中文技术调研报告，"
            "确保每句话都有可追溯的引用支撑"
        ),
        llm=llm,
        backstory=(
            "你是技术文档专家，善于将复杂的多框架对比转化为易读、专业的中文报告。"
            "你严格保留每个论断的引用标记，确保报告的所有结论都可以追溯到原始来源。"
        ),
        verbose=True,
        allow_delegation=False,
    )


def create_write_task(
    agent: Agent,
    analysis: str,
    critic_feedback: str,
    plan: str,
) -> Task:
    return Task(
        description=(
            f"## 原始调研计划\n{plan}\n\n"
            f"## Analyst 的对比分析\n{analysis}\n\n"
            f"## Critic 的质疑\n{critic_feedback}\n\n"
            "请撰写一份完整的中文技术调研报告，包含以下章节：\n"
            "1. **摘要** — 简要概述调研内容和结论\n"
            "2. **背景与动机** — 为什么要做这项技术选型\n"
            "3. **框架概述** — 逐一介绍 CrewAI、LangGraph、AutoGen\n"
            "4. **对比矩阵** — 多维度系统性对比（含表格）\n"
            "5. **差距与缺口分析** — 每个框架的不足、适用边界\n"
            "6. **推荐方向** — 基于场景的选型建议\n"
            "7. **结论与展望**\n\n"
            "严格要求：\n"
            "1. 响应 Critic 的质疑，补全缺失的引用\n"
            "2. 每个论断必须保留原始引用标记 [S{id}#{chunk_seq}]\n"
            "3. 引用标记不能丢失、不能编造\n"
            "4. 输出纯 Markdown 格式（不做 LaTeX）"
        ),
        expected_output=(
            "完整的中文调研报告（Markdown），含 7 个标准章节，"
            "引用标记在正文中完整保留。"
        ),
        agent=agent,
    )
