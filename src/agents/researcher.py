"""Researcher — 检索取证员"""

from crewai import Agent, Task
from src.tools.search import search_web, search_arxiv
from src.tools.fetch import fetch_webpage


def create_researcher(llm: dict = None) -> Agent:
    return Agent(
        role="检索取证员",
        goal="根据调研计划，使用搜索工具从互联网、arXiv 收集高质量技术资料，并抓取核心内容",
        llm=llm,
        backstory=(
            "你精通各种检索工具，能从 Bing、arXiv 等来源快速定位到最相关的技术文献、"
            "开源项目和技术博客。你每次引用都会携带来源 ID 标记以确保可追溯。"
        ),
        verbose=True,
        allow_delegation=False,
        tools=[search_web, search_arxiv, fetch_webpage],
    )


def create_research_tasks(agent: Agent, plan: str) -> list[Task]:
    """
    根据调研计划生成多个并行检索任务。
    每个任务聚焦一个子方向，返回带引用的检索摘要。
    """
    # 拆解计划中的关键词方向
    # 这里简化：返回三个固定子任务
    return [
        Task(
            description=(
                f"调研计划：\n{plan}\n\n"
                "【检索任务 1/3 — CrewAI 框架】\n"
                "搜索 CrewAI 的架构设计、核心概念（Agent/Task/Crew/Pipeline）、"
                "最新版本特性、优缺点和社区评价。\n\n"
                "要求：\n"
                "1. 至少引用 3 个独立来源\n"
                "2. 每个论断标注来源引用标记 [S{id}#{chunk_seq}]\n"
                "3. 输出 Markdown 格式"
            ),
            expected_output="CrewAI 框架的检索摘要，含至少 3 条带引用标记的发现。",
            agent=agent,
        ),
        Task(
            description=(
                f"调研计划：\n{plan}\n\n"
                "【检索任务 2/3 — LangGraph 框架】\n"
                "搜索 LangGraph 的架构设计（有向图编排）、StateGraph/MessageGraph 概念、"
                "条件分支、循环机制、与 LangChain 的关系和社区评价。\n\n"
                "要求：\n"
                "1. 至少引用 3 个独立来源\n"
                "2. 每个论断标注来源引用标记 [S{id}#{chunk_seq}]\n"
                "3. 输出 Markdown 格式"
            ),
            expected_output="LangGraph 框架的检索摘要，含至少 3 条带引用标记的发现。",
            agent=agent,
        ),
        Task(
            description=(
                f"调研计划：\n{plan}\n\n"
                "【检索任务 3/3 — AutoGen 框架】\n"
                "搜索 Microsoft AutoGen 的架构设计（多Agent对话）、"
                "AssistantAgent/UserProxyAgent 机制、人机协作模式、"
                "最新版本特性和社区评价。\n\n"
                "要求：\n"
                "1. 至少引用 3 个独立来源\n"
                "2. 每个论断标注来源引用标记 [S{id}#{chunk_seq}]\n"
                "3. 输出 Markdown 格式"
            ),
            expected_output="AutoGen 框架的检索摘要，含至少 3 条带引用标记的发现。",
            agent=agent,
        ),
    ]
