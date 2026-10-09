"""Researcher — 检索取证员

★ 并行纪律（重要，勿改回单 Agent）：
CrewAI 的每个 Agent 实例只持有一个 Executor，而 Executor 不允许并发调用
（`crewai/experimental/agent_executor.py:2869` "Executor is already running.
Cannot invoke the same executor instance concurrently."）。
因此**并行任务必须各自绑定独立的 Agent 实例**——三个任务共用一个 agent
再开 async_execution 会直接抛 RuntimeError（实测）。

★ 工具循环上限：默认 max_iter=25 时 3 个任务最坏 75 轮 LLM+工具调用，
叠加 arXiv 429 / 抓取超时会被拖到阶段超时（实测 240s 不够）。
"""

from crewai import Agent, Task

from src.tools.search import search_web, search_arxiv
from src.tools.fetch import fetch_webpage

# 3 个并行子方向：前 2 个异步并发，第 3 个同步收尾。
# 最坏耗时 = max(90, 90) + 90 = 180s，落在 stage_timeout_sec(300s) 内；
# 若三个全串行则最坏 270s，余量太小。
_TASK_MAX_EXEC_TIME = 90


def create_researcher(llm=None, tag: str = "") -> Agent:
    """创建一个检索取证员（tag 用于区分并行实例的角色名）。"""
    return Agent(
        role=f"检索取证员{tag}",
        goal="根据调研计划，使用搜索工具从互联网、arXiv 收集高质量技术资料，并抓取核心内容",
        llm=llm,
        backstory=(
            "你精通各种检索工具，能从 Bing、arXiv 等来源快速定位到最相关的技术文献、"
            "开源项目和技术博客。你每次引用都会携带来源 ID 标记以确保可追溯。"
        ),
        verbose=True,
        allow_delegation=False,
        tools=[search_web, search_arxiv, fetch_webpage],
        max_iter=6,                       # 每个任务最多 6 轮工具调用
        max_execution_time=_TASK_MAX_EXEC_TIME,  # 单任务墙钟上限（秒）
        max_retry_limit=0,                # 工具失败不重试，由工具自身返回明确失败信息
    )


# 每个子任务：(方向名, 检索说明, 期望产出, 是否异步)
_SUBDIRECTIONS = [
    (
        "CrewAI",
        "搜索 CrewAI 的架构设计、核心概念（Agent/Task/Crew/Pipeline）、"
        "最新版本特性、优缺点和社区评价",
        "CrewAI 框架的检索摘要，含至少 3 条带引用标记的发现。",
        True,
    ),
    (
        "LangGraph",
        "搜索 LangGraph 的架构设计（有向图编排）、StateGraph/MessageGraph 概念、"
        "条件分支、循环机制、与 LangChain 的关系和社区评价",
        "LangGraph 框架的检索摘要，含至少 3 条带引用标记的发现。",
        True,
    ),
    (
        "AutoGen",
        "搜索 Microsoft AutoGen 的架构设计（多Agent对话）、"
        "AssistantAgent/UserProxyAgent 机制、人机协作模式、"
        "最新版本特性和社区评价",
        "AutoGen 框架的检索摘要，含至少 3 条带引用标记的发现。",
        False,   # 最后一个必须同步：CrewAI 禁止末尾连续多个异步任务
    ),
]


def create_research_agents(llm=None, llm_factory=None) -> list[Agent]:
    """为每个子方向创建**独立**的 Researcher 实例（并行前提）。

    传 llm_factory（无参可调用）可为每个 Agent 各建一个 LLM 实例，
    避免并发下共享客户端状态；否则所有 Agent 复用同一个 llm。
    """
    out: list[Agent] = []
    for name, _, _, _ in _SUBDIRECTIONS:
        this_llm = llm_factory() if llm_factory is not None else llm
        out.append(create_researcher(llm=this_llm, tag=f"（{name}）"))
    return out


def create_research_tasks(agents: list[Agent], plan: str) -> list[Task]:
    """根据调研计划生成检索任务，与 agents 一一对应。

    `agents` 必须是**不同实例**，否则 async_execution 会触发
    "Executor is already running" 错误。

    ★ 注意：实际编排走 `create_single_research_crew`（每个子方向一个独立
    Crew，由 orchestrator 用 asyncio.gather 并发 + 单独硬超时）。
    CrewAI 自带的 `async_execution=True` 路径实测不可靠——3 个任务会
    远超单任务耗时之和（单任务实测 18.8s，3 任务却撞穿 300s 阶段超时），
    且单个任务超时无法隔离。本函数保留供串行/调试使用。
    """
    if len(agents) != len(_SUBDIRECTIONS):
        raise ValueError(f"需要 {len(_SUBDIRECTIONS)} 个独立 Agent，收到 {len(agents)}")

    tasks: list[Task] = []
    total = len(_SUBDIRECTIONS)
    for i, (agent, (name, desc, expected, is_async)) in enumerate(
        zip(agents, _SUBDIRECTIONS), start=1
    ):
        tasks.append(Task(
            description=(
                f"调研计划：\n{plan}\n\n"
                f"【检索任务 {i}/{total} — {name} 框架】\n"
                f"{desc}\n\n"
                "要求：\n"
                "1. 至少引用 3 个独立来源\n"
                "2. 每个论断标注来源引用标记 [S{id}#{chunk_seq}]"
                "（必须使用工具返回的真实标记，不得编造）\n"
                "3. 输出 Markdown 格式"
            ),
            expected_output=expected,
            agent=agent,
            async_execution=is_async,
        ))
    return tasks


def subdirection_names() -> list[str]:
    """所有检索子方向名称（供编排层打印/统计）。"""
    return [name for name, _, _, _ in _SUBDIRECTIONS]


def create_single_research_crew(llm, index: int, plan: str):
    """为**单个**子方向建一个独立 Crew（一个 Agent + 一个同步 Task）。

    这是 research 阶段的实际编排单元：orchestrator 并发调用本函数产出的
    多个 Crew，每个都有独立的 Agent 实例、独立的 LLM 客户端、独立的超时，
    单个子方向失败不会拖垮其余方向。
    """
    from crewai import Crew, Process

    if not 0 <= index < len(_SUBDIRECTIONS):
        raise IndexError(f"子方向索引越界: {index}")

    name, desc, expected, _ = _SUBDIRECTIONS[index]
    total = len(_SUBDIRECTIONS)
    agent = create_researcher(llm=llm, tag=f"（{name}）")
    task = Task(
        description=(
            f"调研计划（供你理解上下文，不必逐条执行）：\n{plan}\n\n"
            f"【你的检索任务 {index + 1}/{total} — {name} 框架】\n"
            f"{desc}\n\n"
            "要求：\n"
            "1. 至少引用 3 个独立来源\n"
            "2. 每个论断标注来源引用标记 [S{id}#{chunk_seq}]"
            "（必须使用工具返回的真实标记，不得编造）\n"
            "3. 输出 Markdown 格式"
        ),
        expected_output=expected,
        agent=agent,
        async_execution=False,
    )
    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )
