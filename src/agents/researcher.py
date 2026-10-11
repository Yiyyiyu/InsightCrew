"""Researcher — 检索取证员

提示词外置在 `src/prompts/researcher.yaml`（M1 遗留项 1）：
`role`/`goal`/`backstory`、`task`（串行调试路径）、`single_task`（实际编排路径）
与 `subdirections`（3 个子方向）全部来自 YAML。

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

from src.prompts import agent_block, raw, task_block
from src.tools.fetch import fetch_webpage
from src.tools.pdf import parse_pdf
from src.tools.search import search_arxiv, search_web
from src.tools.vision import list_pdf_images, read_image

# 3 个并行子方向：前 2 个异步并发，第 3 个同步收尾。
# 最坏耗时 = max(90, 90) + 90 = 180s，落在 stage_timeout_sec(300s) 内；
# 若三个全串行则最坏 270s，余量太小。
_TASK_MAX_EXEC_TIME = 90


def _subdirections() -> list[dict]:
    """从 YAML 读子方向定义（每次读缓存，mtime 变更自动重载）。

    返回 [{name, desc, expected, async}, ...]，顺序即执行顺序。
    """
    block = raw("researcher").get("subdirections")
    if not isinstance(block, list) or not block:
        raise ValueError("researcher.yaml 缺少 subdirections 列表")
    return block


def create_researcher(llm=None, tag: str = "") -> Agent:
    """创建一个检索取证员（tag 用于区分并行实例的角色名）。"""
    spec = agent_block("researcher", tag=tag)
    return Agent(
        role=spec["role"],
        goal=spec["goal"],
        llm=llm,
        backstory=spec["backstory"],
        verbose=True,
        allow_delegation=False,
        tools=[
            search_web, search_arxiv, fetch_webpage,
            parse_pdf, list_pdf_images, read_image,
        ],
        max_iter=6,                       # 每个任务最多 6 轮工具调用
        max_execution_time=_TASK_MAX_EXEC_TIME,  # 单任务墙钟上限（秒）
        max_retry_limit=0,                # 工具失败不重试，由工具自身返回明确失败信息
    )


def create_research_agents(llm=None, llm_factory=None) -> list[Agent]:
    """为每个子方向创建**独立**的 Researcher 实例（并行前提）。

    传 llm_factory（无参可调用）可为每个 Agent 各建一个 LLM 实例，
    避免并发下共享客户端状态；否则所有 Agent 复用同一个 llm。
    """
    out: list[Agent] = []
    for sub in _subdirections():
        this_llm = llm_factory() if llm_factory is not None else llm
        out.append(create_researcher(llm=this_llm, tag=f"（{sub['name']}）"))
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
    subs = _subdirections()
    if len(agents) != len(subs):
        raise ValueError(f"需要 {len(subs)} 个独立 Agent，收到 {len(agents)}")

    tasks: list[Task] = []
    total = len(subs)
    for i, (agent, sub) in enumerate(zip(agents, subs), start=1):
        spec = task_block(
            "researcher",
            task_key="task",
            plan=plan,
            index=i,
            total=total,
            name=sub["name"],
            desc=sub["desc"],
            expected=sub["expected"],
        )
        tasks.append(Task(
            description=spec["description"],
            expected_output=spec["expected_output"],
            agent=agent,
            async_execution=bool(sub.get("async", False)),
        ))
    return tasks


def subdirection_names() -> list[str]:
    """所有检索子方向名称（供编排层打印/统计）。"""
    return [sub["name"] for sub in _subdirections()]


def create_single_research_crew(llm, index: int, plan: str):
    """为**单个**子方向建一个独立 Crew（一个 Agent + 一个同步 Task）。

    这是 research 阶段的实际编排单元：orchestrator 并发调用本函数产出的
    多个 Crew，每个都有独立的 Agent 实例、独立的 LLM 客户端、独立的超时，
    单个子方向失败不会拖垮其余方向。
    """
    from crewai import Crew, Process

    subs = _subdirections()
    if not 0 <= index < len(subs):
        raise IndexError(f"子方向索引越界: {index}")

    sub = subs[index]
    total = len(subs)
    spec = task_block(
        "researcher",
        task_key="single_task",
        plan=plan,
        index=index + 1,
        total=total,
        name=sub["name"],
        desc=sub["desc"],
        expected=sub["expected"],
    )
    agent = create_researcher(llm=llm, tag=f"（{sub['name']}）")
    task = Task(
        description=spec["description"],
        expected_output=spec["expected_output"],
        agent=agent,
        async_execution=False,
    )
    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )
