"""Researcher — 检索取证员

提示词外置在 `src/prompts/researcher.yaml`（M1 遗留项 1）：
`role`/`goal`/`backstory`、`task`（串行调试路径）与 `single_task`（实际编排路径）
全部来自 YAML。

★ 检索子方向**不再来自 YAML**（P0 硬编码缺陷）：由 Planner 的调研计划动态派生，
见 `src/core/subdirections.py::derive_subdirections`。此前 YAML 里写死了
CrewAI / LangGraph / AutoGen 三条，任何课题都被检索成那三个框架且不报错。

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

from src.core.subdirections import derive_subdirections
from src.prompts import agent_block, task_block
from src.tools.fetch import fetch_webpage
from src.tools.pdf import parse_pdf
from src.tools.search import search_arxiv, search_web
from src.tools.vision import list_pdf_images, read_image

# ★ 这里曾经设 `max_execution_time=90`（Agent 级单任务墙钟上限），C1 真实预演
#   （任务 990172）证明它**有害**，故移除，只保留外层的单一墙钟上限
#   （`_run_research_fanout` 的 per_timeout）。理由：
#   1) crewai 的 `kickoff_async` 实为 `asyncio.to_thread(self.kickoff)`
#      （crewai/crew.py:1187），走的是**同步** `execute_task` →
#      `_execute_with_timeout`，实现是
#      `with ThreadPoolExecutor() as ex: ex.submit(...).result(timeout=90)`
#      （crewai/agent/core.py:1045-1059）。超时抛的 TimeoutError 要等 `with`
#      退出（`shutdown(wait=True)`）才冒出来，而 shutdown 会阻塞到工作线程
#      自然跑完 —— 所以它**既不是硬中断，抛出时刻也远晚于 90s**
#      （实测 990172 第 4 方向 143.0s 才抛，990170 是 134.8s）。
#   2) 抛异常时 `future.cancel()` 停不掉已在运行的线程，**已经产出的结果被
#      丢弃**：这就是那两个方向「跑了 143s 却一个字都没留下」的原因。
#   3) 它与外层 per_timeout（240s）构成双重超时，而 Python 3.11 起
#      `asyncio.TimeoutError is TimeoutError`（实测 True），外层 except 无法
#      按类型区分 → 日志谎报「超时（>240s）」，掩盖真实原因。
#   `max_iter=6` 仍然限定每个任务的工具调用轮数，配合 llm_timeout_sec 已足以
#   约束单方向耗时；真正的总闸是外层墙钟。


def _subdirections(plan: str = "", topic: str = "") -> list[dict]:
    """从调研计划派生检索子方向（见 `src/core/subdirections.py`）。

    返回 [{name, desc, expected, async}, ...]，顺序即执行顺序。
    **永不返回空**：计划解析不出来时会降级成「单方向覆盖全课题」，
    而不是回落到任何固定主题。
    """
    return derive_subdirections(plan, topic)


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
        # ★ 刻意不设 max_execution_time：它的超时既非硬中断、又会丢弃已产出的
        #   结果（见文件头注释），单方向耗时统一由 `_run_research_fanout` 的
        #   外层墙钟（per_timeout）约束。
        max_retry_limit=0,                # 工具失败不重试，由工具自身返回明确失败信息
    )


def create_research_agents(llm=None, llm_factory=None, plan: str = "",
                           topic: str = "") -> list[Agent]:
    """为每个子方向创建**独立**的 Researcher 实例（并行前提）。

    传 llm_factory（无参可调用）可为每个 Agent 各建一个 LLM 实例，
    避免并发下共享客户端状态；否则所有 Agent 复用同一个 llm。
    """
    out: list[Agent] = []
    for sub in _subdirections(plan, topic):
        this_llm = llm_factory() if llm_factory is not None else llm
        out.append(create_researcher(llm=this_llm, tag=f"（{sub['name']}）"))
    return out


def create_research_tasks(agents: list[Agent], plan: str,
                          topic: str = "") -> list[Task]:
    """根据调研计划生成检索任务，与 agents 一一对应。

    `agents` 必须是**不同实例**，否则 async_execution 会触发
    "Executor is already running" 错误。

    ★ 注意：实际编排走 `create_single_research_crew`（每个子方向一个独立
    Crew，由 orchestrator 用 asyncio.gather 并发 + 单独硬超时）。
    CrewAI 自带的 `async_execution=True` 路径实测不可靠——3 个任务会
    远超单任务耗时之和（单任务实测 18.8s，3 任务却撞穿 300s 阶段超时），
    且单个任务超时无法隔离。本函数保留供串行/调试使用。
    """
    subs = _subdirections(plan, topic)
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


def subdirection_names(plan: str = "", topic: str = "") -> list[str]:
    """所有检索子方向名称（供编排层打印/统计）。"""
    return [sub["name"] for sub in _subdirections(plan, topic)]


def create_single_research_crew(llm, index: int, plan: str, topic: str = ""):
    """为**单个**子方向建一个独立 Crew（一个 Agent + 一个同步 Task）。

    这是 research 阶段的实际编排单元：orchestrator 并发调用本函数产出的
    多个 Crew，每个都有独立的 Agent 实例、独立的 LLM 客户端、独立的超时，
    单个子方向失败不会拖垮其余方向。

    ★ `plan` 必须与调用方 `subdirection_names(plan)` 用的是同一个计划字符串，
    否则子方向数量/顺序不一致会导致索引错位（把 A 方向的提示词配给 B 方向）。
    """
    from crewai import Crew, Process

    subs = _subdirections(plan, topic)
    if not 0 <= index < len(subs):
        raise IndexError(f"子方向索引越界: {index}（共 {len(subs)} 个）")

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
