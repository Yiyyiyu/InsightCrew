"""Planner — 调研规划师

提示词外置在 `src/prompts/planner.yaml`（M1 遗留项 1）。
改提示词只改 YAML，不用动本文件、不用重启进程（loader 按 mtime 热加载）。
"""

from crewai import Agent, Task

from src.prompts import agent_block, task_block


def create_planner(llm: dict = None) -> Agent:
    spec = agent_block("planner")
    return Agent(
        role=spec["role"],
        goal=spec["goal"],
        llm=llm,
        backstory=spec["backstory"],
        verbose=True,
        allow_delegation=False,
    )


def create_plan_task(
    agent: Agent,
    topic: str,
    dimensions: str = "",
    constraints: str = "",
) -> Task:
    """构造调研规划任务。

    `dimensions` / `constraints` 由用户在新建任务时填写（M1 遗留项 5）。
    留空时填入「（未指定，由你自行判断）」而不是让 `$dimensions` 字面量
    留在提示词里——loader 用的是 `safe_substitute`，缺变量会原样保留。
    """
    spec = task_block(
        "planner",
        topic=topic,
        dimensions=(dimensions or "").strip() or "（未指定，由你自行判断）",
        constraints=(constraints or "").strip() or "（未指定）",
    )
    return Task(
        description=spec["description"],
        expected_output=spec["expected_output"],
        agent=agent,
    )
