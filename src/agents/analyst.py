"""Analyst — 对比分析师

提示词外置在 `src/prompts/analyst.yaml`（M1 遗留项 1）。
"""

from crewai import Agent, Task

from src.prompts import agent_block, task_block


def create_analyst(llm: dict = None) -> Agent:
    spec = agent_block("analyst")
    return Agent(
        role=spec["role"],
        goal=spec["goal"],
        llm=llm,
        backstory=spec["backstory"],
        verbose=True,
        allow_delegation=False,
    )


def create_analysis_task(agent: Agent, research_outputs: str) -> Task:
    spec = task_block("analyst", research_outputs=research_outputs)
    return Task(
        description=spec["description"],
        expected_output=spec["expected_output"],
        agent=agent,
    )


def create_rebuttal_task(agent: Agent, analysis: str, critic_output: str) -> Task:
    """质疑回应任务：把 Critic 的质疑回灌给 Analyst 本人重做分析。

    ★ 这是「Critic 的质疑必须被 Analyst 显式回应」的落点。
    此前质疑只被塞进 Writer 的提示词里（`create_write_task` 的
    `critic_feedback`），由写手在正文中敷衍，等于没回应。
    """
    spec = task_block("analyst", task_key="rebuttal", analysis=analysis, critic=critic_output)
    return Task(
        description=spec["description"],
        expected_output=spec["expected_output"],
        agent=agent,
    )
