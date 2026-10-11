"""Reviewer — 质量审核员

提示词外置在 `src/prompts/reviewer.yaml`（M1 遗留项 1）。
"""

from crewai import Agent, Task

from src.prompts import agent_block, task_block


def create_reviewer(llm: dict = None) -> Agent:
    spec = agent_block("reviewer")
    return Agent(
        role=spec["role"],
        goal=spec["goal"],
        llm=llm,
        backstory=spec["backstory"],
        verbose=True,
        allow_delegation=False,
    )


def create_review_task(agent: Agent, report: str) -> Task:
    spec = task_block("reviewer", report=report)
    return Task(
        description=spec["description"],
        expected_output=spec["expected_output"],
        agent=agent,
    )
