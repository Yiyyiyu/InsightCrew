"""Critic — 红队质疑者

提示词外置在 `src/prompts/critic.yaml`（M1 遗留项 1）。
"""

from crewai import Agent, Task

from src.prompts import agent_block, task_block


def create_critic(llm: dict = None) -> Agent:
    spec = agent_block("critic")
    return Agent(
        role=spec["role"],
        goal=spec["goal"],
        llm=llm,
        backstory=spec["backstory"],
        verbose=True,
        allow_delegation=False,
    )


def create_critic_task(agent: Agent, analysis_output: str) -> Task:
    spec = task_block("critic", analysis=analysis_output)
    return Task(
        description=spec["description"],
        expected_output=spec["expected_output"],
        agent=agent,
    )
