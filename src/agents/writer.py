"""Writer — 报告撰写员

提示词外置在 `src/prompts/writer.yaml`（M1 遗留项 1）。
"""

from crewai import Agent, Task

from src.prompts import agent_block, task_block


def create_writer(llm: dict = None) -> Agent:
    spec = agent_block("writer")
    return Agent(
        role=spec["role"],
        goal=spec["goal"],
        llm=llm,
        backstory=spec["backstory"],
        verbose=True,
        allow_delegation=False,
    )


def create_write_task(
    agent: Agent,
    analysis: str,
    critic_feedback: str,
    plan: str,
) -> Task:
    spec = task_block(
        "writer",
        plan=plan,
        analysis=analysis,
        critic_feedback=critic_feedback,
    )
    return Task(
        description=spec["description"],
        expected_output=spec["expected_output"],
        agent=agent,
    )
