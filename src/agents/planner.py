"""Planner — 调研规划师"""

from crewai import Agent, Task


def create_planner(llm: dict = None) -> Agent:
    return Agent(
        role="调研规划师",
        goal="将用户的研究课题拆解为结构化的调研计划，确定检索关键词、对比维度和预期产出",
        llm=llm,
        backstory=(
            "你是经验丰富的技术调研专家，擅长将模糊的研究方向转化为清晰的调研路线图。"
            "你制定的计划涵盖：调研目标、核心问题、检索策略（中英文关键词）、"
            "对比维度、预期产出结构。你的计划是后续所有Agent的工作纲领。"
        ),
        verbose=True,
        allow_delegation=False,
    )


def create_plan_task(agent: Agent, topic: str) -> Task:
    return Task(
        description=(
            f"课题：{topic}\n\n"
            "请制定一份结构化的多Agent协作调研计划，包含：\n"
            "1. 调研目标与核心问题\n"
            "2. 关键对比维度（至少5个维度）\n"
            "3. 中英文检索关键词（至少3组）\n"
            "4. 目标信息来源（arXiv、GitHub、技术博客等）\n"
            "5. 预期报告结构大纲\n\n"
            "输出格式为 Markdown。"
        ),
        expected_output=(
            "一份完整的 Markdown 格式调研计划，包含上述五个部分，"
            "每个部分有 2-5 个要点。"
        ),
        agent=agent,
    )
