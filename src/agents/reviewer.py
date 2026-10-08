"""Reviewer — 质量审核员"""

from crewai import Agent, Task


def create_reviewer(llm: dict = None) -> Agent:
    return Agent(
        role="质量审核员",
        goal="审核报告质量，检查引用可追溯性、论证逻辑完整性和格式规范",
        backstory=(
            "你是质量控制专家，拥有学术出版背景，确保每份报告都达到可发表的标准。"
            "你重点检查：引用是否可追溯、论证是否严密、结构是否完整、语言是否专业。"
            "只有通过你的审核，报告才能交付。"
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def create_review_task(agent: Agent, report: str) -> Task:
    return Task(
        description=(
            f"以下是 Writer 产出的调研报告：\n\n{report}\n\n"
            "请从以下维度审核报告质量：\n"
            "1. **引用完整性**：每个论断是否有引用标记？标记格式是否正确 [S{id}#{chunk}]？\n"
            "2. **逻辑连贯性**：章节之间是否有清晰的逻辑递进？\n"
            "3. **结构完整性**：是否包含全部 7 个标准章节？\n"
            "4. **语言质量**：是否存在语病、错别字或表达不清？\n"
            "5. **结论支撑**：推荐结论是否有足够的引用支撑？\n"
            "6. **Critic 回应**：是否合理回应了 Critic 提出的质疑？\n\n"
            "输出审核报告，包含：\n"
            "- 总体评价（通过/有条件通过/不通过）\n"
            "- 各项评分（1-5分）\n"
            "- 具体问题清单（如有）\n"
            "- 修改建议（如有）"
        ),
        expected_output="格式化的审核报告，含总体评价、各项评分和问题清单。",
        agent=agent,
    )
