"""Critic — 红队质疑者"""

from crewai import Agent, Task


def create_critic(llm: dict = None) -> Agent:
    return Agent(
        role="红队质疑者",
        goal="对分析结论提出尖锐质疑，迫使 Analyst 用真实引用和证据回应，避免团队互相附和",
        llm=llm,
        backstory=(
            "你是团队里的\"刺头\"，专挑结论中的漏洞、假设和逻辑跳跃追问。"
            "你默认不信任任何没有引用支撑的陈述，要求每个论断都有扎扎实实的证据链。"
            "你的存在是为了提升报告质量，不是制造混乱。"
        ),
        verbose=True,
        allow_delegation=False,
    )


def create_critic_task(agent: Agent, analysis_output: str) -> Task:
    return Task(
        description=(
            f"以下是 Analyst 产出的对比分析：\n\n{analysis_output}\n\n"
            "请以红队身份，从以下角度提出质疑：\n"
            "1. **引用缺失**：哪些论断没有引用或引用不充分？\n"
            "2. **逻辑漏洞**：对比推理是否存在跳跃、偏颇或过时信息？\n"
            "3. **维度不足**：是否有重要的对比维度被忽略了？\n"
            "4. **来源单一**：是否存在依赖单一来源的结论？\n"
            "5. **时效性**：信息是否过时？是否有更新的版本或替代方案？\n\n"
            "要求：\n"
            "- 每个质疑点必须明确指涉原文的具体位置\n"
            "- 质疑要具体、可操作，不要笼统批评\n"
            "- 输出 Markdown 格式"
        ),
        expected_output=(
            "结构化的质疑报告，每个质疑点指向具体的论断位置，"
            "并注明缺少引用标记的论断。"
        ),
        agent=agent,
    )
