"""Analyst — 对比分析师"""

from crewai import Agent, Task


def create_analyst(llm: dict = None) -> Agent:
    return Agent(
        role="对比分析师",
        goal=(
            "整合多个检索来源的信息，构建系统的对比矩阵，"
            "识别技术方案之间的差距与各自优势"
        ),
        llm=llm,
        backstory=(
            "你擅长从多个维度系统性地对比技术方案。你能从看似不相关的信息中"
            "发现共性与差异，产出有洞察力的对比分析。每个对比结论都必须有引用支撑。"
        ),
        verbose=True,
        allow_delegation=False,
    )


def create_analysis_task(agent: Agent, research_outputs: str) -> Task:
    return Task(
        description=(
            f"以下是 Researcher 检索到的三个框架的信息：\n\n{research_outputs}\n\n"
            "请完成以下分析：\n"
            "1. **对比矩阵**：从架构设计、学习曲线、灵活性、工具集成、"
            "调试能力、社区活跃度、生产就绪度等维度构建对比表\n"
            "2. **各自优势**：每个框架的独特优势（至少 3 点/框架）\n"
            "3. **差距分析**：每个框架当前存在的明显不足或局限性\n"
            "4. **适用场景**：分别适合什么样的项目和团队\n\n"
            "要求：\n"
            "- 对比矩阵使用 Markdown 表格\n"
            "- 每个结论必须引用原始来源 [S{id}#{chunk_seq}]\n"
            "- 输出 Markdown 格式"
        ),
        expected_output=(
            "结构化的对比分析报告，包含对比矩阵表、优势/差距清单和适用场景建议，"
            "全部带有引用标记。"
        ),
        agent=agent,
    )
