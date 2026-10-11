"""
Agent 角色定义与编排

编排策略：每个阶段用一个独立的 Crew 运行，阶段之间由 orchestrator 控制
（实现 Flow 层的 HITL 闸门、条件分支和状态持久化）。
"""

from crewai import Crew, Process

from src.agents.planner import create_planner, create_plan_task
from src.agents.researcher import (
    create_researcher,
    create_research_agents,
    create_research_tasks,
    create_single_research_crew,
    subdirection_names,
)
from src.agents.analyst import create_analyst, create_analysis_task, create_rebuttal_task
from src.agents.critic import create_critic, create_critic_task
from src.agents.writer import create_writer, create_write_task
from src.agents.reviewer import create_reviewer, create_review_task

def _resolve_agent_config(agent_name: str):
    """获取该 Agent 角色在当前任务里应使用的 CrewAI LLM 实例。

    ★ 这是全项目**唯一**构造 CrewAI LLM 的地方，所以模型选择只在这里接一次。
    七个 crew 工厂与 orchestrator 的 research fan-out 都经由本函数，不会漏网。

    ★ M2.5 起模型由**用户逐角色选定**：``execute_stage`` 把本任务落库的
    ``model_plan`` 解析成 ``{role: ModelOption}`` 写进 ContextVar
    （``src/core/context.py::current_model_plan``），这里按角色取用。
    没有任务上下文（开发/测试）时回落到内置默认目录。

    ★ **运行时跨厂商降级已移除**（用户 m12784 拍板）：模型不通属于异常情况，
    由新建任务时的 preflight 连通性预检拦截，不在执行中途偷偷换模型。
    """
    from src.core.llm_fallback import build_llm_with_fallback

    return build_llm_with_fallback(agent_name)


def create_plan_crew(topic: str, dimensions: str = "", constraints: str = "") -> Crew:
    """阶段1: 调研规划 Crew

    `dimensions` / `constraints` 是用户在新建任务时填的评估维度与技术约束
    （M1 遗留项 5），直接进 Planner 提示词，让规划阶段就按用户口径拆解。
    """
    llm = _resolve_agent_config("Planner")
    agent = create_planner(llm=llm)
    task = create_plan_task(agent, topic, dimensions=dimensions, constraints=constraints)
    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )


def create_research_crew(plan: str) -> Crew:
    """阶段2: 检索取证 Crew（3 个检索任务，串行执行）

    ★ 实际编排不走这里：research 阶段由 orchestrator 调
    `create_single_research_crew` 为每个子方向各建一个独立 Crew，
    再用 asyncio.gather 并发 + 各自硬超时（单方向失败可隔离）。
    本函数保留给串行/调试场景。

    注意：每个任务必须绑**独立**的 Researcher 实例——CrewAI 一个 Agent 只持有
    一个 Executor，共享实例并发会抛 "Executor is already running"。
    """
    agents = create_research_agents(llm_factory=lambda: _resolve_agent_config("Researcher"))
    tasks = create_research_tasks(agents, plan)
    return Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )


def create_analysis_crew(research_outputs: str) -> Crew:
    """阶段3: 对比分析 Crew"""
    llm = _resolve_agent_config("Analyst")
    agent = create_analyst(llm=llm)
    task = create_analysis_task(agent, research_outputs)
    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )


def create_critic_crew(analysis_output: str) -> Crew:
    """阶段4: 红队质疑 Crew"""
    llm = _resolve_agent_config("Critic")
    agent = create_critic(llm=llm)
    task = create_critic_task(agent, analysis_output)
    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )


def create_rebuttal_crew(analysis: str, critic_output: str) -> Crew:
    """阶段5: 质疑回应 Crew

    ★ Critic 的质疑回灌给 Analyst 本人重做分析，产出「质疑 → 回应」配对。
    这一步是「多 Agent 协作」卖点的落点：没有它，质疑就只是写手正文里
    一句敷衍，防 sycophancy 的机制形同虚设。
    """
    llm = _resolve_agent_config("Analyst")
    agent = create_analyst(llm=llm)
    task = create_rebuttal_task(agent, analysis, critic_output)
    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )


def create_write_crew(plan: str, analysis: str, critic_feedback: str) -> Crew:
    """阶段6: 报告撰写 Crew"""
    llm = _resolve_agent_config("Writer")
    agent = create_writer(llm=llm)
    task = create_write_task(agent, analysis, critic_feedback, plan)
    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )


def create_review_crew(report: str) -> Crew:
    """阶段7: 质量审核 Crew"""
    llm = _resolve_agent_config("Reviewer")
    agent = create_reviewer(llm=llm)
    task = create_review_task(agent, report)
    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )


__all__ = [
    "create_plan_crew",
    "create_research_crew",
    "create_analysis_crew",
    "create_critic_crew",
    "create_rebuttal_crew",
    "create_write_crew",
    "create_review_crew",
]
