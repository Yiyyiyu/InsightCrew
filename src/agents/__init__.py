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
from src.agents.analyst import create_analyst, create_analysis_task
from src.agents.critic import create_critic, create_critic_task
from src.agents.writer import create_writer, create_write_task
from src.agents.reviewer import create_reviewer, create_review_task

from src.core.model_router import router as model_router


def _resolve_agent_config(agent_name: str):
    """通过 model_router 获取 Agent 的 CrewAI LLM 实例"""
    from crewai import LLM
    from src.core.config import settings

    route = model_router.route(agent=agent_name)
    llm_kwargs = {
        "model": route.model_id,
        "api_key": route.api_key,
        "temperature": route.temperature,
        "max_tokens": route.max_tokens,
        # 硬超时：避免单次 LLM 请求永久挂起（litellm 会中断底层 HTTP）
        "timeout": settings.llm_timeout_sec,
    }
    if route.api_base:
        llm_kwargs["base_url"] = route.api_base
    return LLM(**llm_kwargs)


def create_plan_crew(topic: str) -> Crew:
    """阶段1: 调研规划 Crew"""
    llm = _resolve_agent_config("Planner")
    agent = create_planner(llm=llm)
    task = create_plan_task(agent, topic)
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


def create_write_crew(plan: str, analysis: str, critic_feedback: str) -> Crew:
    """阶段5: 报告撰写 Crew"""
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
    """阶段6: 质量审核 Crew"""
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
    "create_write_crew",
    "create_review_crew",
]
