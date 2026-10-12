"""
工作流编排器 — 多 Agent 调研的全生命周期（**兼容门面**）

原 `orchestrator.py` 1279 行，超出 NFR「单文件不超过 400 行」的限制，
按职责机械拆分为：

    events.py     SSE 事件推送（落库 + 内存队列 + Redis Pub/Sub + 回放）
    queue.py      arq 入队、worker 健康门控、阶段常量
    stages.py     单阶段执行、Crew 超时与记账、research fan-out
    artifacts.py  中间产物与评测指标的落库（task_events JSON payload）
    critique.py   质疑回应率（Critic 质疑被 Analyst 回应的比例）
    gates.py      HITL 闸门（确认/驳回/回退）、收尾、状态更新
    pipeline.py   任务创建/状态查询/停滞任务回收（对外入口）

**本模块只做再导出，不含任何逻辑**：所有既有 `from src.core.orchestrator
import X` 与 `orchestrator.X` 用法保持不变，拆分因此是零行为变更。

★ 模块级可变全局（`_redis_pool` / `_redis_sub_pool` / `_arq_pool`）与会被
  monkeypatch 的函数（`_worker_alive`）住在其它模块里。测试与验证脚本却按
  老习惯写 `orchestrator.X = ...`（`tests/conftest.py` 重置连接池、
  `tools/verify_phase_a.py` 伪造 worker 缺席）。**若只在门面上赋值，真正的
  持有模块看不到改动** —— 池不会被清掉（下一个测试报
  `RuntimeError: Event loop is closed`），`_ensure_executable` 也仍会调用
  真的 `_worker_alive`。因此本模块用一个自定义 ModuleType 子类把读写都
  代理到真正的持有者（见文末 `_Facade`）。
"""

import sys
import types

from src.core import events as _events
from src.core import queue as _queue
from src.core import stages as _stages
from src.core import artifacts as _artifacts
from src.core import gates as _gates
from src.core import critique as _critique
from src.core import pipeline as _pipeline

# ── 常量 ──
from src.core.events import TERMINAL_EVENTS
from src.core.queue import STAGES, STAGE_LABELS, GATE_STAGES
from src.core.critique import _count_critique_items, _REBUTTAL_SECTION_RE
from src.core.pipeline import STALE_FACTOR

# ── 事件通道 ──
from src.core.events import (
    _event_channel,
    _get_redis,
    _get_sub_redis,
    _persist_event,
    _push_event,
    _push_redis_event,
    _replay_events,
    _task_is_terminal,
    event_stream,
    subscribe,
    unsubscribe,
)

# ── 队列 ──
from src.core.queue import (
    _ensure_executable,
    _enqueue_stage,
    _get_arq_pool,
    _mark_exec_path,
    _worker_alive,
)

# ── 阶段执行 ──
from src.core.stages import (
    _flatten_crew_output,
    _run_crew_safe,
    _run_research_fanout,
    execute_stage,
)
from src.core.artifacts import (
    _load_intermediate,
    _save_evaluation,
    _save_intermediate,
)
from src.core.critique import _critique_resolution_rate

# ── 闸门与收尾 ──
from src.core.gates import (
    _create_gate,
    _find_pending_review,
    _finish_pipeline,
    _reopen_gate,
    _run_stage,
    _update_task_status,
    abort_pipeline,
    advance_stage,
    approve_gate,
    delete_task,
    pause_pipeline,
    reject_gate,
)

# ── 对外入口 ──
from src.core.pipeline import (
    get_task_status,
    reap_stale_tasks,
    start_pipeline,
)

# 需要读写代理的名字 → 真正的持有模块。
# 只有"模块级可变状态"与"会被 monkeypatch 的函数"必须在此列；普通函数与
# 常量直接再导出即可（读走本模块 __dict__，不经过代理）。
_OWNER = {
    "_sse_queues": _events,
    "_redis_pool": _events,
    "_redis_sub_pool": _events,
    "_arq_pool": _queue,
    "_worker_alive": _queue,
}


class _Facade(types.ModuleType):
    """把 `_OWNER` 里列出的名字的读写都转发给真正的持有模块。"""

    def __getattr__(self, name: str):
        owner = _OWNER.get(name)
        if owner is not None:
            return getattr(owner, name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __setattr__(self, name: str, value) -> None:
        owner = _OWNER.get(name)
        if owner is None:
            super().__setattr__(name, value)
            return
        setattr(owner, name, value)
        # 去掉本模块里的影子副本，让后续读取也走代理（否则读到的是旧对象）
        self.__dict__.pop(name, None)

    def __delattr__(self, name: str) -> None:
        owner = _OWNER.get(name)
        if owner is None:
            super().__delattr__(name)
            return
        delattr(owner, name)
        self.__dict__.pop(name, None)


# ★ 必须在所有 import 完成后才替换模块类，否则 import 期间的属性写入会
#   被 __setattr__ 拦下。
sys.modules[__name__].__class__ = _Facade
