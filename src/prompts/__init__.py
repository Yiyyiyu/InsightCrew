"""提示词外置加载器（M1 遗留项 1）。

**为什么用 `string.Template` 而不是 `str.format` / `format_map`（勿改回去）**：
提示词里天然含有 `[S{id}#{chunk_seq}]` 这种引用标记。`str.format` 会把 `{id}`
当成占位符去解析，遇到 `#` 直接抛 `Invalid format specifier`（实测踩过这个坑，
且它在 mock 模式下必然复现）。`string.Template` 只认 `$name`，花括号对它没有
特殊含义，引用标记原样保留。

同时用 `safe_substitute` 而非 `substitute`：
  * 没提供的变量原样留在文本里，不会静默变成空串 —— 提示词缺变量的 bug 会
    在输出里一眼看见，而不是变成一句读不通的话。

**热加载**：按文件 mtime 判断是否重读，所以改 YAML 不用重启进程 —— 这正是
"把提示词从代码里搬出来"要换来的东西。mtime 检查本身只 stat 一次文件，开销可忽略。
"""

from __future__ import annotations

import string
import threading
from pathlib import Path
from typing import Any

import yaml

_PROMPT_DIR = Path(__file__).resolve().parent

# 文件名 -> (原始 dict, mtime)
_cache: dict[str, tuple[dict[str, Any], float]] = {}
_lock = threading.Lock()


class PromptNotFoundError(KeyError):
    """请求了不存在的提示词文件或键。"""


def _load_file(name: str) -> dict[str, Any]:
    """读取并缓存一个 YAML 提示词文件（按 mtime 热加载）。"""
    path = _PROMPT_DIR / f"{name}.yaml"
    if not path.exists():
        raise PromptNotFoundError(
            f"提示词文件不存在: {path.name}（可用: {sorted(available())}）"
        )

    mtime = path.stat().st_mtime
    cached = _cache.get(name)
    if cached is not None and cached[1] == mtime:
        return cached[0]

    with _lock:
        # 双检：并发下可能已有别的线程读好了
        cached = _cache.get(name)
        if cached is not None and cached[1] == path.stat().st_mtime:
            return cached[0]

        with path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise PromptNotFoundError(f"{path.name} 顶层应为映射（dict），实际 {type(raw).__name__}")

        _cache[name] = (raw, path.stat().st_mtime)
        return raw


def available() -> list[str]:
    """所有可用的提示词名（文件名去扩展名）。"""
    return sorted(p.stem for p in _PROMPT_DIR.glob("*.yaml"))


def render(text: str, **variables: Any) -> str:
    """用 `$name` 占位符渲染一段文本；未提供的变量原样保留。"""
    if not isinstance(text, str):
        return text
    return string.Template(text).safe_substitute(**variables)


def agent_block(prompt: str, **variables: Any) -> dict[str, str]:
    """取某个 Agent 的 role / goal / backstory（已渲染）。

    Args:
        prompt: 提示词文件名，如 ``"planner"``。
        variables: 渲染变量，如 ``tag="（CrewAI）"``。

    Raises:
        PromptNotFoundError: 文件不存在，或缺少 role/goal/backstory 字段。
    """
    raw = _load_file(prompt)
    out: dict[str, str] = {}
    for key in ("role", "goal", "backstory"):
        if key not in raw:
            raise PromptNotFoundError(f"{prompt}.yaml 缺少字段 {key!r}")
        out[key] = render(raw[key], **variables)
    return out


def task_block(prompt: str, task_key: str = "task", **variables: Any) -> dict[str, str]:
    """取某个 Task 的 description / expected_output（已渲染）。

    ★ 第一个参数叫 `prompt` 而不是 `name`：researcher 的 YAML 里有一个
    `$name` 变量（子方向名），若形参也叫 name，调用 `task_block("researcher",
    name="CrewAI")` 会抛 `TypeError: got multiple values for argument 'name'`。

    Args:
        prompt: 提示词文件名，如 ``"planner"``。
        task_key: YAML 里的任务块键名，默认 ``"task"``；researcher 另有
            ``"single_task"``。
        variables: 渲染变量，如 ``topic="..."``。
    """
    raw = _load_file(prompt)
    block = raw.get(task_key)
    if not isinstance(block, dict):
        raise PromptNotFoundError(f"{prompt}.yaml 缺少任务块 {task_key!r}")
    out: dict[str, str] = {}
    for key in ("description", "expected_output"):
        if key not in block:
            raise PromptNotFoundError(f"{prompt}.yaml 的 {task_key} 缺少字段 {key!r}")
        out[key] = render(block[key], **variables)
    return out


def raw(prompt: str) -> dict[str, Any]:
    """取某个提示词文件的原始 dict（不渲染，供需要结构化数据的场景用）。"""
    return _load_file(prompt)


def clear_cache() -> None:
    """清空缓存（测试用）。"""
    with _lock:
        _cache.clear()


__all__ = [
    "PromptNotFoundError",
    "agent_block",
    "available",
    "clear_cache",
    "raw",
    "render",
    "task_block",
]
