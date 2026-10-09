"""CrewAI 启动前置引导（必须在 `import crewai` 之前执行）。

## 背景

CrewAI 在 **import 期**就会初始化遥测/tracing（`crewai/events/event_listener.py`
模块级 `event_listener = EventListener()`），该初始化会调用
`crewai_core.settings.get_writable_config_path()`，依次尝试：

1. `~/.config/crewai/settings.json`
2. `%TEMP%/crewai_settings.json`
3. `./crewai_settings.json`

问题：CrewAI 用 `tempfile.mkstemp()` 探测目录可写性。当进程运行在受限文件沙箱
（只允许写工作区）时，对 `~/.config/crewai` 的写操作**不是抛异常，而是无限阻塞**，
于是 fallback 链永远走不到第 2、3 项 → `import crewai` 永久挂死。

此外 `crewai_core/paths.py::db_storage_path()` 经 `appdirs.user_data_dir()` 取
`%LOCALAPPDATA%`（Windows 下走 pywin32 注册表），也需要可写。

## 做法

启动时快速探测 `~/.config/crewai` 与 `%LOCALAPPDATA%` 是否可写
（用直接 `open()`——它会在权限不足时**立刻抛 PermissionError**，不像 mkstemp 会挂）。
若不可写，就把 HOME/USERPROFILE/APPDATA/LOCALAPPDATA/TEMP 指向工作区内的
`.dev-home/`（并预建所需子目录），使 CrewAI 与 appdirs 都落到可写位置。

真实部署（服务器/开发机）上这些目录本就可写，本模块不做任何改动。
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_ENV_MARKER = "INSIGHTCREW_BOOTSTRAPPED"

# 项目根目录：src/core/bootstrap.py → src/core → src → 项目根
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_DEV_HOME = _PROJECT_ROOT / ".dev-home"

# .dev-home 下需要预建的子目录
_DEV_SUBDIRS = (
    ".config/crewai",
    "AppData/Local",
    "AppData/Roaming",
    "AppData/Local/Temp",
)


def _is_writable(directory: Path) -> bool:
    """探测目录是否可写。

    必须用直接 open()：在受限沙箱下 `tempfile.mkstemp()` 对不可写目录会
    **无限阻塞**，而 open() 会立刻抛 PermissionError。
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".insightcrew_write_probe"
        with open(probe, "w", encoding="utf-8") as f:
            f.write("x")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _needs_redirect() -> bool:
    """判断是否需要把 HOME 等重定向到工作区。"""
    home = Path.home()
    if not _is_writable(home / ".config" / "crewai"):
        return True

    # Windows 上 appdirs 走 %LOCALAPPDATA%
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata and not _is_writable(Path(local_appdata)):
        return True

    return False


def _redirect_to_dev_home() -> None:
    """把 CrewAI/appdirs 依赖的目录环境变量指向 .dev-home。"""
    for sub in _DEV_SUBDIRS:
        (_DEV_HOME / sub).mkdir(parents=True, exist_ok=True)

    os.environ["USERPROFILE"] = str(_DEV_HOME)
    os.environ["HOME"] = str(_DEV_HOME)
    os.environ["APPDATA"] = str(_DEV_HOME / "AppData" / "Roaming")
    os.environ["LOCALAPPDATA"] = str(_DEV_HOME / "AppData" / "Local")

    dev_temp = str(_DEV_HOME / "AppData" / "Local" / "Temp")
    os.environ["TEMP"] = dev_temp
    os.environ["TMP"] = dev_temp
    tempfile.tempdir = dev_temp

    # appdirs 在 Windows 下优先用 pywin32 的 SHGetFolderPath（读注册表，不受
    # 环境变量影响），这里显式告知 crewai_core 使用工作区存储。
    os.environ.setdefault("CREWAI_STORAGE_DIR", _PROJECT_ROOT.name)


def ensure_crewai_environment() -> None:
    """幂等入口：确保 CrewAI 能在当前沙箱下正常 import。"""
    if os.environ.get(_ENV_MARKER) == "1":
        return
    os.environ[_ENV_MARKER] = "1"

    # 关闭 CrewAI 遥测/tracing：否则首次执行会弹出
    # "Share this execution trace with CrewAI? [y/N] (20s timeout)" 交互提示。
    # 在 arq Worker 等无 tty 环境下该提示虽会被跳过，但服务端不应依赖这一点，
    # 且交互终端下会白等 20 秒。这里显式关闭。
    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

    try:
        if _needs_redirect():
            _redirect_to_dev_home()
            print(
                "[bootstrap] 检测到 ~/.config/crewai 或 %LOCALAPPDATA% 不可写，"
                f"已将 CrewAI 配置目录重定向到 {_DEV_HOME}",
                file=sys.stderr,
            )
    except Exception as e:  # 引导失败不应阻止进程启动
        print(f"[bootstrap] 环境引导失败（忽略）: {e}", file=sys.stderr)


# import src.core.bootstrap 即生效
ensure_crewai_environment()
