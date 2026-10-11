"""领域异常类型。

设计原则（重要）：
    这条流水线只有**一条**正常路径 —— arq 入队 → worker 执行。
    其余一切情况（Redis 不可用、worker 未启动、队列池建不起来）都属于
    **服务异常**，必须显式抛出、被 API 层转成 503 让调用方看见，
    绝不允许任何代码悄悄绕行到"本进程接着跑"的降级路径。

    历史教训：曾经入队函数返回 bool，`enqueue_job` 不抛异常就 return True，
    而 True 只证明"事件进了 Redis 列表"，不证明"有 worker 在取"。
    结果 worker 缺席时任务永久停在 running，前端一直转圈、无错误、无超时。
"""


class ServiceUnavailableError(RuntimeError):
    """依赖服务不可用，任务无法进入正常执行路径。

    由 orchestrator 的执行前置检查抛出，API 层捕获后转 HTTP 503。
    """

    def __init__(self, service: str, detail: str = ""):
        # service: "redis" | "worker" | "queue"
        self.service = service
        self.detail = detail
        super().__init__(f"{service} 不可用: {detail}" if detail else f"{service} 不可用")

    def as_dict(self) -> dict:
        """给 API 层直接塞进 HTTPException(detail=...) 的结构化描述"""
        return {
            "code": "service_unavailable",
            "service": self.service,
            "message": str(self),
        }


class ModelUnreachableError(RuntimeError):
    """模型连通性预检失败（M2.5-5）。

    由 ``src/core/preflight.py`` 抛出，API 层转 HTTP 400。含义是「你选的这些
    模型此刻用不了」——Key 错、地址错、模型名不存在、网络不通，任何一种都算。
    与 ``ServiceUnavailableError`` 的区别：那个是**我们这边**的服务故障
    （Redis/worker），这个是**用户填的配置**有问题。
    """

    def __init__(self, failures: list[dict]):
        self.failures = list(failures or [])
        super().__init__(f"{len(self.failures)} 个模型的连通性预检失败")

    def as_dict(self) -> dict:
        return {"code": "model_unreachable", "failures": self.failures}


class BuiltinLockedError(RuntimeError):
    """内置模型未解锁，或该用户一个可用模型都没有（M2.5-5）。

    API 层转 HTTP 403。两种情形共用一个异常是因为它们的前置动作相同
    （去解锁，或去添加自己的来源与模型），用 ``reason`` 区分：
      * ``locked``   —— 计划里引用了 ``builtin:*`` 但当前用户没解锁；
      * ``no_model`` —— 目录为空：没解锁 + 也没有任何自建模型。
    """

    def __init__(self, roles: list[str], reason: str = "locked"):
        self.roles = list(roles or [])
        self.reason = reason
        if reason == "no_model":
            message = (
                "你没有可用的模型：内置模型需要先用密码解锁，"
                "或先在「管理来源与模型」里添加自己的来源与模型"
            )
        else:
            message = f"内置模型未解锁（涉及角色：{', '.join(self.roles) or '全部'}）"
        super().__init__(message)

    def as_dict(self) -> dict:
        return {
            "code": "builtin_locked",
            "reason": self.reason,
            "roles": self.roles,
            "message": str(self),
        }
