"""数据库 ORM 模型（SQLAlchemy async）"""

from datetime import datetime
from sqlalchemy import (Column, Integer, String, Text, DateTime, Float,
                        ForeignKey, JSON, Index, text)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False)
    # 邮箱可选：老账号（如 demo）没有邮箱，注册页填了才写。
    # 唯一但允许 NULL —— SQLite 里多行 NULL 不冲突，正合「可选」语义。
    email = Column(String(128), unique=True, nullable=True)
    # ★ 邮箱验证时间（用户 m26742）：非空才表示这个邮箱真的归用户所有。
    #   为什么单独一列而不是「有 email 就算绑定」：邮箱是找回密码的唯一凭据，
    #   写错一个字母的邮箱与「没绑邮箱」一样收不到信，但前者看起来像绑定成功，
    #   用户到忘记密码那天才发现 —— 那时已经进不来了。
    #   注册/绑定/换绑都必须先收到验证码并校验通过才写这一列。
    #   老库补列后为 NULL：既有用户（含 demo）需要重新验证一次邮箱才可改密。
    email_verified_at = Column(DateTime, nullable=True)
    hashed_password = Column(String(256), nullable=False)
    # ★ 令牌版本号：改密码时 +1，使此前签发的所有 JWT 立即失效。
    #   JWT 是无状态的，服务端无法「撤回」一个已发出的 token；把版本号写进
    #   载荷、校验时与库中值比对，就把无状态令牌变成了可作废令牌。
    #   老库补列默认 0，老 token 载荷里没有该字段时也按 0 处理，两边一致。
    token_version = Column(Integer, default=0, nullable=False, server_default="0")
    created_at = Column(DateTime, default=datetime.utcnow)


class PasswordReset(Base):
    """找回密码的一次性凭据。

    ★ 只存 ``sha256(token)``，明文 token 仅存在于那封邮件里。
      数据库泄露也拿不到可用的重置链接（和密码只存 bcrypt 哈希同理）。
    ★ ``used_at`` 非空即视为已作废 —— 链接只能点一次，重放直接拒绝。
    """

    __tablename__ = "password_resets"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    token_hash = Column(String(64), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    used_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class EmailCode(Base):
    """邮箱验证码（用户 m26742：注册 / 绑定邮箱 / 换邮箱 / 改密码都要验证）。

    ## 为什么是一张独立表，而不是塞进 users

    验证码的生命周期与用户不同：一个用户可能反复申请（改错邮箱重来、换绑、
    改密各一次），是**一对多**；而且注册时**用户还不存在**，没有 user_id 可挂。
    所以用独立表，`user_id` 可空。

    ## 为什么要分 purpose

    同一个人可能同时有「验证当前邮箱」和「换绑到新邮箱」两个待办。若只按
    email 查码，换绑时用一个发给旧邮箱的码就能通过 —— 验证的意义就没了。
    `purpose` 把不同用途的码隔离开，校验必须 (email, purpose) 双命中。

    ## 为什么存 sha256 而不是明文

    与重置令牌同理：库被读走也拿不到可用的验证码（码只有 6 位，但配合
    5 次尝试上限与 10 分钟有效期，暴力枚举窗口极小）。用 HMAC(密钥) 而非裸
    sha256 —— 6 位数字的裸哈希一秒就能撞出全表，必须带密钥。
    """

    __tablename__ = "email_codes"
    id = Column(Integer, primary_key=True)
    #: 验证码发往的邮箱（小写归一）。注册时还没有 user_id，只能靠它定位。
    email = Column(String(128), nullable=False, index=True)
    #: 关联用户。注册流程为 NULL（账号还没建）。
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    #: register / bind / verify / password —— 见上方「为什么要分 purpose」。
    purpose = Column(String(32), nullable=False)
    code_hash = Column(String(64), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    #: 已用掉的码不能再用（一次性）。
    used_at = Column(DateTime, nullable=True)
    #: 连续错误次数。达到上限即作废 —— 否则 6 位码可以被慢慢试出来。
    attempts = Column(Integer, default=0, nullable=False, server_default="0")
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        # 按 (email, purpose) 查最新的那条；索引用非唯一即可，
        # 「同一邮箱同一用途只保留一条有效」由写入时的作废逻辑保证。
        Index("ix_email_codes_email_purpose", "email", "purpose"),
    )


class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    topic = Column(String(256), nullable=False)
    status = Column(String(32), default="pending")
    current_stage = Column(String(32), default="plan")
    gate_status = Column(String(32), default="none")
    result_report_id = Column(Integer, nullable=True)
    total_cost_cny = Column(Float, default=0.0)
    exec_path = Column(String(32), default="")   # "arq"（唯一正常值）；旧数据可能残留 "fallback(...)"
    # ── 任务级参数（M1 遗留项 5：新建任务不再只能填主题）──
    dimensions = Column(Text, nullable=True)        # 用户指定的评估维度（逗号/换行分隔）
    constraints = Column(Text, nullable=True)       # 用户指定的技术约束
    budget_limit_cny = Column(Float, nullable=True)      # 本任务预算上限；NULL 表示用全局配置
    # ── M2.5：每个阶段用哪个模型（用户逐角色自选）──
    # JSON 文本：{"Planner": "builtin:deepseek-pro", "Researcher": "user:3", ..., "vision": "builtin:glm-4v-flash"}
    # 7 个键 = 6 个 Agent 角色 + 1 个读图档。NULL 表示未指定（用默认计划）。
    # ★ M2.5-7 删除了旧的 `model_strategy` 列（同构/异构开关，已被本字段取代）。
    model_plan = Column(Text, nullable=True)
    # ── 暂停请求（闸门收敛配套：任意阶段边界可临时变成闸门）──
    # 1 = 用户请求在**下一个阶段边界**暂停（不是立刻打断）。worker 在
    # `advance_stage` 读到它就建闸门；建闸时清零。见 `src/core/gates.py::pause_pipeline`。
    pause_requested = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TaskEvent(Base):
    __tablename__ = "task_events"
    id = Column(Integer, primary_key=True)
    task_id = Column(Integer, ForeignKey("tasks.id"))
    event_type = Column(String(64))
    payload = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class HumanReview(Base):
    __tablename__ = "human_reviews"
    id = Column(Integer, primary_key=True)
    task_id = Column(Integer, ForeignKey("tasks.id"))
    gate_name = Column(String(64))
    status = Column(String(32), default="pending")
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)


class Source(Base):
    __tablename__ = "sources"
    id = Column(Integer, primary_key=True)
    # ★ url 必须唯一：get_or_create_source 按 url 复用来源行，若并发下插入两行，
    #   同一个 URL 就有了两个 source_id，引用台账的可追溯链路会分叉。
    url = Column(String(1024), unique=True, index=True)
    title = Column(String(256))
    source_type = Column(String(32))  # arxiv, web, pdf
    fetched_at = Column(DateTime, default=datetime.utcnow)
    raw_text = Column(Text, nullable=True)


class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey("sources.id"))
    seq = Column(Integer)
    content = Column(Text)
    page_num = Column(Integer, nullable=True)
    # ★ (source_id, seq) 是引用标记 [S{source_id}#{seq}] 的解析键，必须唯一。
    #   若允许重复，同一个标记会指向多个 chunk，verify_citations / record_claims /
    #   报告页证据抽屉三处的 scalar_one_or_none() 会直接抛 MultipleResultsFound，
    #   或静默取到错误的 chunk——可追溯性（PRD 硬约束）当场失效。
    #   索引名沿用 SQLAlchemy 默认命名，与 create_all 生成的一致，便于老库补建。
    __table_args__ = (
        Index("ix_chunks_source_id_seq", "source_id", "seq", unique=True),
    )


class Claim(Base):
    __tablename__ = "claims"
    id = Column(Integer, primary_key=True)
    # ★ task_id 是用户隔离的必需字段：没有它，论断就是全局的，
    #   任务详情/证据台账页只能全表查 → 把别人的论断展示给当前用户。
    #   record_claims(text, task_id) 原先收下 task_id 却没落库（形参被忽略），
    #   这里补上列并在 ledger 里真正写入。
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=True, index=True)
    text = Column(Text)
    evidence_level = Column(String(1))  # A=全文验证, B=摘要级, C=待复核
    created_at = Column(DateTime, default=datetime.utcnow)


class ClaimEvidence(Base):
    __tablename__ = "claim_evidence"
    id = Column(Integer, primary_key=True)
    claim_id = Column(Integer, ForeignKey("claims.id"))
    chunk_id = Column(Integer, ForeignKey("chunks.id"))
    source_id = Column(Integer, ForeignKey("sources.id"))


class LlmCall(Base):
    __tablename__ = "llm_calls"
    id = Column(Integer, primary_key=True)
    task_id = Column(Integer, ForeignKey("tasks.id"))
    model = Column(String(64))
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    cost_cny = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True)
    task_id = Column(Integer, ForeignKey("tasks.id"))
    content_md = Column(Text)
    bibtex = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Evaluation(Base):
    __tablename__ = "evaluations"
    id = Column(Integer, primary_key=True)
    task_id = Column(Integer, ForeignKey("tasks.id"))
    metric = Column(String(64))
    value = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


# ══════════════════════════════════════════════════════════════════
# M2.5 用户自定义模型（用户 m12784：平台开放、用户自管）
#
# 内置 5 个模型**不入库**，定义在 config/builtin_models.yaml（只读）；
# 下面三张表只存「用户自建」的东西。
# ══════════════════════════════════════════════════════════════════


class Provider(Base):
    """用户自建的 API 来源（一个来源下有多个模型）。

    protocol 目前只有 'openai'（DeepSeek/智谱/linoapi/OneAPI/Ollama
    全部是 OpenAI 兼容协议），留字段是为了将来接 Anthropic 原生协议。
    """

    __tablename__ = "providers"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(64), nullable=False)
    api_base = Column(String(512), nullable=False)
    # ★ 只存 Fernet 密文，明文永不落库、永不返回前端（见 src/core/crypto.py）
    api_key_encrypted = Column(Text, nullable=False)
    protocol = Column(String(32), default="openai")
    created_at = Column(DateTime, default=datetime.utcnow)


class UserModel(Base):
    """用户自建来源下的一个具体模型。"""

    __tablename__ = "models"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    provider_id = Column(Integer, ForeignKey("providers.id"), nullable=False, index=True)
    model_id = Column(String(128), nullable=False)        # 实际发给 API 的模型名
    display_name = Column(String(128), nullable=True)
    max_tokens = Column(Integer, default=4096)
    temperature = Column(Float, default=0.3)
    supports_vision = Column(Integer, default=0)          # 0/1
    disable_thinking = Column(Integer, default=0)         # 0/1
    # 用户自填单价（¥/百万 token）；0 = 只记 token 不计费
    # （不填会导致记账全 ¥0、预算熔断失效，即已修过的缺陷⑥）
    price_in = Column(Float, default=0.0)
    price_out = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class BuiltinUnlock(Base):
    """内置模型的解锁态。

    ★ 以 user_id 为主键 —— 每个用户独立上锁/解锁，站长解锁不影响他人；
      新注册用户没有这一行，即默认锁上（用户 m13118 澄清）。
    """

    __tablename__ = "builtin_unlocks"
    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    unlocked_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
