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
    hashed_password = Column(String(256), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


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
