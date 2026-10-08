"""数据库 ORM 模型（SQLAlchemy async）"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey, JSON
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
    exec_path = Column(String(32), default="")   # "arq" / "fallback(main)" / "fallback(worker)"
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
    url = Column(String(1024))
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


class Claim(Base):
    __tablename__ = "claims"
    id = Column(Integer, primary_key=True)
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
