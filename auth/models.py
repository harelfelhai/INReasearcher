"""SQLAlchemy ORM models for users, research sessions, and Excel exports."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from .db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=_uuid)
    username = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False, default="user")   # "user" | "admin"
    admin_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    credit_balance = Column(Float, nullable=False, default=0.0)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    admin = relationship("User", remote_side=[id], backref="managed_users")
    sessions = relationship(
        "ResearchSession", back_populates="user", cascade="all, delete-orphan"
    )


class ResearchSession(Base):
    __tablename__ = "research_sessions"

    id = Column(String, primary_key=True, default=_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    question = Column(Text, nullable=False)
    entity_type = Column(String, nullable=True)
    entity_list_json = Column(Text, nullable=True)      # JSON-encoded list
    plan_json = Column(Text, nullable=True)             # JSON-encoded ResearchPlan
    results_json = Column(Text, nullable=True)          # JSON-encoded list[entity_done payloads]
    status = Column(String, nullable=False, default="running")
    cost_used = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="sessions")
    exports = relationship(
        "ExcelExport", back_populates="session", cascade="all, delete-orphan"
    )


class ExcelExport(Base):
    __tablename__ = "excel_exports"

    id = Column(String, primary_key=True, default=_uuid)
    session_id = Column(String, ForeignKey("research_sessions.id"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    session = relationship("ResearchSession", back_populates="exports")
