"""Database operations for users, sessions, and exports."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import models, schemas, security


_REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORTS_DIR = Path(os.getenv("EXPORTS_DIR", _REPO_ROOT / "exports"))
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Users ────────────────────────────────────────────────────────────────────

def get_user(db: Session, user_id: str) -> Optional[models.User]:
    return db.get(models.User, user_id)


def get_user_by_username(db: Session, username: str) -> Optional[models.User]:
    return db.scalar(select(models.User).where(models.User.username == username))


def list_users_for_admin(db: Session, admin_id: str) -> list[models.User]:
    return list(
        db.scalars(
            select(models.User)
            .where(models.User.admin_id == admin_id)
            .order_by(models.User.created_at.desc())
        )
    )


def create_user(
    db: Session,
    *,
    username: str,
    password: str,
    role: str = "user",
    admin_id: Optional[str] = None,
    credit_balance: float = 0.0,
) -> models.User:
    user = models.User(
        username=username,
        password_hash=security.hash_password(password),
        role=role,
        admin_id=admin_id,
        credit_balance=credit_balance,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def update_budget(
    db: Session, user: models.User, req: schemas.BudgetUpdateRequest
) -> models.User:
    if req.set_to is not None:
        user.credit_balance = float(req.set_to)
    if req.add is not None:
        user.credit_balance = float(user.credit_balance) + float(req.add)
    db.commit()
    db.refresh(user)
    return user


def deactivate_user(db: Session, user: models.User) -> None:
    user.is_active = False
    db.commit()


def user_stats(db: Session, user_id: str) -> tuple[int, float]:
    """(session_count, total_cost_used) for one user."""
    row = db.execute(
        select(
            func.count(models.ResearchSession.id),
            func.coalesce(func.sum(models.ResearchSession.cost_used), 0.0),
        ).where(models.ResearchSession.user_id == user_id)
    ).one()
    return int(row[0] or 0), float(row[1] or 0.0)


def ensure_default_admin(db: Session) -> None:
    """Bootstrap: if no admin exists, create one from env vars."""
    has_admin = db.scalar(select(func.count(models.User.id)).where(models.User.role == "admin"))
    if has_admin:
        return
    username = os.getenv("DEFAULT_ADMIN_USERNAME", "admin")
    password = os.getenv("DEFAULT_ADMIN_PASSWORD", "admin")
    if get_user_by_username(db, username):
        return
    create_user(db, username=username, password=password, role="admin", credit_balance=0.0)


# ── Research sessions ────────────────────────────────────────────────────────

def create_session(
    db: Session,
    *,
    user_id: str,
    question: str,
    entity_type: Optional[str],
    entity_list: Iterable[str],
    plan_dict: Optional[dict],
) -> models.ResearchSession:
    session = models.ResearchSession(
        user_id=user_id,
        question=question,
        entity_type=entity_type,
        entity_list_json=json.dumps(list(entity_list), ensure_ascii=False),
        plan_json=json.dumps(plan_dict, ensure_ascii=False) if plan_dict else None,
        status="running",
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def complete_session(
    db: Session,
    session: models.ResearchSession,
    *,
    status: str,
    cost_used: float,
) -> models.ResearchSession:
    session.status = status
    session.cost_used = float(cost_used)
    session.completed_at = datetime.utcnow()
    db.commit()
    db.refresh(session)
    return session


def list_sessions_for_user(
    db: Session, user_id: str, limit: int = 100
) -> list[models.ResearchSession]:
    return list(
        db.scalars(
            select(models.ResearchSession)
            .where(models.ResearchSession.user_id == user_id)
            .order_by(models.ResearchSession.created_at.desc())
            .limit(limit)
        )
    )


def get_session(db: Session, session_id: str) -> Optional[models.ResearchSession]:
    return db.get(models.ResearchSession, session_id)


# ── Excel exports ────────────────────────────────────────────────────────────

def record_export(
    db: Session,
    *,
    session_id: str,
    user_id: str,
    filename: str,
    file_path: str,
) -> models.ExcelExport:
    export = models.ExcelExport(
        session_id=session_id,
        user_id=user_id,
        filename=filename,
        file_path=file_path,
    )
    db.add(export)
    db.commit()
    db.refresh(export)
    return export


def get_export(db: Session, export_id: str) -> Optional[models.ExcelExport]:
    return db.get(models.ExcelExport, export_id)
