"""Auth + admin + user routers.

Mounted under /api/auth, /api/admin, /api/user in api/main.py.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from . import crud, schemas, security
from .db import get_db
from .deps import get_current_user, require_admin
from .models import User


# ── /api/auth ────────────────────────────────────────────────────────────────

auth_router = APIRouter(prefix="/api/auth", tags=["auth"])


@auth_router.post("/login", response_model=schemas.TokenResponse)
def login(req: schemas.LoginRequest, db: Session = Depends(get_db)):
    user = crud.get_user_by_username(db, req.username)
    if not user or not user.is_active or not security.verify_password(req.password, user.password_hash):
        raise HTTPException(401, "Invalid username or password")
    token = security.create_access_token(user.id, user.role)
    return schemas.TokenResponse(
        access_token=token,
        user_id=user.id,
        username=user.username,
        role=user.role,  # type: ignore[arg-type]
    )


@auth_router.get("/me", response_model=schemas.UserOut)
def me(user: User = Depends(get_current_user)):
    return user


# ── /api/admin ───────────────────────────────────────────────────────────────

admin_router = APIRouter(prefix="/api/admin", tags=["admin"])


@admin_router.get("/users", response_model=list[schemas.ManagedUserOut])
def list_managed_users(
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    users = crud.list_users_for_admin(db, admin.id)
    out: list[schemas.ManagedUserOut] = []
    for u in users:
        count, cost = crud.user_stats(db, u.id)
        out.append(
            schemas.ManagedUserOut(
                id=u.id, username=u.username, role=u.role,
                admin_id=u.admin_id, credit_balance=u.credit_balance,
                is_active=u.is_active, created_at=u.created_at,
                session_count=count, total_cost_used=cost,
            )
        )
    return out


@admin_router.post("/users", response_model=schemas.UserOut)
def create_managed_user(
    req: schemas.CreateUserRequest,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    if crud.get_user_by_username(db, req.username):
        raise HTTPException(409, "Username already exists")
    # Admin can only create regular users under themselves
    if req.role != "user":
        raise HTTPException(400, "Admins can only create regular users via this endpoint")
    user = crud.create_user(
        db,
        username=req.username,
        password=req.password,
        role="user",
        admin_id=admin.id,
        credit_balance=req.credit_balance,
    )
    return user


@admin_router.put("/users/{user_id}/budget", response_model=schemas.UserOut)
def update_user_budget(
    user_id: str,
    req: schemas.BudgetUpdateRequest,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    target = crud.get_user(db, user_id)
    if not target or target.admin_id != admin.id:
        raise HTTPException(404, "User not found")
    if req.set_to is None and req.add is None:
        raise HTTPException(400, "Provide either 'set_to' or 'add'")
    return crud.update_budget(db, target, req)


@admin_router.delete("/users/{user_id}", status_code=204)
def disable_user(
    user_id: str,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    target = crud.get_user(db, user_id)
    if not target or target.admin_id != admin.id:
        raise HTTPException(404, "User not found")
    crud.deactivate_user(db, target)


@admin_router.get("/users/{user_id}/sessions", response_model=list[schemas.SessionOut])
def list_user_sessions(
    user_id: str,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    target = crud.get_user(db, user_id)
    if not target or target.admin_id != admin.id:
        raise HTTPException(404, "User not found")
    return crud.list_sessions_for_user(db, user_id)


# ── /api/user ────────────────────────────────────────────────────────────────

user_router = APIRouter(prefix="/api/user", tags=["user"])


@user_router.get("/sessions", response_model=list[schemas.SessionOut])
def list_own_sessions(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return crud.list_sessions_for_user(db, user.id)


@user_router.get("/sessions/{session_id}", response_model=schemas.SessionOut)
def get_own_session(
    session_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    sess = crud.get_session(db, session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    # Owner OR the admin of the owner can view
    if sess.user_id != user.id:
        if user.role != "admin":
            raise HTTPException(404, "Session not found")
        owner = crud.get_user(db, sess.user_id)
        if not owner or owner.admin_id != user.id:
            raise HTTPException(404, "Session not found")
    return sess


@user_router.get("/exports/{export_id}")
def download_export(
    export_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    export = crud.get_export(db, export_id)
    if not export:
        raise HTTPException(404, "Export not found")
    # Owner OR the admin of the owner can download
    if export.user_id != user.id:
        if user.role != "admin":
            raise HTTPException(404, "Export not found")
        owner = crud.get_user(db, export.user_id)
        if not owner or owner.admin_id != user.id:
            raise HTTPException(404, "Export not found")
    if not Path(export.file_path).exists():
        raise HTTPException(410, "File no longer available on disk")
    return FileResponse(
        export.file_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=export.filename,
    )
