"""Pydantic schemas for the auth/admin/user API surface."""
from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ── Auth ─────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    user_id: str
    username: str
    role: Literal["user", "admin"]


class UserOut(BaseModel):
    id: str
    username: str
    role: Literal["user", "admin"]
    admin_id: Optional[str] = None
    credit_balance: float
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Admin ────────────────────────────────────────────────────────────────────

class CreateUserRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=4, max_length=128)
    role: Literal["user", "admin"] = "user"
    credit_balance: float = 0.0


class BudgetUpdateRequest(BaseModel):
    """Either set the balance to an absolute number or add to it."""
    set_to: Optional[float] = None
    add: Optional[float] = None


class ManagedUserOut(UserOut):
    session_count: int = 0
    total_cost_used: float = 0.0


# ── Sessions / Exports ───────────────────────────────────────────────────────

class ExcelExportOut(BaseModel):
    id: str
    filename: str
    created_at: datetime

    model_config = {"from_attributes": True}


class SessionOut(BaseModel):
    id: str
    user_id: str
    question: str
    entity_type: Optional[str] = None
    status: str
    cost_used: float
    created_at: datetime
    completed_at: Optional[datetime] = None
    exports: List[ExcelExportOut] = []

    model_config = {"from_attributes": True}


# ── Memory feedback / seeding ─────────────────────────────────────────────────

class CellFeedback(BaseModel):
    entity_name: str
    field_id: str
    is_correct: bool
    correct_value: Optional[str] = None


class FeedbackRequest(BaseModel):
    cells: List[CellFeedback]


class MemorySeedRequest(BaseModel):
    kind: Literal["success", "failure"]
    field_type: str
    field_label: str
    entity: str
    value: str
    quote: str
    source_url: str
    source_domain: str = ""
    reason: Optional[str] = None


class MemoryStatsOut(BaseModel):
    compiler_successes: int
    extraction_successes: int
    extraction_failures: int
    path: str
