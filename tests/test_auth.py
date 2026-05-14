"""Auth + admin + user-router tests using FastAPI TestClient.

Uses an isolated SQLite file (one per test session, dropped + recreated
per test) so we never touch the real data/app.db.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


# ── Test DB setup: must happen BEFORE importing api.main / auth.db ───────────
_TEST_DB = Path(__file__).parent / "_test_auth.db"
if _TEST_DB.exists():
    _TEST_DB.unlink()
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB}"
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")

from fastapi.testclient import TestClient  # noqa: E402

from auth.db import Base, SessionLocal, engine  # noqa: E402
from auth import crud  # noqa: E402
from api.main import app  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_db():
    """Drop and recreate all tables before each test."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        crud.ensure_default_admin(db)
    finally:
        db.close()
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    return TestClient(app)


def _login(client: TestClient, username: str, password: str) -> dict:
    r = client.post("/api/auth/login",
                    json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return r.json()


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ── Auth ─────────────────────────────────────────────────────────────────────

def test_health_is_public(client):
    assert client.get("/api/health").json() == {"ok": True}


def test_login_with_correct_credentials_returns_token(client):
    data = _login(client, "admin", "admin")
    assert data["token_type"] == "bearer"
    assert data["role"] == "admin"
    assert data["access_token"]


def test_login_with_wrong_password_returns_401(client):
    r = client.post("/api/auth/login",
                    json={"username": "admin", "password": "WRONG"})
    assert r.status_code == 401


def test_me_requires_token(client):
    assert client.get("/api/auth/me").status_code == 401


def test_me_returns_current_user(client):
    tok = _login(client, "admin", "admin")["access_token"]
    me = client.get("/api/auth/me", headers=_auth(tok)).json()
    assert me["username"] == "admin"
    assert me["role"] == "admin"


# ── Admin: create user + budget ──────────────────────────────────────────────

def test_admin_can_create_user_and_user_can_login(client):
    admin = _login(client, "admin", "admin")["access_token"]
    r = client.post(
        "/api/admin/users",
        headers=_auth(admin),
        json={"username": "alice", "password": "pw1234",
              "role": "user", "credit_balance": 5.0},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["username"] == "alice"
    assert body["credit_balance"] == 5.0
    assert body["admin_id"]

    # Alice can now log in
    alice = _login(client, "alice", "pw1234")
    assert alice["role"] == "user"


def test_admin_create_user_rejects_duplicate_username(client):
    admin = _login(client, "admin", "admin")["access_token"]
    payload = {"username": "bob", "password": "x1234",
               "role": "user", "credit_balance": 0}
    client.post("/api/admin/users", headers=_auth(admin), json=payload)
    r = client.post("/api/admin/users", headers=_auth(admin), json=payload)
    assert r.status_code == 409


def test_admin_lists_only_its_own_users(client):
    """Two admins, each with one user; admin A must not see admin B's user."""
    admin_a = _login(client, "admin", "admin")["access_token"]

    # Create a second admin via raw CRUD (the /admin endpoint deliberately
    # refuses to create admins)
    db = SessionLocal()
    crud.create_user(db, username="admin2", password="admin2",
                     role="admin", credit_balance=0)
    db.close()
    admin_b = _login(client, "admin2", "admin2")["access_token"]

    client.post("/api/admin/users", headers=_auth(admin_a),
                json={"username": "user_a", "password": "p1234",
                      "role": "user", "credit_balance": 0})
    client.post("/api/admin/users", headers=_auth(admin_b),
                json={"username": "user_b", "password": "p1234",
                      "role": "user", "credit_balance": 0})

    listed_a = {u["username"] for u in
                client.get("/api/admin/users", headers=_auth(admin_a)).json()}
    listed_b = {u["username"] for u in
                client.get("/api/admin/users", headers=_auth(admin_b)).json()}

    assert "user_a" in listed_a and "user_b" not in listed_a
    assert "user_b" in listed_b and "user_a" not in listed_b


def test_budget_add_and_set_modify_balance(client):
    admin = _login(client, "admin", "admin")["access_token"]
    user = client.post("/api/admin/users", headers=_auth(admin),
                       json={"username": "user_one", "password": "p1234",
                             "role": "user", "credit_balance": 10.0}).json()
    uid = user["id"]

    # add 5
    r = client.put(f"/api/admin/users/{uid}/budget",
                   headers=_auth(admin), json={"add": 5.0})
    assert r.status_code == 200
    assert r.json()["credit_balance"] == 15.0

    # set to 2
    r = client.put(f"/api/admin/users/{uid}/budget",
                   headers=_auth(admin), json={"set_to": 2.0})
    assert r.status_code == 200
    assert r.json()["credit_balance"] == 2.0

    # neither → 400
    r = client.put(f"/api/admin/users/{uid}/budget",
                   headers=_auth(admin), json={})
    assert r.status_code == 400


def test_admin_cannot_modify_other_admins_users(client):
    admin_a = _login(client, "admin", "admin")["access_token"]

    db = SessionLocal()
    crud.create_user(db, username="admin2", password="admin2",
                     role="admin", credit_balance=0)
    db.close()
    admin_b = _login(client, "admin2", "admin2")["access_token"]

    u_b = client.post("/api/admin/users", headers=_auth(admin_b),
                      json={"username": "user_b", "password": "p1234",
                            "role": "user", "credit_balance": 0}).json()

    # admin A cannot touch u_b's budget
    r = client.put(f"/api/admin/users/{u_b['id']}/budget",
                   headers=_auth(admin_a), json={"add": 1.0})
    assert r.status_code == 404


# ── Role gating ──────────────────────────────────────────────────────────────

def test_regular_user_cannot_hit_admin_endpoints(client):
    admin = _login(client, "admin", "admin")["access_token"]
    client.post("/api/admin/users", headers=_auth(admin),
                json={"username": "user_one", "password": "p1234",
                      "role": "user", "credit_balance": 0})
    u_tok = _login(client, "user_one", "p1234")["access_token"]
    assert client.get("/api/admin/users", headers=_auth(u_tok)).status_code == 403


def test_admin_create_rejects_admin_role_request(client):
    """The admin /users endpoint only mints regular users — admins must be
    bootstrapped or created via direct DB access."""
    admin = _login(client, "admin", "admin")["access_token"]
    r = client.post("/api/admin/users", headers=_auth(admin),
                    json={"username": "newadmin", "password": "p1234",
                          "role": "admin", "credit_balance": 0})
    assert r.status_code == 400


# ── User sessions are isolated per user ──────────────────────────────────────

def test_user_sessions_endpoint_returns_only_own_sessions(client):
    admin = _login(client, "admin", "admin")["access_token"]
    u1 = client.post("/api/admin/users", headers=_auth(admin),
                     json={"username": "user_one", "password": "p1234",
                           "role": "user", "credit_balance": 0}).json()
    client.post("/api/admin/users", headers=_auth(admin),
                json={"username": "user_two", "password": "p1234",
                      "role": "user", "credit_balance": 0})

    # Insert a session directly for u1
    db = SessionLocal()
    sess = crud.create_session(
        db, user_id=u1["id"], question="who is the mayor",
        entity_type="city", entity_list=["Tel Aviv"], plan_dict={"x": 1},
    )
    db.close()

    u1_tok = _login(client, "user_one", "p1234")["access_token"]
    u2_tok = _login(client, "user_two", "p1234")["access_token"]

    u1_list = client.get("/api/user/sessions", headers=_auth(u1_tok)).json()
    u2_list = client.get("/api/user/sessions", headers=_auth(u2_tok)).json()
    assert any(s["id"] == sess.id for s in u1_list)
    assert all(s["id"] != sess.id for s in u2_list)
    assert len(u2_list) == 0


# ── Inactive user cannot log in ──────────────────────────────────────────────

def test_deactivated_user_cannot_use_existing_token(client):
    admin = _login(client, "admin", "admin")["access_token"]
    u = client.post("/api/admin/users", headers=_auth(admin),
                    json={"username": "victim", "password": "p1234",
                          "role": "user", "credit_balance": 0}).json()
    tok = _login(client, "victim", "p1234")["access_token"]
    # before disable: works
    assert client.get("/api/auth/me", headers=_auth(tok)).status_code == 200
    # disable
    client.delete(f"/api/admin/users/{u['id']}", headers=_auth(admin))
    # token now rejected (user is_active = False)
    assert client.get("/api/auth/me", headers=_auth(tok)).status_code == 401
