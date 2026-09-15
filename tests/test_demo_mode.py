"""A hosted instance is unauthenticated, so it must accept no writes.

These run without a ledger: the refusal has to land *before* the ledger check,
or a demo host with no database would return 409 and hide the real reason.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from ledgerlens.api.main import app

    return TestClient(app)


@pytest.fixture
def demo(monkeypatch):
    monkeypatch.setenv("LEDGERLENS_DEMO", "1")


def test_reads_still_work_in_demo(client, demo):
    assert client.get("/health").json()["demo"] is True
    assert client.get("/").status_code == 200


def test_upload_is_refused(client, demo):
    response = client.post("/ingest", files={"file": ("s.csv", b"a,b\n1,2\n", "text/csv")})
    assert response.status_code == 403
    assert "read-only demo" in response.json()["detail"]


def test_proposing_a_change_is_refused(client, demo):
    response = client.post("/approvals", json={"action": "recategorize", "params": {}})
    assert response.status_code == 403


def test_deciding_an_approval_is_refused(client, demo):
    response = client.post("/approvals/any-thread/decide", json={"approved": True})
    assert response.status_code == 403


def test_writes_are_open_when_the_flag_is_unset(client, monkeypatch):
    """The flag is opt-in. Unset, the app behaves exactly as it always has —
    otherwise local development pays for a production concern."""
    monkeypatch.delenv("LEDGERLENS_DEMO", raising=False)
    assert client.get("/health").json()["demo"] is False
    # Not a 403: it gets far enough to complain about the request itself.
    assert client.post("/approvals", json={"action": "nope", "params": {}}).status_code != 403


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes"])
def test_truthy_spellings(client, monkeypatch, value):
    monkeypatch.setenv("LEDGERLENS_DEMO", value)
    assert client.get("/health").json()["demo"] is True


@pytest.mark.parametrize("value", ["", "0", "false", "no"])
def test_falsy_spellings(client, monkeypatch, value):
    monkeypatch.setenv("LEDGERLENS_DEMO", value)
    assert client.get("/health").json()["demo"] is False
