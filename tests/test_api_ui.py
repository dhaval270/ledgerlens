"""The /ui page itself. Needs no ledger, so it is not gated on one."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from ledgerlens.api.main import app

    return TestClient(app)


def test_the_page_is_never_cached(client):
    """Observed live: fresh JSON from the API rendered by stale JavaScript, so
    the answer text showed an edit and the badge beside it did not."""
    headers = client.get("/ui").headers
    assert "no-store" in headers.get("cache-control", "")


def test_the_page_is_served(client):
    body = client.get("/ui").text
    assert "<title>LedgerLens</title>" in body


def test_the_page_is_self_contained(client):
    """A strict-CSP-style rule by hand: no external hosts, so the UI works
    offline and cannot leak a page view to a CDN."""
    body = client.get("/ui").text
    for marker in ("http://", "https://", "//cdn"):
        assert marker not in body.replace("http://127.0.0.1", "")
