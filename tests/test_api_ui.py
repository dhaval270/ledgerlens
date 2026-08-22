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


# --- the page and the API must agree ----------------------------------------

def _paths_the_page_calls(body: str) -> set[str]:
    """Every path the JavaScript fetches, normalised to its route shape.

    `/approvals/${openThread}/decide` becomes `/approvals/{}/decide` so it can
    be compared against the route table without evaluating the template.
    """
    import re

    found = set()
    for raw in re.findall(r"""(?:api|fetch)\(\s*[`'"]([^`'"]+)""", body):
        path = raw.split("?")[0].rstrip("/") or "/"
        found.add(re.sub(r"\$\{[^}]*\}", "{}", path))
    return found


def _route_shapes(app) -> set[str]:
    import re

    return {re.sub(r"\{[^}]*\}", "{}", r.path.rstrip("/") or "/")
            for r in app.routes if hasattr(r, "path")}


def test_every_path_the_page_calls_is_a_real_route(client):
    """The bug this exists for is silent in both directions.

    A page calling a route that does not exist shows an error only to whoever
    clicks that control; a route with no caller shows nothing at all. §8's
    approval endpoints, /meta — built explicitly "for the approval forms" — and
    the staleness guard were all live and complete, and the page had no way to
    reach any of them. The README's central safety claim was true and
    unreachable.
    """
    from ledgerlens.api.main import app

    body = client.get("/ui").text
    missing = _paths_the_page_calls(body) - _route_shapes(app)
    assert not missing, f"page calls routes that do not exist: {sorted(missing)}"


def test_the_page_reaches_the_approval_endpoints(client):
    """Named explicitly, because the generic check above passes on a page that
    calls nothing at all."""
    body = client.get("/ui").text
    called = _paths_the_page_calls(body)
    assert "/approvals" in called
    assert "/approvals/{}/decide" in called
    assert "/meta" in called


def test_a_write_is_never_one_click_away(client):
    """Approve must be rendered from a proposal, not wired to a bare action.

    The whole §8 property is that the diff comes first. A page that could POST
    a decision without having shown a proposal would satisfy the route check
    above and violate the thing the routes are for.
    """
    body = client.get("/ui").text
    decide_at = body.index("/decide")
    assert "renderProposal" in body[:decide_at]
    assert body.count("awaiting your approval") == 1
