"""Conversation history — the follow-up path. No network.

The property under test is narrow and load-bearing: history reaches the planner
so a pronoun can be resolved, and reaches nothing else, so a figure from an
earlier turn can never be presented as this turn's answer.
"""

from __future__ import annotations

import pytest

from ledgerlens.agent import memory


@pytest.fixture(autouse=True)
def clean():
    memory._threads.clear()
    yield
    memory._threads.clear()


# --- storage -----------------------------------------------------------------

def test_turns_come_back_oldest_first():
    memory.remember("t", "first?", "1")
    memory.remember("t", "second?", "2")
    assert [turn["question"] for turn in memory.history("t")] == ["first?", "second?"]


def test_threads_do_not_bleed_into_each_other():
    memory.remember("a", "mine", "1")
    memory.remember("b", "yours", "2")
    assert [t["question"] for t in memory.history("a")] == ["mine"]


def test_no_thread_id_means_no_history():
    """Every eval calls ask() without one, so a benchmark score can never be
    lifted by the question that happened to run before it."""
    memory.remember(None, "q", "a")
    assert memory.history(None) == []
    assert memory._threads == {}


def test_a_thread_is_bounded():
    for i in range(memory.MAX_TURNS + 4):
        memory.remember("t", f"q{i}", "a")
    turns = memory.history("t")
    assert len(turns) == memory.MAX_TURNS
    assert turns[-1]["question"] == f"q{memory.MAX_TURNS + 3}"    # newest kept


def test_the_store_is_bounded():
    """An unbounded dict keyed by client-supplied ids is a memory leak with a
    user-facing trigger."""
    for i in range(memory.MAX_THREADS + 10):
        memory.remember(f"t{i}", "q", "a")
    assert len(memory._threads) == memory.MAX_THREADS


def test_activity_keeps_a_thread_alive():
    memory.remember("keep", "q", "a")
    for i in range(memory.MAX_THREADS - 1):
        memory.remember(f"filler{i}", "q", "a")
    memory.remember("keep", "again", "a")            # moves it to newest
    memory.remember("overflow", "q", "a")            # evicts the oldest
    assert memory.history("keep")


def test_forgetting_actually_forgets():
    memory.remember("t", "q", "a")
    memory.forget("t")
    assert memory.history("t") == []


def test_a_refusal_is_still_a_turn():
    """"Why not?" is a follow-up. A history that omits the turn it refers to
    would have the planner resolve the pronoun against the wrong question."""
    memory.remember("t", "What is my credit score?", "I can't answer that: not in bank data")
    assert len(memory.history("t")) == 1


# --- rendering ---------------------------------------------------------------

def test_no_turns_renders_nothing():
    """An empty history must not leave a stray heading in the prompt."""
    assert memory.as_prompt([]) == ""


def test_the_prompt_forbids_reusing_figures():
    text = memory.as_prompt([{"question": "travel in April?", "answer": "-500.00"}])
    assert "Do NOT reuse" in text
    assert "stands on" in text


def test_long_answers_are_truncated():
    """Answers are here to say what the turn was about. A paragraph of figures
    in the planner's prompt is a paragraph of figures available to copy."""
    text = memory.as_prompt([{"question": "q", "answer": "9" * 5000}], max_chars=40)
    assert "…" in text
    assert len(text) < 500


# --- the boundary ------------------------------------------------------------

def test_history_reaches_the_planner_and_no_other_node():
    """State is a dict every node can read. The guarantee that a remembered
    figure cannot become an answer rests on only one node consulting it, which
    is a fact about the source rather than about the type."""
    import pathlib

    root = pathlib.Path(memory.__file__).parent
    readers = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if path.name not in {"memory.py", "state.py", "graph.py"}
        and ('state["history"]' in path.read_text() or 'get("history")' in path.read_text())
    ]
    assert readers == ["nodes/planner.py"], readers


def test_the_planner_passes_history_through(monkeypatch):
    from ledgerlens.agent.nodes import planner as planner_mod

    seen = {}

    def capture(question, feedback="", history=None):
        seen["history"] = history
        return planner_mod.Plan(answerable=False, refusal_reason="stub")

    monkeypatch.setattr(planner_mod, "make_plan", capture)
    turns = [{"question": "travel in April?", "answer": "-500.00"}]
    planner_mod.planner({"question": "and in May?", "history": turns})
    assert seen["history"] == turns


# --- the API surface ---------------------------------------------------------

@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from ledgerlens.api.main import app

    return TestClient(app)


@pytest.fixture
def captured(monkeypatch):
    """Stub the graph. What matters here is the thread id, not the answer.

    The ledger guard is stubbed too. Threading is a property of the endpoint
    and not of the data, so gating these on a built ledger would drop them in
    CI — where there is no ledger — and that is precisely where the wiring
    between page, endpoint and store most needs checking.
    """
    from ledgerlens.api import main

    monkeypatch.setattr(main, "_require_ledger", lambda: None)
    seen = {}

    def fake_ask(question, thread_id=None):
        seen["thread_id"] = thread_id
        return {"answer": "ok", "verified": True, "no_data": False, "verdict": {},
                "plan": [], "tool_results": [], "answerable": True,
                "thread_id": thread_id}

    monkeypatch.setattr("ledgerlens.agent.graph.ask", fake_ask)
    return seen


def test_a_first_question_is_given_a_thread(client, captured):
    """Minted server-side, so a thread is something this process handed out
    rather than any string a caller invents."""
    body = client.post("/ask", json={"question": "q"}).json()
    assert body["thread_id"]
    assert body["thread_id"] == captured["thread_id"]


def test_a_follow_up_stays_on_the_same_thread(client, captured):
    first = client.post("/ask", json={"question": "travel in April?"}).json()
    second = client.post("/ask", json={"question": "and in May?",
                                       "thread_id": first["thread_id"]}).json()
    assert second["thread_id"] == first["thread_id"]


def test_a_thread_can_be_forgotten(client):
    """DELETE touches no ledger, so it needs no guard stubbed."""
    memory.remember("chat-x", "q", "a")
    assert client.delete("/ask/chat-x").json()["status"] == "forgotten"
    assert memory.history("chat-x") == []
