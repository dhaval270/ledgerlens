"""The routing harness's own arithmetic — no network, no model.

A benchmark that miscounts is worse than no benchmark, because the number still
looks like a number. These are the two ways this one has already been wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.run_routing_eval import is_rate_limited, run    # noqa: E402

RATE_LIMIT = ("RateLimitError: Error code: 429 - {'error': {'message': "
              "'Rate limit reached for model ... on tokens per day (TPD)'}}")


def test_a_429_is_recognised():
    assert is_rate_limited(RATE_LIMIT)
    assert not is_rate_limited(None)
    assert not is_rate_limited("OperationalError: no such table: anomaly_results")


def _fake_run(monkeypatch, outcomes):
    """Drive `run` with a stubbed agent. `outcomes` maps question -> result."""
    import evals.run_routing_eval as harness

    def fake_ask(question):
        result = outcomes[question]
        if isinstance(result, Exception):
            raise result
        return result

    module = type(sys)("ledgerlens.agent.graph")
    module.ask = fake_ask
    monkeypatch.setitem(sys.modules, "ledgerlens.agent.graph", module)
    return harness


def _entry(qid, question, tools, expected):
    return {"id": qid, "question": question, "tools": tools,
            "expected_value": expected, "expect_refusal": False}


def _outcome(tool, value):
    return {"tool_results": [{"tool": tool, "query": "SELECT 1",
                              "rows": [{"v": value}]}],
            "verified": True, "no_data": False, "answerable": True}


class _Throttled(Exception):
    def __str__(self):
        return RATE_LIMIT


def test_a_throttled_query_is_not_scored_as_a_wrong_answer(monkeypatch):
    """The failure this exists for.

    A 429 lands at the end of a run, where the daily budget runs out. Counted
    as an ordinary failure it is indistinguishable from a bad plan and drags
    every headline down by however many queries were left — one run reported
    27.8% routing accuracy over 18 queries of which the last 5 never reached
    the model at all.
    """
    harness = _fake_run(monkeypatch, {
        "good": _outcome("sql", -100.0),
        "throttled": _Throttled(),
    })
    report = harness.run([_entry("a", "good", ["sql"], -100.0),
                          _entry("b", "throttled", ["sql"], -200.0)])

    assert report["n"] == 1                    # scored over what ran
    assert report["attempted"] == 2
    assert report["rate_limited"] == 1
    assert report["complete"] is False
    assert report["answer_accuracy_pct"] == 100.0
    assert report["routing_accuracy_pct"] == 100.0


def test_a_complete_run_says_so(monkeypatch):
    harness = _fake_run(monkeypatch, {"good": _outcome("sql", -100.0)})
    report = harness.run([_entry("a", "good", ["sql"], -100.0)])
    assert report["complete"] is True
    assert report["rate_limited"] == 0


def test_a_real_error_is_still_a_failure(monkeypatch):
    """Only throttling is excused. A crash is a result."""
    harness = _fake_run(monkeypatch, {"boom": RuntimeError("no such table")})
    report = harness.run([_entry("a", "boom", ["sql"], -100.0)])
    assert report["n"] == 1
    assert report["answer_accuracy_pct"] == 0.0


def test_a_fully_throttled_run_measures_nothing(monkeypatch):
    """Rather than dividing by zero, or printing 0.0% as if it were a score."""
    harness = _fake_run(monkeypatch, {"throttled": _Throttled()})
    report = harness.run([_entry("a", "throttled", ["sql"], -100.0)])
    assert report["n"] == 0
    assert report["complete"] is False
    assert "answer_accuracy_pct" not in report
