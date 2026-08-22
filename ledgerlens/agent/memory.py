"""Conversation history for follow-up questions.

Every question used to arrive with no past. "What did I spend on travel in
April?" worked; "and in May?" was unanswerable, and the honest refusal made it
look like a data problem rather than a missing feature. Two of the benchmark's
own anomaly queries are follow-ups — "how much did I actually spend in *that*
abnormal travel month", "my normal travel spend, *ignoring the spike*" — and
both route to plain SQL, because standing alone that is what they look like.

**History is context for the planner, never a source of figures.** Prior turns
are shown so a pronoun can be resolved into a question that stands on its own;
the tools then re-retrieve everything from the ledger. Nothing in §6.7 changes:
a figure still has to appear in a row retrieved during *this* turn or the
verifier rejects it. Answering "and in May?" from a number remembered out of
April's turn is precisely the fabrication the verifier exists to catch, so the
history reaches the planner and no other node.

In process and bounded, which is the honest scope: history dies with the
server, and a deployment with more than one worker will hand consecutive
questions to different memories. Persisting it means a table and a retention
policy for what is, by §10, sensitive data — a decision worth making
deliberately rather than inheriting from a dict.
"""

from __future__ import annotations

import itertools
import threading

MAX_TURNS = 6          # per thread, oldest dropped
MAX_THREADS = 200      # total, least-recently-used dropped

_lock = threading.Lock()
_threads: dict[str, list[dict]] = {}
_counter = itertools.count(1)


def new_thread_id() -> str:
    return f"chat-{next(_counter)}"


def history(thread_id: str | None) -> list[dict]:
    """Prior turns, oldest first. Unknown threads are simply empty."""
    if not thread_id:
        return []
    with _lock:
        return list(_threads.get(thread_id, ()))


def remember(thread_id: str | None, question: str, answer: str) -> None:
    """Record one completed turn.

    Refusals and failures are recorded too. "Why not?" is a follow-up like any
    other, and a history that quietly omits the turn it refers to is worse than
    no history — the planner would resolve the pronoun against the wrong turn.
    """
    if not thread_id:
        return
    with _lock:
        turns = _threads.pop(thread_id, [])       # pop+insert = move to newest
        turns.append({"question": question, "answer": answer})
        _threads[thread_id] = turns[-MAX_TURNS:]
        while len(_threads) > MAX_THREADS:
            _threads.pop(next(iter(_threads)))


def forget(thread_id: str) -> None:
    with _lock:
        _threads.pop(thread_id, None)


def as_prompt(turns: list[dict], max_chars: int = 400) -> str:
    """Render turns for the planner, or "" when there are none.

    Answers are truncated because their only job here is to say what the last
    turn was about. A full answer is a paragraph of figures, and putting
    figures in the planner's prompt is how one gets copied into a plan.
    """
    if not turns:
        return ""
    lines = ["", "Earlier in this conversation (oldest first):"]
    for turn in turns:
        answer = turn["answer"].replace("\n", " ")
        if len(answer) > max_chars:
            answer = answer[:max_chars] + "…"
        lines.append(f"  Q: {turn['question']}")
        lines.append(f"  A: {answer}")
    lines += [
        "",
        "Use it only to resolve what this question refers to — \"that month\",",
        "\"the spike\", \"and in May?\". Write every sub-question so it stands on",
        "its own, naming the period, category or merchant in full. Do NOT reuse",
        "figures from above: the tools re-retrieve everything, and a number",
        "carried over from an earlier turn has no support in this one.",
        "",
    ]
    return "\n".join(lines)
