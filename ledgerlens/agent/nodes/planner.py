"""Planner node — §6.3.

In:  question + schema DDL + available tools + data window.
Out: strict JSON list of sub-questions, each tagged with a tool and a one-line
     rationale. No prose, no answer.

Routing guidance, per §6.3:
  countable / summable / comparable        → sql
  free-text recall ("that trip to Boston") → semantic
  "is this unusual", "did anything change" → anomaly

Two things this node has to get right, both learned from measurement:

**Minimal decomposition.** An early smoke test had the 8B split "how much did I
spend on groceries in March?" into four sub-questions fanned across all three
tools. Every extra sub-question is another LLM call, another set of rows for the
verifier to reconcile, and another chance to be wrong. The schema caps the plan
and the prompt demands the fewest steps that answer the question.

**Refusal is a plan.** Seven of the golden questions are unanswerable by
design — forecasts, credit scores, advice, periods outside the data — and the
correct behaviour is to decline, not to route them somewhere and let a tool
invent a number. That decision belongs here, before any tool runs, because it is
the only node that sees the question before the machinery starts.
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, Field

from ..state import AgentState

MAX_REPLANS = 2
MAX_STEPS = 4


class Tool(str, enum.Enum):
    SQL = "sql"
    SEMANTIC = "semantic"
    ANOMALY = "anomaly"


class Step(BaseModel):
    sub_question: str = Field(description="One self-contained question")
    tool: Tool = Field(description="Which tool answers it")
    rationale: str = Field(description="One line: why this tool")


class Plan(BaseModel):
    answerable: bool = Field(
        description="False if the ledger cannot answer this even in principle"
    )
    refusal_reason: str = Field(
        default="", description="If not answerable, one line saying why"
    )
    steps: list[Step] = Field(
        default_factory=list, description="Empty when not answerable"
    )


PROMPT = """You plan how to answer questions about a personal finance ledger.

The ledger holds bank transactions from {date_min} to {date_max}: dates,
amounts, merchants, categories, and detected recurring charges. Nothing else.

Tools:
- sql      counting, summing, averaging, ranking, comparing, filtering by date
- semantic free-text recall where the wording does not name a category or
           merchant directly ("that trip to Boston", "anything medical-looking")
- anomaly  "is this unusual", "did anything change", spikes against a baseline

Set answerable=false, with no steps, when the ledger cannot answer it:
- the future ("what will I spend next month") — there is no forecasting
- facts not in bank data (credit score, salary negotiations, account balance)
- advice ("should I invest", "am I saving enough") — this system reports, it
  does not advise
- periods outside {date_min}..{date_max} — say so rather than answering 0

Otherwise produce the FEWEST steps that answer the question, at most {max_steps}.
- A single total, count or ranking is ONE step. Do not add extra steps to
  double-check it, and do not route the same sub-question to several tools.
- Split only when the answer genuinely needs separate retrievals, e.g.
  "did I spend more on food this semester than last" is two sums plus a
  comparison the caller performs.
- Never collapse a comparison into one combined figure. "Compare June and July"
  asks for both sides; a sub-question reading "total for June and July" has
  quietly discarded the question. Either keep the two sides as separate steps,
  or write one step that returns a row per side.
- Prefer sql. Use semantic only when no category or merchant name fits the
  wording, and anomaly only for genuine "is this unusual" questions.

Question: {question}
{feedback}"""

FEEDBACK = """
A previous attempt failed verification:
  {reason}
Plan differently: retrieve what is needed to support every figure."""


def _bounds() -> tuple[str, str]:
    from ...db import connect_readonly

    with connect_readonly() as conn:
        row = conn.execute(
            "SELECT MIN(posted_date), MAX(posted_date) FROM transactions"
        ).fetchone()
    return (row[0] or "unknown", row[1] or "unknown")


def make_plan(question: str, feedback: str = "") -> Plan:
    from ...llm import invoke_structured

    date_min, date_max = _bounds()
    plan = invoke_structured(
        Plan,
        PROMPT.format(
            question=question,
            date_min=date_min,
            date_max=date_max,
            max_steps=MAX_STEPS,
            feedback=feedback,
        ),
    )

    # Trim rather than trust: the schema cannot enforce a maximum length, and an
    # over-long plan costs a tool call per extra step.
    if len(plan.steps) > MAX_STEPS:
        plan.steps = plan.steps[:MAX_STEPS]
    if not plan.answerable:
        plan.steps = []

    return plan


def planner(state: AgentState) -> AgentState:
    verdict = state.get("verifier_verdict") or {}
    feedback = ""
    if verdict and not verdict.get("pass"):
        feedback = FEEDBACK.format(reason=verdict.get("reason", "unspecified"))

    plan = make_plan(state["question"], feedback)

    # Enforced here as well as in make_plan: a refusal that still carries steps
    # would run tools for a question already judged unanswerable, which is how a
    # refusal turns into an invented number.
    steps = plan.steps if plan.answerable else []

    return {
        **state,
        "plan": [step.model_dump(mode="json") for step in steps],
        "replan_count": state.get("replan_count", 0) + (1 if feedback else 0),
        "answerable": plan.answerable,
        "refusal_reason": plan.refusal_reason,
    }
