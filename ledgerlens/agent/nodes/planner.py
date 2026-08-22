"""Planner node — §6.3.

In:  question + schema DDL + available tools + data window.
Out: strict JSON list of sub-questions, each tagged with a tool and a one-line
     rationale. No prose, no answer.

Routing guidance, per §6.3:
  names a real category or merchant        → sql
  names the rows by concept ("takeout", "rideshare") → semantic, then sql
  "is this unusual", "did anything change" → anomaly

The distinction is not subject matter but whether a literal string match can
find the rows, which is why the prompt carries the ledger's actual category
and merchant vocabulary — see `_vocabulary`.

Two things this node has to get right, both learned from measurement:

**Minimal decomposition.** An early smoke test had the 8B split "how much did I
spend on groceries in March?" into four sub-questions fanned across all three
tools. Every extra sub-question is another LLM call, another set of rows for the
verifier to reconcile, and another chance to be wrong. The schema caps the plan
and the prompt demands the fewest steps that answer the question.

**Follow-ups are resolved here.** The planner is the only node that sees the
conversation, because it is the only one whose job is deciding what the question
means. Every node after it works from sub-questions that already stand alone.

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
- semantic finds rows by meaning when no category or merchant name matches the
           wording; it returns ids only, so sql always follows it
- anomaly  spikes and dips measured against a per-category trailing baseline

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

Choosing a tool is about how the rows get FOUND, not what the question is
about. Every question ends with sql doing the arithmetic.

The category field is a closed vocabulary. These are the only values that
exist:
{categories}
Merchant names are literal bank strings. Currently in the ledger:
{merchants}

- sql alone when the wording names one of those, or needs no name at all
  ("last month", "my biggest purchase", "how many transactions").
- semantic first, then sql, when the wording describes rows by concept
  instead: "rideshare home", "streaming services", "takeout". No LIKE pattern
  matches those. raw_descriptor holds UBER *TRIP 8QX2K, not the word
  "rideshare". A concept LIKE returns zero rows and reports $0.00 spent, which
  reads like an answer and is not one. Undecided between the two? Take
  semantic: an extra retrieval costs one call, a missed one costs the answer.
- anomaly for "unusual", "abnormal", "a spike", "normal for me", "did anything
  change", "went up in price" — and for a follow-up naming something only the
  detector can identify ("that abnormal month", "ignoring the spike"). sql can
  total any month but cannot say which month is abnormal, nor what the
  baseline looks like with the spike excluded.

{history}
Question: {question}
{feedback}"""

FEEDBACK = """
A previous attempt failed verification:
  {reason}
Plan differently: retrieve what is needed to support every figure."""


MAX_MERCHANTS = 40


def _bounds() -> tuple[str, str]:
    from ...db import connect_readonly

    with connect_readonly() as conn:
        row = conn.execute(
            "SELECT MIN(posted_date), MAX(posted_date) FROM transactions"
        ).fetchone()
    return (row[0] or "unknown", row[1] or "unknown")


def _vocabulary() -> tuple[str, str]:
    """The category and merchant names the ledger actually holds.

    Routing turned on a fact the planner could not see. Told only that a
    `category` column exists, it read "what do I pay for my gym membership?"
    as a category question and wrote LIKE '%gym%' — which matches nothing,
    because the descriptor says PLANET FITNESS and the category says
    subscriptions. Nine of ten semantic questions failed this way. Showing the
    closed vocabulary turns the choice into a lookup: the word is in the list
    or it is not.

    Merchants are capped because the list grows with the ledger while the
    prompt budget does not; the busiest are the ones worth naming.

    The prompt's worked examples are deliberately not golden-set questions.
    Illustrating the rule with "gym membership" would be writing the answer
    key into the prompt and then scoring it.
    """
    from ...db import connect_readonly

    with connect_readonly() as conn:
        categories = [r[0] for r in conn.execute(
            "SELECT name FROM categories ORDER BY name"
        )]
        merchants = [r[0] for r in conn.execute(
            """SELECT m.canonical_name FROM merchants m
               LEFT JOIN transactions t ON t.merchant_id = m.id
               GROUP BY m.id ORDER BY COUNT(t.id) DESC, m.canonical_name
               LIMIT ?""",
            (MAX_MERCHANTS,),
        )]

    listed = ", ".join(sorted(merchants)) or "(none resolved yet)"
    if len(merchants) == MAX_MERCHANTS:
        listed += ", and others not listed here"
    return ", ".join(categories) or "(none)", listed


def make_plan(question: str, feedback: str = "",
              history: list[dict] | None = None) -> Plan:
    from ...llm import invoke_structured
    from ..memory import as_prompt

    date_min, date_max = _bounds()
    categories, merchants = _vocabulary()
    plan = invoke_structured(
        Plan,
        PROMPT.format(
            question=question,
            date_min=date_min,
            date_max=date_max,
            categories=categories,
            merchants=merchants,
            max_steps=MAX_STEPS,
            history=as_prompt(history or []),
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

    plan = make_plan(state["question"], feedback, state.get("history"))

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
