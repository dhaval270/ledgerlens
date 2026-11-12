"""Groq chat model factory.

Every node that calls an LLM goes through here so model choice, temperature, and
retry policy stay in one place. Groq's rate limits make retry a real failure mode:
one question can fan out to ~8 calls (planner + 3 SQL repairs + verifier, twice
over on replan), so 429s are expected, not exceptional.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_groq import ChatGroq

load_dotenv()

# Hosted models are not a stable substrate. The original choice here was
# llama-3.3-70b-versatile, picked by measurement over llama-3.1-8b-instant:
# 67.4% vs 52.2% execution accuracy, 100% vs 95.7% first-attempt SQL validity,
# 7.6s vs 13.7s median latency. Groq has since retired both, and the endpoint
# now returns 404 for them — so every figure measured on the 70B is a historical
# record, not a reproducible result. Anything quoted from a run predating this
# line was produced on a model no longer obtainable.
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Groq exposes two routes to a typed response and they are not interchangeable.
# LangChain's default is tool-calling, which gpt-oss silently declines: asked for
# a schema it returns prose and Groq raises `tool_use_failed`, a 400 that reads
# like a transport error but is really a capability mismatch. Native JSON-schema
# decoding works on every current model, so it is pinned rather than defaulted.
STRUCTURED_METHOD = "json_schema"


@lru_cache(maxsize=8)
def get_llm(model: str = DEFAULT_MODEL, temperature: float = 0.0) -> ChatGroq:
    """Structured-output work (planning, SQL, categorization) wants temperature 0."""
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set — copy .env.example to .env")

    return ChatGroq(
        model=model,
        temperature=temperature,
        max_retries=5,  # Groq 429s under the free tier
    )


@lru_cache(maxsize=8)
def get_json_llm(model: str = DEFAULT_MODEL) -> ChatGroq:
    """Constrained JSON decoding — valid JSON, but *any* shape.

    Measured: asked for bare JSON, llama-3.1-8b wrapped it in a ``` fence and
    broke json.loads. This mode fixes that. It does not fix schema drift — the
    same model returned {"WHOLEFDS": {"canonical_name": ...}}, nesting the answer
    under the input. Valid JSON, wrong shape. Prefer get_structured_llm below
    wherever a specific shape is required.
    """
    return get_llm(model).bind(response_format={"type": "json_object"})


@lru_cache(maxsize=16)
def get_structured_llm(schema: type, model: str = DEFAULT_MODEL,
                       temperature: float = 0.0):
    """Schema-enforced output. Returns instances of `schema`, never raw text.

    This is the structural version of non-negotiable #1: don't ask the model to
    be reliable at something the type system can guarantee.

    Prefer `invoke_structured` at call sites — the guarantee holds only when the
    model returns *an instance*, and sometimes it returns the schema instead.
    """
    return get_llm(model, temperature).with_structured_output(
        schema, method=STRUCTURED_METHOD
    )


class StructuredOutputError(RuntimeError):
    """The model would not produce an instance of the requested schema."""


# The observed failure is specific and worth naming: asked for a `Plan`, the
# model echoes back the JSON Schema it was given — `{"properties": {...},
# "required": [...], "title": "Plan"}` — which is valid JSON, matches no
# instance, and fails validation on a missing `answerable`. It is not a rate
# limit and not a bad prompt, so nothing upstream retries it, and the traceback
# reaches the user as a 500.
#
# Measured on gpt-oss-120b: 2 of 18 planner calls, and 1 of 5 live questions.
# Roughly one in eight, which for an interactive tool is not a tail case.
_SUFFIX = """

IMPORTANT: return a JSON object whose fields hold real values for this specific
question. Do not return the schema. Do not emit "properties", "required",
"title" or "$defs" — those describe the format, they are not the answer."""


def invoke_structured(schema: type, prompt: str, attempts: int = 3,
                      model: str = DEFAULT_MODEL) -> Any:
    """Invoke with schema enforcement, retrying when the model returns a schema.

    Two things change between attempts. The prompt gains an explicit
    instruction, because the default failure is the model misreading what was
    asked. And temperature lifts off zero, because a deterministic retry of a
    deterministic failure is just the same failure again — the whole reason
    temperature is 0 elsewhere is that it makes retries pointless.
    """
    last: Exception | None = None

    for attempt in range(attempts):
        try:
            return get_structured_llm(
                schema, model, temperature=0.0 if attempt == 0 else 0.3
            ).invoke(prompt if attempt == 0 else prompt + _SUFFIX)
        except OutputParserException as exc:
            last = exc
        except Exception as exc:
            # Groq surfaces the same miss as a 400 when the model declines to
            # call the tool at all. Everything else — 429s, auth, transport —
            # belongs to the caller, which has its own budget for it.
            if "tool_use_failed" not in str(exc):
                raise
            last = exc

    raise StructuredOutputError(
        f"{schema.__name__} not produced after {attempts} attempts: {last}"
    ) from last
