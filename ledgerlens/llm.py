"""Groq chat model factory.

Every node that calls an LLM goes through here so model choice, temperature, and
retry policy stay in one place. Groq's rate limits make retry a real failure mode:
one question can fan out to ~8 calls (planner + 3 SQL repairs + verifier, twice
over on replan), so 429s are expected, not exceptional.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Chosen by measurement, not by §2's sizing guidance — which assumed local
# inference, where a 70B is impractical. On hosted Groq the 70B is strictly
# better on the golden set: 67.4% vs 52.2% execution accuracy, 100% vs 95.7%
# first-attempt SQL validity, and 7.6s vs 13.7s median latency (it needs no
# repair round trips). Two prompt rewrites on the 8B moved accuracy by -4.4 and
# -2.2 points; the model swap moved it +15.2.
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


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
def get_structured_llm(schema: type, model: str = DEFAULT_MODEL):
    """Schema-enforced output. Returns instances of `schema`, never raw text.

    This is the structural version of non-negotiable #1: don't ask the model to
    be reliable at something the type system can guarantee.
    """
    return get_llm(model).with_structured_output(schema)
