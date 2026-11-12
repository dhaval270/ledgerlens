"""Structured-output retry — the crash that reached users as a 500.

Asked for a `Plan`, the model sometimes returns the JSON Schema it was given
instead of an instance of it. Valid JSON, wrong shape, fails validation on a
missing required field. It is not a rate limit and not a malformed prompt, so
nothing upstream retried it and the traceback — schema and all — was rendered
into the answer box.

Measured on gpt-oss-120b: 2 of 18 planner calls in the routing eval, 1 of 5
live questions. No network here; the model is a stub that fails on cue.
"""

from __future__ import annotations

import pytest
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel

import ledgerlens.llm as llm_module
from ledgerlens.llm import StructuredOutputError, invoke_structured


class Plan(BaseModel):
    answerable: bool


class FakeLLM:
    """Fails `failures` times, then returns `result`. Records prompts and temps."""

    def __init__(self, failures: int, result=None, exc=None):
        self.failures = failures
        self.result = result or Plan(answerable=True)
        self.exc = exc or OutputParserException("Failed to parse Plan from completion")
        self.prompts: list[str] = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if len(self.prompts) <= self.failures:
            raise self.exc
        return self.result


@pytest.fixture
def patched(monkeypatch):
    """Swap the factory so no key and no network are needed."""
    made: dict = {}

    def install(fake: FakeLLM):
        def factory(schema, model=None, temperature=0.0):
            made.setdefault("temperatures", []).append(temperature)
            return fake
        monkeypatch.setattr(llm_module, "get_structured_llm", factory)
        return made

    return install


def test_a_clean_call_does_not_retry(patched):
    fake = FakeLLM(failures=0)
    patched(fake)
    assert invoke_structured(Plan, "question").answerable is True
    assert len(fake.prompts) == 1


def test_a_schema_echo_is_retried(patched):
    fake = FakeLLM(failures=1)
    patched(fake)
    assert invoke_structured(Plan, "question").answerable is True
    assert len(fake.prompts) == 2


def test_the_retry_tells_the_model_what_it_did_wrong(patched):
    """A bare retry of a temperature-0 failure is the same failure again."""
    fake = FakeLLM(failures=1)
    patched(fake)
    invoke_structured(Plan, "question")
    assert "Do not return the schema" in fake.prompts[1]
    assert "Do not return the schema" not in fake.prompts[0]


def test_temperature_lifts_off_zero_after_the_first_attempt(patched):
    fake = FakeLLM(failures=1)
    made = patched(fake)
    invoke_structured(Plan, "question")
    assert made["temperatures"][0] == 0.0
    assert made["temperatures"][1] > 0.0


def test_groqs_400_for_a_declined_tool_call_counts_as_the_same_failure(patched):
    """Groq reports the same miss two different ways depending on the model."""
    fake = FakeLLM(failures=1, exc=Exception("Error code: 400 - tool_use_failed"))
    patched(fake)
    assert invoke_structured(Plan, "question").answerable is True


def test_exhausted_attempts_raise_a_named_error(patched):
    """So the API can say a sentence instead of printing the schema."""
    fake = FakeLLM(failures=99)
    patched(fake)
    with pytest.raises(StructuredOutputError) as caught:
        invoke_structured(Plan, "question", attempts=3)
    assert "Plan" in str(caught.value)
    assert len(fake.prompts) == 3


def test_unrelated_failures_are_not_retried(patched):
    """A 429 has its own budget upstream; burning attempts here hides it."""
    fake = FakeLLM(failures=99, exc=Exception("Error code: 429 - rate limit reached"))
    patched(fake)
    with pytest.raises(Exception) as caught:
        invoke_structured(Plan, "question")
    assert "429" in str(caught.value)
    assert len(fake.prompts) == 1
