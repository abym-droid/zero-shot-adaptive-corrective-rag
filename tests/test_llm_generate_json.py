"""Tests for the structured-JSON generation contract (llm/base.py).

Both the router and the gate lean on ``LLMBackend.generate_json``
(prompt → validate → bounded retry). These tests pin the parse / retry /
fallback behaviour using FakeBackend only.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from adarag.llm.base import GenResult, LLMBackend, LLMJsonError
from adarag.llm.fake import FakeBackend


class Item(BaseModel):
    """Minimal schema for generate_json round-trips."""

    name: str
    count: int


GOOD = '{"name": "widget", "count": 3}'


def test_fake_backend_is_llm_backend():
    assert issubclass(FakeBackend, LLMBackend)


def test_generate_returns_genresult():
    fake = FakeBackend(["hello"])
    result = fake.generate("say hi")
    assert isinstance(result, GenResult)
    assert result.text == "hello"
    assert isinstance(result.prompt_tokens, int) and result.prompt_tokens >= 0
    assert isinstance(result.completion_tokens, int) and result.completion_tokens >= 0
    assert result.latency_s >= 0.0


def test_generate_json_clean_json():
    fake = FakeBackend([GOOD])
    item, gen = fake.generate_json("give me an item", Item)
    assert isinstance(item, Item)
    assert (item.name, item.count) == ("widget", 3)
    assert isinstance(gen, GenResult)


def test_generate_json_strips_code_fences():
    fake = FakeBackend([f"```json\n{GOOD}\n```"])
    item, _ = fake.generate_json("item please", Item)
    assert (item.name, item.count) == ("widget", 3)


def test_generate_json_extracts_first_json_block_from_prose():
    fake = FakeBackend([f"Sure, here you go: {GOOD} - hope that helps!"])
    item, _ = fake.generate_json("item please", Item)
    assert (item.name, item.count) == ("widget", 3)


def test_generate_json_retries_after_invalid_then_succeeds():
    fake = FakeBackend(["this is not json at all", GOOD])
    item, _ = fake.generate_json("item please", Item, max_retries=2)
    assert (item.name, item.count) == ("widget", 3)
    # FakeBackend records calls, so the retry must show up as a second call
    assert len(fake.calls) == 2


def test_generate_json_retry_feeds_error_back_into_prompt():
    fake = FakeBackend(["nope", GOOD])
    fake.generate_json("item please", Item, max_retries=2)
    # The second call's prompt must differ from the first (error appended).
    assert fake.calls[0] != fake.calls[1]


def test_generate_json_raises_after_exhausting_retries():
    fake = FakeBackend(["garbage"] * 5)
    with pytest.raises(LLMJsonError):
        fake.generate_json("item please", Item, max_retries=2)


def test_generate_json_schema_mismatch_raises():
    # Valid JSON but wrong shape must also count as a failure.
    fake = FakeBackend(['{"unexpected": true}'] * 5)
    with pytest.raises(LLMJsonError):
        fake.generate_json("item please", Item, max_retries=1)
