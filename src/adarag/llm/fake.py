"""Deterministic fake backend for offline tests.

Lets the router, gate, tier runners and orchestrator be exercised with
scripted model outputs and zero network / model downloads - including the
``generate_json`` retry ladder (feed one malformed reply, then a valid one)
and the escalation-bound tests. Purely a test double, never part of the
evaluated system.
"""
from __future__ import annotations

import time
from typing import Callable, Union

from adarag.llm.base import GenResult, LLMBackend

ResponseFn = Callable[[str, Union[str, None]], str]


class FakeBackend(LLMBackend):
    """Scripted, deterministic LLMBackend.

    ``responses`` is either a list of canned replies consumed in order (a
    RuntimeError fires if the script runs out - provide exactly as many
    replies as calls) or a callable ``fn(prompt, system) -> str``. Every
    ``generate`` call is recorded in ``self.calls``.
    """

    name = "fake"

    def __init__(self, responses: list[str] | ResponseFn):
        if callable(responses):
            self._fn: ResponseFn | None = responses
            self._queue: list[str] = []
        else:
            self._fn = None
            self._queue = list(responses)
        self._cursor = 0
        self.calls: list[dict] = []

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> GenResult:
        """Return the next scripted response (or ``fn(prompt, system)``).

        Token counts are naive whitespace word counts - stable and cheap,
        enough to assert that trace accounting is wired up.
        """
        self.calls.append(
            {
                "prompt": prompt,
                "system": system,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if self._fn is not None:
            text = self._fn(prompt, system)
        else:
            if self._cursor >= len(self._queue):
                raise RuntimeError(
                    f"FakeBackend script exhausted after {len(self._queue)} "
                    f"response(s); generate was called again with prompt: "
                    f"{prompt[:120]!r}"
                )
            text = self._queue[self._cursor]
            self._cursor += 1

        start = time.perf_counter()
        return GenResult(
            text=text,
            prompt_tokens=len(prompt.split()) + (len(system.split()) if system else 0),
            completion_tokens=len(text.split()),
            latency_s=time.perf_counter() - start,
        )
