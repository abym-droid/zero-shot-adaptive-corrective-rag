"""Apple Silicon MLX backend - the local 4-bit path the deliverable runs on.

Wraps ``mlx_lm`` to run 4-bit open-weight 7B models (default
``mlx-community/Qwen2.5-7B-Instruct-4bit``) natively on the MacBook. We use
``stream_generate`` rather than plain ``generate`` because its responses
carry exact prompt/generation token counts, which feed the token-cost
accounting. mlx imports are deferred to ``__init__`` so this module stays
importable on CUDA-only boxes where mlx is absent.
"""
from __future__ import annotations

import time

from adarag.llm.base import GenResult, LLMBackend


class MLXBackend(LLMBackend):
    """LLM backend running 4-bit MLX conversions (``mlx-community/*`` ids);
    raises ImportError off Apple Silicon."""

    name = "mlx"

    def __init__(self, model_id: str):
        try:
            from mlx_lm import load, stream_generate
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:  # pragma: no cover - platform dependent
            raise ImportError(
                "mlx_lm is required for MLXBackend (Apple silicon only). "
                "On CUDA/CPU boxes use adarag.llm.hf_backend.HFBackend instead."
            ) from exc

        self.model_id = model_id
        self._stream_generate = stream_generate
        self._make_sampler = make_sampler
        self.model, self.tokenizer = load(model_id)

    def _render_prompt(self, prompt: str, system: str | None) -> str:
        """Render (system, user) turns through the model's chat template.

        Some templates reject the system role outright (Mistral-7B-Instruct
        raises a TemplateError), so on failure fold the system text into the
        user turn and retry, the conventional workaround for those models.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception:
            if not system:
                raise
            merged = [{"role": "user", "content": f"{system}\n\n{prompt}"}]
            return self.tokenizer.apply_chat_template(
                merged, add_generation_prompt=True, tokenize=False
            )

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> GenResult:
        """Generate with MLX streaming decode; token counts come straight
        from mlx_lm's ``GenerationResponse`` (temp=0.0 is greedy argmax)."""
        rendered = self._render_prompt(prompt, system)
        sampler = self._make_sampler(temp=temperature)

        start = time.perf_counter()
        text_parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        for response in self._stream_generate(
            self.model,
            self.tokenizer,
            rendered,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            text_parts.append(response.text)
            prompt_tokens = response.prompt_tokens
            completion_tokens = response.generation_tokens
        latency = time.perf_counter() - start

        return GenResult(
            text="".join(text_parts),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=latency,
        )
