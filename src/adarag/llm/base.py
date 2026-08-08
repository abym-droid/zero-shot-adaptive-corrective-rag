"""Backend-agnostic LLM interface for the pipeline.

The router, the corrective gate and the answer generators all talk to one
abstract LLMBackend, so the same pipeline runs on MLX locally, HF
transformers on Colab/Kaggle, or a scripted fake in tests. The load-bearing
part is generate_json: 7B instruct models like to wrap JSON in code fences
or add chatty preambles, so it runs a parse ladder (fence stripping ->
first balanced object -> pydantic validation) with a bounded error-feedback
retry before raising LLMJsonError. Every call also reports token counts and
latency via GenResult, since token cost and latency are metrics we track.
"""
from __future__ import annotations

import abc
import json
import re
from dataclasses import dataclass

import pydantic


class LLMJsonError(Exception):
    """Raised when ``generate_json`` cannot obtain schema-valid JSON.

    Carries the last raw model output and the last parse/validation error so
    callers can log the failure into the trace before applying their fallback
    (router -> SINGLE_STEP, gate -> "ambiguous").
    """

    def __init__(self, message: str, *, raw_text: str = "", last_error: str = ""):
        super().__init__(message)
        self.raw_text = raw_text
        self.last_error = last_error


@dataclass
class GenResult:
    """Token/latency bookkeeping for a single LLM generation call.

    For ``generate_json`` the counts aggregate across all attempts (retries
    are real cost and must show up in the accounting); ``text`` is the final
    attempt's output.
    """

    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


# --- JSON robustness helpers -------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    """Return the content of the first fenced code block, or ``text`` as-is.

    Instruct models often wrap JSON in ```json ... ``` even when told not to.
    """
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    # Handle an unterminated opening fence (model hit max_tokens mid-block).
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
    return stripped.strip()


def _extract_first_json_object(text: str) -> str | None:
    """Extract the first balanced ``{...}`` block, or None if none exists
    or the braces never balance (truncated output).

    The brace counter is string/escape aware, so braces inside JSON string
    values do not break matching.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


_JSON_SYSTEM_SUFFIX = (
    "You must respond with ONLY a single valid JSON object matching the "
    "requested schema. No markdown, no code fences, no explanations before "
    "or after the JSON."
)


class LLMBackend(abc.ABC):
    """Abstract text-generation backend (MLX, HF transformers, or fake).

    Subclasses set ``name`` (used in traces and run metadata) and implement
    ``generate``; ``generate_json`` is shared by all backends.
    """

    name: str = "abstract"

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> GenResult:
        """Generate a completion for ``prompt``.

        ``system`` is rendered via the model's chat template where the
        backend supports one. temperature=0.0 means greedy decoding - the
        default everywhere here, for reproducibility.
        """

    def generate_json(
        self,
        prompt: str,
        schema: type[pydantic.BaseModel],
        *,
        system: str | None = None,
        max_tokens: int = 512,
        max_retries: int = 2,
    ) -> tuple[pydantic.BaseModel, GenResult]:
        """Generate output validated against a pydantic ``schema``.

        This is the structured-output contract the router and gate rely on:
        the model is prompted for JSON-only output (schema included, JSON
        boilerplate added here - don't put it in ``prompt``); the reply is
        cleaned (fence stripping, first balanced ``{...}``) and validated.
        On failure the parse error is fed back into the prompt and the call
        retried up to ``max_retries`` more times.

        Returns ``(instance, gen_result)`` where the GenResult aggregates
        token counts/latency over all attempts; raises LLMJsonError if no
        attempt produced schema-valid JSON.
        """
        schema_json = json.dumps(schema.model_json_schema(), indent=None)
        base_prompt = (
            f"{prompt}\n\n"
            f"Return ONLY a JSON object conforming to this JSON schema:\n"
            f"{schema_json}"
        )
        full_system = (
            f"{system.rstrip()}\n\n{_JSON_SYSTEM_SUFFIX}" if system else _JSON_SYSTEM_SUFFIX
        )

        attempt_prompt = base_prompt
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_latency = 0.0
        last_text = ""
        last_error = ""

        for _attempt in range(1 + max_retries):
            result = self.generate(
                attempt_prompt,
                system=full_system,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            total_prompt_tokens += result.prompt_tokens
            total_completion_tokens += result.completion_tokens
            total_latency += result.latency_s
            last_text = result.text

            candidate = _strip_code_fences(result.text)
            extracted = _extract_first_json_object(candidate)
            if extracted is None:
                last_error = "no JSON object found in output"
            else:
                try:
                    instance = schema.model_validate_json(extracted)
                    agg = GenResult(
                        text=result.text,
                        prompt_tokens=total_prompt_tokens,
                        completion_tokens=total_completion_tokens,
                        latency_s=total_latency,
                    )
                    return instance, agg
                except pydantic.ValidationError as exc:
                    last_error = str(exc)

            # Retry: feed the error back so the model can self-correct.
            attempt_prompt = (
                f"{base_prompt}\n\n"
                f"Your previous reply was not valid:\n{last_error}\n"
                f"Previous reply (for reference): {last_text[:500]}\n"
                f"Respond again with ONLY the corrected JSON object."
            )

        raise LLMJsonError(
            f"generate_json failed after {1 + max_retries} attempt(s): {last_error}",
            raw_text=last_text,
            last_error=last_error,
        )


def load_backend(model_id: str, device: str = "auto") -> LLMBackend:
    """Factory selecting the right backend for ``model_id`` and platform.

    ``mlx-community/*`` ids need MLX; otherwise, if the resolved device is
    ``mps`` and ``mlx_lm`` is importable, MLX is still preferred (local Mac
    default); anything else goes to HFBackend (Colab/Kaggle CUDA or CPU).
    Returns a ready-to-use backend with weights already loaded.
    """
    import importlib.util

    mlx_available = importlib.util.find_spec("mlx_lm") is not None

    if model_id.startswith("mlx-community/"):
        if not mlx_available:
            raise ImportError(
                f"Model '{model_id}' is an MLX conversion but mlx_lm is not "
                "installed. Install mlx-lm (Apple silicon only) or use the "
                "non-MLX hub id (e.g. Qwen/Qwen2.5-7B-Instruct) for the HF backend."
            )
        from adarag.llm.mlx_backend import MLXBackend

        return MLXBackend(model_id)

    from adarag.device import get_device

    resolved = get_device(device)
    if resolved == "mps" and mlx_available:
        from adarag.llm.mlx_backend import MLXBackend

        return MLXBackend(model_id)

    from adarag.llm.hf_backend import HFBackend

    return HFBackend(model_id, device=resolved)
