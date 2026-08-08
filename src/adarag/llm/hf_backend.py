"""HF transformers backend - the Colab/Kaggle CUDA (or CPU) path.

Runs the same open-weight generators (e.g. ``Qwen/Qwen2.5-7B-Instruct``) as
the local MLX path: bfloat16 + ``device_map="auto"`` on CUDA, optional
bitsandbytes 4-bit quantization when available (mirrors the local 4-bit MLX
setting), chat prompts via ``tokenizer.apply_chat_template``. Heavy imports
(torch, transformers) are deferred into ``__init__`` so merely importing
this module stays cheap on any platform.
"""
from __future__ import annotations

import importlib.util
import time

from adarag.llm.base import GenResult, LLMBackend


class HFBackend(LLMBackend):
    """LLM backend running open-weight models via HF transformers.

    ``load_in_4bit`` quantizes with bitsandbytes NF4 on CUDA (keeps a 7B
    within Colab T4 memory and matches the local 4-bit MLX setting); it is
    silently ignored if bitsandbytes or CUDA is absent.
    """

    name = "hf"

    def __init__(self, model_id: str, device: str = "auto", load_in_4bit: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from adarag.device import get_device

        self.model_id = model_id
        self.device = get_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        quantization_config = None
        if (
            load_in_4bit
            and self.device == "cuda"
            and importlib.util.find_spec("bitsandbytes") is not None
        ):
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=quantization_config,
            )
        else:
            dtype = torch.float16 if self.device == "mps" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
            self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> GenResult:
        """Generate with ``model.generate``; completion tokens are the newly
        generated ids (prompt slice removed before decoding)."""
        import torch

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        rendered = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs: dict = {"max_new_tokens": max_tokens}
        if temperature > 0.0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False)

        start = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                pad_token_id=self.tokenizer.pad_token_id
                or self.tokenizer.eos_token_id,
                **gen_kwargs,
            )
        latency = time.perf_counter() - start

        new_ids = output_ids[0][prompt_len:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return GenResult(
            text=text,
            prompt_tokens=prompt_len,
            completion_tokens=int(new_ids.shape[0]),
            latency_s=latency,
        )
