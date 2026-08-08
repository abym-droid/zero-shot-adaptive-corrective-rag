"""LLM backend package: one interface, three implementations.

MLXBackend is the local Apple Silicon 4-bit path, HFBackend covers
Colab/Kaggle CUDA (or CPU), FakeBackend is a scripted double for offline
tests, and load_backend picks the right one for a model id and platform.
Importing this package is cheap: the heavy libraries (mlx, torch,
transformers) only load when a real backend is instantiated.
"""
from adarag.llm.base import GenResult, LLMBackend, LLMJsonError, load_backend
from adarag.llm.fake import FakeBackend
from adarag.llm.hf_backend import HFBackend
from adarag.llm.mlx_backend import MLXBackend

__all__ = [
    "FakeBackend",
    "GenResult",
    "HFBackend",
    "LLMBackend",
    "LLMJsonError",
    "MLXBackend",
    "load_backend",
]
