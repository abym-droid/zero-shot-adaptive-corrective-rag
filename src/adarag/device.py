"""Device selection for constrained-compute local deployment.

On the local MacBook the target accelerator is Apple MPS (no CUDA). For local 7B-class
generation the primary path is MLX (Apple-native, 4-bit), which manages its own
Metal device; this helper is for the PyTorch-side components (embedders, rerankers).
"""
from __future__ import annotations


def get_device(prefer: str = "auto") -> str:
    """Return the best available torch device string: 'mps' | 'cuda' | 'cpu'.

    prefer='auto' (default) picks mps on Apple silicon, cuda on a GPU box, else cpu.
    Pass an explicit value to force it.
    """
    if prefer != "auto":
        return prefer
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
