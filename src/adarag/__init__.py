"""adarag - Adaptive RAG with zero-shot LLM routing and corrective escalation.

Deliverable-system package for the MSc thesis (LJMU), Abhishek Mukherjee.
"""
__version__ = "0.0.1"

from .device import get_device  # noqa: F401
from .config import Settings, RetrievalTier  # noqa: F401
