"""Central config for the adarag system

Encodes the locked design decisions from the proposal as typed settings so the
router/gate/orchestrator share one source of truth. Nothing here loads
a model, it just declares the three retrieval tiers and default model ids.
"""
from __future__ import annotations
from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrievalTier(str, Enum):
    """The three complexity tiers the zero-shot LLM router chooses between,
    and the escalation ladder for the corrective gate (A -> B -> C, one step max)."""
    NO_RETRIEVAL = "no_retrieval"     # A: answer from parametric knowledge
    SINGLE_STEP = "single_step"       # B: one retrieval pass
    ITERATIVE = "iterative"           # C: multi-step iterative retrieval (IRCoT-style)


# escalation ladder: gate promotes at most one tier on an 'incorrect' verdict
ESCALATION_NEXT = {
    RetrievalTier.NO_RETRIEVAL: RetrievalTier.SINGLE_STEP,
    RetrievalTier.SINGLE_STEP: RetrievalTier.ITERATIVE,
    RetrievalTier.ITERATIVE: None,    # top tier - no further escalation
}

class Settings(BaseSettings):
    """Runtime settings; override via env vars (prefix ADARAG_) or a .env file."""
    model_config = SettingsConfigDict(env_prefix="ADARAG_", env_file=".env", extra="ignore")

    # --- local generator (open-weight 7B, reduced precision, MLX on Apple Silicon) ---
    router_model: str = Field(
        default="mlx-community/Qwen2.5-7B-Instruct-4bit",
        description="Zero-shot LLM router (structured JSON output).",
    )
    generator_model: str = Field(
        default="mlx-community/Qwen2.5-7B-Instruct-4bit",
        description="Answer generator, swappable to Mistral-7B/Gemma-class.",
    )
    # candidate open-weight alternatives kept for the router prompt/model ablation
    router_model_alt: str = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
    
    # --- retrieval ---
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # dense baseline
    colbert_model: str = "colbert-ir/colbertv2.0"                   # late-interaction option
    top_k: int = 5

    # --- escalation bound (locked: at most one tier promotion per query) ---
    max_escalations: int = 1

    # --- paths (all git-ignored) ---
    data_dir: Path = Path("data")
    index_dir: Path = Path("indices")
    runs_dir: Path = Path("runs")

    # --- device ---
    device: str = "auto"   # resolved by adarag.device.get_device


settings = Settings()
