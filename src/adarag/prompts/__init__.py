"""Prompt registry for the adarag system.

Router prompts are re-exported here; gate prompts live in
``adarag.prompts.gate_prompts`` and are imported defensively so this package
still imports in a partial checkout - prefer importing them directly.
"""
from adarag.prompts.router_prompts import (  # noqa: F401
    PROMPT_VARIANTS,
    ROUTER_PROMPTS,
    ROUTER_SYSTEM,
    render_router_prompt,
)

__all__ = [
    "PROMPT_VARIANTS",
    "ROUTER_PROMPTS",
    "ROUTER_SYSTEM",
    "render_router_prompt",
]

try:  # tolerate gate_prompts being absent so a partial tree still imports
    from adarag.prompts.gate_prompts import (  # noqa: F401
        GATE_ANSWER_PROMPT,
        GATE_DOC_PROMPT,
    )

    __all__ += ["GATE_ANSWER_PROMPT", "GATE_DOC_PROMPT"]
except ImportError:  # pragma: no cover
    pass
