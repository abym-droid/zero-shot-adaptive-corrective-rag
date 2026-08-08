#!/usr/bin/env python3
"""adarag environment self-check.

Verifies the local thesis-system stack imports and that the Apple MPS backend
actually computes. Pure checks — no network, no model download — unless you pass
--gen, which additionally loads a small MLX model to prove local 7B-class
inference works end-to-end.

Usage:
    python environment/check_env.py            # import + MPS + component smoke tests
    python environment/check_env.py --gen      # also load & run a tiny MLX model (downloads weights)
"""
from __future__ import annotations
import argparse, importlib, platform, sys

GREEN, RED, YEL, RST = "\033[92m", "\033[91m", "\033[93m", "\033[0m"
def ok(m):   print(f"  {GREEN}ok{RST}    {m}")
def bad(m):  print(f"  {RED}FAIL{RST}  {m}")
def warn(m): print(f"  {YEL}warn{RST}  {m}")

results = {"ok": 0, "fail": 0, "warn": 0}

def check(label, fn, optional=False):
    try:
        detail = fn()
        ok(f"{label}{(' — ' + detail) if detail else ''}")
        results["ok"] += 1
    except Exception as e:  # noqa: BLE001
        if optional:
            warn(f"{label} — optional, not available: {type(e).__name__}: {e}")
            results["warn"] += 1
        else:
            bad(f"{label} — {type(e).__name__}: {e}")
            results["fail"] += 1

def ver(mod):
    m = importlib.import_module(mod)
    return f"{mod} {getattr(m, '__version__', '?')}"

# ---------------------------------------------------------------- platform ---
def sec(t): print(f"\n{t}\n" + "-" * 68)

sec("platform")
print(f"  python   {sys.version.split()[0]}")
print(f"  machine  {platform.machine()}  ({platform.platform()})")

# ------------------------------------------------------------------- torch ---
sec("torch + Apple MPS")
def _torch_mps():
    import torch
    assert torch.backends.mps.is_available(), "MPS backend not available"
    x = torch.randn(512, 512, device="mps")
    y = (x @ x).sum().item()      # force a real compute on the GPU
    assert y == y, "NaN from MPS matmul"
    return f"torch {torch.__version__}, MPS matmul ok"
check("torch MPS compute", _torch_mps)

# --------------------------------------------------------- models / HF stack ---
sec("models / HuggingFace stack")
for m in ["transformers", "accelerate", "tokenizers", "safetensors",
          "huggingface_hub", "sentence_transformers", "datasets"]:
    check(m, lambda m=m: ver(m))

# ------------------------------------------------------- local MLX inference ---
sec("local inference (MLX, Apple silicon)")
check("mlx", lambda: ver("mlx"))
check("mlx_lm import", lambda: (__import__("mlx_lm"), "load/generate available")[1])

# ---------------------------------------------------------------- retrieval ---
sec("retrieval")
def _faiss():
    import faiss, numpy as np
    idx = faiss.IndexFlatIP(16)
    idx.add(np.random.rand(10, 16).astype("float32"))
    D, I = idx.search(np.random.rand(1, 16).astype("float32"), 3)
    assert I.shape == (1, 3)
    return f"faiss {faiss.__version__}, flat index search ok"
check("faiss-cpu", _faiss)
def _bm25():
    from rank_bm25 import BM25Okapi
    bm = BM25Okapi([["a", "b"], ["b", "c"]])
    _ = bm.get_scores(["b"])
    return "rank_bm25 ok"
check("rank-bm25", _bm25)

# -------------------------------------------------------------- orchestration ---
sec("orchestration (LangGraph state machine)")
def _langgraph():
    from langgraph.graph import StateGraph, END
    from typing import TypedDict
    class S(TypedDict):
        n: int
    g = StateGraph(S)
    g.add_node("inc", lambda s: {"n": s["n"] + 1})
    g.set_entry_point("inc")
    g.add_edge("inc", END)
    app = g.compile()
    out = app.invoke({"n": 0})
    assert out["n"] == 1
    import langgraph
    return f"langgraph {getattr(langgraph,'__version__','?')}, compiled graph ran"
check("langgraph compile+invoke", _langgraph)
for m in ["langchain", "langchain_core", "langchain_community"]:
    check(m, lambda m=m: ver(m), optional=True)

# ---------------------------------------------------------- structured output ---
sec("structured router output")
def _pydantic():
    from pydantic import BaseModel
    from enum import Enum
    class Tier(str, Enum):
        no_retrieval = "no_retrieval"; single = "single_step"; iterative = "iterative"
    class Route(BaseModel):
        tier: Tier; confidence: float
    r = Route.model_validate_json('{"tier":"single_step","confidence":0.9}')
    assert r.tier == Tier.single
    return "pydantic v2 JSON route validation ok"
check("pydantic route schema", _pydantic)

# ----------------------------------------------------------------- evaluation ---
sec("evaluation")
# RAGAS/ARES live in the isolated `adarag-eval` env (legacy-langchain pin clash);
# not expected in core. Verify them with: conda activate adarag-eval && python -c "import ragas".
check("ragas (expected only in adarag-eval env)", lambda: ver("ragas"), optional=True)
check("evaluate", lambda: ver("evaluate"), optional=True)
check("outlines (constrained decode)", lambda: ver("outlines"), optional=True)
check("ragatouille (ColBERTv2)", lambda: ver("ragatouille"), optional=True)
check("ares-ai", lambda: ver("ares"), optional=True)

# -------------------------------------------------------------- optional --gen ---
args = argparse.ArgumentParser(add_help=False)
args.add_argument("--gen", action="store_true")
ns, _ = args.parse_known_args()
if ns.gen:
    sec("live local generation (MLX 4-bit 7B-class)")
    def _gen():
        from mlx_lm import load, generate
        model, tok = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
        out = generate(model, tok, prompt="Reply with one word: ok", max_tokens=8)
        return f"generated: {out!r}"
    check("mlx-lm load+generate (downloads ~4GB)", _gen)

# ---------------------------------------------------------------------- summary ---
sec("summary")
print(f"  ok={results['ok']}  warn(optional)={results['warn']}  fail={results['fail']}")
sys.exit(1 if results["fail"] else 0)
