"""Route a dev500 slice with a chosen router model and prompt variant, and
record one tier per question. No retrieval, no generation.

The routed system's answer at a given tier is deterministic and identical
to the fixed-tier arm's answer for that question, so the tiers written here
can be scored end to end by lookup into the fixed A/B/C arms (see
scripts/lookup_router_eval.py). Questions are rendered exactly as the
evaluation runner renders them (MCQ options appended).

Usage:
    python scripts/router_dev500.py mlx-community/Qwen2.5-14B-Instruct-4bit v1 teleqna hotpotqa
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def render(ex: dict) -> str:
    q = ex["question"]
    opts = (ex.get("meta") or {}).get("options")
    if opts and isinstance(opts, dict):
        block = "\n".join(f"{k}. {v}" for k, v in opts.items())
        return f"{q}\n\nOptions:\n{block}\n\nAnswer with the letter of the single best option."
    return q


def main(model: str, variant: str, datasets: list[str]) -> int:
    from adarag.llm import MLXBackend
    from adarag.router import Router

    router = Router(MLXBackend(model), prompt_variant=variant)
    tag = model.split("/")[-1].replace("-Instruct-4bit", "").replace("Qwen2.5-", "qwen")
    for ds in datasets:
        src = ROOT / "data" / "processed" / f"{ds}.dev500.jsonl"
        rows = [json.loads(l) for l in src.open()]
        out = {"model": model, "prompt_variant": variant, "dataset": ds, "tiers": {}, "reasons": {}}
        t0 = time.perf_counter()
        for i, ex in enumerate(rows, 1):
            d, _ = router.route(render(ex))
            out["tiers"][ex["qid"]] = d.tier.value
            out["reasons"][ex["qid"]] = d.reason
            if i % 50 == 0:
                print(f"  [{ds}] {i}/{len(rows)} ({(time.perf_counter()-t0)/i:.2f}s/q)", flush=True)
        dst = ROOT / "runs" / "router_dev500" / f"{ds}_{tag}_{variant}.json"
        dst.parent.mkdir(exist_ok=True)
        dst.write_text(json.dumps(out, indent=1))
        print("wrote", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3:]))
