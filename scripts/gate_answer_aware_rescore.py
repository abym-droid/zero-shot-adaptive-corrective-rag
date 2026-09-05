"""Post-hoc ablation: an answer-aware gate over the stored predictions of a
verdict-only (no-escalation) run.

The frozen gate judges only the question and the retrieved documents on
tiers B and C; it never sees the draft answer. This script re-judges every
retrieval-tier row of an existing run with a prompt that also shows the
draft answer, using the same model and JSON contract, and writes one JSON
file with the old and new verdict per question. No pipeline is re-run, so
the answers and EM are unchanged; only the verdict changes. Tier A rows
(answer already judged by the frozen gate) are copied through.

Usage:
    python scripts/gate_answer_aware_rescore.py runs/p4_teleqna_noesc
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from adarag.gate import GateVerdict  # noqa: E402
from adarag.prompts.gate_prompts import GATE_JSON_INSTRUCTIONS  # noqa: E402

PROMPT = (
    """You are a strict evidence-quality evaluator inside a retrieval-augmented
question-answering system. Your job is to judge whether the retrieved
documents below support the draft answer to the question. You are NOT
answering the question yourself.

Question:
{question}

Draft answer:
{answer}

Retrieved documents (each line starts with its id in square brackets):
{docs_block}

Verdict definitions:
- "correct": at least one document contains evidence that directly supports
  the draft answer.
- "ambiguous": the documents are partially relevant but do not clearly
  confirm or contradict the draft answer.
- "incorrect": the documents contradict the draft answer, or are irrelevant
  and the draft answer looks implausible; better retrieval is needed.

In "useful_doc_ids", list ONLY the ids of documents genuinely useful for
answering the question (an empty list if none). Copy ids exactly as shown in
the square brackets. Keep "reason" to one short sentence.

"""
    + GATE_JSON_INSTRUCTIONS
)


def main(run_dir: str, model: str | None = None, limit: int | None = None) -> int:
    from adarag.config import settings
    from adarag.llm import MLXBackend

    run = Path(run_dir)
    rows = [json.loads(l) for l in (run / "predictions.jsonl").open()]
    if limit:
        rows = rows[:limit]
    model = model or settings.generator_model
    backend = MLXBackend(model)
    out = {"run": str(run), "model": model, "prompt": "answer-aware doc gate", "per_qid": {}}
    t0 = time.perf_counter()
    for i, r in enumerate(rows, 1):
        rec = {"tier_final": r["tier_final"], "old_verdict": r["verdict"], "em": r["em"]}
        if r["tier_final"] == "no_retrieval" or not r["contexts"]:
            rec["new_verdict"] = r["verdict"]
            rec["copied"] = True
        else:
            docs = "\n".join(
                f"[d{j}] {' '.join(c.split())[:1500]}" for j, c in enumerate(r["contexts"], 1)
            )
            prompt = PROMPT.format(question=r["question"], answer=r["answer"], docs_block=docs)
            try:
                parsed, gen = backend.generate_json(prompt, GateVerdict, max_tokens=256)
                rec["new_verdict"] = parsed.verdict
                rec["reason"] = parsed.reason
                rec["tokens"] = gen.prompt_tokens + gen.completion_tokens
            except Exception as exc:  # noqa: BLE001
                rec["new_verdict"] = "ambiguous"
                rec["error"] = type(exc).__name__
        out["per_qid"][r["qid"]] = rec
        if i % 50 == 0:
            print(f"  {run.name} {i}/{len(rows)} ({(time.perf_counter()-t0)/i:.2f}s/q)", flush=True)
    dst = run / "gate_answer_aware.json"
    dst.write_text(json.dumps(out, indent=1))
    print("wrote", dst)
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit(main(a[0], a[1] if len(a) > 1 and a[1] != "-" else None, int(a[2]) if len(a) > 2 else None))
