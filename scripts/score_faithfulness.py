"""Faithfulness scoring for finished eval runs (RAGAS, isolated env).

Runs in the separate `adarag-eval` conda env, never in `adarag` (the two
stacks pin conflicting langchain versions). Reads a run directory's
predictions.jsonl, scores it with RAGAS and writes faithfulness.json next to
it. The judge defaults to gpt-4o-mini; the thesis policy allows proprietary
APIs as judge tooling, and the judge sits outside the deliverable pipeline.

    export OPENAI_API_KEY=...
    conda run -n adarag-eval python scripts/score_faithfulness.py runs/<dir> [runs/<dir> ...]

A local judge via an OpenAI-compatible endpoint is selectable with
--judge-base-url/--judge-model, but tried and rejected as the default:
Qwen2.5-7B 4-bit served by mlx does not terminate on RAGAS's structured
prompts (every generation runs to the token cap, so every score is NaN),
regardless of max_tokens. Documented in the lab notebook, 18 Aug.

MCQ handling (the telecom-adapted variant): a bare-letter answer ("B") gives
a faithfulness judge nothing to check, so for MCQ rows the letter is resolved
to its option text from the gold list before judging. Rows without retrieved
contexts (tier A answers, runs made before contexts were recorded) are
skipped and counted.

ARES is not run here; it needs its own trained judge setup and is an
optional secondary metric.
"""
from __future__ import annotations

import argparse
import json
import re
import string
import sys
from pathlib import Path

_LETTERS = set(string.ascii_uppercase[:8])


def _mcq_answer_text(answer: str, golds: list[str]) -> str | None:
    """If the answer is a bare option letter, return the gold option text.

    Only resolves when the letter matches the gold letter (golds store both
    the letter and the option text); a wrong-letter answer stays a letter,
    which the judge will correctly score as unfaithful/irrelevant.
    """
    m = re.match(r"^\s*([A-H])\b\s*[.):]?\s*$", answer.strip(), re.I)
    if not m:
        return None
    letter = m.group(1).upper()
    gold_letter = next((g.strip().upper() for g in golds
                        if len(g.strip()) == 1 and g.strip().upper() in _LETTERS), None)
    texts = [g for g in golds if len(g.strip()) > 1]
    if letter == gold_letter and texts:
        return texts[0]
    return None


def load_rows(run_dir: Path, limit: int | None) -> tuple[list[dict], int]:
    """Return (scoreable rows, n_skipped_no_context)."""
    rows, skipped = [], 0
    with (run_dir / "predictions.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            contexts = [c for c in (r.get("contexts") or []) if c.strip()]
            if not contexts:
                skipped += 1
                continue
            answer = str(r.get("answer", ""))
            golds = [str(g) for g in (r.get("gold") or [])]
            resolved = _mcq_answer_text(answer, golds) if r.get("is_mcq") else None
            rows.append({
                "qid": r.get("qid"),
                "user_input": str(r.get("question", "")),
                "response": resolved or answer,
                "retrieved_contexts": contexts,
                "reference": next((g for g in golds if len(g.strip()) > 1), golds[0] if golds else ""),
            })
            if limit and len(rows) >= limit:
                break
    return rows, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dirs", nargs="+", type=Path,
                    help="run directories containing predictions.jsonl")
    ap.add_argument("--judge-base-url", default=None,
                    help="OpenAI-compatible judge endpoint; omit for api.openai.com")
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--judge-api-key", default=None,
                    help="judge API key; omit to use OPENAI_API_KEY from the env")
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="local embedding model for answer relevancy")
    ap.add_argument("--judge-max-tokens", type=int, default=4096,
                    help="max completion tokens for judge calls (RAGAS prompts "
                         "produce long statement lists; too low -> NaN scores)")
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N scoreable rows per run")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the datasets and report counts, no judge calls")
    args = ap.parse_args()

    # dataset build + counts first, so --dry-run needs no heavy imports
    prepared = []
    for run_dir in args.run_dirs:
        if not (run_dir / "predictions.jsonl").exists():
            print(f"[{run_dir}] no predictions.jsonl, skipping")
            continue
        rows, skipped = load_rows(run_dir, args.limit)
        print(f"[{run_dir.name}] {len(rows)} scoreable rows"
              f" ({skipped} skipped, no retrieved contexts)")
        if rows:
            prepared.append((run_dir, rows, skipped))
    if args.dry_run or not prepared:
        return 0

    from datasets import Dataset
    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.run_config import RunConfig
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import faithfulness

    judge_kwargs = dict(model=args.judge_model, temperature=0.0, timeout=300,
                        max_tokens=args.judge_max_tokens)
    if args.judge_base_url:
        judge_kwargs["base_url"] = args.judge_base_url
    if args.judge_api_key:
        judge_kwargs["api_key"] = args.judge_api_key
    judge = LangchainLLMWrapper(ChatOpenAI(**judge_kwargs))

    metrics = [faithfulness]
    embeddings = None
    try:  # answer relevancy needs local embeddings; optional
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.metrics import answer_relevancy
        embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name=args.embedder))
        metrics.append(answer_relevancy)
    except Exception as exc:  # sentence-transformers not installed etc.
        print(f"answer_relevancy disabled ({type(exc).__name__}: {exc}); "
              f"pip install sentence-transformers in adarag-eval to enable")

    for run_dir, rows, skipped in prepared:
        ds = Dataset.from_list([{k: v for k, v in r.items() if k != "qid"}
                                for r in rows])
        result = evaluate(ds, metrics=metrics, llm=judge, embeddings=embeddings,
                          run_config=RunConfig(timeout=600, max_workers=2))
        scores = result.to_pandas()
        agg = {m.name: float(scores[m.name].mean()) for m in metrics
               if m.name in scores}
        out = {
            "n_scored": len(rows),
            "n_skipped_no_context": skipped,
            "judge_model": args.judge_model,
            "judge_base_url": args.judge_base_url,
            **agg,
            "per_example": [
                {"qid": rows[i]["qid"],
                 **{m.name: (None if scores[m.name].isna()[i]
                             else float(scores[m.name][i]))
                    for m in metrics if m.name in scores}}
                for i in range(len(rows))
            ],
        }
        path = run_dir / "faithfulness.json"
        path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"[{run_dir.name}] " +
              "  ".join(f"{k}={v:.3f}" for k, v in agg.items()) +
              f"  -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
