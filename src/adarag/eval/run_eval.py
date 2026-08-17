"""Evaluation runner: drive AdaptiveRAG over a dataset and write run artefacts.

Each run writes two files under runs/<timestamp>_<dataset>/:
predictions.jsonl (one row per example with the answer, scores and the full
routing/gate/escalation trace) and summary.json (aggregates plus routing
accuracy against the silver labels and the run config). The --no-escalation
flag is the gate ablation - it sets max_escalations=0 so the corrective gate
can never promote a tier. Cross-module imports (pipeline, llm, retrieval,
data) are deliberately lazy so this module imports cleanly without the heavy
extras installed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import typer

from adarag.config import ESCALATION_NEXT, RetrievalTier, settings
from adarag.eval.metrics import aggregate, em, f1, mcq_em, mcq_f1, routing_accuracy
from adarag.eval.silver_labels import silver_labels_for

__all__ = ["app", "run"]

app = typer.Typer(
    add_completion=False,
    help="Evaluate the adaptive RAG pipeline on a processed QA dataset.",
)

# Reverse escalation ladder (tier value -> previous tier value), used to
# recover the router's initial decision from the final tier + escalation count.
_TIER_PREV: dict[str, str] = {
    nxt.value: cur.value for cur, nxt in ESCALATION_NEXT.items() if nxt is not None
}

# Default generator for the HF (transformers) backend.
_HF_DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class _NullRetriever:
    """Retriever stub returning no documents (tier-A-only / FakeBackend runs)."""

    name = "null"

    def search(self, query: str, k: int = 5) -> list:
        return []


def _load_examples(dataset: Path) -> list[Any]:
    """Load QAExamples via adarag.data.loaders, with a minimal jsonl fallback.

    The fallback keeps the evaluator usable for smoke tests even if the data
    package is unavailable; rows must already be in the unified QAExample
    jsonl schema (qid/question/answers/domain/dataset/meta).
    """
    try:
        from adarag.data.loaders import load_dataset_file

        return list(load_dataset_file(dataset))
    except ImportError:
        rows: list[Any] = []
        with dataset.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows


def _field(ex: Any, name: str, default: Any = None) -> Any:
    """Read attribute or dict key from a QAExample-like object."""
    if isinstance(ex, dict):
        return ex.get(name, default)
    return getattr(ex, name, default)


def _make_backend(backend: str, model: Optional[str], device: str) -> Any:
    """Instantiate the requested LLM backend (lazy imports).

    ``fake`` builds a deterministic FakeBackend whose responses are keyed off
    the prompt content (router prompts get a tier JSON, gate prompts a
    verdict JSON, everything else a short answer) - enough to smoke-test the
    whole graph offline.
    """
    backend = backend.lower()
    if backend == "fake":
        from adarag.llm.fake import FakeBackend

        def _respond(prompt: str, *args: Any, **kwargs: Any) -> str:
            text = prompt.lower()
            # Gate check must come first: rendered gate prompts also mention
            # tiers/JSON, so a tier-based test would misroute them.
            if "verdict" in text:
                return '{"verdict": "correct", "reason": "fake gate", "useful_doc_ids": []}'
            if "tier" in text and ("route" in text or "json" in text):
                return '{"tier": "single_step", "reason": "fake route"}'
            return "fake answer"

        return FakeBackend(responses=_respond)

    try:
        from adarag.llm import load_backend  # type: ignore[attr-defined]
    except ImportError:
        from adarag.llm.base import load_backend

    if backend == "mlx":
        return load_backend(model or settings.generator_model, device=device)
    if backend == "hf":
        return load_backend(model or _HF_DEFAULT_MODEL, device=device)
    raise typer.BadParameter(f"unknown backend {backend!r} (expected mlx|hf|fake)")


def _load_retriever(index: Optional[Path], kind: str) -> Any:
    """Load a persisted retriever from indices/<name>/ (lazy imports)."""
    if index is None:
        return _NullRetriever()
    kind = kind.lower()
    if kind == "bm25":
        from adarag.retrieval.bm25 import BM25Retriever

        return BM25Retriever.load(str(index / "bm25"))
    if kind == "dense":
        from adarag.retrieval.dense import DenseRetriever

        return DenseRetriever.load(str(index / "dense"))
    raise typer.BadParameter(f"unknown retriever {kind!r} (expected bm25|dense)")


def _initial_tier(final_tier: str, escalations: int) -> str:
    """Recover the router's pre-escalation tier by walking the ladder back.

    The escalation ladder is linear (A -> B -> C), so the initial decision is
    just the final tier stepped back ``escalations`` times.
    """
    tier = final_tier
    for _ in range(int(escalations)):
        tier = _TIER_PREV.get(tier, tier)
    return tier


def _sum_trace(trace: list[dict[str, Any]], key: str) -> int:
    """Sum an integer token field across all trace entries."""
    return int(sum(float(t.get(key, 0) or 0) for t in trace))


def _dataset_tag(dataset: Path) -> str:
    """Short dataset tag for the run directory (strips .jsonl / .dev500)."""
    return dataset.name.split(".", 1)[0]


@app.command()
def run(
    dataset: Path = typer.Option(
        ..., "--dataset", exists=True, dir_okay=False, readable=True,
        help="Processed dataset jsonl (QAExample rows), e.g. data/processed/squad.dev500.jsonl",
    ),
    index: Optional[Path] = typer.Option(
        None, "--index", exists=True, file_okay=False,
        help="Index directory (indices/<name>) built by `adarag index`. "
        "Omit only for retrieval-free smoke runs.",
    ),
    backend: str = typer.Option(
        "fake", "--backend", help="LLM backend: mlx | hf | fake."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="Override the backend's default model id."
    ),
    retriever: str = typer.Option(
        "bm25", "--retriever", help="Retriever to load from the index: bm25 | dense."
    ),
    prompt_variant: str = typer.Option(
        "v1", "--prompt-variant",
        help="Router prompt variant (v1|v2) - the prompt-sensitivity ablation.",
    ),
    escalation: bool = typer.Option(
        True, "--escalation/--no-escalation",
        help="Corrective escalation gate on/off (the gate ablation; off sets "
        "max_escalations=0).",
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", min=1, help="Evaluate only the first N examples."
    ),
    device: str = typer.Option(
        "auto", "--device", help="Device hint for the backend (auto|mps|cuda|cpu)."
    ),
    out_dir: Optional[Path] = typer.Option(
        None, "--out-dir",
        help="Run directory override (default runs/<timestamp>_<dataset>/).",
    ),
) -> Path:
    """Run AdaptiveRAG over a dataset; write predictions.jsonl + summary.json.

    Returns the run directory containing the two artefacts.
    """
    examples = _load_examples(dataset)
    if limit is not None:
        examples = examples[:limit]
    if not examples:
        raise typer.BadParameter(f"no examples found in {dataset}")

    backend_obj = _make_backend(backend, model, device)
    retriever_obj = _load_retriever(index, retriever)

    eff_settings = settings.model_copy(
        update={"max_escalations": settings.max_escalations if escalation else 0}
    )

    from adarag.pipeline.graph import AdaptiveRAG

    rag = AdaptiveRAG(
        backend_obj,
        retriever_obj,
        settings=eff_settings,
        prompt_variant=prompt_variant,
    )

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir or (eff_settings.runs_dir / f"{ts}_{_dataset_tag(dataset)}")
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_path = run_dir / "predictions.jsonl"

    rows: list[dict[str, Any]] = []
    with pred_path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            qid = str(_field(ex, "qid"))
            question = str(_field(ex, "question"))
            golds = list(_field(ex, "answers") or [])
            meta = _field(ex, "meta") or {}
            options = meta.get("options") if isinstance(meta, dict) else None
            is_mcq = bool(options)

            # MCQ questions must reach the model WITH their options - without
            # this the model answers free-form and option-letter EM scores it
            # blind (sub-random accuracy even from a working system).
            pipeline_question = question
            if is_mcq and isinstance(options, dict):
                opts_block = "\n".join(f"{k}. {v}" for k, v in options.items())
                pipeline_question = (
                    f"{question}\n\nOptions:\n{opts_block}\n\n"
                    "Answer with the letter of the single best option."
                )

            t0 = time.perf_counter()
            state = rag.answer(pipeline_question)
            latency = time.perf_counter() - t0

            answer = str(state.get("answer", "") or "")
            tier_final = str(state.get("tier", RetrievalTier.SINGLE_STEP.value))
            escalations = int(state.get("escalations", 0) or 0)
            trace = list(state.get("trace", []) or [])
            # retrieved chunk texts, needed downstream by the faithfulness
            # scorer (scripts/score_faithfulness.py, adarag-eval env)
            contexts = [str(d.get("text", "")) for d in (state.get("docs") or [])
                        if isinstance(d, dict)]

            score_em = mcq_em(answer, golds, options) if is_mcq else em(answer, golds)
            score_f1 = mcq_f1(answer, golds, options) if is_mcq else f1(answer, golds)
            row: dict[str, Any] = {
                "qid": qid,
                "question": question,
                "gold": golds,
                "answer": answer,
                "tier_initial": _initial_tier(tier_final, escalations),
                "tier_final": tier_final,
                "escalated": escalations > 0,
                "verdict": state.get("verdict"),
                "em": score_em,
                "f1": score_f1,
                "latency_s": round(latency, 4),
                "prompt_tokens": _sum_trace(trace, "prompt_tokens"),
                "completion_tokens": _sum_trace(trace, "completion_tokens"),
                "contexts": contexts,
                "trace": trace,
                # bookkeeping (aggregation + silver labelling)
                "dataset": _field(ex, "dataset", _dataset_tag(dataset)),
                "domain": _field(ex, "domain"),
                "is_mcq": is_mcq,
                # e.g. TeleQnA question category - lets the summary slice
                # standards-derived vs out-of-corpus questions from one run
                "category": meta.get("category") if isinstance(meta, dict) else None,
            }
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    summary = aggregate(rows)
    # per-category breakdown when the dataset carries one (TeleQnA: only ~30%
    # of questions are standards-derived, so full-set and per-category numbers
    # both matter - one run, sliced post-hoc)
    categories = sorted({r["category"] for r in rows if r.get("category")})
    if categories:
        summary["by_category"] = {
            c: aggregate([r for r in rows if r.get("category") == c])
            for c in categories
        }
    silver = silver_labels_for(examples)
    summary["routing_accuracy_vs_silver"] = routing_accuracy(
        {r["qid"]: r["tier_initial"] for r in rows},
        {q: t.value for q, t in silver.items()},
    )
    summary["config"] = {
        "dataset": str(dataset),
        "index": str(index) if index else None,
        "backend": backend,
        "model": model,
        "retriever": retriever if index else "null",
        "prompt_variant": prompt_variant,
        "escalation": escalation,
        "max_escalations": eff_settings.max_escalations,
        "top_k": eff_settings.top_k,
        "limit": limit,
        "device": device,
        "timestamp": ts,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    typer.echo(
        f"[adarag eval] {len(rows)} examples -> {run_dir}\n"
        f"  EM={summary.get('em', 0.0):.3f}  F1={summary.get('f1', 0.0):.3f}  "
        f"escalation_rate={summary.get('escalation_rate', 0.0):.3f}  "
        f"routing_acc={summary['routing_accuracy_vs_silver']:.3f}"
    )
    return run_dir


if __name__ == "__main__":
    app()
