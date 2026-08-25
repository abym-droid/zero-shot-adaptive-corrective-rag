"""RQ1 head-to-head: zero-shot router vs the retrained T5 classifier on the
authors' own test subsampled questions, scored against one shared set of
silver labels.

Both routers see the identical 500 questions per dataset (parsed from the
Adaptive-RAG prediction bundle under runs/predictions/), and both are scored
against the same silver labels, built with this repo's protocol
(src/adarag/eval/silver_labels.py): dataset provenance assigns tier B to
single-hop sets and tier C to multi-hop sets, then the upgrade-to-A rule
relabels a question NO_RETRIEVAL when the closed-book FLAN-T5-XL arm from the
same bundle already answered it correctly (EM = 1 under this repo's
normalization). FLAN-T5-XL is the regime the T5 classifier was trained for,
so the shared silver favours neither side. This removes the two
incomparabilities of cross-slice comparison: different question samples and
different silver protocols.

The T5 side needs no model - its per-question routing choices are read from
the bundle's <dataset>_option.json files (option A/B/C). Only the zero-shot
router runs live, one short structured-output call per question, no retrieval
or generation.

Usage:
    python scripts/rq1_router_headtohead.py --dry-run
    python scripts/rq1_router_headtohead.py                  # full 6 x 500
    python scripts/rq1_router_headtohead.py --datasets squad musique --limit 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from adarag.config import RetrievalTier  # noqa: E402
from adarag.eval.metrics import em  # noqa: E402
from adarag.eval.silver_labels import silver_label  # noqa: E402

_BUNDLE = ROOT / "runs" / "predictions"
_CLS_ROOT = (_BUNDLE / "classifier" / "t5-large" / "flan_t5_xl"
             / "epoch" / "25" / "2026_08_17" / "19_30_22")
_TEST = _BUNDLE / "test"
_CLOSED_BOOK_ARM = "nor_qa_flan_t5_xl_{ds}____prompt_set_1"

#: bundle dataset key -> (this repo's dataset name for provenance labels)
DATASETS = {
    "nq": "nq",  # not in SINGLE_HOP_DATASETS; explicit tier below
    "trivia": "triviaqa",
    "squad": "squad",
    "musique": "musique",
    "hotpotqa": "hotpotqa",
    "2wikimultihopqa": "2wikimultihopqa",
}

_OPTION_TO_TIER = {
    "A": RetrievalTier.NO_RETRIEVAL,
    "B": RetrievalTier.SINGLE_STEP,
    "C": RetrievalTier.ITERATIVE,
}


def parse_chains(path: Path) -> dict[str, str]:
    """qid -> question text from a nor_qa chains.txt (qid line, question
    line, then the Q:/A: block; stanzas separated by blank lines)."""
    out: dict[str, str] = {}
    stanza: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines() + [""]:
        if raw.strip():
            stanza.append(raw)
            continue
        if len(stanza) >= 2:
            out[stanza[0].strip()] = stanza[1].strip()
        stanza = []
    return out


def load_dataset(ds: str) -> tuple[dict[str, str], dict[str, list[str]],
                                   dict[str, RetrievalTier], dict[str, float]]:
    """Return (questions, golds, t5_tiers, closed_book_em) keyed by qid."""
    arm = _TEST / _CLOSED_BOOK_ARM.format(ds=ds)
    stem = f"{ds}_to_{ds}__test_subsampled"
    questions = parse_chains(arm / f"prediction__{stem}_chains.txt")
    golds = {
        qid: [str(a) for a in answers] if isinstance(answers, list) else [str(answers)]
        for qid, answers in json.loads(
            (arm / f"ground_truth__{stem}.json").read_text(encoding="utf-8")
        ).items()
    }
    cb_preds = json.loads(
        (arm / f"prediction__{stem}.json").read_text(encoding="utf-8"))
    closed_book_em = {
        qid: em(str(pred), golds.get(qid, [])) for qid, pred in cb_preds.items()
    }
    options = json.loads(
        (_CLS_ROOT / ds / f"{ds}_option.json").read_text(encoding="utf-8"))
    t5_tiers = {qid: _OPTION_TO_TIER[rec["option"]] for qid, rec in options.items()}
    return questions, golds, t5_tiers, closed_book_em


def build_silver(ds: str, qids: list[str],
                 closed_book_em: dict[str, float]) -> dict[str, RetrievalTier]:
    provenance = (RetrievalTier.SINGLE_STEP if ds == "nq"
                  else silver_label(DATASETS[ds]))
    return {
        qid: (RetrievalTier.NO_RETRIEVAL
              if closed_book_em.get(qid, 0.0) >= 1.0 else provenance)
        for qid in qids
    }


def accuracy(pred: dict[str, RetrievalTier], silver: dict[str, RetrievalTier],
             qids: list[str]) -> float:
    return sum(pred[q] == silver[q] for q in qids) / len(qids) if qids else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS),
                    choices=list(DATASETS))
    ap.add_argument("--model", default=None,
                    help="router model id (default: settings.router_model)")
    ap.add_argument("--prompt-variant", default="v1", choices=["v1", "v2"])
    ap.add_argument("--limit", type=int, default=None,
                    help="first N questions per dataset")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "runs" / "rq1_router_headtohead.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify joins and silver distributions, no model")
    args = ap.parse_args()

    loaded = {}
    for ds in args.datasets:
        questions, golds, t5_tiers, cb_em = load_dataset(ds)
        qids = sorted(set(questions) & set(t5_tiers) & set(golds))
        if args.limit:
            qids = qids[: args.limit]
        silver = build_silver(ds, qids, cb_em)
        n_a = sum(1 for q in qids if silver[q] is RetrievalTier.NO_RETRIEVAL)
        print(f"[{ds}] {len(qids)} joined "
              f"(questions={len(questions)} t5={len(t5_tiers)} gold={len(golds)}) "
              f"silver: A={n_a} rest={silver[qids[0]].value if qids else '-'}"
              f" | t5 acc vs shared silver: "
              f"{accuracy(t5_tiers, silver, qids):.3f}")
        loaded[ds] = (qids, questions, t5_tiers, silver)
    if args.dry_run:
        return 0

    from adarag.config import settings
    from adarag.llm import MLXBackend
    from adarag.router import Router

    model = args.model or settings.router_model
    print(f"loading router backend: {model} (variant {args.prompt_variant})")
    router = Router(MLXBackend(model), prompt_variant=args.prompt_variant)

    report = {"model": model, "prompt_variant": args.prompt_variant,
              "closed_book_arm": "nor_qa_flan_t5_xl",
              "silver_protocol": "provenance + upgrade-to-A (this repo)",
              "per_dataset": {}, "overall": {}}
    all_t5_hits = all_zs_hits = all_n = 0
    for ds, (qids, questions, t5_tiers, silver) in loaded.items():
        zs_tiers: dict[str, RetrievalTier] = {}
        t0 = time.perf_counter()
        for i, qid in enumerate(qids, 1):
            decision, _ = router.route(questions[qid])
            zs_tiers[qid] = decision.tier
            if i % 50 == 0:
                rate = (time.perf_counter() - t0) / i
                print(f"  [{ds}] {i}/{len(qids)} ({rate:.2f}s/q)", flush=True)
        t5_acc = accuracy(t5_tiers, silver, qids)
        zs_acc = accuracy(zs_tiers, silver, qids)
        all_t5_hits += sum(t5_tiers[q] == silver[q] for q in qids)
        all_zs_hits += sum(zs_tiers[q] == silver[q] for q in qids)
        all_n += len(qids)
        report["per_dataset"][ds] = {
            "n": len(qids),
            "t5_routing_acc": t5_acc,
            "zeroshot_routing_acc": zs_acc,
            "silver_dist": dict(Counter(silver[q].value for q in qids)),
            "t5_dist": dict(Counter(t5_tiers[q].value for q in qids)),
            "zeroshot_dist": dict(Counter(zs_tiers[q].value for q in qids)),
            "per_qid": {q: {"silver": silver[q].value,
                            "t5": t5_tiers[q].value,
                            "zeroshot": zs_tiers[q].value} for q in qids},
        }
        print(f"[{ds}] t5={t5_acc:.3f} zeroshot={zs_acc:.3f}  "
              f"({time.perf_counter() - t0:.0f}s)")

    report["overall"] = {"n": all_n,
                         "t5_routing_acc": all_t5_hits / all_n,
                         "zeroshot_routing_acc": all_zs_hits / all_n}
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\noverall: t5={all_t5_hits / all_n:.3f} "
          f"zeroshot={all_zs_hits / all_n:.3f}  -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
