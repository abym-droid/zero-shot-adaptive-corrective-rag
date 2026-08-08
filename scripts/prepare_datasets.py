"""Normalise the manually downloaded raw datasets into the unified jsonl format.

Raw files are downloaded by hand (sources and exact filenames are listed in my
notes) into data/raw/<name>/. This script only does the processing: it reads
each raw file, converts the rows to the shared QAExample schema via
adarag.data.loaders, and writes three files per dataset under data/processed/:

    <name>.jsonl          full normalised set
    <name>.dev500.jsonl   seeded 500-row dev slice (the local eval workhorse)
    <name>.toy.jsonl      seeded 20-row slice for quick smoke tests

Retrieval corpora (<name>.corpus.jsonl, for the index builder): SciFact and
TSpec-LLM ship their own; squad/hotpotqa/musique/2wikimultihopqa get a
gold-context corpus built from their own passages (triviaqa and popqa ship no
contexts - they stay closed-book); `medical-corpus` builds the bounded
medical corpus (PubMedQA contexts + MedQA textbooks if downloaded).
A summary of what was written goes to data/manifest.json.

Usage:
    python scripts/prepare_datasets.py all
    python scripts/prepare_datasets.py squad teleqna
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from adarag.config import settings
from adarag.data import loaders
from adarag.data.schema import write_jsonl

RAW = settings.data_dir / "raw"
PROCESSED = settings.data_dir / "processed"
MANIFEST = settings.data_dir / "manifest.json"

SEED = 42
DEV_N = 500
TOY_N = 20


def read_parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sample(examples: list, n: int) -> list:
    # deterministic order-preserving sample so slices are reproducible
    if len(examples) <= n:
        return list(examples)
    rng = random.Random(SEED)
    idx = sorted(rng.sample(range(len(examples)), n))
    return [examples[i] for i in idx]


def update_manifest(name: str, entry: dict) -> None:
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest[name] = entry
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def finalize(name: str, examples: list, source: str, license_note: str) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    n_full = write_jsonl(examples, PROCESSED / f"{name}.jsonl")
    n_dev = write_jsonl(sample(examples, DEV_N), PROCESSED / f"{name}.dev500.jsonl")
    n_toy = write_jsonl(sample(examples, TOY_N), PROCESSED / f"{name}.toy.jsonl")
    update_manifest(name, {
        "rows": n_full, "dev_rows": n_dev, "toy_rows": n_toy,
        "source": source, "license": license_note,
    })
    print(f"[{name}] {n_full} rows (dev500: {n_dev}, toy: {n_toy})")


def dedupe_docs(docs: list) -> list:
    # gold-context corpora repeat passages across questions; keep first seen
    seen, out = set(), []
    for d in docs:
        key = (d["title"], d["text"])
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def finalize_corpus(name: str, docs: list, source: str, license_note: str) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / f"{name}.corpus.jsonl"
    n = loaders.write_corpus_jsonl(docs, path)
    update_manifest(f"{name}__corpus", {
        "rows": n, "source": source, "license": license_note,
    })
    print(f"[{name}] {n} corpus docs -> {path.name}")


# --------------------------------------------------------------------------- #
# one function per dataset; raw filename conventions match my download notes
# --------------------------------------------------------------------------- #
def teleqna() -> None:
    data = read_json(RAW / "teleqna" / "TeleQnA.json")
    finalize("teleqna", loaders.normalize_teleqna(data),
             "TeleQnA (github netop-team/TeleQnA, extracted from TeleQnA.zip)",
             "Research use; archive is password-protected (see repo README).")


def tspec_llm() -> None:
    # corpus only - the 3GPP markdown tree from HF rasoul-nikbakht/TSpec-LLM
    # (gated; fetched with `hf download` after accepting the terms)
    root = RAW / "tspec_llm"
    docs = []
    for md in sorted(root.rglob("*.md")):
        try:
            text = md.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        docs.append({"doc_id": md.relative_to(root).as_posix(),
                     "title": md.stem, "text": text})
    finalize_corpus("tspec_llm", docs,
                    "rasoul-nikbakht/TSpec-LLM (HF, gated), Rel-16 + Rel-17 subset",
                    "Research use (3GPP-derived); see dataset card.")


def medqa() -> None:
    rows = read_parquet(RAW / "medqa" / "test.parquet")
    finalize("medqa", loaders.normalize_medqa(rows),
             "GBaker/MedQA-USMLE-4-options (test split)", "MIT (dataset card).")


def pubmedqa() -> None:
    rows = read_parquet(RAW / "pubmedqa" / "train.parquet")
    finalize("pubmedqa", loaders.normalize_pubmedqa(rows),
             "qiaojin/PubMedQA (pqa_labeled, the 1k expert-labelled set)",
             "MIT (dataset card).")


def mirage() -> None:
    data = read_json(RAW / "mirage" / "benchmark.json")
    finalize("mirage", loaders.normalize_mirage(data),
             "github Teddy-XiongGZ/MIRAGE benchmark.json",
             "See MIRAGE repo (per-subset upstream licenses).")


def scifact() -> None:
    claims = read_parquet(RAW / "scifact" / "claims.parquet")
    finalize("scifact", loaders.normalize_scifact_claims(claims),
             "allenai/scifact (claims, validation split, parquet export)",
             "CC BY-NC 2.0 (dataset card).")
    corpus = read_parquet(RAW / "scifact" / "corpus.parquet")
    finalize_corpus("scifact", loaders.normalize_scifact_corpus(corpus),
                    "allenai/scifact (corpus, parquet export)",
                    "CC BY-NC 2.0 (dataset card).")


def _wiki_paragraphs(rows: list) -> list:
    # hotpotqa / 2wiki context format: {"title": [...], "sentences": [[...]]}
    docs = []
    for r in rows:
        ctx = r.get("context") or {}
        for title, sents in zip(ctx.get("title") or [], ctx.get("sentences") or []):
            text = "".join(sents).strip()
            if text:
                docs.append({"doc_id": "", "title": str(title), "text": text})
    docs = dedupe_docs(docs)
    for n, d in enumerate(docs):
        d["doc_id"] = f"p{n}"
    return docs


def hotpotqa() -> None:
    rows = read_parquet(RAW / "hotpotqa" / "validation.parquet")
    finalize("hotpotqa", loaders.normalize_hotpotqa(rows),
             "hotpotqa/hotpot_qa (distractor, validation)", "CC BY-SA 4.0.")
    # gold-context corpus: all distractor-setting paragraphs, deduped
    finalize_corpus("hotpotqa", _wiki_paragraphs(rows),
                    "hotpotqa/hotpot_qa (validation paragraphs)", "CC BY-SA 4.0.")


def musique() -> None:
    rows = read_parquet(RAW / "musique" / "validation.parquet")
    finalize("musique", loaders.normalize_musique(rows),
             "dgslibisey/MuSiQue (validation)", "CC BY 4.0.")
    docs = dedupe_docs([
        {"doc_id": "", "title": str(p.get("title", "")), "text": str(p.get("paragraph_text", "")).strip()}
        for r in rows for p in (r.get("paragraphs") or []) if str(p.get("paragraph_text", "")).strip()
    ])
    for n, d in enumerate(docs):
        d["doc_id"] = f"p{n}"
    finalize_corpus("musique", docs,
                    "dgslibisey/MuSiQue (validation paragraphs)", "CC BY 4.0.")


def twowiki() -> None:
    rows = read_parquet(RAW / "2wikimultihopqa" / "validation.parquet")
    finalize("2wikimultihopqa", loaders.normalize_2wikimultihopqa(rows),
             "framolfese/2WikiMultihopQA (validation)", "Apache-2.0.")
    finalize_corpus("2wikimultihopqa", _wiki_paragraphs(rows),
                    "framolfese/2WikiMultihopQA (validation paragraphs)", "Apache-2.0.")


def squad() -> None:
    rows = read_parquet(RAW / "squad" / "validation.parquet")
    finalize("squad", loaders.normalize_squad(rows),
             "rajpurkar/squad (v1.1, validation)", "CC BY-SA 4.0.")
    docs = dedupe_docs([
        {"doc_id": "", "title": str(r.get("title", "")), "text": str(r.get("context", "")).strip()}
        for r in rows if str(r.get("context", "")).strip()
    ])
    for n, d in enumerate(docs):
        d["doc_id"] = f"p{n}"
    finalize_corpus("squad", docs,
                    "rajpurkar/squad (validation paragraphs)", "CC BY-SA 4.0.")


def triviaqa() -> None:
    rows = read_parquet(RAW / "triviaqa" / "validation.parquet")
    finalize("triviaqa", loaders.normalize_triviaqa(rows),
             "mandarjoshi/trivia_qa (rc.nocontext, validation)", "Apache-2.0.")


def popqa() -> None:
    rows = read_parquet(RAW / "popqa" / "test.parquet")
    finalize("popqa", loaders.normalize_popqa(rows),
             "akariasai/PopQA (test)", "MIT (dataset card).")


def medical_corpus() -> None:
    # Bounded medical corpus - a documented scoped deviation from MIRAGE's
    # PubMed-scale setup: PubMedQA abstract contexts plus the MedQA English
    # textbooks when they have been downloaded (separate archive, see notes).
    docs = []
    for r in read_parquet(RAW / "pubmedqa" / "train.parquet"):
        ctxs = (r.get("context") or {}).get("contexts") or []
        text = "\n".join(c.strip() for c in ctxs if c.strip())
        if text:
            docs.append({"doc_id": f"pubmed-{r['pubid']}",
                         "title": f"PMID {r['pubid']}", "text": text})
    books_dir = RAW / "medqa" / "textbooks" / "en"
    if books_dir.is_dir():
        for book in sorted(books_dir.glob("*.txt")):
            text = book.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                docs.append({"doc_id": f"medqa-book-{book.stem}",
                             "title": book.stem, "text": text})
    else:
        print("  note: no MedQA textbooks at data/raw/medqa/textbooks/en -"
              " medical corpus is PubMedQA-only until they are added")
    finalize_corpus("medical", docs,
                    "PubMedQA pqa_labeled contexts + MedQA en textbooks (bounded corpus)",
                    "MIT (PubMedQA) / MedQA release terms.")


DATASETS = {
    "teleqna": teleqna,
    "tspec-llm": tspec_llm,
    "medqa": medqa,
    "pubmedqa": pubmedqa,
    "mirage": mirage,
    "scifact": scifact,
    "hotpotqa": hotpotqa,
    "musique": musique,
    "2wikimultihopqa": twowiki,
    "squad": squad,
    "triviaqa": triviaqa,
    "popqa": popqa,
    "medical-corpus": medical_corpus,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="+",
                        help="dataset names, or 'all' (%s)" % ", ".join(DATASETS))
    args = parser.parse_args()

    names = list(DATASETS) if args.names == ["all"] else args.names
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        parser.error(f"unknown dataset(s): {unknown}")

    failed = []
    for name in names:
        try:
            DATASETS[name]()
        except FileNotFoundError as exc:
            print(f"[{name}] SKIPPED - raw file missing: {exc.filename}")
            print(f"          (download it first; see my dataset notes)")
            failed.append(name)
    if failed:
        print(f"\nnot processed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
