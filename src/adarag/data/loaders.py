"""Per-dataset normalisers plus the unified reader for the processed jsonl files.

Each ``normalize_*`` function turns one dataset's native records into
QAExample rows; ``load_dataset_file`` is the single reader everything else
uses. Conventions worth remembering: MCQ sets store both the gold option
letter and the option text in ``answers`` (with the full option map in
``meta["options"]``) so scoring can go either way; free-text sets store the
gold string plus aliases for EM/F1; SciFact and TSpec-LLM additionally emit a
corpus jsonl for the retriever index builder. The pinned HF dataset ids sit
next to each normaliser and in ``scripts/download_datasets.py``.
"""
from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Any, Iterable

from adarag.data.schema import QAExample

# Option index (1-based, TeleQnA style) / 0-based -> letter helpers.
_LETTERS = string.ascii_uppercase


def _letter(idx0: int) -> str:
    """Map a 0-based option index to an uppercase letter (0->A, 1->B, ...)."""
    return _LETTERS[idx0]


# --------------------------------------------------------------------------- #
# Unified reader (used by the evaluation harness)
# --------------------------------------------------------------------------- #
def load_dataset_file(path: str | Path) -> list[QAExample]:
    """Load a processed jsonl file into a list of QAExamples, in file order."""
    from adarag.data.schema import read_jsonl

    return list(read_jsonl(path))


# --------------------------------------------------------------------------- #
# Corpus writer (SciFact + TSpec-LLM retrieval corpora)
# --------------------------------------------------------------------------- #
def write_corpus_jsonl(
    docs: Iterable[dict[str, Any]], path: str | Path
) -> int:
    """Write retrieval-corpus docs ({"doc_id","title","text"}) to jsonl.

    This is the input contract for the chunker/index builder
    (adarag.corpus.chunk.chunk_corpus). Returns the number of docs written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for d in docs:
            row = {
                "doc_id": str(d["doc_id"]),
                "title": d.get("title", "") or "",
                "text": d.get("text", "") or "",
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


# --------------------------------------------------------------------------- #
# Telecom
# --------------------------------------------------------------------------- #
def normalize_teleqna(data: dict[str, Any]) -> list[QAExample]:
    """Normalise the TeleQnA question bank (10k MCQ, domain telecom).

    Source: HF ``netop/TeleQnA`` (gated) - practical fallback is the GitHub
    release ``netop-team/TeleQnA`` (``TeleQnA.zip``, AES-encrypted with
    password ``teleqnadataset`` per the repo README, an anti-contamination
    measure). Native records are keyed ``"question <n>"`` with 1-based
    ``"option N"`` fields and an ``answer`` of the form ``"option <k>: <text>"``.
    """
    out: list[QAExample] = []
    for key, rec in data.items():
        opts: dict[str, str] = {}
        # TeleQnA uses up to 5 "option N" fields, 1-based.
        for n in range(1, 6):
            val = rec.get(f"option {n}")
            if val:
                opts[_letter(n - 1)] = str(val)
        answer_raw = str(rec.get("answer", ""))
        # "answer" is "option <k>: <text>"; recover the 1-based option number.
        gold_letter = ""
        gold_text = ""
        if ":" in answer_raw:
            head, gold_text = answer_raw.split(":", 1)
            gold_text = gold_text.strip()
            digits = "".join(ch for ch in head if ch.isdigit())
            if digits:
                gold_letter = _letter(int(digits) - 1)
        answers = [a for a in (gold_letter, gold_text) if a]
        qid = key.replace(" ", "-")  # "question 0" -> "question-0"
        out.append(
            QAExample(
                qid=f"teleqna-{qid}",
                question=str(rec.get("question", "")),
                answers=answers or [answer_raw],
                domain="telecom",
                dataset="teleqna",
                meta={
                    "mcq": True,
                    "options": opts,
                    "category": rec.get("category", ""),
                    "explanation": rec.get("explanation", ""),
                },
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Medical
# --------------------------------------------------------------------------- #
def normalize_medqa(rows: Iterable[dict[str, Any]]) -> list[QAExample]:
    """Normalise MedQA-USMLE (4-option) test rows, domain medical.

    Source: HF ``GBaker/MedQA-USMLE-4-options`` (config ``default``, split
    ``test``); fallback ``bigbio/med_qa``. Native rows carry ``options``
    (letter->text), ``answer_idx`` (gold letter) and ``answer`` (gold text).
    """
    out: list[QAExample] = []
    for i, rec in enumerate(rows):
        opts = dict(rec.get("options", {}) or {})
        letter = str(rec.get("answer_idx", "")).strip()
        text = str(rec.get("answer", "")).strip()
        answers = [a for a in (letter, text) if a]
        out.append(
            QAExample(
                qid=f"medqa-{i}",
                question=str(rec.get("question", "")),
                answers=answers or [text],
                domain="medical",
                dataset="medqa",
                meta={
                    "mcq": True,
                    "options": opts,
                    "meta_info": rec.get("meta_info", ""),
                },
            )
        )
    return out


def normalize_pubmedqa(rows: Iterable[dict[str, Any]]) -> list[QAExample]:
    """Normalise PubMedQA (expert-labelled ``pqa_labeled``) rows, domain medical.

    Source: HF ``qiaojin/PubMedQA``, config ``pqa_labeled`` (1k expert), split
    ``train`` (the only split shipped). The task is a 3-way yes/no/maybe
    decision; casting it as MCQ with a fixed option map
    ``{"A":"yes","B":"no","C":"maybe"}`` lets it score under the same
    option-letter path as the other MCQ sets. The abstract goes into
    ``meta["context"]`` as the gold passage.
    """
    decision_to_letter = {"yes": "A", "no": "B", "maybe": "C"}
    opts = {"A": "yes", "B": "no", "C": "maybe"}
    out: list[QAExample] = []
    for rec in rows:
        decision = str(rec.get("final_decision", "")).strip().lower()
        letter = decision_to_letter.get(decision, "")
        ctx = rec.get("context", {})
        # context is {"contexts": [...], "labels": [...], ...}; join passages.
        if isinstance(ctx, dict):
            context_text = " ".join(ctx.get("contexts", []) or [])
        else:
            context_text = str(ctx)
        answers = [a for a in (letter, decision) if a]
        out.append(
            QAExample(
                qid=f"pubmedqa-{rec.get('pubid', len(out))}",
                question=str(rec.get("question", "")),
                answers=answers or [decision],
                domain="medical",
                dataset="pubmedqa",
                meta={
                    "mcq": True,
                    "options": opts,
                    "context": context_text,
                    "long_answer": rec.get("long_answer", ""),
                },
            )
        )
    return out


def normalize_mirage(data: dict[str, Any]) -> list[QAExample]:
    """Normalise the MIRAGE benchmark question sets, domain medical.

    Source: GitHub ``Teddy-XiongGZ/MIRAGE`` raw ``benchmark.json``, a dict of
    subset -> {qid: record} (subsets: medqa, medmcqa, pubmedqa, bioasq, mmlu).
    Everything comes out as MCQ; the originating subset is kept in
    ``meta["mirage_subset"]``.
    """
    out: list[QAExample] = []
    for subset, items in data.items():
        for qid, rec in items.items():
            opts = dict(rec.get("options", {}) or {})
            letter = str(rec.get("answer", "")).strip()
            text = opts.get(letter, "")
            answers = [a for a in (letter, text) if a]
            out.append(
                QAExample(
                    qid=f"mirage-{subset}-{qid}",
                    question=str(rec.get("question", "")),
                    answers=answers or [letter],
                    domain="medical",
                    dataset="mirage",
                    meta={
                        "mcq": True,
                        "options": opts,
                        "mirage_subset": subset,
                        "pmid": rec.get("PMID", []),
                    },
                )
            )
    return out


# --------------------------------------------------------------------------- #
# Scientific
# --------------------------------------------------------------------------- #
def normalize_scifact_claims(
    rows: Iterable[dict[str, Any]]
) -> list[QAExample]:
    """Normalise SciFact claims into SUPPORT/CONTRADICT/NOINFO form, domain scientific.

    Source: HF ``allenai/scifact`` via its parquet-conversion branch
    (``refs/convert/parquet/claims/<split>``) - the original loader is a
    dataset *script*, which ``datasets`` 5.x no longer runs. The companion
    corpus is handled by normalize_scifact_corpus. A blank
    ``evidence_label`` means no verifying evidence in the corpus (NOINFO);
    cited/evidence doc ids are kept in ``meta`` for Recall@K.
    """
    out: list[QAExample] = []
    for rec in rows:
        label = str(rec.get("evidence_label", "")).strip().upper() or "NOINFO"
        gold_doc_ids = [str(x) for x in (rec.get("cited_doc_ids", []) or [])]
        evi = rec.get("evidence_doc_id", "")
        if evi:
            gold_doc_ids = list(dict.fromkeys([str(evi), *gold_doc_ids]))
        out.append(
            QAExample(
                qid=f"scifact-{rec.get('id', len(out))}",
                question=str(rec.get("claim", "")),
                answers=[label],
                domain="scientific",
                dataset="scifact",
                meta={
                    "mcq": True,
                    "options": {
                        "A": "SUPPORT",
                        "B": "CONTRADICT",
                        "C": "NOINFO",
                    },
                    "task": "claim_verification",
                    "gold_doc_ids": gold_doc_ids,
                    "evidence_sentences": rec.get("evidence_sentences", []),
                },
            )
        )
    return out


def normalize_scifact_corpus(
    rows: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Normalise SciFact corpus rows to the retriever corpus schema.

    Source: HF ``allenai/scifact`` parquet branch, ``corpus/train``. Abstracts
    arrive as sentence lists and get joined into one text field for
    write_corpus_jsonl.
    """
    docs: list[dict[str, Any]] = []
    for rec in rows:
        abstract = rec.get("abstract", [])
        if isinstance(abstract, list):
            text = " ".join(abstract)
        else:
            text = str(abstract)
        docs.append(
            {
                "doc_id": str(rec.get("doc_id", "")),
                "title": str(rec.get("title", "")),
                "text": text,
            }
        )
    return docs


# --------------------------------------------------------------------------- #
# General-domain calibration sets
# --------------------------------------------------------------------------- #
def normalize_hotpotqa(rows: Iterable[dict[str, Any]]) -> list[QAExample]:
    """Normalise HotpotQA (distractor, validation) multi-hop rows.

    Source: HF ``hotpotqa/hotpot_qa``, config ``distractor``, split
    ``validation``. Supporting-fact titles are kept in ``meta["gold_titles"]``.
    """
    out: list[QAExample] = []
    for rec in rows:
        sf = rec.get("supporting_facts", {}) or {}
        titles = list(dict.fromkeys(sf.get("title", []) or []))
        out.append(
            QAExample(
                qid=f"hotpotqa-{rec.get('id', len(out))}",
                question=str(rec.get("question", "")),
                answers=[str(rec.get("answer", ""))],
                domain="general_multihop",
                dataset="hotpotqa",
                meta={
                    "type": rec.get("type", ""),
                    "level": rec.get("level", ""),
                    "gold_titles": titles,
                },
            )
        )
    return out


def normalize_musique(rows: Iterable[dict[str, Any]]) -> list[QAExample]:
    """Normalise MuSiQue (answerable, validation) multi-hop rows.

    Source: HF ``dgslibisey/MuSiQue``, config ``default``, split ``validation``;
    fallback GitHub ``stonybrooknlp/musique``. Answer + aliases go into
    ``answers``.
    """
    out: list[QAExample] = []
    for rec in rows:
        ans = str(rec.get("answer", ""))
        aliases = [str(a) for a in (rec.get("answer_aliases", []) or [])]
        answers = list(dict.fromkeys([ans, *aliases])) if ans else aliases
        out.append(
            QAExample(
                qid=f"musique-{rec.get('id', len(out))}",
                question=str(rec.get("question", "")),
                answers=answers or [ans],
                domain="general_multihop",
                dataset="musique",
                meta={"answerable": rec.get("answerable", True)},
            )
        )
    return out


def normalize_2wikimultihopqa(
    rows: Iterable[dict[str, Any]]
) -> list[QAExample]:
    """Normalise 2WikiMultiHopQA (validation) multi-hop rows.

    Source: HF ``framolfese/2WikiMultihopQA``, config ``default``, split
    ``validation`` (the original ``xanhho/2WikiMultihopQA`` is a dataset
    script, unusable under ``datasets`` 5.x).
    """
    out: list[QAExample] = []
    for rec in rows:
        out.append(
            QAExample(
                qid=f"2wikimultihopqa-{rec.get('id', len(out))}",
                question=str(rec.get("question", "")),
                answers=[str(rec.get("answer", ""))],
                domain="general_multihop",
                dataset="2wikimultihopqa",
                meta={"type": rec.get("type", "")},
            )
        )
    return out


def normalize_squad(rows: Iterable[dict[str, Any]]) -> list[QAExample]:
    """Normalise SQuAD v1.1 (validation) single-hop rows.

    Source: HF ``rajpurkar/squad``, config ``plain_text``, split
    ``validation``. The gold context is kept in ``meta["context"]``.
    """
    out: list[QAExample] = []
    for rec in rows:
        ans_field = rec.get("answers", {}) or {}
        texts = list(dict.fromkeys(ans_field.get("text", []) or []))
        out.append(
            QAExample(
                qid=f"squad-{rec.get('id', len(out))}",
                question=str(rec.get("question", "")),
                answers=texts or [""],
                domain="general_singlehop",
                dataset="squad",
                meta={
                    "title": rec.get("title", ""),
                    "context": rec.get("context", ""),
                },
            )
        )
    return out


def normalize_triviaqa(rows: Iterable[dict[str, Any]]) -> list[QAExample]:
    """Normalise TriviaQA (rc.nocontext, validation) single-hop rows.

    Source: HF ``mandarjoshi/trivia_qa``, config ``rc.nocontext``, split
    ``validation``. The native ``answer`` field is a dict with
    ``value``/``aliases``/``normalized_aliases``.
    """
    out: list[QAExample] = []
    for rec in rows:
        a = rec.get("answer", {}) or {}
        value = str(a.get("value", ""))
        aliases = [str(x) for x in (a.get("aliases", []) or [])]
        answers = list(dict.fromkeys([value, *aliases])) if value else aliases
        out.append(
            QAExample(
                qid=f"triviaqa-{rec.get('question_id', len(out))}",
                question=str(rec.get("question", "")),
                answers=answers or [value],
                domain="general_singlehop",
                dataset="triviaqa",
                meta={},
            )
        )
    return out


def normalize_popqa(rows: Iterable[dict[str, Any]]) -> list[QAExample]:
    """Normalise PopQA (test) single-hop rows.

    Source: HF ``akariasai/PopQA``, config ``default``, split ``test``.
    ``possible_answers`` is a JSON-encoded string of acceptable surface forms.
    """
    out: list[QAExample] = []
    for rec in rows:
        raw = rec.get("possible_answers", "[]")
        try:
            answers = [str(x) for x in json.loads(raw)] if isinstance(raw, str) else [
                str(x) for x in raw
            ]
        except (json.JSONDecodeError, TypeError):
            answers = []
        obj = str(rec.get("obj", ""))
        if obj and obj not in answers:
            answers.insert(0, obj)
        out.append(
            QAExample(
                qid=f"popqa-{rec.get('id', len(out))}",
                question=str(rec.get("question", "")),
                answers=answers or [obj],
                domain="general_singlehop",
                dataset="popqa",
                meta={
                    "prop": rec.get("prop", ""),
                    "s_pop": rec.get("s_pop", None),
                },
            )
        )
    return out
