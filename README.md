# Zero Shot Adaptive Corrective RAG System

**MSc thesis system:** *Adaptive Retrieval-Augmented Generation with Zero-shot
LLM-based Query Routing and Corrective Feedback for Domain-Specific Knowledge
Tasks.*

**Author:** Abhishek Mukherjee
**Degree:** MSc Artificial Intelligence & Machine Learning

This repository is the thesis deliverable, the novel pipeline itself.
Reproduction of the published baselines it is compared against lives in the
sibling repository `ljmu-thesis-baseline-reproduction`.

## What the system does

For each query:

1. A **zero-shot LLM router** (structured JSON output, no training) picks a
   retrieval tier: A `no_retrieval`, B `single_step`, or C `iterative`
   multi-step retrieval.
2. The tier executes and produces a draft answer with its evidence.
3. A **CRAG-style three-way gate** judges the evidence (or, on tier A, the
   parametric draft): `correct`, `ambiguous`, or `incorrect`.
4. On `incorrect`, the orchestrator escalates the query to the next-higher
   tier only — A→B or B→C, at most one promotion per query — and re-runs.

The whole loop is a single-agent state machine (LangGraph), built from
locally deployable, training-free open-weight models: a 4-bit quantised
Qwen2.5-7B via MLX on the local MacBook (no CUDA), or the same model through
HF transformers on a cloud GPU. Proprietary APIs are not part of the system and
they are permitted later only as comparator/tooling in the evaluation.

## Quickstart (Google Colab)

The reference environment is a stock **Google Colab GPU runtime** (T4 is
sufficient). Every command in this README is expected to work there. Open
`notebooks/adarag_colab_cuda.ipynb` in Colab, set *Runtime → Change runtime
type → GPU*, and run it top to bottom. The notebook:

1. Checks the runtime and GPU,
2. Clones this repository and installs the pinned CUDA dependency stack,
3. Downloads sample datasets and prepares them
   (`scripts/prepare_datasets.py`),
4. Builds a BM25 and FAISS index over the SciFact corpus,
5. Smoke-tests the full route -> retrieve -> generate -> gate loop offline
   (`adarag ask ... --fake`), and
6. Runs a small real evaluation with the HF backend, writing
   `predictions.jsonl` + `summary.json` under `runs/`.

To run the offline test suite on the same runtime (no models, no network):

```bash
pip install pytest && python -m pytest tests -q
```

`environment/colab_check.ipynb` is the standalone environment check for a
fresh VM. A local path for Apple Silicon (conda env via
`environment/setup_env.sh`, MLX 4-bit generation, no CUDA) exists as well.
See *Compute*, but Colab is the environment to reproduce results in.

## CLI

| Command | Purpose |
|---|---|
| `adarag ask` | one question through the full pipeline (`--fake` offline demo; `--no-escalation` ablation) |
| `adarag route` | router only (RQ1); `--prompt-variant v1\|v2` for the prompt-sensitivity ablation |
| `adarag index` | chunk a corpus jsonl into BM25 + FAISS indices |
| `adarag eval` | run a dataset through the harness; EM/F1, routing accuracy, latency, token counts |
| `adarag info` | environment, settings, model-cache status |

## Layout

```
src/adarag/
  config.py          settings, retrieval tiers, the escalation map
  device.py          mps/cuda/cpu selection
  llm/               backend interface + MLX / HF / fake backends, JSON decoding
  retrieval/         retriever interface; BM25, dense (FAISS), ColBERT (optional)
  corpus/            chunking and index building
  router.py          zero-shot LLM router (RQ1)
  gate.py            CRAG-style corrective gate (RQ2)
  prompts/           router prompt variants + gate prompts
  pipeline/          pipeline state, tier executors, LangGraph orchestrator
  data/              QA example schema + per-dataset normalisers
  eval/              metrics, silver routing labels, eval runner
  cli.py             the adarag entry point
scripts/             dataset preparation (raw downloads are manual; the script
                     normalises them and cuts the dev500/toy slices)
tests/               offline pytest suite (fake backend, no network)
notebooks/           Colab notebooks: system runner (CUDA) and the RQ1
                     Adaptive-RAG classifier training/eval
environment/         requirements files, setup script, env self-checks
configs/models.yaml  candidate model registry for the router-model ablation
```

`data/`, `indices/`, `models/` and `runs/` are produced at runtime and are
not committed.

## Compute

The reference execution environment is **Google Colab** (CUDA GPU, HF
transformers backend) - `notebooks/adarag_colab_cuda.ipynb` stands the
system up on a fresh VM. Development also runs locally on a MacBook (Apple
Silicon, no CUDA) which is PyTorch on MPS for embedders, MLX for 4-bit 7B
generation. The design constraint is mainatained throughout namely locally deployable,
training-free models operating under constrained compute.

## Environments

Two conda environments, kept apart on purpose:

| Env | Purpose |
|---|---|
| `adarag` | the system itself: router, gate, orchestration, retrieval, local generation |
| `adarag-eval` | RAGAS / ARES faithfulness scoring |

RAGAS pins an older langchain-community that conflicts with the LangGraph
version the orchestrator uses, so evaluation runs in its own env against the
prediction files the system writes to disk (`runs/**/predictions.jsonl`).

## How this maps to the research questions

- RQ1 (zero-shot router vs trained classifier): `router.py`, `adarag route`,
  and `notebooks/adaptive_rag_classifier_rq1.ipynb`, which trains the
  Adaptive-RAG t5-large classifier to produce the comparison numbers.
- RQ2 (corrective escalation vs one-shot routing): `gate.py` and the bounded
  gate→route back-edge in `pipeline/graph.py`; `--no-escalation` is the
  ablation switch.
- RQ3 (cross-domain behaviour): the dataset normalisers and `adarag eval`
  over the telecom / medical / scientific / general QA sets.
- RQ4 (quality vs latency vs cost under constrained compute): per-node token
  and latency traces recorded by the eval harness.
