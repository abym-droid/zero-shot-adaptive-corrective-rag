"""Command-line entry point for the adarag system (`adarag ...`).

Heavy dependencies are imported inside each command, so `adarag info` and
--help keep working even when optional packages (mlx-lm, ragatouille, faiss)
are missing. --fake on ask/route swaps in the deterministic FakeBackend plus
an in-memory demo retriever, so the whole route -> retrieve -> generate ->
gate loop can be shown offline without downloading a model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer

from adarag.config import RetrievalTier, settings

app = typer.Typer(
    name="adarag",
    help=(
        "Adaptive RAG with zero-shot LLM routing and corrective escalation "
        "(MSc thesis, LJMU)."
    ),
    no_args_is_help=True,
    add_completion=False,
)

# Canned FakeBackend responses used by --fake demos (deterministic, offline).
_FAKE_ROUTE_JSON = '{"tier": "single_step", "reason": "FakeBackend canned decision (demo)"}'
_FAKE_GATE_JSON = (
    '{"verdict": "correct", "reason": "FakeBackend canned verdict (demo)", '
    '"useful_doc_ids": ["demo-0"]}'
)
_FAKE_ANSWER = "This is a deterministic demo answer produced by FakeBackend."


# --------------------------------------------------------------------------- #
# helpers (lazy by construction: no heavy imports at module scope)
# --------------------------------------------------------------------------- #
def _console():
    """Return a rich Console (imported lazily to keep CLI startup light)."""
    from rich.console import Console

    return Console()


def _fail(msg: str) -> "typer.Exit":
    """Print an error message and return a non-zero :class:`typer.Exit`."""
    _console().print(f"[bold red]error:[/bold red] {msg}")
    return typer.Exit(code=1)


def _repo_root() -> Path:
    """Best-effort repo root: src/adarag/cli.py -> parents[2] (editable install)."""
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").exists():
        return root
    return Path.cwd()


def _load_llm_backend(fake: bool, model: Optional[str], default_model: str,
                      fake_responses: list[str]):
    """Return an LLM backend: FakeBackend when ``fake`` else a real one.

    Args:
        fake: If True, build a deterministic FakeBackend (offline demo).
        model: Explicit model id from the CLI, or None for the default.
        default_model: Settings default used when ``model`` is None.
        fake_responses: Canned responses for the FakeBackend.

    Returns:
        An ``adarag.llm.base.LLMBackend`` instance.
    """
    if fake:
        from adarag.llm.fake import FakeBackend

        return FakeBackend(fake_responses)
    from adarag.llm.base import load_backend

    return load_backend(model or default_model, device=settings.device)


def _demo_retriever():
    """Build a tiny in-memory retriever for ``--fake`` demos (no index needed)."""
    from adarag.retrieval.base import Doc, Retriever

    class _DemoRetriever(Retriever):
        """Fixed three-document retriever used only by the --fake demo path."""

        name = "demo"

        def search(self, query: str, k: int = 5) -> list[Doc]:
            docs = [
                Doc(
                    doc_id=f"demo-{i}",
                    text=(
                        f"Demo passage {i}: canned offline context for the "
                        f"query '{query}'. Used only by `adarag ask --fake`."
                    ),
                    score=1.0 - 0.1 * i,
                    meta={"title": "Demo corpus", "chunk_id": f"demo-{i}"},
                )
                for i in range(3)
            ]
            return docs[:k]

    return _DemoRetriever()


def _load_retriever(index_dir: Path, kind: str):
    """Load a persisted retriever from an ``adarag index`` output directory.

    Accepts either the index root (``indices/<name>``, containing ``bm25/`` /
    ``dense/`` subdirectories per the corpus/build layout) or the kind
    subdirectory itself.

    Args:
        index_dir: Path passed via ``--index``.
        kind: ``"bm25"`` or ``"dense"``.

    Returns:
        A loaded ``adarag.retrieval.base.Retriever``.
    """
    path = index_dir / kind if (index_dir / kind).exists() else index_dir
    if kind == "bm25":
        from adarag.retrieval.bm25 import BM25Retriever

        return BM25Retriever.load(str(path))
    if kind == "dense":
        from adarag.retrieval.dense import DenseRetriever

        return DenseRetriever.load(str(path))
    raise typer.BadParameter(f"unknown retriever kind {kind!r} (use bm25|dense)")


def _forward_to_typer_app(module_app, args: list[str], prog: str) -> None:
    """Invoke another typer app with forwarded argv (passthrough delegation)."""
    # Typer instances are click-callables: app(args) -> BaseCommand.main(args).
    module_app(args=args, prog_name=prog)


def _print_state(state: dict, as_json: bool) -> None:
    """Render a final pipeline RAGState to the terminal.

    Args:
        state: Final ``RAGState`` mapping returned by ``AdaptiveRAG.answer``.
        as_json: If True, dump raw JSON instead of the rich summary.
    """
    if as_json:
        typer.echo(json.dumps(state, indent=2, default=str))
        return
    from rich.panel import Panel
    from rich.table import Table

    console = _console()
    console.print(Panel(state.get("answer", "<no answer>"), title="answer"))
    summary = Table(show_header=False, box=None)
    summary.add_row("tier (final)", str(state.get("tier", "?")))
    summary.add_row("route reason", str(state.get("route_reason", "")))
    summary.add_row("gate verdict", str(state.get("verdict", "-")))
    summary.add_row("gate reason", str(state.get("gate_reason", "")))
    summary.add_row("escalations", str(state.get("escalations", 0)))
    summary.add_row("docs retrieved", str(len(state.get("docs", []))))
    console.print(summary)
    trace = state.get("trace", [])
    if trace:
        t = Table(title="trace", header_style="bold")
        t.add_column("node")
        t.add_column("latency_s", justify="right")
        t.add_column("prompt_tok", justify="right")
        t.add_column("completion_tok", justify="right")
        for step in trace:
            t.add_row(
                str(step.get("node", "?")),
                f"{step.get('latency_s', 0.0):.3f}",
                str(step.get("prompt_tokens", 0)),
                str(step.get("completion_tokens", 0)),
            )
        console.print(t)


# --------------------------------------------------------------------------- #
# ask
# --------------------------------------------------------------------------- #
@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to run through the pipeline."),
    fake: bool = typer.Option(
        False, "--fake", help="Use the deterministic FakeBackend + demo retriever (offline demo)."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help=f"Generator model id (default: {settings.generator_model})."
    ),
    index: Optional[Path] = typer.Option(
        None, "--index", help="Index directory from `adarag index` (required unless --fake)."
    ),
    retriever: str = typer.Option("bm25", "--retriever", help="Retriever kind: bm25 | dense."),
    prompt_variant: str = typer.Option(
        "v1", "--prompt-variant", help="Router prompt variant (v1 | v2, the prompt-sensitivity ablation)."
    ),
    escalation: bool = typer.Option(
        True,
        "--escalation/--no-escalation",
        help="Enable/disable the corrective escalation gate loop (the headline ablation).",
    ),
    top_k: int = typer.Option(settings.top_k, "--top-k", help="Passages per retrieval call."),
    json_out: bool = typer.Option(False, "--json", help="Print the full final state as JSON."),
) -> None:
    """Answer one question through the full adaptive-RAG pipeline.

    Route -> execute tier -> gate -> (bounded) escalate. With ``--fake`` the
    whole loop runs offline on canned responses - no models, no index.
    """
    try:
        from adarag.pipeline.graph import AdaptiveRAG
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise _fail(f"pipeline unavailable ({exc})") from exc

    if fake:
        from adarag.llm.fake import FakeBackend

        backend = FakeBackend([_FAKE_ANSWER] * 16)
        router_backend = FakeBackend([_FAKE_ROUTE_JSON] * 8)
        gate_backend = FakeBackend([_FAKE_GATE_JSON] * 8)
        retr = _demo_retriever()
    else:
        if index is None:
            raise _fail("--index is required unless --fake is given")
        backend = _load_llm_backend(False, model, settings.generator_model, [])
        router_backend = None
        gate_backend = None
        try:
            retr = _load_retriever(index, retriever)
        except FileNotFoundError as exc:
            raise _fail(f"could not load {retriever} index from {index}: {exc}") from exc

    run_settings = settings.model_copy(
        update={"top_k": top_k, "max_escalations": settings.max_escalations if escalation else 0}
    )
    rag = AdaptiveRAG(
        backend,
        retr,
        gate_backend=gate_backend,
        router_backend=router_backend,
        settings=run_settings,
        prompt_variant=prompt_variant,
    )
    state = rag.answer(question)
    _print_state(dict(state), json_out)


# --------------------------------------------------------------------------- #
# route
# --------------------------------------------------------------------------- #
@app.command()
def route(
    question: str = typer.Argument(..., help="Question to route (no retrieval/generation)."),
    fake: bool = typer.Option(False, "--fake", help="Use the deterministic FakeBackend."),
    model: Optional[str] = typer.Option(
        None, "--model", help=f"Router model id (default: {settings.router_model})."
    ),
    prompt_variant: str = typer.Option(
        "v1", "--prompt-variant", help="Router prompt variant (v1 | v2, the prompt-sensitivity ablation)."
    ),
    force_min_tier: Optional[RetrievalTier] = typer.Option(
        None,
        "--force-min-tier",
        help="Minimum tier hint (simulates the gate's escalation back-edge).",
    ),
    json_out: bool = typer.Option(False, "--json", help="Print the decision as JSON."),
) -> None:
    """Run only the zero-shot LLM router on a question.

    Prints the chosen tier, the model's reason, and token/latency cost of the
    routing call.
    """
    try:
        from adarag.router import Router
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise _fail(f"router unavailable ({exc})") from exc

    backend = _load_llm_backend(fake, model, settings.router_model, [_FAKE_ROUTE_JSON] * 4)
    decision, gen = Router(backend, prompt_variant=prompt_variant).route(
        question, force_min_tier=force_min_tier
    )
    if json_out:
        typer.echo(
            json.dumps(
                {
                    "tier": decision.tier.value,
                    "reason": decision.reason,
                    "prompt_tokens": gen.prompt_tokens,
                    "completion_tokens": gen.completion_tokens,
                    "latency_s": gen.latency_s,
                },
                indent=2,
            )
        )
        return
    from rich.table import Table

    t = Table(show_header=False, box=None)
    t.add_row("tier", f"[bold]{decision.tier.value}[/bold]")
    t.add_row("reason", decision.reason)
    t.add_row("backend", getattr(backend, "name", "?"))
    t.add_row("tokens", f"{gen.prompt_tokens} prompt / {gen.completion_tokens} completion")
    t.add_row("latency", f"{gen.latency_s:.3f}s")
    _console().print(t)


# --------------------------------------------------------------------------- #
# index
# --------------------------------------------------------------------------- #
@app.command()
def index(
    corpus: Path = typer.Argument(
        ..., exists=True, readable=True,
        help='Corpus jsonl ({"doc_id","title","text"} rows) to index.',
    ),
    out_dir: Optional[Path] = typer.Option(
        None, "--out-dir", help="Output directory (default: indices/<corpus stem>)."
    ),
    kinds: str = typer.Option(
        "bm25,dense", "--kinds", help="Comma-separated index kinds: bm25,dense."
    ),
    embedder: str = typer.Option(
        settings.embedder_model, "--embedder", help="Sentence-transformers model for the dense index."
    ),
) -> None:
    """Chunk a corpus and build on-disk retrieval indices.

    Delegates to ``adarag.corpus.build.build_indices`` which writes
    ``<out_dir>/{bm25/, dense/, meta.json}``.
    """
    try:
        from adarag.corpus.build import build_indices
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise _fail(f"corpus builder unavailable ({exc})") from exc

    out = out_dir or (settings.index_dir / corpus.stem)
    kind_tuple = tuple(k.strip() for k in kinds.split(",") if k.strip())
    _console().print(f"building {kind_tuple} indices for [bold]{corpus}[/bold] -> {out}")
    build_indices(corpus, out, embedder, kinds=kind_tuple)
    _console().print("[green]done[/green]")


# --------------------------------------------------------------------------- #
# eval passthrough
# --------------------------------------------------------------------------- #
@app.command(
    "eval",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def eval_cmd(ctx: typer.Context) -> None:
    """Run the evaluation harness.

    Passthrough to ``adarag.eval.run_eval``; all arguments are forwarded
    verbatim, e.g.::

        adarag eval --dataset data/processed/squad.dev500.jsonl \\
                    --index indices/squad --backend fake --limit 20
    """
    try:
        from adarag.eval import run_eval
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise _fail(f"eval harness unavailable ({exc})") from exc

    eval_app = getattr(run_eval, "app", None)
    if eval_app is None:
        raise _fail("adarag.eval.run_eval does not expose a typer `app`")
    _forward_to_typer_app(eval_app, list(ctx.args), prog="adarag eval")


# --------------------------------------------------------------------------- #
# info
# --------------------------------------------------------------------------- #
@app.command()
def info() -> None:
    """Show environment, settings and model-cache status.

    Works with only core dependencies installed: every optional package is
    probed via package metadata and failures are reported, never raised.
    """
    import platform as _platform
    from importlib.metadata import PackageNotFoundError, version

    from rich.table import Table

    console = _console()

    def _ver(dist: str) -> str:
        try:
            return version(dist)
        except PackageNotFoundError:
            return "[red]not installed[/red]"
        except Exception as exc:  # pragma: no cover - defensive
            return f"[red]error: {exc}[/red]"

    env = Table(title="environment", header_style="bold")
    env.add_column("component")
    env.add_column("value")
    env.add_row("python", f"{_platform.python_version()} ({sys.executable})")
    env.add_row("platform", _platform.platform())
    env.add_row("adarag", _ver("adarag"))
    for dist in (
        "torch", "transformers", "sentence-transformers", "mlx-lm", "langgraph",
        "langchain-core", "faiss-cpu", "rank-bm25", "datasets", "huggingface-hub",
        "ragatouille",
    ):
        env.add_row(dist, _ver(dist))
    try:
        from adarag.device import get_device

        env.add_row("resolved device", get_device(settings.device))
    except Exception as exc:  # torch missing/broken
        env.add_row("resolved device", f"[red]unavailable ({exc})[/red]")
    console.print(env)

    cfg = Table(title="settings (env prefix ADARAG_)", header_style="bold")
    cfg.add_column("setting")
    cfg.add_column("value")
    cfg.add_row("router_model", settings.router_model)
    cfg.add_row("generator_model", settings.generator_model)
    cfg.add_row("embedder_model", settings.embedder_model)
    cfg.add_row("colbert_model", settings.colbert_model)
    cfg.add_row("top_k", str(settings.top_k))
    cfg.add_row("max_escalations", str(settings.max_escalations))
    for label, p in (
        ("data_dir", settings.data_dir),
        ("index_dir", settings.index_dir),
        ("runs_dir", settings.runs_dir),
    ):
        cfg.add_row(label, f"{p} ({'exists' if Path(p).exists() else 'missing'})")
    console.print(cfg)

    cache = Table(title="HF model cache", header_style="bold")
    cache.add_column("model")
    cache.add_column("cached")
    wanted = {
        settings.router_model,
        settings.generator_model,
        settings.embedder_model,
        settings.colbert_model,
    }
    try:
        from huggingface_hub import scan_cache_dir

        cached = {repo.repo_id: repo.size_on_disk_str for repo in scan_cache_dir().repos}
        for model in sorted(wanted):
            cache.add_row(
                model,
                f"[green]yes[/green] ({cached[model]})" if model in cached else "[yellow]no[/yellow]",
            )
    except Exception as exc:  # hub missing or cache unreadable
        for model in sorted(wanted):
            cache.add_row(model, f"[red]unknown ({exc})[/red]")
    console.print(cache)


if __name__ == "__main__":  # pragma: no cover
    app()
