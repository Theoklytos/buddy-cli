"""Command-line interface for Bud RAG Pipeline."""

import json
import os
from pathlib import Path

import click

from bud import __version__
from bud.config import (
    CONFIG_FILE,
    get_config_dir,
    get_data_dir,
    get_output_dir,
    load_config,
    save_config,
    validate_config,
)


@click.group()
@click.version_option(version=__version__, prog_name="bud")
def main():
    """Bud RAG Pipeline - A personal conversation RAG system."""
    pass


@main.command()
def configure():
    """Interactive configuration shell."""
    from rich.console import Console
    from rich.prompt import Prompt, Confirm

    console = Console()

    console.print("\n[bold cyan]Bud RAG Pipeline Configuration[/bold cyan]\n")
    console.print("This will guide you through setting up your configuration.\n")

    config = load_config()

    # Data directory
    current_data = config.get("data_dir", "")
    default_data_dir = str(Path.home() / "data" / "conversations")
    data_dir = Prompt.ask(
        "[green]Data directory[/green] (where your conversation data is stored)",
        default=current_data or default_data_dir,
    )
    config["data_dir"] = data_dir

    # Output directory
    current_output = config.get("output_dir", "")
    default_output_dir = str(Path.home() / "data" / "bud_output")
    output_dir = Prompt.ask(
        "[green]Output directory[/green] (where pipeline outputs will be stored)",
        default=current_output or default_output_dir,
    )
    config["output_dir"] = output_dir

    # LLM configuration
    console.print("\n[bold]LLM Configuration[/bold]\n")

    current_llm_provider = config.get("llm", {}).get("provider", "ollama")
    llm_provider = Prompt.ask(
        "[blue]LLM Provider[/blue]",
        choices=["ollama", "openai", "anthropic"],
        default=current_llm_provider,
    )
    config.setdefault("llm", {})["provider"] = llm_provider

    current_llm_url = config.get("llm", {}).get("base_url", "http://localhost:11434")
    llm_base_url = Prompt.ask(
        "[blue]LLM Base URL[/blue]", default=current_llm_url
    )
    config.setdefault("llm", {})["base_url"] = llm_base_url

    current_llm_model = config.get("llm", {}).get("model", "gemini-3-flash-preview:latest")
    llm_model = Prompt.ask(
        "[blue]LLM Model[/blue]", default=current_llm_model
    )
    config.setdefault("llm", {})["model"] = llm_model

    # Embeddings configuration
    console.print("\n[bold]Embeddings Configuration[/bold]\n")

    current_emb_provider = config.get("embeddings", {}).get("provider", "ollama")
    emb_provider = Prompt.ask(
        "[blue]Embeddings Provider[/blue]",
        choices=["ollama", "openai"],
        default=current_emb_provider,
    )
    config.setdefault("embeddings", {})["provider"] = emb_provider

    current_emb_url = config.get("embeddings", {}).get("base_url", "http://localhost:11434")
    emb_base_url = Prompt.ask(
        "[blue]Embeddings Base URL[/blue]", default=current_emb_url
    )
    config.setdefault("embeddings", {})["base_url"] = emb_base_url

    current_emb_model = config.get("embeddings", {}).get("model", "nomic-embed-text")
    emb_model = Prompt.ask(
        "[blue]Embeddings Model[/blue]", default=current_emb_model
    )
    config.setdefault("embeddings", {})["model"] = emb_model

    # Show resolved model specs so the user knows what limits will be applied
    from bud.lib.model_registry import resolve_embedding_model
    model_cfg = resolve_embedding_model(emb_model)
    if model_cfg["known"]:
        console.print(
            f"  [green]✓[/green] {model_cfg['description']}\n"
            f"  [dim]chunk_max_tokens: {model_cfg['chunk_max_tokens']}  ·  "
            f"max_embed_chars: {model_cfg['max_embed_chars']}[/dim]"
        )
    else:
        console.print(
            f"  [yellow]⚠  '{emb_model}' is not in the model registry — "
            f"conservative 512-token defaults will be used.[/yellow]\n"
            f"  [dim]chunk_max_tokens: {model_cfg['chunk_max_tokens']}  ·  "
            f"max_embed_chars: {model_cfg['max_embed_chars']}[/dim]\n"
            f"  [dim]Tip: run 'bud models' to see all supported models.[/dim]"
        )

    # Save and validate
    console.print("\n[bold]Saving configuration...[/bold]")
    save_config(config)

    is_valid, errors = validate_config(config)

    if is_valid:
        console.print(
            f"\n[green]Configuration saved successfully![/green]"
        )
        console.print(f"[dim]Location: {CONFIG_FILE}[/dim]")
    else:
        console.print("\n[red]Configuration has errors:[/red]")
        for error in errors:
            console.print(f"  - {error}")
        console.print("\n[dim]Configuration still saved. Review and fix errors.[/dim]")


@main.command("models")
def models_command():
    """List all supported embedding models and their configuration parameters."""
    from bud.lib.model_registry import list_known_models

    table = Table(title="Supported Embedding Models", show_lines=True)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Dims", style="magenta", justify="right")
    table.add_column("Context (tokens)", style="yellow", justify="right")
    table.add_column("max_embed_chars", style="green", justify="right")
    table.add_column("chunk_max_tokens", style="blue", justify="right")

    for entry in list_known_models():
        table.add_row(
            entry["model"],
            str(entry["dimension"]),
            str(entry["context_tokens"]),
            str(entry["max_embed_chars"]),
            str(entry["chunk_max_tokens"]),
        )

    console.print(table)
    console.print(
        "\n[dim]These limits are applied automatically when you run "
        "'bud configure' or 'bud process'.[/dim]"
    )


@main.command()
@click.option(
    "--data-dir", "-d",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Path to input data directory (overrides config)",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(file_okay=False, resolve_path=True),
    help="Path to output directory (overrides config)",
)
@click.option(
    "--force/--no-force", "-f",
    default=False,
    help="Re-parse all files even if output is already up-to-date",
)
def parse(data_dir, output_dir, force):
    """Parse conversation JSON files into structured JSONL.

    Reads conversations_*.json from the data directory and writes
    parsed JSONL files to <output_dir>/parsed/.  Files whose output
    is already newer than the source are skipped unless --force is given.
    """
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
    )

    console = Console()

    if data_dir:
        data_dir = Path(data_dir)
    else:
        data_dir = get_data_dir()

    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = get_output_dir()

    output_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir = output_dir / "parsed"

    console.print("\n[bold cyan]Bud RAG Pipeline — Parse[/bold cyan]\n")
    console.print(f"[dim]Data directory:   {data_dir}[/dim]")
    console.print(f"[dim]Output directory: {parsed_dir}[/dim]")
    if force:
        console.print("[dim]Force: re-parsing all files[/dim]")
    console.print("")

    conv_files = sorted(data_dir.glob("conversations_*.json"))
    if not conv_files:
        console.print("[yellow]No conversations_*.json files found in data directory.[/yellow]")
        return

    console.print(f"[green]✓ Found {len(conv_files)} conversation file(s)[/green]\n")

    from bud.stages.parse import parse_all

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=8,
    ) as progress:
        task = progress.add_task("[cyan]Parsing...[/cyan]", total=None)

        def on_progress(total_so_far):
            progress.update(
                task,
                description=f"[cyan]Parsed {total_so_far} conversations so far[/cyan]",
            )

        total = parse_all(data_dir, parsed_dir, progress_callback=on_progress, force=force)

    console.print(f"\n[green]✓ Parse complete — {total} conversations[/green]")
    console.print(f"[dim]Output: {parsed_dir}[/dim]\n")


@main.command()
@click.argument("iterations", type=int, default=5, required=False)
@click.option(
    "--output-dir", "-o",
    type=click.Path(file_okay=False, resolve_path=True),
    help="Path to output directory (overrides config)",
)
@click.option(
    "--stability", "-t",
    type=float,
    default=0.75,
    help="Stability threshold to stop early (default: 0.75)",
)
@click.option(
    "--samples", "-s",
    type=int,
    default=5,
    help="Conversations to sample per iteration (default: 5)",
)
@click.option(
    "--blend/--no-blend",
    default=False,
    help="Use cross-boundary blending instead of whole-conversation sampling.",
)
@click.option(
    "--progressive/--no-progressive",
    default=False,
    help="Progressive cursor-based blending for exhaustive coverage.",
)
@click.option("--reset-cursor", is_flag=True, default=False, help="Reset the progressive blend cursor.")
@click.option("--blend-slices", "-S", type=int, default=6, hidden=True)
@click.option("--blend-width", "-W", type=int, default=8, hidden=True)
def discover(iterations, output_dir, stability, samples, blend, progressive, reset_cursor, blend_slices, blend_width):
    """Run iterative pattern discovery.

    \b
    Usage:
      bud discover          # 5 iterations (default)
      bud discover 10       # 10 iterations
      bud discover 3        # 3 more on top of previous runs

    Always resumes from existing discovery map if present.
    Stops early when stability >= threshold.
    """
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
    )

    console = Console()

    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = get_output_dir()

    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config()

    from bud.lib.model_registry import resolve_embedding_model as _resolve
    _model_cfg = _resolve(config.get("embeddings", {}).get("model", ""))
    config.setdefault("pipeline", {
        "chunk_min_tokens": _model_cfg["chunk_min_tokens"],
        "chunk_max_tokens": _model_cfg["chunk_max_tokens"],
        "schema_evolution_confidence_threshold": 5,
    })

    from bud.stages.index import IndexManager
    index_mgr = IndexManager(output_dir, config)
    index_mgr.ensure_directories()

    parsed_dir = output_dir / "parsed"
    if not parsed_dir.exists() or not any(parsed_dir.glob("*.jsonl")):
        console.print(
            "[red]No parsed conversations found.[/red]\n"
            "[dim]Run 'bud parse' first to generate parsed/*.jsonl files.[/dim]"
        )
        return

    from bud.lib.llm import LLMClient
    from bud.stages.discover import DiscoveryMap, run_discovery

    llm = LLMClient(config)

    # Always resume from existing map
    concept_map = DiscoveryMap(index_mgr.discovery_map_path).load()
    prior = concept_map.iterations_completed

    console.print("\n[bold cyan]Bud RAG Pipeline — Discovery[/bold cyan]\n")
    console.print(f"[dim]Output: {output_dir}[/dim]")
    console.print(f"[dim]Iterations: {iterations}  |  Stability threshold: {stability}[/dim]")
    if prior > 0:
        console.print(
            f"[green]✓ Resuming from {prior} prior iterations "
            f"(stability={concept_map.stability_score:.2f})[/green]"
        )
    if progressive:
        console.print(f"[dim]Mode: progressive[/dim]")
    elif blend:
        console.print(f"[dim]Mode: blend[/dim]")
    console.print("")

    # Progressive cursor
    cursor = None
    if progressive:
        from bud.stages.blend import BlendCursor
        cursor = BlendCursor(index_mgr.blend_cursor_path)
        if reset_cursor:
            cursor.reset()
            console.print("[yellow]⚠  Blend cursor reset[/yellow]\n")
        else:
            cursor.load()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=8,
    ) as progress:
        task = progress.add_task("Sampling...", total=None)

        def on_sampling(iteration_num, max_iter):
            mode_tag = "progressive" if progressive else ("blend" if blend else "sample")
            progress.update(
                task,
                description=f"[dim]iter {iteration_num}/{max_iter}  {mode_tag}  waiting for LLM...[/dim]",
            )

        def on_iteration(iteration_num, score, cmap, prompt_tokens=0, response_tokens=0):
            signals = len(cmap.data.get("boundary_signals", []))
            archetypes = len(cmap.data.get("chunk_archetypes", []))
            anchors = len(cmap.data.get("coherence_anchors", []))
            bar = "▓" * int(score * 10) + "░" * (10 - int(score * 10))
            color = "green" if score >= stability else "yellow"
            total_tok = prompt_tokens + response_tokens

            def _fmt(n):
                return f"{n / 1000:.1f}k" if n >= 1000 else str(n)

            progress.update(
                task,
                description=(
                    f"iter {iteration_num}/{iterations}  "
                    f"[{color}]{bar}[/{color}] "
                    f"stability={score:.2f}  "
                    f"[dim]signals={signals}  archetypes={archetypes}  anchors={anchors}  "
                    f"tokens={_fmt(total_tok)} ({_fmt(prompt_tokens)}↑ {_fmt(response_tokens)}↓)[/dim]"
                ),
            )

        concept_map = run_discovery(
            parsed_dir=str(parsed_dir),
            concept_map=concept_map,
            llm=llm,
            n_samples=samples,
            stability_threshold=stability,
            max_iterations=iterations,
            on_iteration=on_iteration,
            on_sampling=on_sampling,
            use_blend=blend,
            blend_slices=blend_slices,
            blend_width=blend_width,
            use_progressive=progressive,
            cursor=cursor,
        )

    console.print(f"\n[green]✓ Discovery complete![/green]")
    console.print(f"  Total iterations: {concept_map.iterations_completed}")
    console.print(f"  Stability: {concept_map.stability_score:.2f}")
    console.print(f"  Signals: {len(concept_map.data.get('boundary_signals', []))}  "
                  f"Archetypes: {len(concept_map.data.get('chunk_archetypes', []))}  "
                  f"Anchors: {len(concept_map.data.get('coherence_anchors', []))}")
    console.print(f"\n[dim]Saved to: {index_mgr.discovery_map_path}[/dim]")

    if progressive and cursor is not None:
        from bud.stages.blend import _load_turns
        file_totals = {
            f.name: len(_load_turns(f))
            for f in sorted((output_dir / "parsed").glob("*.jsonl"))
        }
        coverage = cursor.coverage(file_totals)
        if coverage:
            console.print("\n  [dim]Blend cursor coverage:[/dim]")
            for fname, pct in sorted(coverage.items()):
                bar = "▓" * int(pct * 20) + "░" * (20 - int(pct * 20))
                console.print(f"    {bar} {pct*100:.0f}%  {fname}")

    console.print("")


@main.command()
@click.argument("iterations", type=int, default=3, required=False)
@click.option(
    "--data-dir", "-d",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Path to input data directory (overrides config)",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(file_okay=False, resolve_path=True),
    help="Path to output directory (overrides config)",
)
@click.option(
    "--stability", "-t",
    type=float,
    default=0.75,
    help="Stability threshold to stop early (default: 0.75)",
)
@click.option(
    "--prompt", "-p",
    type=click.Choice(["conversational", "factual", "mythic", "synthesis"]),
    default="conversational",
    help="Prompt preset to use for chunking",
)
@click.option(
    "--batch-size", "-b",
    type=int,
    default=0,
    help="Max conversations per chunking pass (0 = all, default: all)",
)
@click.option(
    "--concurrency", "-c",
    type=int,
    default=1,
    help="Concurrent LLM requests (default: 1 = sequential)",
)
def chunk(iterations, data_dir, output_dir, stability, prompt, batch_size, concurrency):
    """Iterative chunking with self-refinement.

    \b
    Usage:
      bud chunk             # 3 iterations (default)
      bud chunk 5           # 5 iterations
      bud chunk 1           # single-pass, no refinement

    Loads the discovery map (from 'bud discover') and uses it to inform
    chunking. Then runs multiple passes — after each pass, the LLM reviews
    sample chunks and provides feedback that improves the next pass.

    Outputs a stability score showing how confident the final chunking is.
    """
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
    )

    console = Console()

    if data_dir:
        data_dir = Path(data_dir)
    else:
        data_dir = get_data_dir()

    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = get_output_dir()

    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config()

    from bud.lib.model_registry import resolve_embedding_model as _resolve
    _model_cfg = _resolve(config.get("embeddings", {}).get("model", ""))
    config.setdefault("pipeline", {
        "chunk_min_tokens": _model_cfg["chunk_min_tokens"],
        "chunk_max_tokens": _model_cfg["chunk_max_tokens"],
        "schema_evolution_confidence_threshold": 5,
    })

    from bud.stages.index import IndexManager
    index_mgr = IndexManager(output_dir, config)
    index_mgr.ensure_directories()

    # Load schema
    from bud.lib.schema_manager import SchemaManager
    schema_mgr = SchemaManager(index_mgr.schema_path)
    schema = schema_mgr.load()
    if not schema_mgr.validate(schema):
        schema = schema_mgr.get_default_schema()
        schema_mgr.save(schema)

    console.print("\n[bold cyan]Bud RAG Pipeline — Iterative Chunking[/bold cyan]\n")
    console.print(f"[dim]Output: {output_dir}[/dim]")
    console.print(f"[dim]Iterations: {iterations}  |  Stability threshold: {stability}[/dim]")
    console.print(f"[dim]Prompt preset: {prompt}[/dim]")
    console.print(f"[dim]Schema: v{schema['version']}[/dim]\n")

    # Load discovery map
    from bud.stages.discover import DiscoveryMap
    dm = DiscoveryMap(index_mgr.discovery_map_path).load()
    concept_map_summary = None
    if dm.is_empty():
        console.print("[yellow]⚠ No discovery map found — chunking without it.[/yellow]")
        console.print("[dim]Run 'bud discover' first for better results.[/dim]\n")
    else:
        concept_map_summary = dm.to_summary()
        console.print(
            f"[green]✓ Loaded discovery map "
            f"({dm.iterations_completed} iterations, "
            f"stability={dm.stability_score:.2f})[/green]\n"
        )

    # Parse if needed, then load conversations
    parsed_dir = output_dir / "parsed"
    if not parsed_dir.exists() or not any(parsed_dir.glob("*.jsonl")):
        # Try parsing first
        from bud.stages.parse import parse_all
        conv_files = sorted(data_dir.glob("conversations_*.json"))
        if not conv_files:
            console.print("[red]No conversation files found.[/red]")
            return
        console.print("[cyan]→ Parsing conversations first...[/cyan]")
        parse_all(data_dir, parsed_dir, force=False)

    all_conversations = []
    for pf in sorted(parsed_dir.glob("*.jsonl")):
        with open(pf) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_conversations.append(json.loads(line))

    if not all_conversations:
        console.print("[red]No parsed conversations found.[/red]")
        return

    # Apply batch limit
    if batch_size > 0:
        all_conversations = all_conversations[:batch_size]

    from bud.stages.chunk import estimate_tokens
    total_tokens = sum(
        estimate_tokens(t["text"])
        for conv in all_conversations
        for t in conv.get("turns", [])
    )
    console.print(
        f"[green]✓ {len(all_conversations)} conversations loaded  |  "
        f"~{total_tokens:,} tokens[/green]\n"
    )

    # Initialize LLM and prompt
    from bud.lib.llm import LLMClient
    from bud.lib.prompt_loader import PromptLoader

    llm = LLMClient(config, concurrency=concurrency)
    if concurrency > 1:
        console.print(f"[dim]Concurrency: {concurrency}[/dim]")
    conv_files = sorted(data_dir.glob("conversations_*.json"))
    prompts_dir = str(Path(__file__).parent / "prompts")
    prompt_loader = PromptLoader(prompts_dir)
    system_prompt = prompt_loader.load(prompt, {
        "owner_name": "User",
        "schema": json.dumps(schema["dimensions"], indent=2),
        "file_context": f"{len(conv_files)} conversation files",
    })

    # Run iterative chunking
    from bud.stages.chunk_refine import run_iterative_chunking

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=8,
    ) as progress:
        task = progress.add_task("[cyan]Chunking...[/cyan]", total=None)

        def _on_pass_start(pass_num, max_iter):
            progress.update(
                task,
                description=(
                    f"[yellow]pass {pass_num}/{max_iter}[/yellow]  "
                    f"[dim]chunking {len(all_conversations)} conversations...[/dim]"
                ),
            )

        def _on_pass_complete(pass_num, n_chunks, n_convs):
            progress.update(
                task,
                description=(
                    f"[blue]pass {pass_num}[/blue]  "
                    f"[dim]{n_chunks} chunks from {n_convs} conversations — reviewing...[/dim]"
                ),
            )

        def _on_review_complete(pass_num, score, state):
            bar = "▓" * int(score * 10) + "░" * (10 - int(score * 10))
            color = "green" if score >= stability else "yellow"
            progress.update(
                task,
                description=(
                    f"pass {pass_num}  "
                    f"[{color}]{bar}[/{color}] "
                    f"stability={score:.2f}"
                ),
            )

        final_chunks, refine_state = run_iterative_chunking(
            conversations=all_conversations,
            schema=schema,
            llm=llm,
            config=config,
            system_prompt=system_prompt,
            prompt_preset=prompt,
            schema_version=schema["version"],
            concept_map_summary=concept_map_summary,
            max_iterations=iterations,
            stability_threshold=stability,
            on_pass_start=_on_pass_start,
            on_pass_complete=_on_pass_complete,
            on_review_complete=_on_review_complete,
        )

    # Save chunks to output
    chunks_path = output_dir / "chunks.jsonl"
    with open(chunks_path, "w") as f:
        for c in final_chunks:
            f.write(json.dumps(c) + "\n")

    # Save refinement report
    report = refine_state.to_report()
    report_path = output_dir / "chunk_refinement_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Collect schema proposals
    for c in final_chunks:
        for proposal in c.get("schema_proposals", []):
            schema_mgr.propose_candidate(
                proposal["dimension"],
                proposal["value"],
                proposal.get("rationale", ""),
            )
    promoted = schema_mgr.apply_promotions(config)

    # Summary
    console.print(f"\n[green]✓ Iterative chunking complete![/green]")
    console.print(f"  Passes: {refine_state.pass_count}")
    console.print(f"  Final stability: {refine_state.stability_score:.2f}")
    console.print(f"  Total chunks: {len(final_chunks)}")
    if promoted:
        console.print(f"  Schema evolved: {', '.join(promoted)}")
    console.print(f"\n[dim]Chunks: {chunks_path}[/dim]")
    console.print(f"[dim]Report: {report_path}[/dim]\n")


@main.command()
@click.option(
    "--data-dir", "-d",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Path to input data directory (overrides config)",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(file_okay=False, resolve_path=True),
    help="Path to output directory (overrides config)",
)
@click.option(
    "--resume/--no-resume", "-r",
    default=False,
    hidden=True,
    help="(Deprecated) Resume is now automatic. Kept for backwards compatibility.",
)
@click.option(
    "--batch-size", "-b",
    type=int,
    default=50,
    help="Number of conversations per batch (default: 50)",
)
@click.option(
    "--prompt", "-p",
    type=click.Choice(["conversational", "factual", "mythic", "synthesis"]),
    default="conversational",
    help="Prompt preset to use for chunking",
)
@click.option(
    "--with-discovery/--no-discovery",
    default=False,
    hidden=True,
    help="(Deprecated) Use --discover instead.",
)
@click.option(
    "--discover", "-D",
    type=int,
    default=None,
    help=(
        "Run inline discovery before chunking. The value sets the number of "
        "iterations (e.g. --discover 5). Uses 3 iterations by default when "
        "the flag is given without a value. An existing discovery map is "
        "loaded and extended, so you can run 'bud process --discover 3' "
        "repeatedly to accumulate more insight."
    ),
    is_flag=False,
    flag_value=3,
)
@click.option(
    "--discover-stability", "-T",
    type=float,
    default=0.75,
    help="Stability threshold for inline discovery (default: 0.75)",
)
@click.option(
    "--concurrency", "-c",
    type=int,
    default=1,
    help="Concurrent LLM requests for chunking (default: 1 = sequential)",
)
def process(data_dir, output_dir, resume, batch_size, prompt, with_discovery, discover, discover_stability, concurrency):
    """Run the full RAG pipeline.

    Processes conversation data and builds the vector index.

    Use --discover N to run N discovery iterations before chunking.
    The discovery map is persisted and reused across runs.
    """
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn,
        BarColumn, MofNCompleteColumn, TimeElapsedColumn,
    )

    console = Console()

    # Use overrides or config
    if data_dir:
        data_dir = Path(data_dir)
    else:
        data_dir = get_data_dir()

    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = get_output_dir()

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold cyan]Bud RAG Pipeline[/bold cyan]\n")

    # Load config for LLM/embedding settings
    config = load_config()

    # Derive chunk/embed limits from the configured embedding model
    from bud.lib.model_registry import resolve_embedding_model as _resolve
    _model_cfg = _resolve(config.get("embeddings", {}).get("model", ""))
    config.setdefault("pipeline", {
        "chunk_min_tokens": _model_cfg["chunk_min_tokens"],
        "chunk_max_tokens": _model_cfg["chunk_max_tokens"],
        "schema_evolution_confidence_threshold": 5,
    })

    console.print(f"[dim]Data directory: {data_dir}[/dim]")
    console.print(f"[dim]Output directory: {output_dir}[/dim]")
    console.print(f"[dim]Prompt preset: {prompt}[/dim]")
    console.print(f"[dim]Batch size: {batch_size}[/dim]")
    if discover is not None:
        console.print(f"[dim]Discovery: {discover} iterations (stability threshold: {discover_stability})[/dim]")
    elif with_discovery:
        console.print(f"[dim]Discovery: load existing map[/dim]")
    console.print("")

    # Initialize index manager
    from bud.stages.index import IndexManager
    index_mgr = IndexManager(output_dir, config)

    # Create output directories
    index_mgr.ensure_directories()

    # Load or create schema
    from bud.lib.schema_manager import SchemaManager
    schema_mgr = SchemaManager(index_mgr.schema_path)
    schema = schema_mgr.load()
    if not schema_mgr.validate(schema):
        schema = schema_mgr.get_default_schema()
        schema_mgr.save(schema)

    console.print(f"[green]✓ Schema loaded (v{schema['version']})[/green]")

    # Find conversation files
    conv_files = sorted(data_dir.glob("conversations_*.json"))
    if not conv_files:
        console.print("[yellow]No conversation files found[/yellow]")
        return

    console.print(f"[green]✓ Found {len(conv_files)} conversation file(s)[/green]\n")

    # Initialize vector store — auto-load existing index when present
    index_path = str(index_mgr.index_dir / "chunks")
    from bud.lib.store import VectorStore
    store = VectorStore(index_path, dim=_model_cfg["dimension"])
    has_existing_index = os.path.exists(f"{index_path}.faiss")
    if has_existing_index:
        store.load()
        console.print(f"[green]✓ Loaded existing index ({store.count()} chunks)[/green]")

    # Build progress tracking
    from bud.lib.progress import ProgressTracker
    tracker = ProgressTracker(index_mgr.progress_path)

    # Auto-detect prior progress (resume is always on — --resume flag kept for
    # backwards compatibility but is no longer required)
    from bud.stages.embed import load_embed_queue
    failed_chunks = load_embed_queue(index_mgr.embed_queue_path)
    if failed_chunks:
        console.print(f"[yellow]✓ Resuming {len(failed_chunks)} failed embeddings[/yellow]")

    # Initialize LLM and embedding clients
    from bud.lib.llm import LLMClient
    from bud.lib.embeddings import EmbeddingClient
    llm = LLMClient(config, concurrency=concurrency)
    embedding_client = EmbeddingClient(config)

    # Initialize prompt loader
    from bud.lib.prompt_loader import PromptLoader
    prompts_dir = str(Path(__file__).parent / "prompts")
    prompt_loader = PromptLoader(prompts_dir)
    system_prompt = prompt_loader.load(prompt, {
        "owner_name": "User",
        "schema": json.dumps(schema["dimensions"], indent=2),
        "file_context": f"{len(conv_files)} conversation files",
    })

    # Parse conversations first (needed for both discovery and chunking)
    # Skips files whose output is already up-to-date
    parsed_dir = output_dir / "parsed"
    from bud.stages.parse import parse_conversations_file, parse_all

    console.print("[cyan]→ Parsing conversations[/cyan]")
    total_conversations = parse_all(data_dir, parsed_dir, force=False)

    # Load parsed conversations
    parsed_files = sorted(parsed_dir.glob("conversations_*.jsonl"))
    all_conversations = []
    for pf in parsed_files:
        with open(pf) as f:
            for line in f:
                all_conversations.append(json.loads(line.strip()))

    console.print(f"[green]✓ Parsed {total_conversations} conversations[/green]\n")

    # --- Discovery (inline or load existing) ---
    # --discover N  : run N iterations of discovery, then use the map for chunking
    # --with-discovery : (legacy) load existing map without running new iterations
    use_discovery = discover is not None or with_discovery
    concept_map_summary = None

    if use_discovery:
        from bud.stages.discover import DiscoveryMap, run_discovery

        dm = DiscoveryMap(index_mgr.discovery_map_path).load()
        prior = dm.iterations_completed

        if discover is not None and discover > 0:
            console.print(
                f"[cyan]→ Running inline discovery "
                f"({discover} iteration{'s' if discover != 1 else ''}"
                f"{f', resuming from {prior}' if prior > 0 else ''})[/cyan]"
            )

            from rich.progress import Progress as DiscProgress, SpinnerColumn, TextColumn, TimeElapsedColumn

            with DiscProgress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
                refresh_per_second=8,
            ) as disc_progress:
                disc_task = disc_progress.add_task("[cyan]Discovery...[/cyan]", total=None)

                def _disc_on_sampling(iter_num, max_iter):
                    disc_progress.update(
                        disc_task,
                        description=f"[dim]iter {iter_num}/{max_iter}  waiting for LLM...[/dim]",
                    )

                def _disc_on_iteration(iter_num, score, cmap, prompt_tokens=0, response_tokens=0):
                    signals = len(cmap.data.get("boundary_signals", []))
                    archetypes = len(cmap.data.get("chunk_archetypes", []))
                    bar = "▓" * int(score * 10) + "░" * (10 - int(score * 10))
                    color = "green" if score >= discover_stability else "yellow"
                    total_tok = prompt_tokens + response_tokens
                    tok_str = f"{total_tok / 1000:.1f}k" if total_tok >= 1000 else str(total_tok)
                    disc_progress.update(
                        disc_task,
                        description=(
                            f"iter {iter_num}/{discover}  "
                            f"[{color}]{bar}[/{color}] "
                            f"stability={score:.2f}  "
                            f"[dim]signals={signals} archetypes={archetypes} tokens={tok_str}[/dim]"
                        ),
                    )

                dm = run_discovery(
                    parsed_dir=str(parsed_dir),
                    concept_map=dm,
                    llm=llm,
                    n_samples=5,
                    stability_threshold=discover_stability,
                    max_iterations=discover,
                    on_iteration=_disc_on_iteration,
                    on_sampling=_disc_on_sampling,
                )

            console.print(
                f"[green]✓ Discovery complete "
                f"({dm.iterations_completed} total iterations, "
                f"stability={dm.stability_score:.2f})[/green]"
            )

        if dm.is_empty():
            console.print(
                "[yellow]⚠ Discovery map is empty — chunking without it.[/yellow]\n"
            )
        else:
            concept_map_summary = dm.to_summary()
            if discover is None:
                # --with-discovery (legacy): just loaded, didn't run new iterations
                console.print(
                    f"[green]✓ Loaded discovery map "
                    f"({dm.iterations_completed} iterations, "
                    f"stability={dm.stability_score:.2f})[/green]"
                )
        console.print("")

    # Process in batches
    from bud.stages.chunk import chunk_conversation, chunk_conversations_batch
    from bud.stages.embed import embed_chunks, clear_embed_queue

    use_batch_chunk = llm.concurrency > 1
    if use_batch_chunk:
        console.print(f"[dim]LLM concurrency: {llm.concurrency}[/dim]")

    total_chunks = 0
    errors = 0

    n_convs = len(all_conversations)
    n_batches = max(1, (n_convs + batch_size - 1) // batch_size)

    console.print(f"[cyan]→ Chunking and embedding {n_convs} conversations[/cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=28),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=8,
    ) as progress:
        conv_task  = progress.add_task("[cyan]Conversations[/cyan]", total=n_convs)
        op_task    = progress.add_task("", total=None)

        conv_idx = 0
        for i in range(0, n_convs, batch_size):
            batch = all_conversations[i:i + batch_size]
            batch_num = i // batch_size + 1
            filename = f"conversations_{batch_num}.jsonl"

            # Skip already-processed batches (auto-resume)
            if tracker.is_complete(filename, batch_num):
                progress.update(conv_task, advance=len(batch))
                conv_idx += len(batch)
                progress.update(
                    op_task,
                    description=f"[dim]skipped batch {batch_num}/{n_batches} (already done)[/dim]",
                )
                continue

            # --- Chunking ---
            batch_chunks = []

            if use_batch_chunk:
                # Concurrent chunking
                progress.update(
                    op_task,
                    description=(
                        f"[yellow]chunk[/yellow]  "
                        f"batch {batch_num}/{n_batches}  "
                        f"[dim]{len(batch)} conversations (×{llm.concurrency} concurrent)[/dim]"
                    ),
                )
                completed = [0]

                def _on_chunk_done(idx, chunks_result, error, _bn=batch_num):
                    completed[0] += 1
                    progress.update(conv_task, advance=1)
                    progress.update(
                        op_task,
                        description=(
                            f"[yellow]chunk[/yellow]  "
                            f"batch {_bn}/{n_batches}  "
                            f"[dim]{completed[0]}/{len(batch)} done[/dim]"
                        ),
                    )

                batch_results = chunk_conversations_batch(
                    batch, schema, llm, config, system_prompt,
                    prompt_preset=prompt,
                    schema_version=schema["version"],
                    concept_map_summary=concept_map_summary,
                    on_complete=_on_chunk_done,
                )
                for conv_chunks in batch_results:
                    batch_chunks.extend(conv_chunks)
                    for chunk in conv_chunks:
                        for proposal in chunk.get("schema_proposals", []):
                            schema_mgr.propose_candidate(
                                proposal["dimension"],
                                proposal["value"],
                                proposal.get("rationale", "")
                            )
                conv_idx += len(batch)
            else:
                # Sequential chunking (local LLM)
                for conv in batch:
                    conv_idx += 1
                    name = (conv.get("conversation_name") or conv["id"])[:52]
                    progress.update(
                        op_task,
                        description=(
                            f"[yellow]chunk[/yellow]  "
                            f"batch {batch_num}/{n_batches}  "
                            f"[dim]{name}[/dim]"
                        ),
                    )
                    try:
                        chunks = chunk_conversation(
                            conv, schema, llm, config, system_prompt, prompt_preset=prompt,
                            schema_version=schema["version"],
                            concept_map_summary=concept_map_summary,
                        )
                        batch_chunks.extend(chunks)

                        for chunk in chunks:
                            for proposal in chunk.get("schema_proposals", []):
                                schema_mgr.propose_candidate(
                                    proposal["dimension"],
                                    proposal["value"],
                                    proposal.get("rationale", "")
                                )
                    except Exception as e:
                        errors += 1
                        progress.print(f"  [red]✗ chunk error  {conv['id']}: {e}[/red]")

                    progress.update(conv_task, advance=1)

            total_chunks += len(batch_chunks)

            # --- Embedding ---
            all_chunks_for_embed = failed_chunks + batch_chunks
            n_embed = len(all_chunks_for_embed)

            def _on_chunk(done, total, _batch=batch_num):
                progress.update(
                    op_task,
                    description=(
                        f"[blue]embed[/blue]   "
                        f"batch {_batch}/{n_batches}  "
                        f"[dim]{done}/{total} chunks[/dim]"
                    ),
                )

            progress.update(
                op_task,
                description=(
                    f"[blue]embed[/blue]   "
                    f"batch {batch_num}/{n_batches}  "
                    f"[dim]0/{n_embed} chunks[/dim]"
                ),
            )
            batch_embed_errors: list[str] = []

            def _on_error(chunk, error_msg, _errs=batch_embed_errors):
                if len(_errs) < 1:
                    _errs.append(error_msg)

            failed = embed_chunks(
                all_chunks_for_embed, embedding_client, store,
                index_mgr.embed_queue_path,
                on_chunk=_on_chunk,
                on_error=_on_error,
                max_chars=_model_cfg["max_embed_chars"],
            )

            if failed < len(all_chunks_for_embed):
                clear_embed_queue(index_mgr.embed_queue_path)
            failed_chunks = []

            embedded = n_embed - failed
            if failed > 0:
                first_err = batch_embed_errors[0] if batch_embed_errors else "unknown error"
                progress.print(
                    f"  [yellow]⚠  batch {batch_num}: {failed}/{n_embed} chunks failed to embed[/yellow]\n"
                    f"    [dim]{first_err}[/dim]"
                )
            progress.update(
                op_task,
                description=(
                    f"[green]✓ batch {batch_num}/{n_batches}[/green]  "
                    f"[dim]{len(batch_chunks)} chunks  "
                    f"{embedded} embedded  "
                    f"{total_chunks} total  "
                    f"{errors} errors[/dim]"
                ),
            )

            tracker.mark_complete(filename, batch_num)

            # Incremental save — crash-safe: index is persisted after every
            # batch so an interruption only loses the current (incomplete) batch.
            if store:
                store.save()

        progress.update(
            op_task,
            description=(
                f"[bold green]done[/bold green]  "
                f"[dim]{total_chunks} chunks  "
                f"{store.count() if store else 0} in index  "
                f"{errors} errors[/dim]"
            ),
        )

    # Apply schema evolution
    promoted = schema_mgr.apply_promotions(config)
    if promoted:
        console.print(f"\n[yellow]Schema evolved! Promoted: {', '.join(promoted)}[/yellow]")
        schema = schema_mgr.load()

    # Final index save
    if store:
        store.save()
        console.print(f"\n[green]✓ Final index saved ({store.count()} total chunks)[/green]")

    # Warn about queued embed failures
    from bud.stages.embed import load_embed_queue
    queued = load_embed_queue(index_mgr.embed_queue_path)
    if queued:
        console.print(
            f"\n[yellow]⚠  {len(queued)} chunk(s) failed to embed and are queued for retry.[/yellow]\n"
            f"   Fix the embedding service, then run: [bold]bud process --resume[/bold]\n"
            f"   Queue file: [dim]{index_mgr.embed_queue_path}[/dim]"
        )

    # Print summary
    console.print(f"\n[bold cyan]Pipeline Complete![/bold cyan]")
    console.print(f"  Conversations: {total_conversations}")
    console.print(f"  Chunks: {total_chunks}")
    console.print(f"  Errors: {errors}")
    console.print(f"  Schema version: v{schema['version']}")

    if promoted:
        console.print(f"  Promoted: {', '.join(promoted)}")

    console.print(f"\n[dim]Output: {output_dir}[/dim]")


@main.command()
@click.argument("query_text")
@click.option(
    "--k",
    type=click.INT,
    default=5,
    help="Number of results to return (default: 5)",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(file_okay=False, resolve_path=True),
    help="Path to output directory (overrides config)",
)
def query(query_text, k, output_dir):
    """Query the vector index for relevant context.

    SEARCH_TEXT: The search query to find relevant conversation context
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Load config
    config = load_config()

    # Use override or config for output_dir
    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = get_output_dir()

    console.print(f"\n[bold cyan]Bud RAG Pipeline - Query[/bold cyan]\n")
    console.print(f"[bold]Query:[/bold] {query_text}")
    console.print(f"[bold]Top-K:[/bold] {k}\n")

    # Build paths
    index_dir = output_dir / "index"
    index_path = str(index_dir / "chunks")
    metadata_path = index_dir / "chunks_metadata.jsonl"

    # Load FAISS index
    from bud.lib.store import VectorStore
    store = VectorStore(index_path, dim=0)  # dim=0 will be inferred from loaded index

    try:
        store.load()
    except Exception as e:
        console.print(f"[red]Error loading index: {e}[/red]")
        console.print(f"[dim]Make sure you've run 'bud process' first.[/dim]")
        return

    if store.count() == 0:
        console.print("[red]No chunks found in the index.[/red]")
        console.print(f"[dim]Index path: {index_path}[/dim]")
        return

    console.print(f"[green]✓ Loaded index with {store.count()} chunks[/green]\n")

    # Embed the user query
    from bud.lib.embeddings import EmbeddingClient
    embedding_client = EmbeddingClient(config)

    try:
        query_embedding = embedding_client.embed(query_text)
    except Exception as e:
        console.print(f"[red]Error embedding query: {e}[/red]")
        return

    # Search for top-k similar chunks
    search_results = store.search(query_embedding, k)

    if not search_results:
        console.print("[yellow]No matching chunks found.[/yellow]")
        return

    # Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(search_results, 1):
        context_parts.append(f"[{i}] {chunk.get('text', '')}")

    context = "\n\n".join(context_parts)

    # Generate answer using LLM
    from bud.lib.llm import LLMClient
    llm = LLMClient(config)

    # Format prompt with context and query
    prompt = f"""You are a helpful assistant answering questions based on conversation context.

Context (from conversation history):
{context}

---
User Question: {query_text}

Instructions:
- Answer based ONLY on the context above
- Be concise and focused
- If context is unclear or incomplete, say so
- Cite source by rank number when relevant
"""

    try:
        answer = llm.complete("You are a helpful assistant.", prompt)
    except Exception as e:
        console.print(f"[red]Error generating answer: {e}[/red]")
        answer = "Unable to generate answer due to LLM error."

    # Display results as table
    table = Table(title="Search Results")
    table.add_column("Rank", style="cyan", no_wrap=True)
    table.add_column("Score", style="magenta", no_wrap=True)
    table.add_column("Source", style="green")

    for i, chunk in enumerate(search_results, 1):
        rank = str(i)
        score = chunk.get("score", "N/A")
        source = chunk.get("source_file", chunk.get("source", "conversations.jsonl"))
        table.add_row(rank, str(score), source)

    console.print(table)

    # Display answer
    console.print(f"\n[bold]Answer:[/bold]")
    console.print(answer)

    console.print(f"\n[dim]Query completed.[/dim]")


@main.command()
@click.option(
    "--output-dir", "-o",
    type=click.Path(file_okay=False, resolve_path=True),
    help="Path to output directory (overrides config)",
)
def status(output_dir):
    """Show pipeline status and configuration info."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()

    # Use override or config
    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = get_output_dir()

    config = load_config()

    # Bud header
    console.print("\n[bold cyan]Bud RAG Pipeline[/bold cyan]")
    console.print(f"[dim]Version: {__version__}[/dim]\n")

    # Configuration panel
    config_panel = Panel(
        f"[bold]Config file:[/bold] {CONFIG_FILE}\n"
        f"[bold]Data directory:[/bold] {config.get('data_dir', 'Not set')}\n"
        f"[bold]Output directory:[/bold] {config.get('output_dir', 'Not set')}\n"
        f"\n[bold]LLM:[/bold]\n"
        f"  Provider: {config.get('llm', {}).get('provider', 'Not set')}\n"
        f"  Model: {config.get('llm', {}).get('model', 'Not set')}\n"
        f"  Base URL: {config.get('llm', {}).get('base_url', 'Not set')}\n"
        f"\n[bold]Embeddings:[/bold]\n"
        f"  Provider: {config.get('embeddings', {}).get('provider', 'Not set')}\n"
        f"  Model: {config.get('embeddings', {}).get('model', 'Not set')}\n"
        f"  Base URL: {config.get('embeddings', {}).get('base_url', 'Not set')}",
        title="Configuration",
        border_style="green",
    )
    console.print(config_panel)

    # Vector index status
    index_dir = output_dir / "index"
    index_path = str(index_dir / "chunks")
    faiss_path = f"{index_path}.faiss"
    metadata_path = f"{index_path}_metadata.jsonl"

    index_exists = os.path.exists(faiss_path)
    if index_exists:
        index_size = os.path.getsize(faiss_path)
        metadata_size = os.path.getsize(metadata_path) if os.path.exists(metadata_path) else 0
    else:
        index_size = 0
        metadata_size = 0

    # Get chunk count from index
    chunk_count = 0
    if index_exists:
        try:
            from bud.lib.store import VectorStore
            store = VectorStore(index_path, dim=0)
            store.load()
            chunk_count = store.count()
        except Exception:
            chunk_count = 0

    # Pipeline status table
    table = Table(title="Pipeline Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="white")

    # Vector Index status
    if index_exists:
        index_status = "[green]Ready[/green]"
        index_details = f"{index_dir}\n{chunk_count} chunks\n{index_size:,} bytes (index)"
        if metadata_size > 0:
            index_details += f", {metadata_size:,} bytes (metadata)"
    else:
        index_status = "[yellow]Not built[/yellow]"
        index_details = f"{index_dir}\nRun 'bud process' to build"

    table.add_row("Vector Index", index_status, index_details)

    # Knowledge Base status
    kb_dir = output_dir / "knowledge_base"
    kb_exists = kb_dir.exists() and any(kb_dir.iterdir())
    if kb_exists:
        kb_status = "[green]Ready[/green]"
        kb_details = str(kb_dir)
    else:
        kb_status = "[yellow]Empty[/yellow]"
        kb_details = str(kb_dir)

    table.add_row("Knowledge Base", kb_status, kb_details)

    # Last Processed - get from progress file
    progress_path = output_dir / "progress.json"
    if progress_path.exists():
        try:
            with open(progress_path) as f:
                progress = json.load(f)
            total_files = len(progress)
            total_batches = sum(len(data.get("completed", [])) for data in progress.values())
            last_file = list(progress.keys())[-1] if progress else "N/A"
            last_status = "Complete" if total_batches == sum(
                len(data.get("completed", [])) for data in progress.values()
            ) and all(
                len(data.get("failed", {})) == 0 for data in progress.values()
            ) else "Partial"
            last_processed = f"{last_file} (batch {total_batches})\n{last_status}"
        except Exception:
            last_processed = "N/A"
    else:
        last_processed = "Never (no processing done)"

    table.add_row("Last Processed", "", last_processed)

    console.print(table)

    # Validation
    is_valid, errors = validate_config(config)

    if is_valid:
        console.print("\n[green]Configuration is valid.[/green]")
    else:
        console.print("\n[red]Configuration errors:[/red]")
        for error in errors:
            console.print(f"  - {error}")


if __name__ == "__main__":
    main()
