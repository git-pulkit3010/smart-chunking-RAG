#!/usr/bin/env python3
"""
paper_rag.py – CLI RAG for research papers with intelligent layout-aware chunking.

Usage:
    python paper_rag.py ingest paper.pdf
    python paper_rag.py query "What is the main contribution?"
    python paper_rag.py list-papers
    python paper_rag.py clear
"""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings
from src.extractor import extract_blocks
from src.embeddings import get_embedding_backend
from src.chunker import chunk_paper
from src.vectorstore import PaperVectorStore
from src.generator import stream_answer

app = typer.Typer(
    help="Intelligent Section-Aware RAG for Research Papers (NVIDIA NIM generation, OpenRouter embeddings)",
    rich_markup_mode="rich",
)
console = Console()


@app.command(name="ingest")
def ingest_cmd(
    pdf: str = typer.Argument(..., help="Path to research paper PDF file"),
    max_tokens: Optional[int] = typer.Option(
        None, "--max-tokens", "-t", help="Max token size for a chunk before semantic sub-splitting"
    ),
    threshold: Optional[float] = typer.Option(
        None, "--threshold", "-s", help="Cosine similarity threshold for semantic boundary detection"
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p", help="Embedding provider: 'openrouter' or 'local'"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Embedding model (e.g. 'openai/text-embedding-3-small' or 'all-MiniLM-L6-v2')"
    ),
):
    """
    Ingest a research paper PDF, apply layout-aware intelligent chunking, and index it into ChromaDB.
    """
    settings = get_settings()
    pdf_path = Path(pdf)

    if not pdf_path.is_file():
        console.print(f"[bold red]Error:[/bold red] File not found: [yellow]{pdf}[/yellow]")
        raise typer.Exit(1)

    max_tok = max_tokens or settings.max_chunk_tokens
    sim_thresh = threshold if threshold is not None else settings.semantic_similarity_threshold
    embed_prov = provider or settings.embed_provider
    embed_model_name = model or settings.embed_model

    console.print(Panel(
        f"[bold cyan]Document:[/bold cyan] {pdf_path.name}\n"
        f"[bold cyan]Embedding Provider:[/bold cyan] {embed_prov}\n"
        f"[bold cyan]Embedding Model:[/bold cyan] {embed_model_name}\n"
        f"[bold cyan]Max Tokens / Chunk:[/bold cyan] {max_tok}\n"
        f"[bold cyan]Similarity Threshold:[/bold cyan] {sim_thresh}",
        title="[bold green]Ingesting Research Paper[/bold green]",
        border_style="green",
    ))

    # Step 1: Extract block-level text & layout metadata
    with console.status("[bold blue]Extracting text and layout metadata via PyMuPDF...[/bold blue]"):
        try:
            blocks = extract_blocks(str(pdf_path))
        except Exception as e:
            console.print(f"[bold red]Extraction failed:[/bold red] {e}")
            raise typer.Exit(1)

    if not blocks:
        console.print("[bold red]No text extracted[/bold red] – is this a scanned or raster PDF?")
        raise typer.Exit(1)

    console.print(f"Extracted [bold]{len(blocks)}[/bold] layout blocks.")

    # Step 2: Initialize Embedding Backend
    with console.status("[bold blue]Initializing embedding backend...[/bold blue]"):
        try:
            embed_backend = get_embedding_backend(
                provider=embed_prov,
                model=embed_model_name,
                api_key=settings.openrouter_api_key,
            )
        except Exception as e:
            console.print(f"[bold red]Embedding backend initialization failed:[/bold red]\n{e}")
            raise typer.Exit(1)

    # Step 3: Intelligent Layout-Aware & Semantic Chunking
    with console.status("[bold blue]Executing intelligent chunking decision tree...[/bold blue]"):
        chunks = chunk_paper(
            blocks=blocks,
            embed_backend=embed_backend,
            max_tokens=max_tok,
            threshold=sim_thresh,
        )

    console.print(f"Intelligently segmented into [bold green]{len(chunks)}[/bold green] chunks.")

    # Show chunk breakdown
    strategy_counts = {}
    for ch in chunks:
        strat = ch.get("strategy", "unknown")
        strategy_counts[strat] = strategy_counts.get(strat, 0) + 1

    strat_summary = ", ".join(f"{count} {strat}" for strat, count in strategy_counts.items())
    console.print(f"Chunking breakdown: [dim]{strat_summary}[/dim]")

    # Step 4: Generate Embeddings
    chunk_texts = [ch["text"] for ch in chunks]
    with console.status(f"[bold blue]Computing embeddings via {embed_backend.model_name}...[/bold blue]"):
        try:
            embeddings = embed_backend.embed_documents(chunk_texts)
        except Exception as e:
            console.print(f"[bold red]Embedding generation failed:[/bold red] {e}")
            raise typer.Exit(1)

    # Step 5: Index into ChromaDB
    with console.status("[bold blue]Storing vectors and metadata in ChromaDB...[/bold blue]"):
        store = PaperVectorStore(
            chroma_path=settings.chroma_path,
            collection_name=settings.collection_name,
        )
        total_added = store.add_paper(
            pdf_path=str(pdf_path),
            chunks=chunks,
            embeddings=embeddings,
        )

    console.print(
        f"[bold green]Successfully indexed {total_added} chunks[/bold green] for "
        f"[cyan]{pdf_path.name}[/cyan] into ChromaDB at [dim]{settings.chroma_path}[/dim].\n"
    )


@app.command(name="query")
def query_cmd(
    q: str = typer.Argument(..., help="Question to ask about the paper"),
    k: Optional[int] = typer.Option(
        None, "--k", "-k", help="Number of context chunks to retrieve"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="NVIDIA generation model (e.g. 'meta/llama-3.3-70b-instruct')"
    ),
    show_sources: bool = typer.Option(
        True, "--show-sources/--no-show-sources", help="Display retrieved chunk citations"
    ),
):
    """
    Query the indexed research paper(s). Retrieves relevant chunks and streams the answer via NVIDIA.
    """
    settings = get_settings()
    k_val = k or settings.top_k_retrieval
    llm_model = (model or settings.nvidia_model).strip()

    store = PaperVectorStore(
        chroma_path=settings.chroma_path,
        collection_name=settings.collection_name,
    )

    if store.count() == 0:
        console.print(
            "[bold yellow]No papers found in the vector database.[/bold yellow]\n"
            "Please ingest a paper first: [green]python paper_rag.py ingest <paper.pdf>[/green]"
        )
        raise typer.Exit(1)

    # Step 1: Embed the user question
    with console.status("[bold blue]Retrieving relevant context...[/bold blue]"):
        try:
            embed_backend = get_embedding_backend(
                provider=settings.embed_provider,
                model=settings.embed_model,
                api_key=settings.openrouter_api_key,
            )
            q_emb = embed_backend.embed_query(q)
        except Exception as e:
            console.print(f"[bold red]Failed to embed query:[/bold red] {e}")
            raise typer.Exit(1)

        # Step 2: Query ChromaDB
        hits = store.query(query_embedding=q_emb, k=k_val)

    if not hits:
        console.print("[yellow]No relevant chunks retrieved from the vector database.[/yellow]")
        return

    # Optionally display citations / sources
    if show_sources:
        table = Table(title="Retrieved Context Chunks", border_style="dim")
        table.add_column("#", justify="right", style="cyan", width=3)
        table.add_column("Paper", style="magenta")
        table.add_column("Section Heading", style="green")
        table.add_column("Page", justify="center", style="yellow", width=6)
        table.add_column("Preview", style="white")

        for idx, hit in enumerate(hits, start=1):
            meta = hit.get("meta", {})
            preview = hit.get("text", "").replace("\n", " ")[:90] + "..."
            table.add_row(
                str(idx),
                str(meta.get("paper_title", "Unknown")),
                str(meta.get("heading", "Document Start")),
                str(meta.get("page", 1)),
                preview,
            )
        console.print(table)
        console.print()

    # Step 3: Stream Answer from NVIDIA
    console.print(f"[bold cyan]Answer (model: {llm_model}):[/bold cyan]\n")
    try:
        stream_answer(
            query=q,
            hits=hits,
            api_key=settings.nvidia_api_key or "",
            model=llm_model,
            base_url=settings.nvidia_base_url,
            token_callback=lambda token: console.print(token, end=""),
        )
        console.print("\n")
    except Exception as e:
        console.print(f"\n[bold red]Generation Error:[/bold red] {e}")
        if not settings.nvidia_api_key:
            console.print("[yellow]Tip: Add NVIDIA_API_KEY=your_key_here to .env[/yellow]")
        elif not settings.nvidia_model:
            console.print("[yellow]Tip: Add NVIDIA_MODEL=your_model_here to .env[/yellow]")
        raise typer.Exit(1)


@app.command(name="list-papers")
def list_papers_cmd():
    """
    List all research papers currently indexed in the vector database.
    """
    settings = get_settings()
    store = PaperVectorStore(
        chroma_path=settings.chroma_path,
        collection_name=settings.collection_name,
    )
    papers = store.list_papers()

    if not papers:
        console.print("[yellow]No papers currently indexed.[/yellow]")
        return

    table = Table(title="Indexed Research Papers", border_style="cyan")
    table.add_column("Paper File", style="green")
    table.add_column("Chunks", justify="center", style="cyan")
    table.add_column("Pages", justify="center", style="yellow")
    table.add_column("Full Path", style="dim")

    for p in papers:
        table.add_row(
            p["paper_title"],
            str(p["chunk_count"]),
            str(p["max_page"]),
            p["source"],
        )

    console.print(table)
    console.print(f"Total chunks in database: [bold cyan]{store.count()}[/bold cyan]")


@app.command(name="clear")
def clear_cmd(
    force: bool = typer.Option(False, "--force", "-f", help="Bypass confirmation prompt")
):
    """
    Clear all indexed papers and chunks from the vector database.
    """
    settings = get_settings()
    store = PaperVectorStore(
        chroma_path=settings.chroma_path,
        collection_name=settings.collection_name,
    )

    count = store.count()
    if count == 0:
        console.print("[yellow]Database is already empty.[/yellow]")
        return

    if not force:
        confirm = typer.confirm(f"Are you sure you want to delete all {count} chunks?")
        if not confirm:
            console.print("[dim]Operation cancelled.[/dim]")
            raise typer.Abort()

    store.clear()
    console.print("[bold green]Vector database cleared successfully.[/bold green]")


@app.command(name="config")
def show_config_cmd():
    """
    Display current active configuration loaded from .env.
    """
    settings = get_settings()

    def _mask(value: Optional[str]) -> str:
        if not value:
            return "[red]NOT SET[/red]"
        if len(value) > 12:
            return f"{value[:8]}...{value[-4:]}"
        return "[green]SET[/green]"

    masked_openrouter_key = _mask(settings.openrouter_api_key)
    masked_nvidia_key = _mask(settings.nvidia_api_key)

    table = Table(title="Smart Chunking RAG Configuration", border_style="blue")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="yellow")

    table.add_row("NVIDIA_API_KEY", masked_nvidia_key)
    table.add_row("NVIDIA_MODEL", settings.nvidia_model or "[red]NOT SET[/red]")
    table.add_row("NVIDIA_BASE_URL", settings.nvidia_base_url)
    table.add_row("OPENROUTER_API_KEY", masked_openrouter_key)
    table.add_row("EMBED_MODEL", settings.embed_model)
    table.add_row("EMBED_PROVIDER", settings.embed_provider)
    table.add_row("CHROMA_PATH", settings.chroma_path)
    table.add_row("COLLECTION_NAME", settings.collection_name)
    table.add_row("MAX_CHUNK_TOKENS", str(settings.max_chunk_tokens))
    table.add_row("SEMANTIC_SIMILARITY_THRESHOLD", str(settings.semantic_similarity_threshold))
    table.add_row("TOP_K_RETRIEVAL", str(settings.top_k_retrieval))

    console.print(table)


if __name__ == "__main__":
    app()
