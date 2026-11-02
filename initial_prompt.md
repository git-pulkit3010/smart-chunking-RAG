# RAG with intelligent chunking

#Projects/smart-chunking


Below is a complete design and implementation plan for a CLI RAG application that ingests research‑paper PDFs, applies an intelligent, layout‑aware chunking strategy, and uses OpenRouter for answer generation. The code is deliberately modular so you can swap the OpenRouter model ID, embedding backend, or vector store later.

---

1. ### Architecture overview

```
User → CLI (ingest / query)
         │
         ├─ PDF Ingestion (PyMuPDF)
         │    └─ Block‑level extraction with font‑size/bold metadata
         │
         ├─ Intelligent Chunker
         │    ├─ Section‑aware splitting (headings)
         │    ├─ Semantic chunking for long sections
         │    └─ Recursive fallback
         │
         ├─ Embedding & Indexing
         │    └─ Local sentence‑transformer → Chroma DB
         │
         └─ Retrieval + OpenRouter Chat
              └─ Top‑k chunks → prompt → streaming answer
```

The “intelligence” of the chunker is a decision tree that inspects the PDF’s structural signals and picks the strategy that best preserves context for that particular paper.

---

2. ### Intelligent chunking strategy (per‑use‑case)

Research papers are highly structured, so the best default is section‑aware chunking combined with semantic splitting for oversized sections. The chunker decides as follows:

1. Detect headings – using font size and boldness from PyMuPDF’s block extraction.
2. If clear headings exist → split by section (Abstract, Introduction, Methods, …). Keep each section whole if it fits within a token budget; if a section is too long, apply semantic chunking inside that section (group sentences by embedding similarity).
3. If no headings → use pure semantic chunking (cosine similarity between consecutive sentence embeddings).
4. If even semantic chunking is unavailable (e.g., embedding model missing) → fall back to recursive character/token splitting.

This matches recent findings that semantic layout chunking – combining structural cues with semantic coherence – outperforms purely semantic or purely structural baselines on formal documents.

For tables and figures you can optionally use pdfplumber or marker to extract them as separate chunks, but the core pipeline below focuses on text.

---

3. ### Implementation

3.1 Dependencies

```bash
pip install typer chromadb sentence-transformers pymupdf openrouter python-dotenv
```

· PyMuPDF (fitz) – fast, block‑level text extraction with font metadata.
· ChromaDB – lightweight local vector store (no server required).
· sentence‑transformers – local embedding model (all-MiniLM-L6-v2 is a good default).
· OpenRouter SDK – official Python client for chat completions.

---

3.2 Code: paper_rag.py

```python
#!/usr/bin/env python3
"""
paper_rag.py – CLI RAG for research papers.
Usage:
    python paper_rag.py ingest paper.pdf
    python paper_rag.py query "What is the main contribution?"
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF
import typer
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from openrouter import OpenRouter
from dotenv import load_dotenv

load_dotenv()
app = typer.Typer()

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "papers"
EMBED_MODEL = "all-MiniLM-L6-v2"          # swap if needed
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")  # fill later
MAX_CHUNK_TOKENS = 512                     # soft limit for section chunks

# ------------------------------------------------------------------
# PDF extraction (block‑level with font info)
# ------------------------------------------------------------------
def extract_blocks(pdf_path: str) -> List[Dict[str, Any]]:
    """Return a flat list of blocks with page, text, font_size, is_bold."""
    doc = fitz.open(pdf_path)
    blocks_out = []
    for page_num, page in enumerate(doc, start=1):
        for block in page.get_text("dict")["blocks"]:
            if "lines" not in block:
                continue
            text_parts = []
            max_size = 0.0
            bold = False
            for line in block["lines"]:
                for span in line["spans"]:
                    text_parts.append(span["text"])
                    max_size = max(max_size, span["size"])
                    if "Bold" in span.get("font", ""):
                        bold = True
            txt = "".join(text_parts).strip()
            if txt:
                blocks_out.append({
                    "page": page_num,
                    "text": txt,
                    "font_size": max_size,
                    "is_bold": bold,
                })
    doc.close()
    return blocks_out

# ------------------------------------------------------------------
# Intelligent chunker
# ------------------------------------------------------------------
def is_heading(block: Dict[str, Any], avg_font: float) -> bool:
    """Heuristic: larger font or bold short lines are likely headings."""
    if block["font_size"] > avg_font + 1.5 and len(block["text"]) < 100:
        return True
    if block["is_bold"] and block["font_size"] >= avg_font and len(block["text"]) < 80:
        return True
    return False

def split_into_sections(blocks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Group blocks into sections delimited by headings."""
    if not blocks:
        return []
    avg_font = sum(b["font_size"] for b in blocks) / len(blocks)
    sections = []
    current_heading = "Document Start"
    current_text = []
    for b in blocks:
        if is_heading(b, avg_font):
            if current_text:
                sections.append({"heading": current_heading,
                                 "text": "\n".join(current_text)})
            current_heading = b["text"]
            current_text = []
        else:
            current_text.append(b["text"])
    if current_text:
        sections.append({"heading": current_heading,
                         "text": "\n".join(current_text)})
    return sections

def semantic_split(text: str, embed_model: SentenceTransformer,
                   threshold: float = 0.65, max_tokens: int = MAX_CHUNK_TOKENS) -> List[str]:
    """
    Split a long section into semantically coherent chunks.
    Simple approach: sentence‑level embeddings + break where similarity drops.
    """
    # Very lightweight sentence tokenizer (good enough for papers)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) <= 1:
        return [text]

    embeddings = embed_model.encode(sentences, normalize_embeddings=True)
    chunks, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        sim = float(embeddings[i - 1] @ embeddings[i])
        # rough token estimate
        approx_tokens = sum(len(s.split()) for s in current + [sentences[i]])
        if sim < threshold or approx_tokens > max_tokens:
            chunks.append(" ".join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    if current:
        chunks.append(" ".join(current))
    return chunks

def chunk_paper(blocks: List[Dict[str, Any]],
                embed_model: SentenceTransformer) -> List[Dict[str, str]]:
    """
    Return a list of chunks with metadata (heading, text, source).
    Strategy: section‑aware first, semantic inside long sections.
    """
    sections = split_into_sections(blocks)
    chunks = []
    for sec in sections:
        # rough token count
        n_tokens = len(sec["text"].split())
        if n_tokens <= MAX_CHUNK_TOKENS:
            chunks.append({"heading": sec["heading"], "text": sec["text"]})
        else:
            for sub in semantic_split(sec["text"], embed_model):
                chunks.append({"heading": sec["heading"], "text": sub})
    return chunks

# ------------------------------------------------------------------
# Indexing
# ------------------------------------------------------------------
def get_chroma():
    client = PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(COLLECTION_NAME)

def ingest(pdf_path: str):
    typer.echo(f"Ingesting {pdf_path} …")
    blocks = extract_blocks(pdf_path)
    if not blocks:
        typer.echo("No text extracted – is this a scanned PDF?")
        raise typer.Exit(1)

    embed_model = SentenceTransformer(EMBED_MODEL)
    chunks = chunk_paper(blocks, embed_model)

    col = get_chroma()
    ids, texts, metadatas = [], [], []
    for i, ch in enumerate(chunks):
        cid = f"{Path(pdf_path).stem}_{i}"
        ids.append(cid)
        texts.append(ch["text"])
        metadatas.append({"heading": ch["heading"], "source": pdf_path})

    embeddings = embed_model.encode(texts, normalize_embeddings=True).tolist()
    col.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    typer.echo(f"Indexed {len(chunks)} chunks from {len(blocks)} blocks.")

# ------------------------------------------------------------------
# Retrieval + OpenRouter answer
# ------------------------------------------------------------------
def retrieve(query: str, k: int = 5) -> List[Dict[str, Any]]:
    embed_model = SentenceTransformer(EMBED_MODEL)
    q_emb = embed_model.encode(query, normalize_embeddings=True).tolist()
    col = get_chroma()
    results = col.query(query_embeddings=[q_emb], n_results=k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return [{"text": d, "meta": m} for d, m in zip(docs, metas)]

def answer(query: str):
    hits = retrieve(query)
    if not hits:
        typer.echo("No relevant chunks found. Have you ingested a paper?")
        return

    context = "\n\n---\n\n".join(
        f"[{h['meta']['heading']}]\n{h['text']}" for h in hits
    )
    system_prompt = (
        "You are a research assistant. Answer the user's question using only "
        "the provided context from the paper. Cite section headings when useful. "
        "If the context does not contain the answer, say so."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        typer.echo("Set OPENROUTER_API_KEY environment variable.")
        raise typer.Exit(1)

    with OpenRouter(api_key=api_key) as client:
        typer.echo("\nAnswer (streaming):\n")
        res = client.chat.send(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
        )
        with res as stream:
            for event in stream:
                delta = event.choices[0].delta.content
                if delta:
                    typer.echo(delta, nl=False)
        typer.echo()

# ------------------------------------------------------------------
# CLI commands
# ------------------------------------------------------------------
@app.command()
def ingest_cmd(pdf: str = typer.Argument(..., help="Path to PDF")):
    ingest(pdf)

@app.command()
def query_cmd(q: str = typer.Argument(..., help="Question about the paper")):
    answer(q)

if __name__ == "__main__":
    app()
```

---

4. ### How the “intelligent” decision works in practice

PDF characteristic Detected by Chunking action
Clear headings (Abstract, 1. Introduction, …) is_heading – font size > average + bold short lines Split by headings; long sections → semantic sub‑split
No headings / continuous prose split_into_sections returns one big section Pure semantic chunking (sentence‑embedding similarity)
Very long single section (e.g., a long Related Work) Token count > MAX_CHUNK_TOKENS Semantic sub‑split inside that section
Extremely sparse/odd layout Fallback (not explicitly coded, but you can add a recursive‑character fallback) Recursive character split

This mirrors the semantic layout chunking approach shown to preserve both structural integrity and semantic flow in formal documents.

---

5. ### Configuration points you can fill later

· OpenRouter model ID – set OPENROUTER_MODEL env var (e.g., anthropic/claude-3.5-sonnet, google/gemini-2.5-pro, etc.).
· Embedding backend – currently local all-MiniLM-L6-v2. You can replace it with an OpenRouter‑hosted embedding model if available, or a larger local model (BAAI/bge-large-en).
· Vector store – ChromaDB is file‑based and zero‑config; swap for FAISS or PGVector if you need scale.
· Chunk size / similarity threshold – tune MAX_CHUNK_TOKENS and semantic_split(threshold=…) per discipline (e.g., smaller chunks for dense methods sections).

---

6. Quick‑start

```bash
export OPENROUTER_API_KEY="sk-or-…"
export OPENROUTER_MODEL="openai/gpt-4o-mini"   # replace later

python paper_rag.py ingest my_paper.pdf
python paper_rag.py query "What dataset did the authors use?"
```

The first run downloads the sentence‑transformer model; subsequent runs are fast.

---

7. Why this satisfies “best possible chunking per use case”

· Research papers are structured – section‑aware splitting keeps Abstract, Methods, Results, etc. semantically intact, which is critical for retrieval.
· Long sections are not forced into arbitrary token windows – semantic sub‑splitting respects topic shifts inside a section.
· The strategy is chosen automatically based on the actual layout of each PDF, not a hard‑coded rule that assumes every paper looks the same.
· Fallback safety – if heading detection fails, semantic chunking still produces coherent passages.

If you later need table‑aware or figure‑aware chunking, you can extend extract_blocks with pdfplumber or marker and treat tables as separate chunk types, but the core decision tree above already covers the majority of research‑paper text.
