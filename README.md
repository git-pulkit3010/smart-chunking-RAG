# Smart Chunking RAG for Research Papers

A CLI Retrieval-Augmented Generation (RAG) system specifically designed for research papers. It combines block-level layout analysis with semantic sentence clustering, uses **NVIDIA NIM** for answer generation, and OpenRouter for embeddings.

---

## 1. Key Features

- **Layout-Aware PDF Extraction (PyMuPDF)**: Extracts block-level text along with structural signals (font sizes, boldness, line boxes, page numbers).
- **Adaptive Chunking Engine**: Inspects the PDF layout and picks the strategy per section, so nothing is ever chunked the "wrong" way:
  1. **Section-Aware Chunking**: Preserves structural units (Abstract, Introduction, Methods, Results, Discussion, etc.) intact if within token budget.
  2. **Reference-Aware Chunking**: A bibliography becomes one chunk per citation entry instead of prose.
  3. **Table-Aware Chunking**: A table is split by rows, and every chunk repeats the caption and the header row so it stands alone.
  4. **Semantic Sub-Splitting**: For long sections exceeding `MAX_CHUNK_TOKENS`, computes consecutive sentence embeddings and splits at semantic boundaries where cosine similarity drops - with a minimum chunk size and one sentence of overlap.
  5. **Continuous Prose Strategy**: If no headings exist, applies pure semantic chunking across the entire paper.
  6. **Deterministic Fallback**: Gracefully falls back to paragraph/sentence packing if embeddings are unavailable.
- **Output Invariants**: every chunk is non-empty, within the token budget, not a duplicate, not pathologically small, and carries its heading breadcrumb and page for retrieval.
- **Upgraded OpenRouter Embedding Model**: Powered by `openai/text-embedding-3-small` through OpenRouter's native `/embeddings` API, with optional offline local fallback (`sentence-transformers`).
- **Seamless `.env` Integration**: Automatically loads configuration and API keys from `.env` on every run—no need to run `export` in your shell every time.
- **Local ChromaDB Vector Database**: Zero-config persistent vector store in `./chroma_db`.
- **Streaming Generation**: Streams answers token-by-token from the NVIDIA OpenAI-compatible chat completions endpoint (`https://integrate.api.nvidia.com/v1`), using the model you set in `NVIDIA_MODEL`.

---

## 2. Embedding Model: Why `openai/text-embedding-3-small`?

In `initial_prompt.md`, the default embedding model was local `all-MiniLM-L6-v2`. This codebase upgrades the primary embedder to OpenRouter's hosted **`openai/text-embedding-3-small`**:

| Feature | `all-MiniLM-L6-v2` (Old) | `openai/text-embedding-3-small` (New) |
| :--- | :--- | :--- |
| **Availability on OpenRouter** | Requires local download | **Native OpenRouter endpoint** (`openai/text-embedding-3-small`) |
| **Context Window** | 256 / 512 tokens | **8,192 tokens** (16× larger context) |
| **Embedding Dimensions** | 384 | **1,536 dimensions** |
| **MTEB Benchmark** | 56.3 | **62.3** (substantially higher retrieval accuracy) |
| **Cost on OpenRouter** | Local compute / memory | **$0.02 / 1M tokens** ($0.00002 / 1,000 tokens) |
| **Estimated Cost per Paper** | Free (CPU/GPU load) | **~$0.0002 – $0.0004** (~1/25th of 1 cent for a 20-page paper) |

### Alternative Supported Models:
- **`baai/bge-m3`**: Leading multilingual open-weights model ($0.01 / 1M tokens, 8,194 context).
- **`openai/text-embedding-3-large`**: Highest-dimension OpenAI model ($0.13 / 1M tokens, 1,536 or 3,072 dims).
- **`all-MiniLM-L6-v2`**: Local offline fallback via `sentence-transformers` (set `EMBED_PROVIDER=local` in `.env`).

---

## 3. Installation & Setup

### 3.1. Virtual Environment (`.venv`)
The project uses the `.venv` virtual environment. Activate it with:

```bash
source .venv/bin/activate
```

To install or reinstall dependencies:
```bash
pip install -r requirements.txt
```

### 3.2. Configuration (`.env`)
The program reads all configuration directly from `.env`. **You do not need to run `export` before each command.**

Simply edit `.env` and set your NVIDIA API key (for answering) and OpenRouter API key (for embeddings):
```env
# Answer generation -> NVIDIA
NVIDIA_API_KEY=nvapi-your-key-here
NVIDIA_MODEL=meta/llama-3.3-70b-instruct
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1

# Embeddings -> OpenRouter
OPENROUTER_API_KEY=sk-or-v1-your-key-here
EMBED_MODEL=openai/text-embedding-3-small
EMBED_PROVIDER=openrouter
CHROMA_PATH=./chroma_db
COLLECTION_NAME=papers
MAX_CHUNK_TOKENS=512
SEMANTIC_SIMILARITY_THRESHOLD=0.65
TOP_K_RETRIEVAL=5
```

> **Note**: The answering model is configured entirely through `NVIDIA_API_KEY` / `NVIDIA_MODEL` — edit those two values in `.env` and every command picks them up, no code changes needed.

> **Note**: If you want to run purely offline without an OpenRouter key for embeddings, set `EMBED_PROVIDER=local` and `EMBED_MODEL=all-MiniLM-L6-v2` in `.env`.

---

## 4. Usage Guide

### 4.1. Check Active Configuration
Inspect your current settings loaded from `.env`:
```bash
python paper_rag.py config
```

### 4.2. Ingest a Research Paper PDF
Extracts blocks, runs the intelligent chunker, computes embeddings, and indexes them into ChromaDB:
```bash
python paper_rag.py ingest path/to/paper.pdf
```

Optional CLI flags:
```bash
# Custom token budget and similarity threshold
python paper_rag.py ingest paper.pdf --max-tokens 512 --threshold 0.70

# Use local embedding model for this paper
python paper_rag.py ingest paper.pdf --provider local --model all-MiniLM-L6-v2
```

### 4.3. Query Ingested Papers
Retrieves the most relevant chunks and streams an answer from the NVIDIA API:
```bash
python paper_rag.py query "What is the main contribution of this paper?"
```

Optional CLI flags:
```bash
# Retrieve top-7 chunks and override the NVIDIA model for this run
python paper_rag.py query "What datasets were used?" -k 7 --model meta/llama-3.1-8b-instruct

# Hide retrieved citation preview table
python paper_rag.py query "What are the limitations?" --no-show-sources
```

### 4.4. List Indexed Papers
View all papers and chunk counts currently indexed in ChromaDB:
```bash
python paper_rag.py list-papers
```

### 4.5. Clear the Database
Clear the vector database to start fresh:
```bash
python paper_rag.py clear
# or bypass confirmation
python paper_rag.py clear --force
```

---

## 5. Adaptive Chunking Engine

The engine does not apply one fixed recipe to every document. It first
*profiles* the document and then picks the strategy that fits each section.
Every stage has a deterministic fallback, so `ingest` never fails because of
chunking.

### 5.1 Pipeline

```
                    [ PDF Document ]
                           │
              PyMuPDF block extraction
              (font size, weight, page, bbox)
                           │
                           ▼
        [ Clean blocks: drop running headers/footers,
          page numbers, arXiv stamps, normalise space ]
                           │
                           ▼
        [ Detect headings: numbering + academic
          vocabulary + font statistics + title shape ]
          (captions and list items are rejected)
                           │
                           ▼
        [ Rebuild the heading hierarchy
          -> breadcrumb path per section ]
                           │
                           ▼
    ┌──────────── per-section strategy selection ────────────┐
    │                                                        │
    │  fits in budget?      -> section_whole                 │
    │  References section?  -> section_references            │
    │                        (one chunk per citation entry)  │
    │  contains a table?    -> section_table                 │
    │                        (caption + header row repeated) │
    │  embeddings available -> section_semantic_subsplit     │
    │  no embeddings        -> section_recursive_subsplit    │
    │  no headings at all   -> pure_semantic                  │
    └────────────────────────────────────────────────────────┘
                           │
                           ▼
        [ Finalise: enforce the output invariants ]
        1. never empty
        2. never above the token budget (hard guarantee)
        3. never a pathologically tiny chunk (merged into a neighbour)
        4. never a duplicate
        5. every chunk carries heading breadcrumb + page
```

### 5.2 Output guarantees

| Guarantee | How it is enforced |
| :--- | :--- |
| Chunk never exceeds `MAX_CHUNK_TOKENS` | conservative token estimate (1.25 tokens/word + surcharge for long identifiers) + hard split in the finalise pass |
| No empty chunks | dropped in the finalise pass |
| No duplicate chunks | normalised-text de-duplication |
| No 5-token chunks | chunks below 25% of the budget are merged into a neighbour |
| Chunks never cross a heading | sections are the hard boundary of the pipeline |
| Sentences are never cut in half | a scanner-based sentence splitter that protects abbreviations, decimals, acronyms, initials, URLs, citations, quotes and bracketed spans |
| Tables stay readable | caption + header row are repeated in every table chunk |
| Bibliographies stay usable | one chunk per reference entry, never per sentence |
| Retrieval context | each chunk is embedded together with its breadcrumb (`Section: 3 Methods > 3.1 Data`, `Page: 7`) |
| Embeddings never silently truncated | the chunk budget is clamped to the embedding model's real context window (`max_input_tokens`) |
| Works without an embedding API | deterministic sentence/paragraph packing fallback |

### 5.3 What changed versus the previous decision tree

* The old engine had one global rule ("section -> semantic split"). Long
  tables were shredded into sentences and long bibliographies into meaningless
  prose. Both are now first-class structures.
* The old semantic splitter could emit single-sentence chunks and had no
  overlap, so a chunk boundary always lost context. The new packer enforces a
  minimum chunk size and repeats the last sentence of the previous chunk.
* The old heading detector treated **any** bold line as a heading, which
  over-segmented papers where bold is used for emphasis. Headings are now
  scored on numbering, vocabulary, font size and title shape.
* The old token estimate (`words * 1.3`) was applied inconsistently and the
  recursive fallback produced chunks that were 30% over budget. Token
  accounting is now conservative and used everywhere.
* The old code never removed running headers/footers, so every chunk carried
  the journal name and page number.

## 6. Running Tests

Run the full pytest suite:
```bash
pytest -v
```

The suite covers PDF block extraction, heading heuristics, scientific sentence splitting, semantic clustering, the adaptive engine invariants (budget, no-empty, no-duplicate, no-tiny), running-header removal, table and reference handling, the recursive fallback, vector store operations and the CLI commands.
