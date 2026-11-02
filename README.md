# Smart Chunking RAG for Research Papers

A CLI Retrieval-Augmented Generation (RAG) system specifically designed for research papers. It combines block-level layout analysis with semantic sentence clustering and connects to OpenRouter for both embeddings and answer generation.

---

## 1. Key Features

- **Layout-Aware PDF Extraction (PyMuPDF)**: Extracts block-level text along with structural signals (font sizes, boldness, line boxes, page numbers).
- **Intelligent Chunking Decision Tree**: Automatically inspects the PDF layout to pick the optimal chunking strategy:
  1. **Section-Aware Chunking**: Preserves structural units (Abstract, Introduction, Methods, Results, Discussion, etc.) intact if within token budget.
  2. **Semantic Sub-Splitting**: For long sections exceeding `MAX_CHUNK_TOKENS`, computes consecutive sentence embeddings and splits at semantic boundaries where cosine similarity drops.
  3. **Continuous Prose Strategy**: If no headings exist, applies pure semantic chunking across the entire paper.
  4. **Recursive Fallback**: Gracefully falls back to token/character recursive splitting if embeddings are unavailable.
- **Upgraded OpenRouter Embedding Model**: Powered by `openai/text-embedding-3-small` through OpenRouter's native `/embeddings` API, with optional offline local fallback (`sentence-transformers`).
- **Seamless `.env` Integration**: Automatically loads configuration and API keys from `.env` on every run—no need to run `export` in your shell every time.
- **Local ChromaDB Vector Database**: Zero-config persistent vector store in `./chroma_db`.
- **Streaming Generation**: Streams answers token-by-token using OpenRouter chat completions (`openai/gpt-4o-mini` by default).

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

Simply edit `.env` and set your OpenRouter API key:
```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_MODEL=openai/gpt-4o-mini
EMBED_MODEL=openai/text-embedding-3-small
EMBED_PROVIDER=openrouter
CHROMA_PATH=./chroma_db
COLLECTION_NAME=papers
MAX_CHUNK_TOKENS=512
SEMANTIC_SIMILARITY_THRESHOLD=0.65
TOP_K_RETRIEVAL=5
```

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
Retrieves the most relevant chunks and streams an answer:
```bash
python paper_rag.py query "What is the main contribution of this paper?"
```

Optional CLI flags:
```bash
# Retrieve top-7 chunks and use Claude 3.5 Sonnet
python paper_rag.py query "What datasets were used?" -k 7 --model anthropic/claude-3.5-sonnet

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

## 5. Intelligent Chunking Decision Tree

```
                           [ PDF Document ]
                                  │
                       PyMuPDF Block Extraction
                     (font size, weight, page, bbox)
                                  │
                                  ▼
                     [ Detect Layout Headings ]
                         (font > avg + bold)
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
      [ Clear Headings Found ]            [ No Headings Found ]
                 │                        (Continuous Prose)
                 ▼                                 │
     Split by Section Headings                     │
                 │                                 │
       ┌─────────┴─────────┐                       │
       ▼                   ▼                       │
Section <= MAX_TOKENS   Section > MAX_TOKENS       │
       │                   │                       │
 Keep Section Whole        ▼                       ▼
 (Structural Chunk)  Sentence Splitting      Sentence Splitting
                     (protecting et al.,     (protecting et al.,
                      decimals, Fig., i.e.)   decimals, Fig., i.e.)
                           │                       │
                           ▼                       ▼
                 [ Semantic Chunking ]   [ Pure Semantic Chunking ]
                 (Cosine similarity      (Cosine similarity
                  drops trigger break)    drops trigger break)
                           │                       │
                           └───────────┬───────────┘
                                       ▼
                       (If embedding API unavailable)
                                       │
                                       ▼
                        [ Recursive Fallback Split ]
```

---

## 6. Running Tests

Run the full pytest suite:
```bash
pytest -v
```

All 15 tests covering PDF block extraction, heading heuristics, scientific sentence splitting, semantic clustering, recursive fallback, vector store operations, and CLI commands will run and verify the codebase.
