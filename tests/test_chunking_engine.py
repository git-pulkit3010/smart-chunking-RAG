"""
Tests for the adaptive chunking engine.

Covers the invariants the engine promises (budget, no-empty, no-duplicates,
no-tiny-chunks), the strategy selection per document structure, and the
robustness of the low level text primitives.
"""

from typing import List
import pytest

from src.chunker import (
    _clean_blocks,
    build_sections,
    chunk_paper,
    is_heading,
    recursive_fallback_split,
    semantic_split,
    split_into_sections,
    split_sentences,
)
from src.embeddings import EmbeddingBackend
from src.textproc import (
    estimate_tokens,
    is_table_caption,
    split_reference_entries,
    words_to_token_budget,
)


class TopicEmbedder(EmbeddingBackend):
    """Embedder that groups sentences by keyword, like a real topic model."""

    TOPICS = {
        "biology": [1.0, 0.0, 0.0],
        "physics": [0.0, 1.0, 0.0],
        "finance": [0.0, 0.0, 1.0],
    }

    @property
    def model_name(self) -> str:
        return "topic-embedder"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out = []
        for t in texts:
            low = t.lower()
            vec = [0.34, 0.34, 0.34]
            for topic, vector in self.TOPICS.items():
                if topic in low:
                    vec = list(vector)
                    break
            out.append(vec)
        return out


class BrokenEmbedder(EmbeddingBackend):
    """Embedding backend that always fails (simulates an API outage)."""

    @property
    def model_name(self) -> str:
        return "broken"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise RuntimeError("embedding service unavailable")


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def _assert_invariants(chunks, max_tokens):
    assert chunks, "engine must never return an empty result for real input"
    seen = set()
    for ch in chunks:
        assert ch["text"].strip(), "chunks must never be empty"
        assert estimate_tokens(ch["text"]) <= max_tokens, (
            f"chunk exceeds budget: {estimate_tokens(ch['text'])} > {max_tokens}"
        )
        assert ch["token_count"] == estimate_tokens(ch["text"])
        assert ch["heading"], "every chunk carries a heading"
        assert ch["section_path"], "every chunk carries a section path"
        assert ch["embedding_text"].startswith(ch["context"])
        key = ch["text"].lower().strip()
        assert key not in seen, "duplicate chunks must be removed"
        seen.add(key)


def test_invariants_on_structured_document():
    blocks = [
        {"text": "Abstract", "font_size": 14.0, "is_bold": True, "page": 1},
        {"text": "We study finance markets and biology of cells in depth. " * 20,
         "font_size": 10.0, "is_bold": False, "page": 1},
        {"text": "1. Introduction", "font_size": 14.0, "is_bold": True, "page": 1},
        {"text": "Quantum mechanics explains physics at small scales. " * 25,
         "font_size": 10.0, "is_bold": False, "page": 2},
    ]
    chunks = chunk_paper(blocks, embed_backend=TopicEmbedder(), max_tokens=120)
    _assert_invariants(chunks, 120)
    assert len(chunks) > 2


def test_invariants_without_embeddings():
    blocks = [
        {"text": "Results", "font_size": 13.0, "is_bold": True, "page": 1},
        {"text": "Our method outperforms the baseline by a wide margin. " * 10,
         "font_size": 10.0, "is_bold": False, "page": 1},
    ]
    chunks = chunk_paper(blocks, embed_backend=None, max_tokens=100)
    _assert_invariants(chunks, 100)
    assert all(c["strategy"] == "section_recursive_subsplit" for c in chunks)


def test_embedding_failure_degrades_gracefully():
    blocks = [
        {"text": "Discussion", "font_size": 13.0, "is_bold": True, "page": 1},
        {"text": "Biology and physics both inform the finance model. " * 8,
         "font_size": 10.0, "is_bold": False, "page": 1},
    ]
    chunks = chunk_paper(blocks, embed_backend=BrokenEmbedder(), max_tokens=100)
    _assert_invariants(chunks, 100)
    assert chunks


def test_empty_and_garbage_input_never_raises():
    assert chunk_paper([], embed_backend=TopicEmbedder()) == []
    assert chunk_paper([{"text": "   ", "page": 1}]) == []
    assert chunk_paper([{"text": "", "font_size": 0.0, "page": 1}]) == []


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------


def test_small_section_is_kept_whole():
    blocks = [
        {"text": "Abstract", "font_size": 14.0, "is_bold": True, "page": 1},
        {"text": "A short abstract.", "font_size": 10.0, "is_bold": False, "page": 1},
    ]
    chunks = chunk_paper(blocks, embed_backend=TopicEmbedder(), max_tokens=512)
    assert len(chunks) == 1
    assert chunks[0]["strategy"] == "section_whole"


def test_prose_document_uses_semantic_strategy():
    blocks = [
        {"text": "Once upon a time in a lab, researchers studied cells. " * 3,
         "font_size": 10.0, "is_bold": False, "page": 1},
        {"text": "Quantum mechanics then explained atomic reactions. " * 3,
         "font_size": 10.0, "is_bold": False, "page": 2},
    ]
    chunks = chunk_paper(blocks, embed_backend=TopicEmbedder(), max_tokens=80)
    assert chunks
    assert all(c["strategy"] == "pure_semantic" for c in chunks)
    _assert_invariants(chunks, 80)


def test_reference_section_is_split_per_entry():
    entries = "\n".join(
        f"Smith, J. and Doe, A. ({1990 + i}). A study of biology and physics. "
        f"Journal of Finance, 12(3), 45-67."
        for i in range(12)
    )
    blocks = [
        {"text": "References", "font_size": 14.0, "is_bold": True, "page": 9},
        {"text": entries, "font_size": 10.0, "is_bold": False, "page": 9},
    ]
    chunks = chunk_paper(blocks, embed_backend=TopicEmbedder(), max_tokens=90)
    assert chunks
    assert all(c["strategy"] == "section_references" for c in chunks)
    _assert_invariants(chunks, 90)
    # every entry must survive somewhere in the output
    joined = " ".join(c["text"] for c in chunks)
    assert "Smith, J. and Doe, A. (1990)" in joined
    assert "Smith, J. and Doe, A. (2001)" in joined


def test_table_chunks_repeat_caption_and_header():
    rows = "\n".join(
        f"Model {i}\tAccuracy {90 + i / 10:.1f}\tLatency {100 + i}\tParams {i}M"
        for i in range(8)
    )
    blocks = [
        {"text": "Results", "font_size": 14.0, "is_bold": True, "page": 4},
        {"text": "Table 1: Accuracy of the proposed model on the benchmark.",
         "font_size": 10.0, "is_bold": False, "page": 4},
        {"text": rows, "font_size": 10.0, "is_bold": False, "page": 4},
    ]
    chunks = chunk_paper(blocks, embed_backend=TopicEmbedder(), max_tokens=90)
    table_chunks = [c for c in chunks if c["strategy"] == "section_table"]
    assert table_chunks, "a table must be detected and chunked by rows"
    for c in table_chunks:
        assert "Table 1: Accuracy of the proposed model" in c["text"]
        assert "Model 0" in c["text"]  # header row repeated
    _assert_invariants(chunks, 90)


# ---------------------------------------------------------------------------
# Layout handling
# ---------------------------------------------------------------------------


def test_running_headers_and_footers_are_removed():
    blocks = []
    for page in range(1, 6):
        blocks.append({"text": "Journal of Finance, Vol 12, 2024", "font_size": 8.0,
                       "is_bold": False, "page": page})
        blocks.append({"text": f"Section body text for page {page}.", "font_size": 10.0,
                       "is_bold": False, "page": page})
        blocks.append({"text": str(page), "font_size": 8.0, "is_bold": False, "page": page})
    cleaned = _clean_blocks(blocks)
    text = " ".join(b["text"] for b in cleaned)
    assert "Journal of Finance, Vol 12, 2024" not in text
    assert "Section body text for page 3." in text


def test_heading_hierarchy_is_rebuilt():
    blocks = [
        {"text": "1. Introduction", "font_size": 14.0, "is_bold": True, "page": 1},
        {"text": "Intro body.", "font_size": 10.0, "is_bold": False, "page": 1},
        {"text": "1.1 Motivation", "font_size": 12.0, "is_bold": True, "page": 1},
        {"text": "Motivation body.", "font_size": 10.0, "is_bold": False, "page": 1},
        {"text": "2. Methods", "font_size": 14.0, "is_bold": True, "page": 2},
        {"text": "Methods body.", "font_size": 10.0, "is_bold": False, "page": 2},
    ]
    sections = build_sections(_clean_blocks(blocks))
    paths = [s["path"] for s in sections]
    assert ["1. Introduction", "1.1 Motivation"] in paths
    assert ["2. Methods"] in paths


def test_table_branch_does_not_lose_the_section_prose():
    blocks = [
        {"text": "2.1 Data", "font_size": 12.0, "is_bold": True, "page": 2},
        {"text": "We use daily returns from 1994 to 2014 for 500 firms. " * 6,
         "font_size": 10.0, "is_bold": False, "page": 2},
        {"text": "Table 1: Benchmark results for the proposed model.",
         "font_size": 10.0, "is_bold": False, "page": 3},
        {"text": "\n".join(
            f"Model {i}\tAccuracy {90 + i / 10:.1f}\tSharpe {1 + i / 10:.1f}\tTurnover {i}"
            for i in range(12)
        ), "font_size": 10.0, "is_bold": False, "page": 3},
    ]
    chunks = chunk_paper(blocks, embed_backend=None, max_tokens=150)
    assert any("daily returns" in c["text"] for c in chunks), "section prose was lost"
    assert any(c["strategy"] == "section_table" for c in chunks)
    _assert_invariants(chunks, 150)


def test_heading_detection_rejects_captions_and_list_items():
    assert is_heading({"text": "Table 1: Results of the ablation", "font_size": 10.0,
                       "is_bold": True}, 10.0) is False
    assert is_heading({"text": "1. We ran three experiments", "font_size": 10.0,
                       "is_bold": False}, 10.0) is False
    assert is_heading({"text": "1. Introduction", "font_size": 10.0,
                       "is_bold": False}, 10.0) is True
    assert is_heading({"text": "2.3 Deep Model Architecture", "font_size": 10.0,
                       "is_bold": False}, 10.0) is True


def test_sections_never_cross_a_heading_boundary():
    blocks = [
        {"text": "Introduction", "font_size": 14.0, "is_bold": True, "page": 1},
        {"text": "Alpha text about biology.", "font_size": 10.0, "is_bold": False, "page": 1},
        {"text": "Conclusion", "font_size": 14.0, "is_bold": True, "page": 2},
        {"text": "Beta text about physics.", "font_size": 10.0, "is_bold": False, "page": 2},
    ]
    sections = split_into_sections(blocks)
    assert len(sections) == 2
    assert "biology" in sections[0]["text"] and "physics" not in sections[0]["text"]
    assert "physics" in sections[1]["text"] and "biology" not in sections[1]["text"]


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------


def test_sentence_splitter_handles_tricky_constructs():
    text = (
        "Smith et al. reported 3.14% gains. "
        "The U.S. team used Fig. 2 and Tab. 1. "
        "See https://example.com/a.b for details. "
        "Values of 1.000 and 0.5 remained stable. "
        "He said \"This is quoted. Still quoted.\" and left. "
        "Finally, J. Smith agreed."
    )
    sentences = split_sentences(text)
    assert sentences[0] == "Smith et al. reported 3.14% gains."
    assert "U.S. team" in sentences[1]
    assert "https://example.com/a.b" in sentences[2]
    assert "1.000" in sentences[3]
    # the quoted sentence must not be split
    assert any('"This is quoted. Still quoted."' in s for s in sentences)
    assert sentences[-1] == "Finally, J. Smith agreed."


def test_reference_entry_splitting():
    text = (
        "Smith, J. (2020). Title one. Journal.\n"
        "continuation line of the first entry.\n"
        "Doe, A. and Roe, B. (2021). Title two. Journal.\n"
        "[3] Third entry with a numeric marker."
    )
    entries = split_reference_entries(text)
    assert len(entries) == 3
    assert entries[0].startswith("Smith, J. (2020)")
    assert "continuation line" in entries[0]
    assert entries[2].startswith("[3]")


def test_table_caption_detection():
    assert is_table_caption("Table 2: Ablation results") is True
    assert is_table_caption("Figure 3: Architecture") is True
    assert is_table_caption("The table below shows results") is False


# ---------------------------------------------------------------------------
# Budget helpers
# ---------------------------------------------------------------------------


def test_token_estimation_is_conservative():
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello world") >= 2
    long_url = "https://example.com/" + "a" * 200
    assert estimate_tokens(long_url) > estimate_tokens("short")
    assert words_to_token_budget(512) >= 100


def test_recursive_fallback_respects_budget_and_overlaps():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = recursive_fallback_split(text, max_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1
    for c in chunks:
        assert estimate_tokens(c) <= 100
    # overlap: consecutive chunks share content
    assert set(chunks[0].split()) & set(chunks[1].split())


def test_semantic_split_survives_a_broken_embedder():
    text = "Biology of cells. Physics of particles. " * 5
    chunks = semantic_split(text, BrokenEmbedder(), threshold=0.65, max_tokens=60)
    assert chunks
    for c in chunks:
        assert estimate_tokens(c) <= 60


def test_chunks_carry_retrieval_context():
    blocks = [
        {"text": "3.2 Model Architecture", "font_size": 13.0, "is_bold": True, "page": 5},
        {"text": "The encoder uses a transformer stack.", "font_size": 10.0,
         "is_bold": False, "page": 5},
    ]
    chunks = chunk_paper(blocks, embed_backend=TopicEmbedder(), max_tokens=512)
    assert chunks
    ctx = chunks[0]["context"]
    assert "3.2 Model Architecture" in ctx
    assert "Page: 5" in ctx
    assert chunks[0]["embedding_text"].startswith("Section:")
