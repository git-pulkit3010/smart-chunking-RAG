"""
Unit tests for intelligent chunking, heading detection, sentence splitting, and fallback.
"""

from typing import List
import pytest

from src.chunker import (
    is_heading,
    split_into_sections,
    split_sentences,
    semantic_split,
    recursive_fallback_split,
    chunk_paper,
)
from src.embeddings import EmbeddingBackend


class MockEmbeddingBackend(EmbeddingBackend):
    """Mock embedder returning simple vectors based on text features."""

    def __init__(self, dim: int = 4):
        self._dim = dim

    @property
    def model_name(self) -> str:
        return "mock-embedder"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Generate orthogonal or similar vectors based on topic
        embeddings = []
        for t in texts:
            if "biology" in t.lower() or "cells" in t.lower():
                embeddings.append([1.0, 0.0, 0.0, 0.0])
            elif "physics" in t.lower() or "quantum" in t.lower():
                embeddings.append([0.0, 1.0, 0.0, 0.0])
            else:
                embeddings.append([0.5, 0.5, 0.5, 0.5])
        return embeddings


def test_is_heading():
    avg_font = 10.0

    # Section keyword with reasonable length
    assert is_heading({"text": "Abstract", "font_size": 10.0, "is_bold": False}, avg_font) is True
    assert is_heading({"text": "1. Introduction", "font_size": 12.0, "is_bold": False}, avg_font) is True
    assert is_heading({"text": "3. Methodology", "font_size": 10.0, "is_bold": True}, avg_font) is True

    # Font size significantly larger
    assert is_heading({"text": "Experimental Results Overview", "font_size": 13.0, "is_bold": False}, avg_font) is True

    # Bold font at average size
    assert is_heading({"text": "Ablation Setup", "font_size": 10.0, "is_bold": True}, avg_font) is True

    # Normal body sentence should NOT be a heading
    assert is_heading(
        {
            "text": "This is a regular sentence describing the background and prior literature in neural models.",
            "font_size": 10.0,
            "is_bold": False,
        },
        avg_font,
    ) is False


def test_split_sentences():
    text = (
        "Smith et al. conducted an experiment showing a 3.14% improvement. "
        "As discussed in Sec. 2 and Fig. 1, the model converged rapidly! "
        "Did it outperform baselines? Yes, it did."
    )
    sentences = split_sentences(text)

    # Citations and abbreviations should NOT cause split.
    # 1: Smith et al... 2: As discussed in Sec. 2... 3: Did it outperform...? 4: Yes, it did.
    assert len(sentences) == 4
    assert "Smith et al. conducted an experiment showing a 3.14% improvement." == sentences[0]
    assert "As discussed in Sec. 2 and Fig. 1, the model converged rapidly!" == sentences[1]
    assert "Did it outperform baselines?" == sentences[2]
    assert "Yes, it did." == sentences[3]


def test_split_into_sections():
    blocks = [
        {"text": "Abstract", "font_size": 14.0, "is_bold": True, "page": 1},
        {"text": "This paper presents our findings on machine learning.", "font_size": 10.0, "is_bold": False, "page": 1},
        {"text": "1. Introduction", "font_size": 14.0, "is_bold": True, "page": 1},
        {"text": "We start by introducing neural networks.", "font_size": 10.0, "is_bold": False, "page": 1},
        {"text": "Neural networks are powerful function approximators.", "font_size": 10.0, "is_bold": False, "page": 2},
    ]

    sections = split_into_sections(blocks)
    assert len(sections) == 2
    assert sections[0]["heading"] == "Abstract"
    assert "findings on machine learning" in sections[0]["text"]
    assert sections[1]["heading"] == "1. Introduction"
    assert "Neural networks are powerful" in sections[1]["text"]


def test_semantic_split_with_mock():
    mock_backend = MockEmbeddingBackend()
    text = (
        "Biology is the study of living cells and organisms. "
        "All cells contain DNA and cellular structures. "
        "Quantum mechanics is a branch of theoretical physics. "
        "The physics of particles behaves differently at quantum scales."
    )
    # The first 2 sentences are biology, the next 2 are physics -> should split into 2 chunks
    chunks = semantic_split(text, mock_backend, threshold=0.65, max_tokens=100)
    assert len(chunks) == 2
    assert "Biology" in chunks[0]
    assert "Quantum mechanics" in chunks[1]


def test_recursive_fallback_split():
    long_text = "word " * 150  # 150 words
    chunks = recursive_fallback_split(long_text, max_tokens=50, overlap_tokens=10)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.split()) <= 50


def test_chunk_paper_decision_tree():
    mock_backend = MockEmbeddingBackend()

    # Case 1: Paper with clear headings and short sections
    blocks_structured = [
        {"text": "Abstract", "font_size": 14.0, "is_bold": True, "page": 1},
        {"text": "Short abstract text.", "font_size": 10.0, "is_bold": False, "page": 1},
        {"text": "1. Introduction", "font_size": 14.0, "is_bold": True, "page": 1},
        {"text": "Short intro text.", "font_size": 10.0, "is_bold": False, "page": 1},
    ]
    chunks = chunk_paper(blocks_structured, embed_backend=mock_backend, max_tokens=512)
    assert len(chunks) == 2
    assert chunks[0]["strategy"] == "section_whole"
    assert chunks[1]["strategy"] == "section_whole"

    # Case 2: Section exceeding max_tokens -> semantic sub-split
    long_section = (
        "Biology is the study of living cells and organisms. "
        "All cells contain DNA and cellular structures. "
        "Quantum mechanics is a branch of theoretical physics. "
        "The physics of particles behaves differently at quantum scales. "
    ) * 10
    blocks_long = [
        {"text": "1. Methodology", "font_size": 14.0, "is_bold": True, "page": 1},
        {"text": long_section, "font_size": 10.0, "is_bold": False, "page": 1},
    ]
    chunks_long = chunk_paper(blocks_long, embed_backend=mock_backend, max_tokens=50)
    assert len(chunks_long) > 1
    assert all(c["strategy"] == "section_semantic_subsplit" for c in chunks_long)

    # Case 3: Paper with NO headings (continuous prose)
    blocks_unstructured = [
        {"text": "Once upon a time in a scientific lab, researchers worked on cells.", "font_size": 10.0, "is_bold": False, "page": 1},
        {"text": "Quantum mechanics then explained the atomic reactions.", "font_size": 10.0, "is_bold": False, "page": 1},
    ]
    chunks_unstructured = chunk_paper(blocks_unstructured, embed_backend=mock_backend, max_tokens=512)
    assert len(chunks_unstructured) >= 1
    assert chunks_unstructured[0]["strategy"] == "pure_semantic"
