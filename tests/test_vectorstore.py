"""
Unit tests for embeddings and vector store.
"""

from pathlib import Path
import pytest
import numpy as np

from src.embeddings import cosine_similarity, get_embedding_backend, OpenRouterEmbeddings
from src.vectorstore import PaperVectorStore


def test_cosine_similarity():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    assert pytest.approx(cosine_similarity(v1, v2)) == 1.0

    v_orthogonal = [0.0, 1.0, 0.0]
    assert pytest.approx(cosine_similarity(v1, v_orthogonal)) == 0.0

    v_opposite = [-1.0, 0.0, 0.0]
    assert pytest.approx(cosine_similarity(v1, v_opposite)) == -1.0


def test_openrouter_embeddings_requires_api_key():
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is required"):
        OpenRouterEmbeddings(api_key="")


def test_vector_store_operations(tmp_path: Path):
    store = PaperVectorStore(
        chroma_path=str(tmp_path / "test_chroma"),
        collection_name="test_papers",
    )

    chunks = [
        {"heading": "Abstract", "text": "This paper presents our neural model.", "page": 1, "strategy": "section_whole"},
        {"heading": "Results", "text": "The model achieved 95% accuracy.", "page": 2, "strategy": "section_whole"},
    ]
    # Simple 3-dimensional embeddings
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    added = store.add_paper("paper_a.pdf", chunks, embeddings)
    assert added == 2
    assert store.count() == 2

    # Test list_papers
    papers = store.list_papers()
    assert len(papers) == 1
    assert papers[0]["paper_title"] == "paper_a.pdf"
    assert papers[0]["chunk_count"] == 2

    # Query with vector close to the first chunk
    hits = store.query(query_embedding=[0.9, 0.1, 0.0], k=1)
    assert len(hits) == 1
    assert "Abstract" in hits[0]["meta"]["heading"]

    # Delete paper
    store.delete_paper("paper_a.pdf")
    assert store.count() == 0

    # Clear
    store.clear()
    assert store.count() == 0
