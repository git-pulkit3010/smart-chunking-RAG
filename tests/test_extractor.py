"""
Unit tests for PDF block extraction.
"""

from pathlib import Path
import pymupdf as fitz
import pytest

from src.extractor import extract_blocks, is_span_bold


def create_sample_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "sample_paper.pdf"
    doc = fitz.open()

    # Page 1
    page1 = doc.new_page()
    page1.insert_text((50, 60), "Deep Learning Architectures in 2026", fontsize=20)
    page1.insert_text((50, 100), "Abstract", fontsize=14)
    page1.insert_text(
        (50, 130),
        "In this work, we propose a novel transformer-based framework that achieves state of the art results.",
        fontsize=10,
    )
    page1.insert_text((50, 170), "1. Introduction", fontsize=14)
    page1.insert_text(
        (50, 200),
        "Transformers have fundamentally reshaped modern natural language processing and computer vision.",
        fontsize=10,
    )

    # Page 2
    page2 = doc.new_page()
    page2.insert_text((50, 60), "2. Methodology", fontsize=14)
    page2.insert_text(
        (50, 90),
        "We introduce a layout-aware block chunking mechanism that preserves document structural context.",
        fontsize=10,
    )

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_is_span_bold():
    assert is_span_bold({"font": "Helvetica-Bold", "flags": 0}) is True
    assert is_span_bold({"font": "Times-Roman", "flags": 16}) is True
    assert is_span_bold({"font": "Arial-Regular", "flags": 0}) is False


def test_extract_blocks(tmp_path):
    pdf_path = create_sample_pdf(tmp_path)
    blocks = extract_blocks(str(pdf_path))

    assert len(blocks) >= 5
    # Check title block
    assert blocks[0]["text"] == "Deep Learning Architectures in 2026"
    assert blocks[0]["font_size"] == 20.0
    assert blocks[0]["page"] == 1

    # Check that both page 1 and page 2 blocks exist
    pages = {b["page"] for b in blocks}
    assert 1 in pages
    assert 2 in pages


def test_extract_nonexistent_pdf():
    with pytest.raises(FileNotFoundError):
        extract_blocks("non_existent_file.pdf")
