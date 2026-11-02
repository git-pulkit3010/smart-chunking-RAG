"""
Unit tests for Typer CLI commands.
"""

from pathlib import Path
from typer.testing import CliRunner
import pymupdf as fitz

from paper_rag import app

runner = CliRunner()


def create_test_pdf(path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 70), "CLI Test Paper", fontsize=18)
    page.insert_text((50, 100), "Abstract", fontsize=14)
    page.insert_text((50, 130), "This is a test paper ingested via the CLI test harness.", fontsize=10)
    page.insert_text((50, 170), "1. Introduction", fontsize=14)
    page.insert_text((50, 200), "Here is an introduction section explaining the concepts.", fontsize=10)
    doc.save(str(path))
    doc.close()
    return path


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Intelligent Section-Aware RAG" in result.output
    assert "ingest" in result.output
    assert "query" in result.output


def test_cli_config():
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert "Smart Chunking RAG Configuration" in result.output
    assert "EMBED_MODEL" in result.output


def test_cli_ingest_and_list_and_clear(tmp_path: Path, monkeypatch):
    test_pdf = create_test_pdf(tmp_path / "test_doc.pdf")
    chroma_dir = str(tmp_path / "cli_chroma")

    monkeypatch.setenv("CHROMA_PATH", chroma_dir)
    monkeypatch.setenv("EMBED_PROVIDER", "local")
    monkeypatch.setenv("EMBED_MODEL", "all-MiniLM-L6-v2")

    # Ingest using local sentence-transformer
    result = runner.invoke(app, ["ingest", str(test_pdf), "--provider", "local", "--model", "all-MiniLM-L6-v2"])
    assert result.exit_code == 0
    assert "Successfully indexed" in result.output

    # List papers
    result_list = runner.invoke(app, ["list-papers"])
    assert result_list.exit_code == 0
    assert "test_doc.pdf" in result_list.output

    # Clear
    result_clear = runner.invoke(app, ["clear", "--force"])
    assert result_clear.exit_code == 0
    assert "cleared successfully" in result_clear.output
