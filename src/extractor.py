"""
PDF layout and text extraction module using PyMuPDF.
Extracts block-level text along with layout metadata (font size, weight, page number).
"""

from pathlib import Path
from typing import Any, Dict, List
import pymupdf as fitz


def is_span_bold(span: Dict[str, Any]) -> bool:
    """Determine if a text span is bold based on font name and flags."""
    font_name = str(span.get("font", "")).lower()
    flags = span.get("flags", 0)
    # Flag 2^4 (16) or 2^1 (2) often indicates bold/italic in PDF fonts
    flag_bold = bool(flags & (1 << 4))
    name_bold = any(b in font_name for b in ("bold", "black", "heavy", "semibold"))
    return flag_bold or name_bold


def extract_blocks(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text blocks from a PDF with layout metadata.

    Returns:
        List of dicts: [
            {
                "page": int,
                "text": str,
                "font_size": float,
                "is_bold": bool,
                "bbox": tuple
            }, ...
        ]
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = fitz.open(str(path))
    blocks_out: List[Dict[str, Any]] = []

    try:
        for page_num, page in enumerate(doc, start=1):
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                # Filter out image blocks and blocks with no lines
                if "lines" not in block:
                    continue

                text_lines: List[str] = []
                max_size = 0.0
                has_bold = False

                for line in block.get("lines", []):
                    line_text_parts: List[str] = []
                    for span in line.get("spans", []):
                        txt = span.get("text", "")
                        if txt:
                            line_text_parts.append(txt)
                            max_size = max(max_size, span.get("size", 0.0))
                            if is_span_bold(span):
                                has_bold = True
                    line_str = "".join(line_text_parts).strip()
                    if line_str:
                        text_lines.append(line_str)

                full_text = "\n".join(text_lines).strip()
                if full_text:
                    blocks_out.append({
                        "page": page_num,
                        "text": full_text,
                        "font_size": round(max_size, 2),
                        "is_bold": has_bold,
                        "bbox": block.get("bbox", (0, 0, 0, 0)),
                    })
    finally:
        doc.close()

    return blocks_out
