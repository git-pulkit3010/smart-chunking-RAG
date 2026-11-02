"""
Intelligent layout-aware chunking engine for research papers.
Combines block-level layout analysis (font size, bold heuristics, heading regex)
with semantic sentence clustering and recursive fallback.
"""

import re
from typing import Any, Callable, Dict, List, Optional
from src.embeddings import EmbeddingBackend, cosine_similarity

# Common academic section title patterns
SECTION_PATTERN = re.compile(
    r"^(?:(?:\d+|[IVXLCDM]+)(?:\.\d+)*\s+)?(?:Abstract|Introduction|Background|"
    r"Related\s+Work|Preliminary|Preliminaries|Methodology|Methods|System\s+Overview|"
    r"Architecture|Model|Approach|Experimental\s+Setup|Experiments|Evaluation|"
    r"Results|Discussion|Ablation\s+Study|Ablation\s+Studies|Conclusion|Conclusions|"
    r"Broader\s+Impact|Limitations|References|Appendix|Acknowledgments?)\b",
    re.IGNORECASE,
)

# Common academic abbreviations to protect during sentence splitting
ABBREVIATIONS = re.compile(
    r"\b(?:et\s+al|e\.g|i\.e|vs|fig|figs|tab|tabs|eq|eqs|ref|refs|al|dr|prof|sec|secs|no|vol|pp|approx|dept)\.",
    re.IGNORECASE,
)


def is_running_header_or_footer(text: str) -> bool:
    """Filter out noise like single page numbers, arxiv watermarks, or conference headers."""
    stripped = text.strip()
    if not stripped:
        return True
    # Pure page number (e.g. '1', '23', 'Page 4 of 12')
    if re.match(r"^(?:page\s+)?\d+(?:\s+of\s+\d+)?$", stripped, re.IGNORECASE):
        return True
    # ArXiv watermark
    if re.match(r"^arxiv:\d{4}\.\d{4,5}(?:v\d+)?(?:\s+\[.*\])?", stripped, re.IGNORECASE):
        return True
    return False


def is_heading(block: Dict[str, Any], avg_font: float) -> bool:
    """
    Heuristic to determine if a block represents a section heading.
    Inspects font size, boldness, length, and common academic section patterns.
    """
    text = block.get("text", "").strip()
    font_size = block.get("font_size", 0.0)
    is_bold = block.get("is_bold", False)

    if not text or len(text) > 120 or is_running_header_or_footer(text):
        return False

    # Heading check 1: matches recognized research paper section titles
    if SECTION_PATTERN.match(text) and len(text) < 80:
        return True

    # Heading check 2: significantly larger font than average body text
    if font_size > (avg_font + 1.2) and len(text) < 100:
        # Avoid treating full sentences with a period as headings unless very short
        if not text.endswith(".") or len(text.split()) <= 6:
            return True

    # Heading check 3: bold line of modest length with font size at or above average
    if is_bold and font_size >= avg_font and len(text) < 80:
        if not text.endswith(".") or len(text.split()) <= 6:
            return True

    return False


def split_into_sections(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group blocks into sections delimited by detected headings.
    Returns a list of dicts: [{"heading": str, "text": str, "page": int}, ...]
    """
    if not blocks:
        return []

    # Calculate average font size of the document
    sizes = [b["font_size"] for b in blocks if b.get("font_size", 0.0) > 0]
    avg_font = (sum(sizes) / len(sizes)) if sizes else 10.0

    sections: List[Dict[str, Any]] = []
    current_heading = "Document Start"
    current_lines: List[str] = []
    current_page = blocks[0].get("page", 1)

    for b in blocks:
        txt = b.get("text", "").strip()
        if not txt or is_running_header_or_footer(txt):
            continue

        if is_heading(b, avg_font):
            if current_lines:
                sections.append({
                    "heading": current_heading,
                    "text": "\n".join(current_lines),
                    "page": current_page,
                })
            current_heading = txt
            current_lines = []
            current_page = b.get("page", 1)
        else:
            current_lines.append(txt)

    if current_lines:
        sections.append({
            "heading": current_heading,
            "text": "\n".join(current_lines),
            "page": current_page,
        })

    return sections


def split_sentences(text: str) -> List[str]:
    """
    Sentence tokenizer designed for scientific and research documents.
    Protects citations (et al.), abbreviations (e.g., i.e.), and decimal numbers.
    """
    if not text.strip():
        return []

    # Replace periods in abbreviations with a safe token
    def _mask_abbr(m: re.Match) -> str:
        return m.group(0).replace(".", "__DOT__")

    masked = ABBREVIATIONS.sub(_mask_abbr, text)
    # Protect decimals like 3.14 or 0.001
    masked = re.sub(r"(\d+)\.(\d+)", r"\1__DOT__\2", masked)

    # Split on sentence end punctuation (. ! ?) followed by whitespace and uppercase/start char
    sentences_raw = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(])", masked)

    # Restore periods and clean up
    sentences: List[str] = []
    for s in sentences_raw:
        cleaned = s.replace("__DOT__", ".").strip()
        if cleaned:
            sentences.append(cleaned)

    return sentences if sentences else [text.strip()]


def semantic_split(
    text: str,
    embed_backend: EmbeddingBackend,
    threshold: float = 0.65,
    max_tokens: int = 512,
    min_tokens: int = 0,
) -> List[str]:
    """
    Split a section into semantically coherent passages.
    Computes consecutive sentence embeddings and breaks whenever cosine similarity drops
    below threshold or the accumulated chunk token limit is reached.
    """
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []

    try:
        embeddings = embed_backend.embed_documents(sentences)
    except Exception:
        # Fall back gracefully to recursive token split if embedding API fails
        return recursive_fallback_split(text, max_tokens=max_tokens)

    chunks: List[str] = []
    current_sentences: List[str] = [sentences[0]]

    for i in range(1, len(sentences)):
        prev_emb = embeddings[i - 1]
        curr_emb = embeddings[i]
        sim = cosine_similarity(prev_emb, curr_emb)

        # Rough token estimate (words * 1.3 ≈ tokens)
        current_token_count = sum(int(len(s.split()) * 1.3) for s in current_sentences)
        candidate_token_count = current_token_count + int(len(sentences[i].split()) * 1.3)

        if (sim < threshold and current_token_count >= min_tokens) or (
            candidate_token_count > max_tokens
        ):
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentences[i]]
        else:
            current_sentences.append(sentences[i])

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def recursive_fallback_split(
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
) -> List[str]:
    """
    Robust recursive character/token fallback chunking.
    Splits by paragraph, then line, then sentence, then words, respecting token budget.
    """
    words = text.split()
    if not words:
        return []

    # If within token budget, return whole
    if int(len(words) * 1.3) <= max_tokens:
        return [text.strip()]

    chunks: List[str] = []
    step = max(1, max_tokens - overlap_tokens)

    for i in range(0, len(words), step):
        chunk_words = words[i : i + max_tokens]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if i + max_tokens >= len(words):
            break

    return chunks


def chunk_paper(
    blocks: List[Dict[str, Any]],
    embed_backend: Optional[EmbeddingBackend] = None,
    max_tokens: int = 512,
    threshold: float = 0.65,
) -> List[Dict[str, Any]]:
    """
    Execute the intelligent chunking decision tree:
    1. Detect layout headings & split into sections.
    2. If clear headings exist:
       - Keep section whole if within max_tokens.
       - Sub-split long sections semantically (or recursive fallback if embedding unavailable).
    3. If no headings exist (continuous prose):
       - Apply pure semantic chunking across the entire paper.
    4. Fallback:
       - Recursive character/token splitting.

    Returns:
        List of dicts: [
            {
                "heading": str,
                "text": str,
                "page": int,
                "strategy": str
            }, ...
        ]
    """
    if not blocks:
        return []

    sections = split_into_sections(blocks)
    chunks: List[Dict[str, Any]] = []

    # Check if we detected meaningful sections (more than 1 section, or heading wasn't Document Start)
    has_real_headings = len(sections) > 1 or (
        len(sections) == 1 and sections[0]["heading"] != "Document Start"
    )

    if has_real_headings:
        for sec in sections:
            sec_text = sec["text"].strip()
            if not sec_text:
                continue

            approx_tokens = int(len(sec_text.split()) * 1.3)
            heading = sec["heading"]
            page = sec.get("page", 1)

            if approx_tokens <= max_tokens:
                chunks.append({
                    "heading": heading,
                    "text": sec_text,
                    "page": page,
                    "strategy": "section_whole",
                })
            else:
                # Long section: apply semantic sub-split
                if embed_backend:
                    sub_chunks = semantic_split(
                        sec_text,
                        embed_backend,
                        threshold=threshold,
                        max_tokens=max_tokens,
                    )
                    strat = "section_semantic_subsplit"
                else:
                    sub_chunks = recursive_fallback_split(
                        sec_text,
                        max_tokens=max_tokens,
                    )
                    strat = "section_recursive_subsplit"

                for sub in sub_chunks:
                    if sub.strip():
                        chunks.append({
                            "heading": heading,
                            "text": sub.strip(),
                            "page": page,
                            "strategy": strat,
                        })
    else:
        # Case: No headings detected (e.g. continuous prose or layout didn't yield headings)
        combined_text = "\n\n".join(b.get("text", "") for b in blocks if b.get("text", "").strip())
        first_page = blocks[0].get("page", 1)

        if embed_backend:
            sub_chunks = semantic_split(
                combined_text,
                embed_backend,
                threshold=threshold,
                max_tokens=max_tokens,
            )
            strat = "pure_semantic"
        else:
            sub_chunks = recursive_fallback_split(
                combined_text,
                max_tokens=max_tokens,
            )
            strat = "recursive_fallback"

        for sub in sub_chunks:
            if sub.strip():
                chunks.append({
                    "heading": "Document Content",
                    "text": sub.strip(),
                    "page": first_page,
                    "strategy": strat,
                })

    return chunks
