"""
Adaptive, layout-aware chunking engine for research papers.

The engine is built around a simple promise: **the chunking strategy is
decided from the document's real structure, and the output always satisfies a
set of hard invariants.**

Decision pipeline
-----------------
1. ``_clean_blocks``   – strip running headers/footers, page numbers, layout
   noise and normalise whitespace.
2. ``build_sections``  – detect headings using font statistics *and* numbering
   *and* academic section vocabulary, then rebuild the heading hierarchy
   (breadcrumb path) for every section.
3. Per-section strategy selection (this is the "dynamic" part):
   * section fits the budget              -> ``section_whole``
   * reference section                    -> ``section_references`` (one chunk
     per bibliographic entry, never per sentence)
   * section contains a table             -> ``section_table`` (caption +
     header row repeated in every chunk so each chunk stands alone)
   * long section + embeddings available  -> ``section_semantic_subsplit``
   * long section, no embeddings          -> ``section_recursive_subsplit``
   * document without any headings        -> ``pure_semantic``
4. ``_finalize`` – enforce the invariants:
   * every chunk is non-empty and within the token budget (hard guarantee)
   * no pathologically small chunks (they are merged into a neighbour)
   * no duplicate chunks
   * every chunk carries its heading breadcrumb + page as retrieval context

Every stage has a deterministic fallback, so ``chunk_paper`` never raises and
never returns a chunk that violates the invariants.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.embeddings import EmbeddingBackend, cosine_similarity
from src.textproc import (
    LIST_MARKER_RE,
    NUMBERED_HEADING_RE,
    TABLE_CAPTION_RE,
    clean_sentence,
    estimate_tokens,
    extract_table_rows,
    is_reference_heading,
    is_running_header_or_footer,
    is_table_caption,
    is_table_row,
    normalize_whitespace,
    split_reference_entries,
    split_sentences,
    words_to_token_budget,
)

# Common academic section title patterns (kept for backwards compatibility)
SECTION_PATTERN = re.compile(
    r"^(?:(?:\d+|[IVXLCDM]+)(?:\.\d+)*\.?\s+)?(?:Abstract|Introduction|Background|"
    r"Related\s+Work|Preliminary|Preliminaries|Methodology|Methods|System\s+Overview|"
    r"Architecture|Model|Approach|Experimental\s+Setup|Experiments|Evaluation|"
    r"Results|Discussion|Ablation\s+Study|Ablation\s+Studies|Conclusion|Conclusions|"
    r"Broader\s+Impact|Limitations|References|Appendix|Acknowledgments?)\b",
    re.IGNORECASE,
)

# Abbreviations protected during sentence splitting (re-exported for callers)
from src.textproc import ABBREVIATIONS  # noqa: E402  (public re-export)

DEFAULT_MAX_TOKENS = 512
DEFAULT_THRESHOLD = 0.65
DEFAULT_OVERLAP_SENTENCES = 1


# ---------------------------------------------------------------------------
# Document statistics & block cleaning
# ---------------------------------------------------------------------------


def _body_font_size(blocks: Sequence[Dict[str, Any]]) -> float:
    """Length-weighted median font size == the size of the body text."""
    weighted: List[Tuple[float, int]] = []
    for b in blocks:
        text = (b.get("text") or "").strip()
        size = float(b.get("font_size") or 0.0)
        if text and size > 0:
            weighted.append((size, len(text)))
    if not weighted:
        return 10.0
    weighted.sort()
    total = sum(length for _, length in weighted) or 1
    seen = 0
    for size, length in weighted:
        seen += length
        if seen >= total / 2:
            return size
    return weighted[-1][0]


def _clean_blocks(blocks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove layout noise and normalise text.

    * Page numbers / arXiv stamps are dropped.
    * Lines that repeat as the first (header) or last (footer) line on many
      pages are dropped, keeping only their first occurrence.
    * Whitespace is normalised, empty blocks are removed.
    """
    if not blocks:
        return []

    pages = {b.get("page", 1) for b in blocks}
    npages = len(pages)

    first_lines: Dict[int, set] = {}
    last_lines: Dict[int, set] = {}
    for b in blocks:
        page = b.get("page", 1)
        lines = [ln.strip() for ln in (b.get("text") or "").split("\n") if ln.strip()]
        if not lines:
            continue
        first_lines.setdefault(page, set()).add(lines[0])
        last_lines.setdefault(page, set()).add(lines[-1])

    votes: Counter = Counter()
    for candidates in list(first_lines.values()) + list(last_lines.values()):
        votes.update(candidates)

    min_votes = max(3, int(0.4 * npages)) if npages >= 3 else 10 ** 6
    repeated = {c for c, v in votes.items() if v >= min_votes and 0 < len(c) <= 120}

    seen: set = set()
    cleaned: List[Dict[str, Any]] = []
    for b in blocks:
        keep: List[str] = []
        for raw in (b.get("text") or "").split("\n"):
            line = raw.rstrip()
            if not line.strip():
                continue
            if is_running_header_or_footer(line.strip()):
                continue
            if line.strip() in repeated:
                continue
            keep.append(line)
        if not keep:
            continue
        new_block = dict(b)
        new_block["text"] = "\n".join(keep)
        cleaned.append(new_block)

    return cleaned


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------


# Words that almost always start a *sentence* rather than a heading.
_SENTENCE_START_WORDS = {
    "we", "our", "i", "this", "that", "it", "they", "he", "she", "you",
    "there", "here", "in", "on", "at", "for", "when", "if", "as", "by", "to",
    "from", "with", "after", "before", "during", "since", "although",
    "because", "while", "however", "moreover", "furthermore", "thus",
}


# Lower-case words that are allowed inside a title ("Related Work", "and",
# "Overview of the Method", ...).  Everything else must be capitalised.
_TITLE_CONNECTORS = {
    "of", "the", "and", "for", "in", "on", "to", "with", "a", "an", "via",
    "using", "based", "from", "at", "by", "vs", "versus", "or", "but", "into",
    "over", "under", "our", "their", "its", "your", "new", "novel", "general",
    "overall", "towards", "toward", "beyond", "across", "per", "as",
}


def _looks_like_title(text: str) -> bool:
    """
    Heuristic: does this string look like a *title* rather than a sentence?

    Headings are capitalised; running prose is not.  This is what stops
    "Model performance improved significantly" from being detected as a
    heading just because it starts with a section keyword.
    """
    words = [w.strip(":,.;()[]\"'").strip() for w in (text or "").split()]
    words = [w for w in words if w]
    if not words:
        return False
    for w in words:
        if w.lower() in _TITLE_CONNECTORS:
            continue
        if w[0].isupper() or w[0].isdigit():
            continue
        return False
    return True


def _numbered_heading_rest(text: str) -> Optional[str]:
    """
    Return the title part of a numbered heading, or ``None``.

    ``"1. Introduction"`` -> ``"Introduction"``
    ``"1. We ran the experiments"`` -> ``None`` (a numbered list item)
    """
    m = NUMBERED_HEADING_RE.match(text or "")
    if not m:
        return None
    rest = m.group("rest").strip()
    if not rest:
        return None
    if text.rstrip().endswith("."):
        return None
    words = rest.split()
    if len(words) > 8:
        return None
    first = words[0].lower().strip(",;:")
    if len(words) >= 3 and first in _SENTENCE_START_WORDS:
        return None
    return rest


def _is_heading_block(block: Dict[str, Any], body_font: float) -> bool:
    """Decide whether a block is a section heading."""
    text = (block.get("text") or "").strip()
    if not text or len(text) > 140:
        return False
    if is_running_header_or_footer(text):
        return False
    if TABLE_CAPTION_RE.match(text):
        return False

    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) > 2:
        return False

    words = text.split()
    n_words = len(words)
    font = float(block.get("font_size") or 0.0)
    bold = bool(block.get("is_bold"))

    # 1. Numbered heading: "1", "2.3", "3.1.4", "IV." ...
    #    Checked *before* the list-item exclusion, because "1. Introduction"
    #    is syntactically identical to a numbered list item.
    rest = _numbered_heading_rest(text)
    if rest is not None and n_words <= 12 and _looks_like_title(rest):
        return True

    # 2. A bulleted / numbered list item is never a heading.
    if LIST_MARKER_RE.match(text):
        return False

    # 3. Canonical academic section title ("Abstract", "Related Work", ...).
    #    The title-like check keeps body sentences such as
    #    "Model performance improved ..." from being mistaken for a heading.
    if SECTION_PATTERN.match(text) and n_words <= 6 and _looks_like_title(text):
        return True

    # 4. Visibly larger than the body text.
    if body_font > 0 and font >= body_font * 1.18 and n_words <= 12:
        if (not text.endswith(".") or n_words <= 5) and _looks_like_title(text):
            return True

    # 5. Bold run-in heading at body size.
    if bold and body_font > 0 and font >= body_font * 0.97 and n_words <= 10:
        if not text.endswith(".") and _looks_like_title(text):
            return True

    return False


def is_heading(block: Dict[str, Any], avg_font: float) -> bool:
    """
    Heuristic to determine if a block represents a section heading.
    (Public API kept for backwards compatibility.)
    """
    return _is_heading_block(block, avg_font or 10.0)


def _heading_level(text: str) -> int:
    """Infer the hierarchy level of a heading from its numbering."""
    m = re.match(r"^\s*(\d{1,3}(?:\.\d{1,3}){0,3})", text)
    if m:
        return len(m.group(1).split("."))
    if re.match(r"^\s*[IVXLC]{1,6}\.", text):
        return 1
    return 2


def build_sections(blocks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group blocks into sections delimited by detected headings.

    Each section carries its heading, hierarchy level, breadcrumb path and the
    body blocks that belong to it.
    """
    if not blocks:
        return []

    body_font = _body_font_size(blocks)
    sections: List[Dict[str, Any]] = []
    stack: List[Tuple[int, str]] = []

    current: Dict[str, Any] = {
        "heading": "Document Start",
        "level": 0,
        "path": [],
        "page": blocks[0].get("page", 1),
        "blocks": [],
    }

    for b in blocks:
        text = (b.get("text") or "").strip()
        if not text:
            continue
        if _is_heading_block(b, body_font):
            if current["blocks"]:
                sections.append(current)
            level = _heading_level(text)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
            current = {
                "heading": text,
                "level": level,
                "path": [h for _, h in stack],
                "page": b.get("page", 1),
                "blocks": [],
            }
        else:
            current["blocks"].append(b)

    if current["blocks"]:
        sections.append(current)

    return sections


def split_into_sections(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group blocks into sections delimited by detected headings.
    Returns a list of dicts: [{"heading": str, "text": str, "page": int}, ...]
    """
    sections = build_sections(_clean_blocks(blocks))
    out: List[Dict[str, Any]] = []
    for sec in sections:
        text = "\n\n".join((b.get("text") or "") for b in sec["blocks"]).strip()
        if not text:
            continue
        out.append({
            "heading": sec["heading"],
            "text": text,
            "page": sec.get("page", 1),
        })
    return out


# ---------------------------------------------------------------------------
# Units (sentences) & packing
# ---------------------------------------------------------------------------


def _build_units(blocks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Turn blocks into sentence-level units, remembering paragraph starts."""
    units: List[Dict[str, Any]] = []
    for b in blocks:
        text = b.get("text") or ""
        if not text.strip():
            continue
        sentences = split_sentences(text)
        if not sentences:
            continue
        for i, sentence in enumerate(sentences):
            units.append({
                "text": sentence,
                "tokens": estimate_tokens(sentence),
                "para_start": i == 0,
            })
    return units


def _overlap_units(
    units: Sequence[Dict[str, Any]],
    overlap: int,
    budget: int,
) -> List[Dict[str, Any]]:
    """Pick the trailing units to carry over into the next chunk."""
    if overlap <= 0 or not units:
        return []
    carry = list(units[-overlap:])
    while carry and sum(u["tokens"] for u in carry) > budget:
        carry.pop(0)
    return carry


def _pack_units(
    units: Sequence[Dict[str, Any]],
    sims: Optional[Sequence[float]],
    max_tokens: int,
    min_tokens: int,
    threshold: float,
    overlap: int,
) -> List[List[Dict[str, Any]]]:
    """
    Pack sentence units into chunks.

    A break is made when
      * the token budget would be exceeded (hard limit), or
      * the semantic similarity between two consecutive sentences drops below
        ``threshold`` **and** the chunk already holds at least ``min_tokens``.

    Consecutive chunks share ``overlap`` trailing sentences for context.
    """
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            groups.append(current)
        current = []
        current_tokens = 0

    for i, unit in enumerate(units):
        unit_tokens = unit["tokens"]

        if current and current_tokens + unit_tokens > max_tokens:
            # Hard budget: close the chunk before it overflows.
            flush()
            carry = _overlap_units(groups[-1], overlap, max_tokens - unit_tokens)
            current = list(carry)
            current_tokens = sum(u["tokens"] for u in current)
        elif (
            current
            and len(current) >= 2
            and current_tokens >= min_tokens
            and sims is not None
            and i - 1 < len(sims)
        ):
            # Semantic boundary between the previous unit and this one.
            sim = sims[i - 1]
            if sim is not None and sim < threshold:
                carry = _overlap_units(current, overlap, max_tokens)
                flush()
                current = list(carry)
                current_tokens = sum(u["tokens"] for u in current)

        current.append(unit)
        current_tokens += unit_tokens

    flush()
    return groups


def _pack_by_items(
    items: Sequence[str],
    max_tokens: int,
    min_tokens: int,
    overlap: int,
) -> List[str]:
    """Pack already-atomic items (reference entries, table rows) into chunks."""
    units = [{"text": t, "tokens": estimate_tokens(t), "para_start": True} for t in items]
    groups = _pack_units(units, None, max_tokens, min_tokens, 0.0, overlap)
    return ["\n".join(u["text"] for u in g) for g in groups]


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _split_table_blocks(
    blocks: Sequence[Dict[str, Any]],
) -> Tuple[Optional[int], List[int], List[int]]:
    """
    Locate a table inside a section.

    Returns ``(caption_index, row_block_indexes, remaining_indexes)``.
    ``caption_index`` is ``None`` when the section holds no table.
    """
    caption_idx: Optional[int] = None
    for i, b in enumerate(blocks):
        if is_table_caption((b.get("text") or "").strip()):
            caption_idx = i
            break
    if caption_idx is None:
        return None, [], list(range(len(blocks)))

    row_idxs: List[int] = []
    for i in range(caption_idx + 1, len(blocks)):
        lines = (blocks[i].get("text") or "").split("\n")
        if any(is_table_row(line) for line in lines):
            row_idxs.append(i)
        elif row_idxs:
            break  # the table ended

    remaining = [i for i in range(len(blocks)) if i != caption_idx and i not in row_idxs]
    return caption_idx, row_idxs, remaining


def _extract_table_chunks(
    blocks: Sequence[Dict[str, Any]],
    caption_idx: int,
    row_idxs: Sequence[int],
    heading: str,
    path: str,
    page: int,
    max_tokens: int,
    min_tokens: int,
) -> List[Dict[str, Any]]:
    """
    Emit self-contained chunks for a table.

    Every chunk repeats the caption and the header row, so a retrieved chunk
    always makes sense on its own.
    """
    caption = ((blocks[caption_idx].get("text") or "").split("\n")[0] or "").strip()
    rows: List[str] = []
    for i in row_idxs:
        for line in (blocks[i].get("text") or "").split("\n"):
            if is_table_row(line):
                rows.append(line.strip())
            elif rows and line.strip():
                # continuation of the previous row (wrapped cell)
                rows[-1] = f"{rows[-1]} {line.strip()}"
    if not rows:
        return []

    header = rows[0]
    body = rows[1:] if len(rows) > 1 else []

    # The caption and the header row are repeated in every chunk, so they
    # must be paid for out of the budget.
    fixed_tokens = estimate_tokens(caption) + estimate_tokens(header) + 4
    row_budget = max_tokens - fixed_tokens

    chunks: List[Dict[str, Any]] = []
    if body and row_budget >= 32:
        for group in _pack_by_items(body, row_budget, min_tokens, 0):
            text = f"{caption}\n{header}\n{group}"
            chunks.append({
                "heading": heading,
                "section_path": path,
                "text": text.strip(),
                "page": page,
                "strategy": "section_table",
            })
    elif body:
        # Not enough room for row chunks: keep the caption + header and let
        # the finalise pass split the rows if they still overflow.
        text = f"{caption}\n{header}\n" + "\n".join(body)
        chunks.append({
            "heading": heading,
            "section_path": path,
            "text": text.strip(),
            "page": page,
            "strategy": "section_table",
        })
    else:
        chunks.append({
            "heading": heading,
            "section_path": path,
            "text": f"{caption}\n{header}".strip(),
            "page": page,
            "strategy": "section_table",
        })
    return chunks


# ---------------------------------------------------------------------------
# Public chunking API
# ---------------------------------------------------------------------------


def semantic_split(
    text: str,
    embed_backend: EmbeddingBackend,
    threshold: float = 0.65,
    max_tokens: int = 512,
    min_tokens: int = 0,
) -> List[str]:
    """
    Split a section into semantically coherent passages.
    Computes consecutive sentence embeddings and breaks whenever cosine
    similarity drops below threshold or the accumulated chunk token limit is
    reached.  (Public API kept for backwards compatibility.)
    """
    if not text or not text.strip():
        return []

    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []

    try:
        embeddings = embed_backend.embed_documents(sentences)
    except Exception:
        return recursive_fallback_split(text, max_tokens=max_tokens)

    if not embeddings or len(embeddings) != len(sentences):
        return recursive_fallback_split(text, max_tokens=max_tokens)

    units = [{"text": s, "tokens": estimate_tokens(s), "para_start": False} for s in sentences]
    sims = [cosine_similarity(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)]
    groups = _pack_units(units, sims, max_tokens, min_tokens, threshold, 0)
    return [" ".join(u["text"] for u in g) for g in groups if g]


def recursive_fallback_split(
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
) -> List[str]:
    """
    Robust recursive character/token fallback chunking.
    Splits by word runs respecting the token budget, with a safety overlap so
    no information is lost at a chunk boundary.
    """
    if not text or not text.strip():
        return []

    words = text.split()
    if not words:
        return []

    if estimate_tokens(text) <= max_tokens:
        return [text.strip()]

    max_words = words_to_token_budget(max_tokens)
    step = max(1, max_words - words_to_token_budget(overlap_tokens))

    chunks: List[str] = []
    i = 0
    while i < len(words):
        chunk_words = words[i: i + max_words]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if i + max_words >= len(words):
            break
        i += step
    return chunks


def _make_chunk(
    heading: str,
    path: str,
    text: str,
    page: int,
    strategy: str,
) -> Dict[str, Any]:
    text = (text or "").strip()
    context = _context_header(heading, path, page)
    return {
        "heading": heading,
        "section_path": path,
        "text": text,
        "page": page,
        "strategy": strategy,
        "token_count": estimate_tokens(text),
        "context": context,
        "embedding_text": f"{context}\n\n{text}" if text else context,
    }


def _context_header(heading: str, path: str, page: int) -> str:
    bits = [f"Section: {path or heading or 'Document'}"]
    if page:
        bits.append(f"Page: {page}")
    return "\n".join(bits)


def _finalize(
    chunks: List[Dict[str, Any]],
    max_tokens: int,
    min_tokens: int,
) -> List[Dict[str, Any]]:
    """
    Enforce the output invariants.

    1. every chunk is non-empty
    2. every chunk fits the token budget (hard guarantee)
    3. no pathologically small chunk survives (they are merged into a neighbour)
    4. no duplicate chunks
    """
    # --- 1/2: non-empty + hard token budget -----------------------------
    prepared: List[Dict[str, Any]] = []
    for ch in chunks:
        text = (ch.get("text") or "").strip()
        if not text:
            continue
        if estimate_tokens(text) > max_tokens:
            for piece in recursive_fallback_split(text, max_tokens=max_tokens):
                if piece.strip():
                    sub = dict(ch)
                    sub["text"] = piece.strip()
                    sub["token_count"] = estimate_tokens(piece.strip())
                    prepared.append(sub)
            continue
        ch["text"] = text
        ch["token_count"] = estimate_tokens(text)
        prepared.append(ch)

    # --- 3: merge tiny chunks into a neighbour ---------------------------
    merged: List[Dict[str, Any]] = []
    for ch in prepared:
        if (
            merged
            and ch["token_count"] < min_tokens
            and merged[-1]["token_count"] + ch["token_count"] <= max_tokens
            and merged[-1].get("heading") == ch.get("heading")
        ):
            prev = merged[-1]
            prev["text"] = f"{prev['text']}\n\n{ch['text']}".strip()
            prev["token_count"] = estimate_tokens(prev["text"])
            continue
        merged.append(ch)

    # a tiny chunk that could not be merged backwards may still merge forward
    i = 0
    while i < len(merged) - 1:
        if (
            merged[i]["token_count"] < min_tokens
            and merged[i]["token_count"] + merged[i + 1]["token_count"] <= max_tokens
            and merged[i].get("heading") == merged[i + 1].get("heading")
        ):
            merged[i]["text"] = f"{merged[i]['text']}\n\n{merged[i + 1]['text']}".strip()
            merged[i]["token_count"] = estimate_tokens(merged[i]["text"])
            del merged[i + 1]
            continue
        i += 1

    # --- 4: de-duplicate --------------------------------------------------
    seen: set = set()
    final: List[Dict[str, Any]] = []
    for idx, ch in enumerate(merged):
        key = re.sub(r"[^a-z0-9]+", "", ch["text"].lower())[:400]
        if not key or key in seen:
            continue
        seen.add(key)
        ch["chunk_index"] = len(final)
        ch["context"] = _context_header(ch.get("heading", ""), ch.get("section_path", ""), ch.get("page", 1))
        ch["embedding_text"] = f"{ch['context']}\n\n{ch['text']}"
        final.append(ch)

    return final


def chunk_paper(
    blocks: List[Dict[str, Any]],
    embed_backend: Optional[EmbeddingBackend] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    threshold: float = DEFAULT_THRESHOLD,
    min_tokens: Optional[int] = None,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
) -> List[Dict[str, Any]]:
    """
    Execute the adaptive chunking pipeline.

    Args:
        blocks: layout blocks produced by :func:`src.extractor.extract_blocks`
        embed_backend: optional embedding backend used for semantic splitting
        max_tokens: hard token budget per chunk
        threshold: cosine similarity below which a semantic break is allowed
        min_tokens: minimum size of a chunk (defaults to 25% of ``max_tokens``)
        overlap_sentences: how many sentences are repeated between neighbours

    Returns:
        List of dicts with keys: heading, section_path, text, page, strategy,
        token_count, context, embedding_text, chunk_index.
    """
    if not blocks:
        return []

    max_tokens = max(32, int(max_tokens or DEFAULT_MAX_TOKENS))
    if min_tokens is None:
        min_tokens = max(16, int(max_tokens * 0.25))
    min_tokens = max(1, min(int(min_tokens), max_tokens - 1))
    overlap = max(0, int(overlap_sentences))

    clean = _clean_blocks(blocks)
    if not clean:
        return []

    sections = build_sections(clean)
    if not sections:
        return []

    has_headings = any(sec["heading"] != "Document Start" for sec in sections)

    chunks: List[Dict[str, Any]] = []

    if has_headings:
        for sec in sections:
            chunks.extend(_chunk_section(
                sec,
                embed_backend=embed_backend,
                max_tokens=max_tokens,
                min_tokens=min_tokens,
                threshold=threshold,
                overlap=overlap,
            ))
    else:
        # Continuous prose: no headings at all -> semantic chunking of the
        # whole document, or a deterministic recursive fallback.
        text = "\n\n".join((b.get("text") or "") for b in clean).strip()
        if not text:
            return []

        page = clean[0].get("page", 1)
        if estimate_tokens(text) <= max_tokens:
            chunks.append(_make_chunk("Document Content", "Document Content", text, page, "pure_semantic"))
        else:
            units = _build_units(clean)
            if embed_backend:
                groups = _pack_units(
                    units,
                    _similarities(units, embed_backend),
                    max_tokens,
                    min_tokens,
                    threshold,
                    overlap,
                )
                strategy = "pure_semantic"
            else:
                groups = _pack_units(units, None, max_tokens, min_tokens, threshold, overlap)
                strategy = "recursive_fallback"
            for group in groups:
                text = " ".join(u["text"] for u in group).strip()
                if text:
                    chunks.append(_make_chunk("Document Content", "Document Content", text, page, strategy))

    return _finalize(chunks, max_tokens, min_tokens)


def _similarities(units: Sequence[Dict[str, Any]], embed_backend: EmbeddingBackend) -> Optional[List[float]]:
    """Cosine similarity between consecutive unit embeddings (None on failure)."""
    if len(units) < 2:
        return None
    try:
        embeddings = embed_backend.embed_documents([u["text"] for u in units])
    except Exception:
        return None
    if not embeddings or len(embeddings) != len(units):
        return None
    try:
        return [cosine_similarity(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)]
    except Exception:
        return None


def _chunk_section(
    section: Dict[str, Any],
    embed_backend: Optional[EmbeddingBackend],
    max_tokens: int,
    min_tokens: int,
    threshold: float,
    overlap: int,
) -> List[Dict[str, Any]]:
    """Choose and apply the right strategy for a single section."""
    blocks = section.get("blocks") or []
    text = "\n\n".join((b.get("text") or "") for b in blocks).strip()
    if not text:
        return []

    heading = section.get("heading", "Document Start")
    path = " > ".join(section.get("path") or []) or heading
    page = section.get("page", 1)

    # 1. The whole section fits -> keep it structurally intact.
    if estimate_tokens(text) <= max_tokens:
        return [_make_chunk(heading, path, text, page, "section_whole")]

    # 2. Bibliography -> one chunk per reference entry (never per sentence).
    if is_reference_heading(heading):
        entries = split_reference_entries(text)
        if entries:
            out = []
            for group in _pack_by_items(entries, max_tokens, min_tokens, overlap):
                out.append(_make_chunk(heading, path, group, page, "section_references"))
            return out

    # 3. Table -> caption + header row repeated in every chunk.  The prose
    #    of the section that is not part of the table is chunked separately.
    caption_idx, row_idxs, remaining_idx = _split_table_blocks(blocks)
    if caption_idx is not None:
        out = _extract_table_chunks(
            blocks, caption_idx, row_idxs, heading, path, page, max_tokens, min_tokens,
        )
        rest = [blocks[i] for i in remaining_idx]
        if rest:
            out.extend(_pack_blocks(
                rest, heading, path, page, embed_backend, max_tokens, min_tokens, threshold, overlap,
            ))
        if out:
            return out

    # 4. Long section -> semantic sub-split (or deterministic fallback).
    return _pack_blocks(
        blocks, heading, path, page, embed_backend, max_tokens, min_tokens, threshold, overlap,
    )


def _pack_blocks(
    blocks: Sequence[Dict[str, Any]],
    heading: str,
    path: str,
    page: int,
    embed_backend: Optional[EmbeddingBackend],
    max_tokens: int,
    min_tokens: int,
    threshold: float,
    overlap: int,
) -> List[Dict[str, Any]]:
    """Chunk the prose of a section (semantic when possible)."""
    units = _build_units(blocks)
    if not units:
        return []

    if embed_backend:
        sims = _similarities(units, embed_backend)
        groups = _pack_units(units, sims, max_tokens, min_tokens, threshold, overlap)
        strategy = "section_semantic_subsplit"
    else:
        groups = _pack_units(units, None, max_tokens, min_tokens, threshold, overlap)
        strategy = "section_recursive_subsplit"

    out: List[Dict[str, Any]] = []
    for group in groups:
        piece = " ".join(u["text"] for u in group).strip()
        if piece:
            out.append(_make_chunk(heading, path, piece, page, strategy))
    return out
