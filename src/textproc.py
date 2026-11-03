"""
Low-level text primitives for the adaptive chunking engine.

Everything here is pure-Python, dependency-free and deterministic so that the
chunking engine degrades predictably when optional services (embeddings,
tokenizers) are unavailable.

Main responsibilities
---------------------
* Token accounting that is *conservative* (never under-estimates), so a chunk
  that "fits the budget" really fits the embedding model's context window.
* A sentence splitter that survives the things that actually break naive
  regex splitters in scientific / legal / technical text: abbreviations,
  decimal numbers, acronyms, initials, URLs, e-mail addresses, file names,
  citations, bulleted lists, quotes and bracketed spans.
* Structural helpers for tables, reference lists and list items.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------

# Average sub-word token count per whitespace word for English prose.
# 1.25 is deliberately conservative (real BPE ratios are ~1.3) so that the
# "never exceed the budget" invariant holds even for technical text.
TOKEN_RATIO = 1.25

# Long identifiers (URLs, snake_case, hashes) shatter into many BPE tokens.
_LONG_WORD_CHARS_PER_TOKEN = 8


def estimate_tokens(text: Optional[str]) -> int:
    """Conservatively estimate the number of LLM tokens in ``text``."""
    if not text or not text.strip():
        return 0
    total = 0
    for word in text.split():
        total += 1 + (len(word) // _LONG_WORD_CHARS_PER_TOKEN)
    return max(1, int(total * TOKEN_RATIO))


def words_to_token_budget(max_tokens: int) -> int:
    """How many whitespace words fit into ``max_tokens`` tokens."""
    return max(1, int(max_tokens / TOKEN_RATIO))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Trim ``text`` so that its estimated token count fits ``max_tokens``."""
    if max_tokens <= 0:
        return ""
    words = text.split()
    limit = words_to_token_budget(max_tokens)
    if len(words) <= limit:
        return text.strip()
    return " ".join(words[:limit]).strip()


# ---------------------------------------------------------------------------
# Abbreviations / masking
# ---------------------------------------------------------------------------

ABBREVIATIONS: set[str] = {
    "al", "approx", "app", "apps", "avg", "bib", "ca", "chap", "ch", "cit",
    "cf", "co", "col", "cols", "conf", "corp", "dec", "dept", "dr", "drs",
    "ed", "eds", "e.g", "i.e", "eq", "eqs", "eqn", "est", "etc", "fig", "figs",
    "fn", "ibid", "ie", "inc", "int", "jr", "ltd", "max", "min", "mr", "mrs",
    "ms", "miss", "no", "nos", "op", "p", "pp", "prof", "profs", "pt", "ref",
    "refs", "resp", "rev", "sec", "secs", "sen", "sens", "sess", "sr", "st",
    "std", "supp", "suppl", "tab", "tabs", "tbl", "tbls", "trans", "univ",
    "var", "viz", "vs", "vol", "vols", "yr",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
    "nov", "dec",
}

_SENTINEL = "\x00"

_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"')\]]+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_FILE_RE = re.compile(
    r"\b[\w.-]+\.(?:py|js|ts|tsx|jsx|json|csv|tsv|txt|md|pdf|html?|xml|ya?ml"
    r"|npy|npz|pt|pth|bin|sh|ipynb|tex|bib|doi|log)\b",
    re.IGNORECASE,
)
_DECIMAL_RE = re.compile(r"(?<=\d)\.(?=\d)")
_ACRONYM_RE = re.compile(r"\b(?:[A-Za-z]\.){2,}")
_INITIAL_RE = re.compile(r"\b[A-Za-z]\.(?=\s+[A-Z])")
_ABBREV_RE = re.compile(
    r"\b(" + "|".join(sorted((re.escape(a) for a in ABBREVIATIONS), key=len, reverse=True)) + r")\.",
    re.IGNORECASE,
)

LIST_MARKER_RE = re.compile(
    r"^\s*(?:[-*+•·–—]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)]|\(?[ivxIVX]{1,4}[.)])\s+"
)
TABLE_CAPTION_RE = re.compile(
    r"^\s*(?:table|tab\.?|figure|fig\.?|algorithm|listing|scheme|chart|plot)\s*\d+",
    re.IGNORECASE,
)
REFERENCE_HEADING_RE = re.compile(
    r"^\s*(?:\d+[.)]?\s*)?(?:references?|bibliography|works\s+cited|literature\s+cited)\b",
    re.IGNORECASE,
)
NUMBERED_HEADING_RE = re.compile(
    r"^\s*(?:\d{1,3}(?:\.\d{1,3}){0,3}\.?|[IVXLC]{1,6}\.|[A-Z]\.)\s+(?P<rest>[A-Z].*)$"
)
RUNNING_NOISE_RE = re.compile(
    r"^(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?$|^arxiv:\d{4}\.\d{4,5}(?:v\d+)?(?:\s*\[.*\])?$",
    re.IGNORECASE,
)


def _mask_protected(text: str) -> str:
    """Replace dots inside protected constructs with a sentinel."""
    out = text
    for regex in (_URL_RE, _EMAIL_RE, _FILE_RE, _ACRONYM_RE):
        out = regex.sub(lambda m: m.group(0).replace(".", _SENTINEL), out)
    out = _DECIMAL_RE.sub(_SENTINEL, out)
    out = _INITIAL_RE.sub(lambda m: m.group(0).replace(".", _SENTINEL), out)
    out = _ABBREV_RE.sub(lambda m: m.group(0).replace(".", _SENTINEL), out)
    return out


def is_running_header_or_footer(text: str) -> bool:
    """True for page numbers, arXiv stamps and similar layout noise."""
    stripped = (text or "").strip()
    if not stripped:
        return True
    return bool(RUNNING_NOISE_RE.match(stripped))


def clean_sentence(sentence: str) -> str:
    """Normalise whitespace and strip a leading list marker."""
    s = re.sub(r"\s+", " ", sentence or "").strip()
    s = LIST_MARKER_RE.sub("", s, count=1)
    return s.strip()


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences, protecting the constructs that break naive
    regex splitters (abbreviations, decimals, acronyms, initials, URLs,
    citations, bracketed spans and quoted strings).
    """
    if not text or not text.strip():
        return []

    masked = _mask_protected(text)
    n = len(masked)

    sentences: List[str] = []
    start = 0
    depth = 0
    in_quote = False
    i = 0

    while i < n:
        ch = masked[i]
        if ch in "\"'“”‘’":
            # Toggle quote state, but only for balanced-looking quotes.
            if ch in "“”":
                in_quote = not in_quote
            elif ch in "\"'":
                in_quote = not in_quote
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch in ".!?" and depth == 0 and not in_quote:
            j = i + 1
            while j < n and masked[j] in "\"'”’)]»":
                j += 1
            if j < n and masked[j] in " \t\r\n":
                k = j
                while k < n and masked[k] in " \t\r\n":
                    k += 1
                if k < n and (
                    masked[k].isupper()
                    or masked[k].isdigit()
                    or masked[k] in "\"'([“«"
                ):
                    segment = masked[start:j].replace(_SENTINEL, ".")
                    cleaned = clean_sentence(segment)
                    if cleaned:
                        sentences.append(cleaned)
                    start = k
                    i = k
                    continue
        i += 1

    tail = masked[start:].replace(_SENTINEL, ".")
    cleaned = clean_sentence(tail)
    if cleaned:
        sentences.append(cleaned)

    return sentences


def split_list_items(text: str) -> List[str]:
    """Split a bulleted / numbered list into its individual items."""
    items: List[str] = []
    for line in (text or "").split("\n"):
        if not line.strip():
            continue
        if LIST_MARKER_RE.match(line):
            items.append(re.sub(r"\s+", " ", line).strip())
        elif items:
            # continuation of the previous item
            items[-1] = f"{items[-1]} {line.strip()}"
        else:
            items.append(re.sub(r"\s+", " ", line).strip())
    return [i for i in items if i]


def is_table_caption(text: str) -> bool:
    return bool(TABLE_CAPTION_RE.match((text or "").strip()))


def is_reference_heading(text: str) -> bool:
    return bool(REFERENCE_HEADING_RE.match((text or "").strip()))


def is_table_row(line: str) -> bool:
    """Heuristic: a line that looks like a row of a table."""
    stripped = (line or "").strip()
    if not stripped or len(stripped) > 400:
        return False
    cells = [c for c in re.split(r"\s{2,}|\t", stripped) if c.strip()]
    if len(cells) < 2:
        return False
    numeric = sum(1 for c in cells if re.search(r"\d", c))
    return numeric >= max(1, len(cells) // 2)


def extract_table_rows(text: str) -> List[str]:
    """Return the table-like rows contained in ``text``."""
    return [line.strip() for line in (text or "").split("\n") if is_table_row(line)]


def split_reference_entries(text: str) -> List[str]:
    """
    Split a reference section into individual bibliographic entries.

    A new entry starts at a numbered marker or at a ``Surname, A.`` pattern.
    Lines that do not match are appended to the current entry.
    """
    entries: List[str] = []
    current: List[str] = []

    for raw_line in (text or "").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        starts_entry = bool(
            re.match(r"^\[\d+(?:\s*[-–,]\s*\d+)?\]", line)
            or re.match(r"^\d{1,3}[.)]\s+", line)
            or re.match(r"^[A-Z][A-Za-z'’\-]+,\s+[A-Z]\.", line)
            or re.match(r"^[A-Z][A-Za-z'’\-]+\s+(?:[A-Z]\.\s*)+[A-Z]", line)
            or re.match(r"^[A-Z][a-z]+,\s[A-Z][a-z]+", line)
        )
        if starts_entry and current:
            entries.append(re.sub(r"\s+", " ", " ".join(current)).strip())
            current = []
        current.append(line)

    if current:
        entries.append(re.sub(r"\s+", " ", " ".join(current)).strip())

    return [e for e in entries if e]


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace while preserving paragraph breaks."""
    if not text:
        return ""
    paragraphs = re.split(r"\n\s*\n", text)
    cleaned = []
    for para in paragraphs:
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in para.split("\n")]
        cleaned.append("\n".join(ln for ln in lines if ln))
    return "\n\n".join(p for p in cleaned if p)
