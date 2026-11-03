"""
Answer generation module using NVIDIA NIM (OpenAI-compatible Chat Completions).
Synthesizes retrieved chunks and streams answers to the user.

Configuration comes from the environment (see `src/config.py`):
    NVIDIA_API_KEY   – bearer token for the NVIDIA endpoint
    NVIDIA_MODEL     – model identifier, e.g. "meta/llama-3.3-70b-instruct"
    NVIDIA_BASE_URL  – base URL (defaults to https://integrate.api.nvidia.com/v1)
"""

import json
from typing import Any, Callable, Dict, Iterator, List, Optional

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"


SYSTEM_PROMPT = (
    "You are an expert research assistant. Answer the user's question accurately using only "
    "the provided context from the research paper(s). Cite section headings and page numbers "
    "when useful. If the context does not contain enough information to answer the question, "
    "clearly state that the information is not present in the paper."
)


def build_rag_prompt(query: str, hits: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a structured context string for the LLM."""
    formatted_chunks = []
    for hit in hits:
        meta = hit.get("meta", {})
        heading = meta.get("section_path") or meta.get("heading", "Section")
        page = meta.get("page", "?")
        source = meta.get("paper_title", "Paper")
        text = hit.get("text", "").strip()
        formatted_chunks.append(
            f"--- [{source} | Section: {heading} | Page: {page}] ---\n{text}"
        )

    context = "\n\n".join(formatted_chunks)
    return f"Context from ingested paper(s):\n\n{context}\n\nQuestion: {query}"


def _iter_sse_content(lines: Iterator[str]) -> Iterator[str]:
    """Yield content deltas from a server-sent-events stream of chat completions."""
    for line in lines:
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if not data or data == "[DONE]":
            if data == "[DONE]":
                return
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        for choice in event.get("choices", []) or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                yield content


def stream_answer(
    query: str,
    hits: List[Dict[str, Any]],
    api_key: str,
    model: str,
    base_url: str = DEFAULT_BASE_URL,
    token_callback: Optional[Callable[[str], None]] = None,
    timeout: float = 300.0,
) -> str:
    """
    Stream a chat completion from the NVIDIA API.

    Args:
        query: User's question
        hits: Retrieved context chunks
        api_key: NVIDIA API key
        model: NVIDIA model identifier (e.g. 'meta/llama-3.3-70b-instruct')
        base_url: NVIDIA API base URL
        token_callback: Optional function invoked with each incoming delta string
        timeout: HTTP timeout in seconds

    Returns:
        The full assembled answer string
    """
    if not api_key:
        raise ValueError(
            "NVIDIA_API_KEY is not set in .env!\n"
            "Please add your NVIDIA API key to the .env file to run queries."
        )
    if not model:
        raise ValueError(
            "NVIDIA_MODEL is not set in .env!\n"
            "Please add the NVIDIA model you want to use (e.g. 'meta/llama-3.3-70b-instruct') "
            "to the .env file to run queries."
        )

    import httpx

    user_prompt = build_rag_prompt(query, hits)
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
        "temperature": 0.2,
    }

    full_response_parts: List[str] = []

    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", url, headers=headers, json=payload) as response:
            if response.status_code >= 400:
                body = response.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"NVIDIA API error {response.status_code} for model '{model}': {body}"
                )

            for content in _iter_sse_content(response.iter_lines()):
                full_response_parts.append(content)
                if token_callback:
                    token_callback(content)

    return "".join(full_response_parts)
