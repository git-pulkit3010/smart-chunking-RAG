"""
Answer generation module using OpenRouter Chat Completions.
Synthesizes retrieved chunks and streams answers to the user.
"""

from typing import Any, Callable, Dict, Iterator, List, Optional
from openrouter import OpenRouter


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
        heading = meta.get("heading", "Section")
        page = meta.get("page", "?")
        source = meta.get("paper_title", "Paper")
        text = hit.get("text", "").strip()
        formatted_chunks.append(
            f"--- [{source} | Section: {heading} | Page: {page}] ---\n{text}"
        )

    context = "\n\n".join(formatted_chunks)
    return f"Context from ingested paper(s):\n\n{context}\n\nQuestion: {query}"


def stream_answer(
    query: str,
    hits: List[Dict[str, Any]],
    api_key: str,
    model: str = "openai/gpt-4o-mini",
    token_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Stream chat completion response from OpenRouter.

    Args:
        query: User's question
        hits: Retrieved context chunks
        api_key: OpenRouter API Key
        model: OpenRouter model identifier (e.g. 'openai/gpt-4o-mini')
        token_callback: Optional function invoked with each incoming delta string

    Returns:
        The full assembled answer string
    """
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set in .env!\n"
            "Please add your OpenRouter API key to the .env file to run queries."
        )

    user_prompt = build_rag_prompt(query, hits)
    full_response_parts: List[str] = []

    with OpenRouter(api_key=api_key) as client:
        stream = client.chat.send(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
        )

        for event in stream:
            choices = getattr(event, "choices", [])
            if choices:
                delta = getattr(choices[0], "delta", None)
                if delta:
                    content = getattr(delta, "content", None)
                    if content:
                        full_response_parts.append(content)
                        if token_callback:
                            token_callback(content)

    return "".join(full_response_parts)
