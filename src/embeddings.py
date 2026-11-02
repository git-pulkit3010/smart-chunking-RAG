"""
Embedding backends for Smart Chunking RAG.
Supports OpenRouter hosted embedding models (e.g. openai/text-embedding-3-small)
and local offline sentence-transformers models.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute the cosine similarity between two 1D vectors."""
    a = np.asarray(v1, dtype=np.float32)
    b = np.asarray(v2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class EmbeddingBackend(ABC):
    """Abstract interface for text embedding providers."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text documents."""
        pass

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        results = self.embed_documents([text])
        if not results:
            raise RuntimeError(f"Embedding failed for query: '{text[:50]}'")
        return results[0]

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name or ID of the embedding model."""
        pass


class OpenRouterEmbeddings(EmbeddingBackend):
    """
    OpenRouter-hosted embedding backend.
    Default model: 'openai/text-embedding-3-small' (8192 context, 1536 dim, $0.02 / 1M tokens).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "openai/text-embedding-3-small",
        batch_size: int = 32,
    ):
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required to use OpenRouter embeddings.\n"
                "Please add your API key to .env, or set EMBED_PROVIDER=local in .env to use local embeddings."
            )
        self.api_key = api_key
        self._model = model
        self.batch_size = batch_size

    @property
    def model_name(self) -> str:
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        from openrouter import OpenRouter

        all_embeddings: List[List[float]] = []

        with OpenRouter(api_key=self.api_key) as client:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                # Filter out pure whitespace or replace with placeholder to avoid API empty text errors
                clean_batch = [t if t.strip() else " " for t in batch]

                response = client.embeddings.generate(
                    model=self._model,
                    input=clean_batch,
                )

                if not hasattr(response, "data") or not response.data:
                    raise RuntimeError(f"OpenRouter embedding returned no data for batch {i//self.batch_size}")

                # Ensure embeddings maintain the original input order
                sorted_items = sorted(
                    response.data,
                    key=lambda item: getattr(item, "index", 0) if getattr(item, "index", 0) is not None else 0,
                )
                for item in sorted_items:
                    all_embeddings.append(item.embedding)

        return all_embeddings


class LocalSentenceTransformerEmbeddings(EmbeddingBackend):
    """
    Local offline embedding backend using sentence-transformers.
    Default model: 'all-MiniLM-L6-v2' or 'BAAI/bge-large-en'.
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model_name = model
        self._model = SentenceTransformer(model)

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()


def get_embedding_backend(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> EmbeddingBackend:
    """
    Factory function to initialize the configured embedding backend.
    """
    from src.config import get_settings

    settings = get_settings()
    provider = (provider or settings.embed_provider).strip().lower()
    model = (model or settings.embed_model).strip()
    api_key = api_key or settings.openrouter_api_key

    if provider == "openrouter":
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set in .env!\n"
                "To use OpenRouter embeddings (like openai/text-embedding-3-small), set OPENROUTER_API_KEY in .env.\n"
                "Alternatively, set EMBED_PROVIDER=local and EMBED_MODEL=all-MiniLM-L6-v2 in .env for offline local use."
            )
        return OpenRouterEmbeddings(api_key=api_key, model=model)
    elif provider == "local":
        return LocalSentenceTransformerEmbeddings(model=model)
    else:
        raise ValueError(f"Unknown EMBED_PROVIDER '{provider}'. Supported providers: 'openrouter', 'local'.")
