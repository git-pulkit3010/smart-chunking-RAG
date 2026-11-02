"""
Configuration module for Smart Chunking RAG.
Automatically loads environment variables from .env without requiring shell exports.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Find .env in the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# Load environment variables with override=False so any explicit CLI/system env vars take precedence,
# but the .env file is always loaded automatically if present.
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()


DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


@dataclass
class Settings:
    openrouter_api_key: Optional[str]
    nvidia_api_key: Optional[str]
    nvidia_model: str
    nvidia_base_url: str
    embed_model: str
    embed_provider: str
    chroma_path: str
    collection_name: str
    max_chunk_tokens: int
    semantic_similarity_threshold: float
    top_k_retrieval: int

    @classmethod
    def load(cls) -> "Settings":
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip() or None
        nvidia_api_key = os.getenv("NVIDIA_API_KEY", "").strip() or None
        nvidia_model = os.getenv("NVIDIA_MODEL", "").strip()
        nvidia_base_url = os.getenv("NVIDIA_BASE_URL", DEFAULT_NVIDIA_BASE_URL).strip().rstrip("/")
        embed_model = os.getenv("EMBED_MODEL", "openai/text-embedding-3-small").strip()
        embed_provider = os.getenv("EMBED_PROVIDER", "openrouter").strip().lower()
        chroma_path = os.getenv("CHROMA_PATH", "./chroma_db").strip()
        collection_name = os.getenv("COLLECTION_NAME", "papers").strip()

        try:
            max_chunk_tokens = int(os.getenv("MAX_CHUNK_TOKENS", "512"))
        except ValueError:
            max_chunk_tokens = 512

        try:
            threshold = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.65"))
        except ValueError:
            threshold = 0.65

        try:
            top_k = int(os.getenv("TOP_K_RETRIEVAL", "5"))
        except ValueError:
            top_k = 5

        return cls(
            openrouter_api_key=api_key,
            nvidia_api_key=nvidia_api_key,
            nvidia_model=nvidia_model,
            nvidia_base_url=nvidia_base_url,
            embed_model=embed_model,
            embed_provider=embed_provider,
            chroma_path=chroma_path,
            collection_name=collection_name,
            max_chunk_tokens=max_chunk_tokens,
            semantic_similarity_threshold=threshold,
            top_k_retrieval=top_k,
        )


def get_settings() -> Settings:
    """Retrieve application settings loaded from .env and environment."""
    return Settings.load()
