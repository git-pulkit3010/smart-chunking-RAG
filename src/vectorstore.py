"""
Vector store management using ChromaDB.
Provides persistent storage, retrieval, and collection management for paper chunks.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.api.models.Collection import Collection


class PaperVectorStore:
    """ChromaDB wrapper for indexing and querying research paper chunks."""

    def __init__(self, chroma_path: str = "./chroma_db", collection_name: str = "papers"):
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Research paper smart chunks"},
        )

    def add_paper(
        self,
        pdf_path: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> int:
        """
        Store chunks and their precomputed embeddings in ChromaDB.
        Overwrites any previous chunks for the same source paper to prevent duplicates.
        """
        if not chunks:
            return 0

        source_str = str(Path(pdf_path).resolve())
        stem = Path(pdf_path).stem

        # Remove existing chunks from the same paper if re-ingesting
        self.delete_paper(source_str)

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for i, ch in enumerate(chunks):
            cid = f"{stem}_{i}_{ch.get('page', 1)}"
            ids.append(cid)
            documents.append(ch["text"])
            metadatas.append({
                "heading": str(ch.get("heading", "Document Start")),
                "source": source_str,
                "paper_title": Path(pdf_path).name,
                "page": int(ch.get("page", 1)),
                "chunk_index": i,
                "strategy": str(ch.get("strategy", "unknown")),
            })

        # Batch insert into ChromaDB
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end_idx = i + batch_size
            self.collection.add(
                ids=ids[i:end_idx],
                documents=documents[i:end_idx],
                embeddings=embeddings[i:end_idx],
                metadatas=metadatas[i:end_idx],
            )

        return len(chunks)

    def query(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter_source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve the top-k most similar chunks for a query embedding."""
        total_chunks = self.collection.count()
        if total_chunks == 0:
            return []

        actual_k = min(k, total_chunks)
        where_clause = {"source": filter_source} if filter_source else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_k,
            where=where_clause,
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0] if "distances" in results else [0.0] * len(docs)

        hits: List[Dict[str, Any]] = []
        for doc, meta, cid, dist in zip(docs, metas, ids, distances):
            hits.append({
                "id": cid,
                "text": doc,
                "meta": meta,
                "distance": dist,
            })
        return hits

    def list_papers(self) -> List[Dict[str, Any]]:
        """Return a summary of all unique papers currently indexed."""
        all_items = self.collection.get(include=["metadatas"])
        metadatas = all_items.get("metadatas") or []

        papers_map: Dict[str, Dict[str, Any]] = {}
        for meta in metadatas:
            source = meta.get("source", "unknown")
            if source not in papers_map:
                papers_map[source] = {
                    "source": source,
                    "paper_title": meta.get("paper_title", Path(source).name),
                    "chunk_count": 0,
                    "max_page": 1,
                }
            papers_map[source]["chunk_count"] += 1
            papers_map[source]["max_page"] = max(
                papers_map[source]["max_page"], meta.get("page", 1)
            )

        return list(papers_map.values())

    def delete_paper(self, source_path: str):
        """Delete all chunks associated with a specific paper source."""
        source_str = str(Path(source_path).resolve())
        try:
            self.collection.delete(where={"source": source_str})
        except Exception:
            pass

    def clear(self):
        """Reset the collection by deleting all entries."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Research paper smart chunks"},
        )

    def count(self) -> int:
        """Return total number of chunks in the collection."""
        return self.collection.count()
