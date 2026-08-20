"""
In-process vector search using FAISS and PyTorch inference optimizations.
"""
import os
import pickle
from typing import List, Dict
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

from config import FAISS_INDEX_PATH, FAISS_METADATA_PATH, EMBEDDING_MODEL

_index = None
_metadata: List[Dict] = []
_embedder = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        torch.set_num_threads(max(1, os.cpu_count() or 4))
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        _embedder.eval()
    return _embedder


def _ensure_loaded():
    """Loads index and metadata into memory once at startup."""
    global _index, _metadata
    if _index is None:
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(FAISS_METADATA_PATH):
            raise RuntimeError(
                f"FAISS index or metadata not found at {FAISS_INDEX_PATH}. "
                f"Run ingest.py first to build the index."
            )
        _index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_METADATA_PATH, "rb") as f:
            _metadata = pickle.load(f)


def build_index(chunks: List[Dict]) -> int:
    """Additive indexing: embeds new chunks and appends to FAISS index."""
    global _index, _metadata

    existing_ids = set()
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_METADATA_PATH):
        print("Existing FAISS index found — loading to add new chunks...")
        _ensure_loaded()
        existing_ids = {m["chunk_id"] for m in _metadata}
        index = _index
        metadata = list(_metadata)
    else:
        index = None
        metadata = []

    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
    skipped = len(chunks) - len(new_chunks)
    if skipped:
        print(f"Skipping {skipped} chunks already present in index.")
    if not new_chunks:
        print("No new chunks to add.")
        return 0

    embedder = _get_embedder()
    texts = [c["text"] for c in new_chunks]
    print(f"Embedding {len(texts)} new chunks with {EMBEDDING_MODEL}...")
    with torch.inference_mode():
        vectors = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    dim = vectors.shape[1]
    if index is None:
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 64
    elif index.d != dim:
        print(f"Warning: Index dimension {index.d} does not match vector dimension {dim}. Creating new index...")
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 64
        metadata = []

    index.add(vectors)
    metadata.extend([
        {
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "text": c["text"],
            "strategy": c["strategy"],
            "metadata": c["metadata"],
        }
        for c in new_chunks
    ])

    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(FAISS_METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    _index = index
    _metadata = metadata

    print(f"Saved FAISS index ({index.ntotal} total vectors) to {FAISS_INDEX_PATH}")
    print(f"Saved metadata ({len(metadata)} entries) to {FAISS_METADATA_PATH}")
    return len(new_chunks)


def vector_search(query: str, top_k: int = 10, language_filter: str = None) -> List[Dict]:
    """In-memory cosine similarity search via normalized inner product."""
    _ensure_loaded()
    if _index.ntotal == 0:
        return []

    embedder = _get_embedder()
    with torch.inference_mode():
        query_vector = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)

    search_k = min(_index.ntotal, top_k * 5 if language_filter else top_k)
    scores, indices = _index.search(query_vector, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or idx >= len(_metadata):
            continue
        item = _metadata[idx]
        item_lang = item.get("language") or item.get("metadata", {}).get("language")
        if language_filter and item_lang != language_filter:
            continue
        results.append({
            "text": item.get("text", ""),
            "chunk_id": item.get("chunk_id", ""),
            "doc_id": item.get("doc_id", ""),
            "score": float(score),
            "metadata": item.get("metadata", {"language": item_lang or "en"}),
        })
        if len(results) >= top_k:
            break

    return results
