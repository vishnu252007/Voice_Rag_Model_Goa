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
        num_cores = os.cpu_count() or 6
        torch.set_num_threads(max(4, min(num_cores, 8)))
        try:
            torch.set_num_interop_threads(2)
        except Exception:
            pass
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        _embedder.eval()
    return _embedder


def _is_lfs_pointer(path: str) -> bool:
    try:
        if os.path.exists(path) and os.path.getsize(path) < 1000:
            with open(path, "rb") as f:
                head = f.read(100)
            return b"version https://git-lfs" in head
    except Exception:
        pass
    return False


_query_embed_cache = {}


def _ensure_loaded():
    """Loads index and metadata into memory once at startup."""
    global _index, _metadata
    if _index is None:
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(FAISS_METADATA_PATH):
            raise RuntimeError(
                f"FAISS index or metadata not found at {FAISS_INDEX_PATH}. "
                f"Run 'python ingest.py --strategy metadata' first to build the index."
            )
        if _is_lfs_pointer(FAISS_INDEX_PATH) or _is_lfs_pointer(FAISS_METADATA_PATH):
            raise RuntimeError(
                f"FAISS index files at {FAISS_INDEX_PATH} are Git LFS pointer files. "
                f"Please run 'git lfs pull' to download the binary vector store, or run 'python ingest.py' to rebuild it."
            )
        _index = faiss.read_index(FAISS_INDEX_PATH)
        try:
            _index.hnsw.efSearch = 24
        except Exception:
            pass
        with open(FAISS_METADATA_PATH, "rb") as f:
            _metadata = pickle.load(f)


def _encode_query(query: str) -> np.ndarray:
    q_key = query.strip()
    if q_key in _query_embed_cache:
        return _query_embed_cache[q_key]
    embedder = _get_embedder()
    with torch.inference_mode():
        vec = embedder.encode([q_key], convert_to_numpy=True, normalize_embeddings=True)
    vec = np.ascontiguousarray(vec, dtype=np.float32)
    if len(_query_embed_cache) > 1000:
        _query_embed_cache.pop(next(iter(_query_embed_cache)))
    _query_embed_cache[q_key] = vec
    return vec


def vector_search(query: str, top_k: int = 10, language_filter: str = None) -> List[Dict]:
    """In-memory cosine similarity search via normalized inner product."""
    _ensure_loaded()
    if _index.ntotal == 0:
        return []

    query_vector = _encode_query(query)
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