"""
In-process vector search using FAISS and PyTorch inference optimizations.
"""
import os
import pickle
import threading
from typing import List, Dict
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

from config import FAISS_INDEX_PATH, FAISS_METADATA_PATH, EMBEDDING_MODEL

_index = None
_metadata: List[Dict] = []
_embedder = None
_query_embed_cache = {}
_embed_cache_lock = threading.Lock()


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


def _try_git_lfs_pull():
    """Attempts to pull Git LFS binary files if run in an environment where git-lfs was not pulled."""
    try:
        import subprocess
        res = subprocess.run(["git", "lfs", "pull"], capture_output=True, text=True, timeout=90)
        print(f"Git LFS pull attempt output: {res.stdout}")
    except Exception as e:
        print(f"Git LFS pull attempt notice: {e}")


def _ensure_loaded():
    """Loads index and metadata into memory once at startup with automatic cloud fallback."""
    global _index, _metadata
    if _index is None or _index.ntotal == 0:
        # 1. Try loading from all pre-built binary candidates
        candidates = [
            (FAISS_INDEX_PATH, FAISS_METADATA_PATH),
            (os.path.join(os.path.dirname(__file__), "kb_faiss.bin"), os.path.join(os.path.dirname(__file__), "kb_metadata.pkl")),
            (os.path.join(os.path.dirname(__file__), "faiss_index.bin"), os.path.join(os.path.dirname(__file__), "faiss_metadata.pkl")),
        ]
        for bin_p, meta_p in candidates:
            if os.path.exists(bin_p) and os.path.exists(meta_p) and not _is_lfs_pointer(bin_p):
                try:
                    loaded_idx = faiss.read_index(bin_p)
                    with open(meta_p, "rb") as f:
                        loaded_meta = pickle.load(f)
                    if loaded_idx.ntotal > 0 and len(loaded_meta) > 0:
                        _index = loaded_idx
                        try:
                            _index.hnsw.efSearch = 24
                        except Exception:
                            pass
                        _metadata = loaded_meta
                        print(f"Loaded FAISS index from {bin_p} ({_index.ntotal} vectors).")
                        return
                except Exception as e:
                    print(f"FAISS load notice for {bin_p}: {e}")

        # 2. JSON Fallback: build from knowledge_base.json if binary files are unreadable
        kb_json = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
        if os.path.exists(kb_json):
            try:
                import json
                print("Building FAISS index from knowledge_base.json fallback...")
                with open(kb_json, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                if chunks:
                    build_index(chunks)
                    return
            except Exception as e:
                print(f"JSON fallback build notice: {e}")

        # 3. Final in-memory fallback
        dim = 768
        _index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        _metadata = []


def build_index(chunks: List[Dict]) -> int:
    """Additive indexing: embeds new chunks and appends to FAISS index."""
    global _index, _metadata

    existing_ids = set()
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_METADATA_PATH):
        try:
            _ensure_loaded()
            existing_ids = {m["chunk_id"] for m in _metadata}
            index = _index
            metadata = list(_metadata)
        except Exception:
            index = None
            metadata = []
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
        index.hnsw.efSearch = 24
    elif index.d != dim:
        print(f"Warning: Index dimension {index.d} does not match vector dimension {dim}. Creating new index...")
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 64
        index.hnsw.efSearch = 24
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


def _encode_query(query: str) -> np.ndarray:
    q_key = query.strip()
    with _embed_cache_lock:
        if q_key in _query_embed_cache:
            return _query_embed_cache[q_key]

    embedder = _get_embedder()
    with torch.inference_mode():
        vec = embedder.encode([q_key], convert_to_numpy=True, normalize_embeddings=True)
    vec = np.ascontiguousarray(vec, dtype=np.float32)

    with _embed_cache_lock:
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