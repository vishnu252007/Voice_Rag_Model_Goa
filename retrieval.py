"""
Step 3: Hybrid Retrieval combining Dense Vector Search + BM25 keyword matching + optional Reranker.
"""
import sys
import os
import math
import pickle
from typing import List, Dict
from rank_bm25 import BM25Okapi

from config import TOP_K_RETRIEVE, TOP_K_FINAL, VECTOR_BACKEND, RERANKER_ENABLED, RERANKER_MODEL

if VECTOR_BACKEND == "faiss":
    from faiss_store import vector_search
else:
    from vectorstore import vector_search

_reranker = None
_bm25_index = None
_bm25_corpus = None


def _try_autoload_bm25():
    """Autoloads BM25 index from cache if available."""
    cache_path = os.path.join(os.path.dirname(__file__), "bm25_chunks.pkl")
    if os.path.exists(cache_path):
        try:
            if os.path.getsize(cache_path) < 1000:
                with open(cache_path, "rb") as f:
                    head = f.read(100)
                if b"version https://git-lfs" in head:
                    print("Notice: bm25_chunks.pkl is a Git LFS pointer. Run 'git lfs pull' or 'python ingest.py' to populate.")
                    return
            with open(cache_path, "rb") as f:
                chunks = pickle.load(f)
            build_bm25_index(chunks)
        except Exception as e:
            print(f"BM25 autoload skipped: {e}")


def build_bm25_index(all_chunks: List[Dict]):
    """Builds an in-memory BM25 index over all indexed chunks."""
    global _bm25_index, _bm25_corpus
    _bm25_corpus = all_chunks
    tokenized = [c["text"].lower().split() for c in all_chunks]
    _bm25_index = BM25Okapi(tokenized) if tokenized else None


def bm25_search(query: str, top_k: int = 10, language_filter: str = None) -> List[Dict]:
    """Runs fast BM25 search with optional language filtering."""
    if _bm25_index is None or not _bm25_corpus:
        return []
    tokens = query.lower().split()
    if not tokens:
        return []
    scores = _bm25_index.get_scores(tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k * 2]
    
    results = []
    for idx in ranked_indices:
        if scores[idx] <= 0:
            break
        item = _bm25_corpus[idx]
        if language_filter and item.get("metadata", {}).get("language") != language_filter:
            continue
        results.append({**item, "score": float(scores[idx])})
        if len(results) >= top_k:
            break
    return results


def _reciprocal_rank_fusion(vector_results: List[Dict], bm25_results: List[Dict], k: int = 60) -> List[Dict]:
    """Merges dense and sparse rankings using Reciprocal Rank Fusion (RRF)."""
    fused_scores = {}
    chunk_lookup = {}

    for rank, item in enumerate(vector_results):
        cid = item["chunk_id"]
        fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        chunk_lookup[cid] = item

    for rank, item in enumerate(bm25_results):
        cid = item["chunk_id"]
        fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        chunk_lookup.setdefault(cid, item)

    merged = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [{**chunk_lookup[cid], "fused_score": score} for cid, score in merged]


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(query: str, candidates: List[Dict], top_k: int = TOP_K_FINAL) -> List[Dict]:
    """Scores candidate passages against the query using a Cross-Encoder."""
    if not candidates:
        return []
    reranker = _get_reranker()
    pairs = [[query, c["text"]] for c in candidates]
    raw_scores = reranker.predict(pairs)
    
    for c, raw in zip(candidates, raw_scores):
        raw_val = float(raw)
        c["rerank_score_raw"] = raw_val
        c["rerank_score"] = float(1.0 / (1.0 + math.exp(-raw_val))) if raw_val < 700 else 1.0

    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]


def hybrid_retrieve(query: str, language_filter: str = None, debug_timing: bool = None) -> List[Dict]:
    """End-to-end hybrid retrieval: Vector + BM25 -> RRF -> (optional) Cross-Encoder."""
    import time
    if debug_timing is None:
        debug_timing = os.getenv("DEBUG_TIMING", "false").lower() == "true"

    t0 = time.perf_counter()
    v_results = vector_search(query, top_k=TOP_K_RETRIEVE, language_filter=language_filter)
    t1 = time.perf_counter()
    b_results = bm25_search(query, top_k=TOP_K_RETRIEVE, language_filter=language_filter)
    t2 = time.perf_counter()
    fused = _reciprocal_rank_fusion(v_results, b_results)
    t3 = time.perf_counter()
    
    candidates = fused[:TOP_K_FINAL]
    if RERANKER_ENABLED and candidates:
        final = rerank(query, candidates, top_k=TOP_K_FINAL)
    else:
        # Pass vector / RRF score directly
        for c in candidates:
            c["rerank_score"] = c.get("score", c.get("fused_score", 0.5))
        final = candidates
    t4 = time.perf_counter()

    if debug_timing:
        print(
            f"    [retrieval] vector={(t1-t0)*1000:.2f}ms  "
            f"bm25={(t2-t1)*1000:.2f}ms  rrf={(t3-t2)*1000:.2f}ms  "
            f"rerank={(t4-t3)*1000:.2f}ms  total={(t4-t0)*1000:.2f}ms",
            flush=True
        )

    return final


_try_autoload_bm25()

if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "What was the Manhattan Project?"
    results = hybrid_retrieve(q)
    for r in results:
        print(f"[{r.get('rerank_score', 0):.3f}] {r['text'][:100]}...")