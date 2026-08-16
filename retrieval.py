"""
Step 3: given a query, get the best possible chunks — not just the top vector match.

Combines:
  - vector_search (meaning-based, from vectorstore.py)
  - BM25 (exact keyword match, catches names/numbers vectors blur)
  - a rerank pass on the merged shortlist (slower but accurate, so we only run it
    on ~20 candidates instead of the whole dataset)

HOW TO TEST THIS FILE ON ITS OWN:
    python retrieval.py "your test question here"
(assumes you've already run vectorstore.py's upload once so there's data to search)
"""
import sys
import math
from typing import List, Dict
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from vectorstore import vector_search
from config import TOP_K_RETRIEVE, TOP_K_FINAL

_reranker = None
_bm25_index = None
_bm25_corpus = None  # keep the original chunk dicts aligned with the BM25 index


def _try_autoload_bm25():
    """If ingest.py has already been run, load its saved chunk cache automatically
    so main.py doesn't need a separate manual step to enable BM25 search."""
    import os, pickle
    cache_path = os.path.join(os.path.dirname(__file__), "bm25_chunks.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            chunks = pickle.load(f)
        build_bm25_index(chunks)


def build_bm25_index(all_chunks: List[Dict]):
    """Call this once after uploading chunks to vectorstore, so BM25 has something
    to search too. In production you'd persist this; for a hackathon, rebuilding
    it at startup from the same chunk list you indexed is simplest."""
    global _bm25_index, _bm25_corpus
    _bm25_corpus = all_chunks
    tokenized = [c["text"].lower().split() for c in all_chunks]
    _bm25_index = BM25Okapi(tokenized)


def bm25_search(query: str, top_k: int = 20) -> List[Dict]:
    if _bm25_index is None:
        return []  # if no index built yet, hybrid search just falls back to vector-only
    scores = _bm25_index.get_scores(query.lower().split())
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [{**_bm25_corpus[i], "score": float(scores[i])} for i in ranked]


def _reciprocal_rank_fusion(vector_results: List[Dict], bm25_results: List[Dict], k: int = 60) -> List[Dict]:
    """Merges two ranked lists into one. Instead of trying to compare raw scores
    from two different methods (which aren't on the same scale), we use each
    result's RANK POSITION — this is the standard trick (reciprocal rank fusion)."""
    fused_scores = {}
    chunk_lookup = {}

    for rank, item in enumerate(vector_results):
        cid = item["chunk_id"]
        fused_scores[cid] = fused_scores.get(cid, 0) + 1.0 / (k + rank + 1)
        chunk_lookup[cid] = item

    for rank, item in enumerate(bm25_results):
        cid = item["chunk_id"]
        fused_scores[cid] = fused_scores.get(cid, 0) + 1.0 / (k + rank + 1)
        chunk_lookup.setdefault(cid, item)

    merged = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [{**chunk_lookup[cid], "fused_score": score} for cid, score in merged]


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        # small, fast cross-encoder — good enough for reranking ~20 candidates in a hackathon
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def rerank(query: str, candidates: List[Dict], top_k: int = TOP_K_FINAL) -> List[Dict]:
    if not candidates:
        return []
    reranker = _get_reranker()
    pairs = [[query, c["text"]] for c in candidates]
    raw_scores = reranker.predict(pairs)
    # The cross-encoder outputs raw, unbounded logits (can be very negative for
    # poor matches, e.g. -8 to -10) — NOT a 0-1 similarity score. We squash them
    # through a sigmoid so MIN_RETRIEVAL_SCORE in config.py (a 0-1 threshold)
    # actually means something. Without this, every query was getting refused.
    for c, raw in zip(candidates, raw_scores):
        c["rerank_score_raw"] = float(raw)
        c["rerank_score"] = float(1 / (1 + math.exp(-raw)))
    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]


def hybrid_retrieve(query: str, language_filter: str = None, debug_timing: bool = True) -> List[Dict]:
    """The main function everything else calls. Returns the final, reranked,
    ready-to-use chunks for a query."""
    import time
    # NOTE: reverted from parallel (ThreadPoolExecutor) back to sequential —
    # testing whether BM25's CPU-heavy pure-Python scoring loop was holding
    # the GIL and starving the Qdrant search thread's actual execution time.
    t0 = time.perf_counter()
    v_results = vector_search(query, top_k=TOP_K_RETRIEVE, language_filter=language_filter)
    t1 = time.perf_counter()
    b_results = bm25_search(query, top_k=TOP_K_RETRIEVE)
    t2 = time.perf_counter()
    fused = _reciprocal_rank_fusion(v_results, b_results)
    t3 = time.perf_counter()
    final = rerank(query, fused[:TOP_K_RETRIEVE])
    t4 = time.perf_counter()

    if debug_timing:
        line = (f"vector_search={(t1-t0)*1000:.0f}ms  bm25_search={(t2-t1)*1000:.0f}ms  "
                f"fusion={(t3-t2)*1000:.0f}ms  rerank={(t4-t3)*1000:.0f}ms\n")
        print(f"    [retrieval breakdown SEQUENTIAL] {line}", flush=True)
        import os
        log_path = os.path.join(os.path.dirname(__file__), "retrieval_debug.log")
        with open(log_path, "a") as f:
            f.write(f"SEQUENTIAL: {line}")

    return final

    return final


_try_autoload_bm25()


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "what monument is in Agra"
    for r in hybrid_retrieve(q):
        print(f"  rerank_score={r.get('rerank_score', 0):.3f}  {r['text']}")