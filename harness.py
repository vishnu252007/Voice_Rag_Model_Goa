"""
The RAG Model Harness:
Structured orchestration connecting Guardrails -> Retrieval -> Guardrails -> Generation -> Guardrails.
Maintains granular stage timing metrics, caching, and recovery.
"""
import sys
import os
from collections import OrderedDict
from retrieval import hybrid_retrieve
from generation import generate_answer
from guardrails import check_input_safety, check_retrieval_confidence, check_grounding
from latency import timed_stage, log_run
from config import CACHE_ENABLED, CACHE_MAX_SIZE

_cache: "OrderedDict[tuple, dict]" = OrderedDict()


def _cache_key(query_text: str, language_filter: str) -> tuple:
    return (query_text.strip().lower(), language_filter)


def _cache_get(key):
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    return None


def _cache_put(key, value):
    _cache[key] = value
    _cache.move_to_end(key)
    if len(_cache) > CACHE_MAX_SIZE:
        _cache.popitem(last=False)


def _refusal(reason: str, stage: str, language: str = None) -> dict:
    default_msg = "I do not have sufficient information to answer this question from the indexed dataset."
    if language == "hi":
        default_msg = "दिए गए संदर्भ से इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी उपलब्ध नहीं है।"
    elif language == "te":
        default_msg = "అందించిన డేటా నుండి ఈ ప్రశ్నకు సమాధానం ఇవ్వడానికి తగినంత సమాచారం లేదు."
    elif language == "ta":
        default_msg = "வழங்கப்பட்ட தரவிலிருந்து இந்த கேள்விக்கு பதிலளிக்க போதுமான தகவல் இல்லை."

    return {
        "answer": default_msg,
        "grounded": False,
        "refused": True,
        "refusal_stage": stage,
        "refusal_reason": reason,
        "chunks_used": [],
    }


def answer_query(query_text: str, language_filter: str = None) -> dict:
    """Orchestrated RAG pipeline with high-precision per-stage latency tracking."""
    cache_key = _cache_key(query_text, language_filter)
    if CACHE_ENABLED:
        cached = _cache_get(cache_key)
        if cached is not None:
            result = dict(cached)
            result["timings"] = {**result.get("timings", {}), "cache_hit": True}
            return result

    timings = {}

    # Stage 1: Input Processing & Safety Guardrail
    with timed_stage() as t_in:
        ok, reason = check_input_safety(query_text)
    timings["query_processing_ms"] = round(t_in["ms"], 2)

    if not ok:
        timings["total_ms"] = round(sum(v for k, v in timings.items() if isinstance(v, (int, float))), 2)
        log_run(timings)
        return {**_refusal(reason, "input_check", language_filter), "timings": timings}

    # Stage 2: Hybrid Retrieval (Dense Vector FAISS + BM25)
    with timed_stage() as t_ret:
        chunks = hybrid_retrieve(query_text, language_filter=language_filter)
        if not chunks and language_filter:
            chunks = hybrid_retrieve(query_text, language_filter=None)
    timings["retrieval_ms"] = round(t_ret["ms"], 2)

    # Stage 3: Retrieval Confidence Check
    with timed_stage() as t_conf:
        ok, reason = check_retrieval_confidence(chunks)
    timings["confidence_check_ms"] = round(t_conf["ms"], 2)

    if not ok:
        timings["total_ms"] = round(sum(v for k, v in timings.items() if isinstance(v, (int, float))), 2)
        log_run(timings)
        return {**_refusal(reason, "retrieval_check", language_filter), "timings": timings}

    # Stage 4: Answer Generation
    with timed_stage() as t_gen:
        result = generate_answer(query_text, chunks)
    timings["generation_ms"] = round(t_gen["ms"], 2)

    # Stage 5: Grounding Validation Guardrail
    with timed_stage() as t_grd:
        ok, reason = check_grounding(result)
    timings["grounding_ms"] = round(t_grd["ms"], 2)

    timings["total_ms"] = round(sum(v for k, v in timings.items() if isinstance(v, (int, float))), 2)
    log_run(timings)

    if not ok:
        ref_resp = _refusal(reason, "grounding_check", language_filter)
        if result.get("answer") and not str(result["answer"]).startswith("Error"):
            ref_resp["answer"] = result["answer"]
        return {**ref_resp, "timings": timings}

    sources = [
        {
            "chunk_id": c["chunk_id"],
            "text": c["text"][:160] + "...",
            "score": round(float(c.get("rerank_score", c.get("score", 0.0))), 3)
        }
        for c in chunks[:3]
    ]

    response = {
        "answer": result["answer"],
        "translations": result.get("translations", {}),
        "grounded": True,
        "refused": False,
        "chunks_used": result.get("used_chunk_ids", []),
        "sources": sources,
        "timings": timings,
    }

    if CACHE_ENABLED:
        _cache_put(cache_key, response)

    return response


if __name__ == "__main__":
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    query = sys.argv[1] if len(sys.argv) > 1 else "What was the Manhattan project?"
    print(answer_query(query))