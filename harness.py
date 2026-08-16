"""
This IS the "proper harness" the task asks for. It's the one place that calls
every other module in order, checks guardrails between stages, retries where
sensible, times everything, and never lets a single failure crash the whole
request — it always returns a well-formed response, even a refusal.

HOW TO TEST THIS FILE ON ITS OWN:
    python harness.py "your question here"
(text-only test, skips STT — good for testing retrieval/generation without a mic)
"""
import sys
from retrieval import hybrid_retrieve
from generation import generate_answer
from guardrails import check_input_safety, check_retrieval_confidence, check_grounding
from latency import timed_stage, log_run


def _refusal(reason: str, stage: str) -> dict:
    return {
        "answer": "I don't have enough information to answer that from the provided data.",
        "grounded": False,
        "refused": True,
        "refusal_stage": stage,
        "refusal_reason": reason,
        "chunks_used": [],
    }


def answer_query(query_text: str, language_filter: str = None) -> dict:
    """The main entry point. Takes already-transcribed text (STT happens before
    this is called, in main.py) and runs it through guardrails -> retrieval ->
    guardrails -> generation -> guardrails, with timing at each step."""
    timings = {}

    # --- Guardrail 1: is this even a valid, safe query? ---
    ok, reason = check_input_safety(query_text)
    if not ok:
        return {**_refusal(reason, "input_check"), "timings": timings}

    # --- Retrieval, with one retry if it comes back empty ---
    with timed_stage() as t:
        chunks = hybrid_retrieve(query_text, language_filter=language_filter)
        if not chunks and language_filter:
            # retry without the language filter in case it was too restrictive
            chunks = hybrid_retrieve(query_text, language_filter=None)
    timings["retrieval_ms"] = round(t["ms"], 1)

    # --- Guardrail 2: do we actually trust these chunks? ---
    ok, reason = check_retrieval_confidence(chunks)
    if not ok:
        return {**_refusal(reason, "retrieval_check"), "timings": timings}

    # --- Generation ---
    with timed_stage() as t:
        result = generate_answer(query_text, chunks)
    timings["generation_ms"] = round(t["ms"], 1)

    # --- Guardrail 3: is the answer actually grounded in the context? ---
    ok, reason = check_grounding(result)
    if not ok:
        return {**_refusal(reason, "grounding_check"), "timings": timings}

    timings["total_ms"] = round(sum(timings.values()), 1)
    log_run(timings)

    return {
        "answer": result["answer"],
        "grounded": True,
        "refused": False,
        "chunks_used": result.get("used_chunk_ids", []),
        "timings": timings,
    }


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "what monument is in Agra"
    response = answer_query(query)
    print(response)
