"""
The three safety checks, run at three different points in the pipeline.
Each function returns (passed: bool, reason: str) so the harness can log
WHY something was refused — important for your demo video.
"""
from typing import List, Dict, Tuple
from config import MIN_RETRIEVAL_SCORE

OFF_TOPIC_KEYWORDS_HINT = (
    "This assumes your dataset is general-knowledge passages (MSMARCO-XI). "
    "Adjust the topic check below to match your actual dataset's domain."
)


def check_input_safety(query: str) -> Tuple[bool, str]:
    """Very basic pre-filter: blocks empty input and obvious junk before we
    waste a retrieval + LLM call on it. Extend this with a proper moderation
    API call if you have time (e.g. an LLM classification call)."""
    if not query or len(query.strip()) < 2:
        return False, "Query is empty or too short."
    if len(query) > 2000:
        return False, "Query is unreasonably long — likely not a real spoken question."
    return True, "ok"


def check_retrieval_confidence(chunks: List[Dict]) -> Tuple[bool, str]:
    """After retrieval: if even our best-matching chunk is a weak match, we
    should refuse rather than force the LLM to answer from irrelevant context."""
    if not chunks:
        return False, "No chunks retrieved at all."
    best_score = chunks[0].get("rerank_score", chunks[0].get("score", 0))
    if best_score < MIN_RETRIEVAL_SCORE:
        return False, f"Best retrieval score {best_score:.3f} is below threshold {MIN_RETRIEVAL_SCORE}."
    return True, f"ok (best score {best_score:.3f})"


def check_grounding(generation_result: Dict) -> Tuple[bool, str]:
    """After generation: trust the LLM's own self-reported 'grounded' flag
    (we forced it to output this in generation.py's JSON schema) as a first
    pass. This is a cheap check — a stronger version would run a second,
    separate LLM call asking 'does this answer follow from this context, yes/no'
    for a real second opinion instead of trusting the same call that produced it."""
    if not generation_result.get("grounded", False):
        return False, "Model flagged its own answer as not grounded in context."
    if not generation_result.get("answer", "").strip():
        return False, "Model returned an empty answer."
    return True, "ok"
