"""
Safety guardrails for Voice-Enabled RAG:
1. Input Safety: Pre-filter for harmful/unsafe patterns and invalid lengths
2. Retrieval Confidence: Blocks generation if context match is too weak
3. Grounding Validation: Verifies output is grounded in retrieved facts
"""
from typing import List, Dict, Tuple
from config import MIN_RETRIEVAL_SCORE

_UNSAFE_PATTERNS = [
    "how to make a bomb", "how to make a weapon", "kill myself", "suicide method",
    "how to make drugs", "child abuse", "how to hack into", "credit card details"
]


def check_input_safety(query: str) -> Tuple[bool, str]:
    """Pre-filter run before retrieval."""
    if not query or len(query.strip()) < 2:
        return False, "Query is empty or too short."
    if len(query) > 2000:
        return False, "Query exceeds maximum allowed length."

    query_lower = query.lower()
    for pattern in _UNSAFE_PATTERNS:
        if pattern in query_lower:
            return False, "Query matched an unsafe content policy rule."

    return True, "ok"


def check_retrieval_confidence(chunks: List[Dict]) -> Tuple[bool, str]:
    """Evaluates whether retrieved chunks have sufficient relevance confidence."""
    if not chunks:
        return False, "No relevant chunks retrieved."
    
    best_score = chunks[0].get("rerank_score", chunks[0].get("score", 0.0))
    if best_score < MIN_RETRIEVAL_SCORE:
        return False, f"Best retrieval score {best_score:.3f} is below threshold {MIN_RETRIEVAL_SCORE}."
    
    return True, f"ok (confidence {best_score:.3f})"


def check_grounding(generation_result: Dict) -> Tuple[bool, str]:
    """Validates answer grounding and non-emptiness."""
    if not generation_result.get("grounded", False):
        return False, "Answer flagged as not grounded in retrieved context."
    if not generation_result.get("answer", "").strip():
        return False, "Answer returned by model was empty."
    return True, "ok"