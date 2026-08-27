"""
Safety guardrails for Voice-Enabled RAG:
1. Input Safety: Pre-filter for harmful/unsafe patterns and invalid lengths
   - Covers English, Hindi, Telugu, and Tamil patterns
   - Basic obfuscation normalization
2. Retrieval Confidence: Blocks generation if context match is too weak
3. Grounding Validation: Verifies output is grounded in retrieved facts
"""
import re
from typing import List, Dict, Tuple
from config import MIN_RETRIEVAL_SCORE

# Multilingual unsafe patterns covering English, Hindi (Devanagari + transliterated),
# Telugu, and Tamil — the four languages this system serves.
_UNSAFE_PATTERNS = [
    # --- English ---
    "how to make a bomb", "how to make a weapon", "how to build a bomb",
    "kill myself", "suicide method", "how to commit suicide", "ways to die",
    "how to make drugs", "how to make meth", "how to cook meth",
    "child abuse", "child exploitation",
    "how to hack into", "how to hack a", "password cracking",
    "credit card details", "steal credit card", "steal identity",
    "how to kill someone", "how to poison someone", "how to murder",

    # --- Hindi (Devanagari) ---
    "बम कैसे बनाएं", "बम बनाने का तरीका", "हथियार कैसे बनाएं",
    "आत्महत्या कैसे करें", "आत्महत्या का तरीका", "खुद को मारना",
    "ड्रग्स कैसे बनाएं", "नशीली दवा बनाना",
    "बच्चों का शोषण", "बाल शोषण",
    "हैक कैसे करें", "पासवर्ड कैसे तोड़ें",
    "किसी को कैसे मारें", "जहर कैसे दें",

    # --- Hindi (transliterated / Romanized) ---
    "bomb kaise banaye", "hatiyar kaise banaye",
    "suicide kaise kare", "khud ko kaise mare",
    "drugs kaise banaye",
    "hack kaise kare",

    # --- Telugu ---
    "బాంబు ఎలా తయారు చేయాలి", "ఆయుధం ఎలా తయారు చేయాలి",
    "ఆత్మహత్య ఎలా చేయాలి", "ఆత్మహత్య చేసుకోవడం",
    "డ్రగ్స్ ఎలా తయారు చేయాలి",
    "హ్యాక్ ఎలా చేయాలి",
    "ఎవరినైనా ఎలా చంపాలి",

    # --- Tamil ---
    "குண்டு எப்படி செய்வது", "ஆயுதம் எப்படி செய்வது",
    "தற்கொலை செய்வது எப்படி",
    "போதைப்பொருள் தயாரிப்பது",
    "ஹேக் செய்வது எப்படி",
]


def _normalize_query(query: str) -> str:
    """Normalizes query for safety matching: lowercase, collapse whitespace,
    strip zero-width characters and common obfuscation separators."""
    text = query.lower().strip()
    # Remove zero-width chars, dots/dashes between letters used for obfuscation
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    # Collapse multiple spaces / special separators
    text = re.sub(r'[\s\-_\.]+', ' ', text)
    return text


def check_input_safety(query: str) -> Tuple[bool, str]:
    """Pre-filter run before retrieval. Checks multiple languages and basic obfuscation."""
    if not query or len(query.strip()) < 2:
        return False, "Query is empty or too short."
    if len(query) > 2000:
        return False, "Query exceeds maximum allowed length."

    normalized = _normalize_query(query)
    for pattern in _UNSAFE_PATTERNS:
        if pattern in normalized:
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