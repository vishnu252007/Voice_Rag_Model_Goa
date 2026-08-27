"""
Bridges THIS project's real embedding + generation functions to the exact
interface rag-local-eval-loop requires (see TARGET_INTERFACE.md).

Point the eval suite at this file for BOTH roles, since everything needed
lives in one file:
    set EVAL_EMBEDDER_MODULE=eval_adapter
    set EVAL_GENERATOR_MODULE=eval_adapter

Then run the smoke test from your project root:
    python -m eval.runner --num-answerable 3 --num-unanswerable 3 --workers 1
(with RAG_PROJECT_ROOT pointed at this folder, per TARGET_INTERFACE.md /
run.ps1's own instructions)
"""
import time
from types import SimpleNamespace

from config import VECTOR_BACKEND, GROQ_MODEL
from generation import generate_answer as _real_generate_answer

# Reuse whichever backend's embedder your real config.py is actually set to —
# same switch retrieval.py already uses, so this always matches what's really
# running in production, not a guess.
if VECTOR_BACKEND == "faiss":
    from faiss_store import _get_embedder
else:
    from vectorstore import _get_embedder


def get_model():
    """The suite calls this once — only the side effect (loading the model
    into memory) matters, per TARGET_INTERFACE.md. Returning it too in case
    that's useful, but nothing requires the return value specifically."""
    return _get_embedder()


def embed(texts: list):
    """texts: list[str] -> array-like, shape (len(texts), dim)."""
    import torch
    model = _get_embedder()
    with torch.inference_mode():
        return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)


def embed_one(text: str):
    """text: str -> array-like, shape (dim,)."""
    import torch
    model = _get_embedder()
    with torch.inference_mode():
        return model.encode(text, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)


def generate_answer(query: str, results: list):
    """results: list of objects with .text and .source (the suite's own
    objects, not ours) — NOT the dict shape our real generate_answer()
    expects, so we convert here. Returns the exact object shape
    TARGET_INTERFACE.md requires: .text, .grounded, .generation_ms, .model
    (an object with attributes, not a dict — hence SimpleNamespace)."""
    chunks = [
        {"chunk_id": getattr(r, "source", f"chunk_{i}"), "text": r.text}
        for i, r in enumerate(results)
    ]

    t0 = time.perf_counter()
    result = _real_generate_answer(query, chunks)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return SimpleNamespace(
        text=result.get("answer", ""),
        # IMPORTANT: pass through your real system's honest belief about
        # whether it found a real answer — never hardcode True here, or the
        # eval suite's reliability ("lying factor") check becomes meaningless
        # against your submission.
        grounded=bool(result.get("grounded", False)),
        generation_ms=elapsed_ms,
        model=GROQ_MODEL,
    )
