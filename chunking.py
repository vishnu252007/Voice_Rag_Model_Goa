"""
Step 2: turn raw passages from the dataset into a list of chunk dicts.

Every strategy returns the SAME shape so the rest of the pipeline doesn't care
which one produced a chunk:

    {
        "text": "...",
        "doc_id": "source document id",
        "chunk_id": "unique id for this chunk",
        "strategy": "fixed" | "semantic" | "metadata",
        "metadata": {"language": "hi", "position": 0, ...}
    }

HOW TO TEST THIS FILE ON ITS OWN:
    python chunking.py
(runs a tiny built-in example so you can see the difference between strategies)
"""
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
import numpy as np

_embedder = None


def _get_embedder():
    # loaded lazily so importing this file doesn't download the model immediately
    global _embedder
    if _embedder is None:
        from config import EMBEDDING_MODEL
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def fixed_size_chunk(text: str, doc_id: str, size: int = 200, overlap: int = 40,
                      language: str = "unknown") -> List[Dict]:
    """Baseline strategy: cut every `size` words, with `overlap` words shared between
    consecutive chunks so a sentence split across the cut point still appears whole
    in at least one chunk."""
    words = text.split()
    chunks = []
    start = 0
    idx = 0
    while start < len(words):
        end = start + size
        piece = " ".join(words[start:end])
        chunks.append({
            "text": piece,
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}_fixed_{idx}",
            "strategy": "fixed",
            "metadata": {"language": language, "position": idx},
        })
        start += size - overlap
        idx += 1
    return chunks


def semantic_chunk(text: str, doc_id: str, similarity_threshold: float = 0.55,
                    language: str = "unknown") -> List[Dict]:
    """Cuts at points where meaning shifts, instead of at a fixed word count.
    Splits into sentences, embeds each one, and starts a new chunk whenever a
    sentence is no longer similar enough to the one before it."""
    sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    if len(sentences) <= 1:
        return [{
            "text": text, "doc_id": doc_id, "chunk_id": f"{doc_id}_semantic_0",
            "strategy": "semantic", "metadata": {"language": language, "position": 0},
        }]

    embedder = _get_embedder()
    embeddings = embedder.encode(sentences, convert_to_tensor=True)

    chunks, current = [], [sentences[0]]
    idx = 0
    for i in range(1, len(sentences)):
        sim = util.cos_sim(embeddings[i - 1], embeddings[i]).item()
        if sim < similarity_threshold:
            chunks.append({
                "text": ". ".join(current) + ".",
                "doc_id": doc_id, "chunk_id": f"{doc_id}_semantic_{idx}",
                "strategy": "semantic", "metadata": {"language": language, "position": idx},
            })
            idx += 1
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    if current:
        chunks.append({
            "text": ". ".join(current) + ".",
            "doc_id": doc_id, "chunk_id": f"{doc_id}_semantic_{idx}",
            "strategy": "semantic", "metadata": {"language": language, "position": idx},
        })
    return chunks


def metadata_aware_chunk(text: str, doc_id: str, language: str, source_type: str = "passage",
                          size: int = 200, overlap: int = 40) -> List[Dict]:
    """Same cut as fixed_size_chunk, but attaches richer metadata so retrieval can
    filter BEFORE searching (e.g. only search Hindi-tagged chunks for a Hindi query),
    which is faster and more accurate than searching everything blindly."""
    base_chunks = fixed_size_chunk(text, doc_id, size, overlap, language)
    for c in base_chunks:
        c["strategy"] = "metadata"
        c["metadata"]["source_type"] = source_type
        c["metadata"]["char_length"] = len(c["text"])
    return base_chunks


def build_all_strategies(text: str, doc_id: str, language: str = "unknown") -> Dict[str, List[Dict]]:
    """Convenience function: runs every strategy on the same text so you can compare
    them side by side (useful for your demo video and README)."""
    return {
        "fixed": fixed_size_chunk(text, doc_id, language=language),
        "semantic": semantic_chunk(text, doc_id, language=language),
        "metadata": metadata_aware_chunk(text, doc_id, language=language),
    }


if __name__ == "__main__":
    sample = (
        "The Taj Mahal is located in Agra, India. It was built by Mughal emperor Shah Jahan. "
        "Construction began in 1632 and took around 21 years to complete. "
        "In contrast, cricket is the most popular sport in India. "
        "The Indian cricket team won the World Cup in 2011 and again in 2023."
    )
    result = build_all_strategies(sample, doc_id="demo_doc", language="en")
    for strategy, chunks in result.items():
        print(f"\n--- {strategy} ({len(chunks)} chunks) ---")
        for c in chunks:
            print(f"  [{c['chunk_id']}] {c['text'][:80]}...")
