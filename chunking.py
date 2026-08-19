"""
Step 2: Turn raw passages from the dataset into a list of chunk dicts.

Supports multiple chunking strategies:
1. fixed: Fixed-size word count chunking with overlap
2. semantic: Meaning-shift boundary chunking using sentence embeddings
3. metadata: Metadata-aware chunking preserving language, source, position, char length

Every strategy returns the same standard shape:
    {
        "text": "...",
        "doc_id": "source document id",
        "chunk_id": "unique id for this chunk",
        "strategy": "fixed" | "semantic" | "metadata",
        "metadata": {"language": "hi", "position": 0, ...}
    }
"""
import re
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from config import EMBEDDING_MODEL
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _split_into_sentences(text: str) -> List[str]:
    """Splits multilingual text into sentences respecting English periods,
    Devanagari dandas (। / ॥), question marks, exclamation marks, and newlines."""
    # Replace common sentence terminators with a unified delimiter
    cleaned = re.sub(r'[\r\n]+', ' ', text)
    # Split by ., !, ?, ।, ॥ or semicolons
    raw_sentences = re.split(r'[.!?।॥;]+', cleaned)
    sentences = [s.strip() for s in raw_sentences if s and len(s.strip()) > 3]
    return sentences if sentences else [text.strip()]


def fixed_size_chunk(
    text: str,
    doc_id: str,
    size: int = 150,
    overlap: int = 30,
    language: str = "unknown"
) -> List[Dict]:
    """Cuts every `size` words, with `overlap` words shared between consecutive chunks."""
    words = text.split()
    if not words:
        return []
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
        start += max(1, size - overlap)
        idx += 1
    return chunks


def semantic_chunk(
    text: str,
    doc_id: str,
    similarity_threshold: float = 0.50,
    language: str = "unknown"
) -> List[Dict]:
    """Cuts at points where meaning shifts, using sentence similarity."""
    sentences = _split_into_sentences(text)
    if len(sentences) <= 1:
        return [{
            "text": text,
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}_semantic_0",
            "strategy": "semantic",
            "metadata": {"language": language, "position": 0},
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
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_semantic_{idx}",
                "strategy": "semantic",
                "metadata": {"language": language, "position": idx},
            })
            idx += 1
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    if current:
        chunks.append({
            "text": ". ".join(current) + ".",
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}_semantic_{idx}",
            "strategy": "semantic",
            "metadata": {"language": language, "position": idx},
        })
    return chunks


def metadata_aware_chunk(
    text: str,
    doc_id: str,
    language: str,
    source_type: str = "passage",
    size: int = 150,
    overlap: int = 30
) -> List[Dict]:
    """Attaches rich metadata (language, source type, char length, position)
    for filtered and accelerated retrieval."""
    base_chunks = fixed_size_chunk(text, doc_id, size, overlap, language)
    for c in base_chunks:
        c["strategy"] = "metadata"
        c["metadata"]["source_type"] = source_type
        c["metadata"]["char_length"] = len(c["text"])
    return base_chunks


def build_all_strategies(text: str, doc_id: str, language: str = "unknown") -> Dict[str, List[Dict]]:
    """Runs all three chunking strategies side by side."""
    return {
        "fixed": fixed_size_chunk(text, doc_id, language=language),
        "semantic": semantic_chunk(text, doc_id, language=language),
        "metadata": metadata_aware_chunk(text, doc_id, language=language),
    }


if __name__ == "__main__":
    sample_en = "The Taj Mahal is in Agra, India. It was built by Mughal emperor Shah Jahan. Construction began in 1632."
    result = build_all_strategies(sample_en, doc_id="demo_en", language="en")
    print(f"Strategies generated: {list(result.keys())}")
