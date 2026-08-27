"""
Builds FAISS index and BM25 index from knowledge_base.json at Docker build time.
This script runs inside the Linux container ensuring cross-platform binary compatibility.
"""
import os
import json
import pickle

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

KB_JSON = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
FAISS_OUT = os.path.join(os.path.dirname(__file__), "kb_faiss.bin")
META_OUT = os.path.join(os.path.dirname(__file__), "kb_metadata.pkl")
BM25_OUT = os.path.join(os.path.dirname(__file__), "kb_bm25.pkl")

print(f"Loading knowledge_base.json ({os.path.getsize(KB_JSON)//1024}KB)...")
with open(KB_JSON, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks. Building FAISS index on Linux...")

from config import EMBEDDING_MODEL
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Load embedder
model = SentenceTransformer(EMBEDDING_MODEL)
model.eval()

texts = [c["text"] for c in chunks]
print(f"Embedding {len(texts)} chunks with {EMBEDDING_MODEL}...")

batch_size = 256
all_vectors = []
with torch.inference_mode():
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vecs = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        all_vectors.append(vecs)
        if (i // batch_size) % 5 == 0:
            print(f"  Embedded {min(i+batch_size, len(texts))}/{len(texts)} chunks...", flush=True)

vectors = np.ascontiguousarray(np.concatenate(all_vectors, axis=0), dtype=np.float32)
dim = vectors.shape[1]

print(f"Building FAISS HNSW index (dim={dim}, {len(vectors)} vectors)...")
index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 64
index.hnsw.efSearch = 24
index.add(vectors)

metadata = [
    {
        "chunk_id": c["chunk_id"],
        "doc_id": c["doc_id"],
        "text": c["text"],
        "strategy": c["strategy"],
        "metadata": c["metadata"],
    }
    for c in chunks
]

faiss.write_index(index, FAISS_OUT)
with open(META_OUT, "wb") as f:
    pickle.dump(metadata, f)

print(f"Saved FAISS index ({index.ntotal} vectors) -> {FAISS_OUT}")
print(f"Saved metadata ({len(metadata)} entries) -> {META_OUT}")

# Build BM25 index
print("Building BM25 index...")
import math
from collections import defaultdict, Counter

postings = defaultdict(list)
doc_lens = []
n_docs = len(chunks)

for idx, chunk in enumerate(chunks):
    tokens = chunk.get("text", "").lower().split()
    doc_lens.append(len(tokens))
    tf_map = Counter(tokens)
    for token, count in tf_map.items():
        postings[token].append((idx, count))

avgdl = sum(doc_lens) / max(1, n_docs)
idf = {}
for token, plist in postings.items():
    df = len(plist)
    idf[token] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

bm25_data = {
    "corpus": chunks,
    "postings": dict(postings),
    "doc_lens": doc_lens,
    "idf": idf,
    "avgdl": avgdl,
}
with open(BM25_OUT, "wb") as f:
    pickle.dump(bm25_data, f)

print(f"Saved BM25 index ({n_docs} chunks) -> {BM25_OUT}")
print("Build complete!")
