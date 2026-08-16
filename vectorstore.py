"""
Step 2b: wraps Qdrant so the rest of the code just calls simple functions
(upload_chunks, search) instead of dealing with the Qdrant client directly.

HOW TO TEST THIS FILE ON ITS OWN:
    python vectorstore.py
(uploads 3 tiny dummy chunks and searches for one, to confirm your Qdrant
credentials work before you point it at the real dataset)
"""
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, EMBEDDING_MODEL, QDRANT_MODE, QDRANT_LOCAL_PATH

_client = None
_embedder = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        if QDRANT_MODE == "local":
            # Embedded mode: no network call at all, data stored in a local folder.
            _client = QdrantClient(path=QDRANT_LOCAL_PATH)
        else:
            if not QDRANT_URL or not QDRANT_API_KEY:
                raise RuntimeError("QDRANT_URL / QDRANT_API_KEY not set. Add them to your .env file, "
                                    "or set QDRANT_MODE=local to skip the cloud cluster entirely.")
            # timeout=60 (default is much shorter) — free-tier clusters can be slow
            # to respond under load, and the default timeout was causing uploads to
            # fail partway through with large batches.
            _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    return _client


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def ensure_collection(vector_size: int = 768):
    client = _get_client()
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    # Qdrant requires a payload index to filter by a field (e.g. language_filter
    # in vector_search below) — without this, any search using that filter fails
    # with "Index required but not found". Safe to call every time; Qdrant just
    # no-ops if the index already exists.
    from qdrant_client.models import PayloadSchemaType
    try:
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="metadata.language",
            field_schema=PayloadSchemaType.KEYWORD,
        )
    except Exception:
        pass  # already exists, nothing to do


def upload_chunks(chunks: List[Dict], batch_size: int = 50):
    """Embeds and uploads a list of chunk dicts (see chunking.py for the shape).
    Uploads in small batches instead of one giant request — a single request with
    hundreds of points can time out on a free-tier cluster or slower connection."""
    embedder = _get_embedder()
    ensure_collection(vector_size=embedder.get_sentence_embedding_dimension())

    texts = [c["text"] for c in chunks]
    vectors = embedder.encode(texts, show_progress_bar=True).tolist()

    points = [
        PointStruct(
            id=i,
            vector=vectors[i],
            payload={
                "text": chunks[i]["text"],
                "doc_id": chunks[i]["doc_id"],
                "chunk_id": chunks[i]["chunk_id"],
                "strategy": chunks[i]["strategy"],
                "metadata": chunks[i]["metadata"],
            },
        )
        for i in range(len(chunks))
    ]

    client = _get_client()
    uploaded = 0
    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]
        # retry once on timeout — free-tier clusters occasionally need a second try
        for attempt in range(2):
            try:
                client.upsert(collection_name=QDRANT_COLLECTION, points=batch)
                uploaded += len(batch)
                break
            except Exception as e:
                if attempt == 0:
                    print(f"  batch {start}-{start+len(batch)} failed ({e}), retrying once...")
                else:
                    raise
        print(f"  uploaded {uploaded}/{len(points)} points so far")

    return uploaded


def vector_search(query: str, top_k: int = 20, language_filter: str = None) -> List[Dict]:
    """Returns top_k chunks closest in meaning to the query, each with a similarity score."""
    import time, os
    t0 = time.perf_counter()
    embedder = _get_embedder()
    query_vector = embedder.encode(query).tolist()
    t1 = time.perf_counter()

    search_filter = None
    if language_filter:
        # Defensive: make sure the payload index exists even if this collection
        # was created before we added index creation to ensure_collection().
        ensure_collection(vector_size=embedder.get_sentence_embedding_dimension())
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        search_filter = Filter(
            must=[FieldCondition(key="metadata.language", match=MatchValue(value=language_filter))]
        )

    results = _get_client().search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=top_k,
        query_filter=search_filter,
    )
    t2 = time.perf_counter()

    log_path = os.path.join(os.path.dirname(__file__), "retrieval_debug.log")
    with open(log_path, "a") as f:
        f.write(f"  [vector_search split] embed={(t1-t0)*1000:.0f}ms  qdrant_network_call={(t2-t1)*1000:.0f}ms\n")

    return [
        {"text": r.payload["text"], "chunk_id": r.payload["chunk_id"],
         "doc_id": r.payload["doc_id"], "score": r.score, "metadata": r.payload["metadata"]}
        for r in results
    ]


if __name__ == "__main__":
    dummy = [
        {"text": "The Taj Mahal is in Agra, India.", "doc_id": "d1", "chunk_id": "d1_0",
         "strategy": "fixed", "metadata": {"language": "en"}},
        {"text": "Cricket is very popular in India.", "doc_id": "d1", "chunk_id": "d1_1",
         "strategy": "fixed", "metadata": {"language": "en"}},
        {"text": "The Ganges is a major river in India.", "doc_id": "d1", "chunk_id": "d1_2",
         "strategy": "fixed", "metadata": {"language": "en"}},
    ]
    n = upload_chunks(dummy)
    print(f"Uploaded {n} test chunks.")
    results = vector_search("what monument is in Agra")
    for r in results:
        print(f"  score={r['score']:.3f}  {r['text']}")