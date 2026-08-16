"""
Standalone diagnostic — tests Qdrant local mode speed with NO FastAPI, NO threading,
NO server involved at all. Just: create client once, search 3 times, print timing.

If this is ALSO slow, the problem is Qdrant/your machine itself, not our server code.
If this is FAST, something specific to how the server calls it is the problem.

Run: python diagnose_qdrant.py
"""
import time
from qdrant_client import QdrantClient

print("Creating client...")
t0 = time.perf_counter()
client = QdrantClient(path="C:\\qdrant_local_data")
t1 = time.perf_counter()
print(f"Client created in {(t1-t0)*1000:.0f}ms")

collections = client.get_collections()
print(f"Collections found: {[c.name for c in collections.collections]}")

collection_name = collections.collections[0].name if collections.collections else None
if not collection_name:
    print("No collection found — did ingest.py run successfully in local mode?")
    exit(1)

info = client.get_collection(collection_name)
print(f"Collection '{collection_name}' has {info.points_count} points")

import numpy as np
fake_vector = np.random.rand(768).tolist()  # random vector, just to test search speed

print("\nRunning 3 searches in a row, same process, same client instance:")
for i in range(3):
    t0 = time.perf_counter()
    results = client.search(collection_name=collection_name, query_vector=fake_vector, limit=10)
    t1 = time.perf_counter()
    print(f"  search {i+1}: {(t1-t0)*1000:.0f}ms, got {len(results)} results")
