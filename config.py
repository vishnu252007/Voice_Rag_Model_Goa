"""
Central configuration file for Voice-Enabled RAG.
Optimized for ultra-low latency (<50ms target, well under the 200ms limit).
"""
import os
from dotenv import load_dotenv

load_dotenv()

def _get_clean_env(key: str, default: str = "") -> str:
    val = os.getenv(key, default)
    return val.strip() if isinstance(val, str) else default

# --- Speech to text ---
SARVAM_API_KEY = _get_clean_env("SARVAM_API_KEY", "")

# --- Generation Mode ---
# "llm"  = intelligent grounded generation via Groq Qwen (high quality, semantic validation)
# "fast" = in-process fast synthesis for high-throughput benchmarks
GENERATION_MODE = _get_clean_env("GENERATION_MODE", "fast").lower()

# --- LLM for generation (when GENERATION_MODE=llm) ---
GROQ_API_KEY = _get_clean_env("GROQ_API_KEY", "")
GROQ_MODEL = _get_clean_env("GROQ_MODEL", "qwen/qwen3.8-27b")

# --- Vector search backend ---
VECTOR_BACKEND = _get_clean_env("VECTOR_BACKEND", "faiss").lower()
FAISS_INDEX_PATH = _get_clean_env("FAISS_INDEX_PATH", "./faiss_index.bin")
FAISS_METADATA_PATH = _get_clean_env("FAISS_METADATA_PATH", "./faiss_metadata.pkl")

# --- Qdrant fallback backend ---
QDRANT_MODE = _get_clean_env("QDRANT_MODE", "cloud")
QDRANT_LOCAL_PATH = _get_clean_env("QDRANT_LOCAL_PATH", "./qdrant_local_data")
QDRANT_URL = _get_clean_env("QDRANT_URL", "")
QDRANT_API_KEY = _get_clean_env("QDRANT_API_KEY", "")
QDRANT_COLLECTION = _get_clean_env("QDRANT_COLLECTION", "hh_goa_rag")

# --- Embeddings ---
EMBEDDING_MODEL = _get_clean_env(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# --- Reranker (Disabled by default for sub-50ms latency target) ---
RERANKER_ENABLED = _get_clean_env("RERANKER_ENABLED", "false").lower() == "true"
RERANKER_MODEL = _get_clean_env("RERANKER_MODEL", "cross-encoder/ms-marco-TinyBERT-L-2-v2")

# --- Guardrail thresholds ---
MIN_RETRIEVAL_SCORE = float(_get_clean_env("MIN_RETRIEVAL_SCORE", "0.08"))
TOP_K_RETRIEVE = int(_get_clean_env("TOP_K_RETRIEVE", "12"))
TOP_K_FINAL = int(_get_clean_env("TOP_K_FINAL", "6"))

# --- Query cache ---
CACHE_ENABLED = _get_clean_env("CACHE_ENABLED", "true").lower() == "true"
CACHE_MAX_SIZE = int(_get_clean_env("CACHE_MAX_SIZE", "500"))