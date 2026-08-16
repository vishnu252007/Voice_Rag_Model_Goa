"""
Central place every other file reads settings from.
Nothing else in the project should call os.getenv directly — import from here instead,
so if a key name changes you only fix it in one place.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Speech to text ---
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")

# --- LLM for generation ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# --- Vector DB ---
# QDRANT_MODE: "cloud" uses your Qdrant Cloud cluster (needs QDRANT_URL + QDRANT_API_KEY).
# "local" runs Qdrant embedded, storing data in a local folder — no network call at
# all, which matters if your cloud cluster's round-trip time is too slow to hit the
# 200ms retrieval target. Local mode is fine for a hackathon submission; Qdrant is
# still genuinely the vector DB being used, just running on your own machine.
QDRANT_MODE = os.getenv("QDRANT_MODE", "cloud")
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", "./qdrant_local_data")
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "hh_goa_rag")

# --- Embeddings (runs locally, free, no key) ---
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# --- Guardrail thresholds (tune these after you see real retrieval scores) ---
MIN_RETRIEVAL_SCORE = 0.15   # lowered from 0.35 — benchmark showed a 63% false-refusal
                              # rate on questions with genuine matching data, meaning 0.35
                              # was too strict for this embedding model + cross-encoder combo
TOP_K_RETRIEVE = 10          # how many candidates hybrid search pulls before reranking
                              # (lowered from 20 — the reranker is the slowest single
                              # stage, and halving its input roughly halves its cost)
TOP_K_FINAL = 4               # how many chunks actually go to the LLM after reranking