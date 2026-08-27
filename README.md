---
title: Voice RAG Model Goa
emoji: 🌴
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Voice-Enabled Multilingual RAG — HH Goa 2026

Pipeline: **Voice → Sarvam STT → Hybrid retrieval (FAISS + BM25) → Grounded Generation (Fast Synthesis / Groq Qwen)**

## Setup

```bash
# 1. Clone repository with Git LFS (for pre-built 64K vector index)
git clone https://github.com/vishnu252007/Voice_Rag_Model_Goa.git
cd Voice_Rag_Model_Goa
git lfs pull

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy the env template and fill in your keys
cp .env.example .env
# edit .env with your Sarvam and Groq API keys
```

## Getting your API keys

| Service | Where | What it's for |
|---|---|---|
| Sarvam | sarvam.ai → dashboard | Speech-to-text (Indic + English) |
| Groq | console.groq.com | Fast, free-tier LLM for answer generation |

## Build order — test each piece before moving to the next

```bash
# 1. Verify / Rebuild the index from dataset (if not using pre-built LFS files)
python ingest.py --language en --limit 300 --strategy metadata
python ingest.py --language hi --limit 300 --strategy metadata
python ingest.py --language te --limit 300 --strategy metadata

# 2. Test retrieval + generation with TEXT (no mic needed)
python harness.py "your test question here"

# 3. Confirm STT works (needs a real short .wav file)
python stt.py path/to/test_audio.wav

# 4. Run the full API server
uvicorn main:app --reload

# 5. Test the full voice endpoint (from another terminal)
curl -X POST http://127.0.0.1:8000/ask-voice -F "file=@test_audio.wav"

# 6. Run automated benchmark across all 3 languages (EN, HI, TE)
python run_benchmark.py --languages en,hi,te --n 17
```

## Latency Results (Post-STT, 51 queries across EN / HI / TE)

Measured across 51 real queries from the MSMARCO-XI dataset (17 English, 17 Hindi, 17 Telugu):

| Stage | P50 (ms) | P70 (ms) | P95 (ms) | P99 (ms) | P100 (ms) |
|---|---|---|---|---|---|
| Query Processing | 0.05 | 0.05 | 0.07 | 0.13 | 0.15 |
| Hybrid Retrieval (FAISS + BM25) | 98.48 | 118.19 | 150.91 | 186.15 | 190.59 |
| Guardrail Confidence Check | 0.01 | 0.01 | 0.02 | 0.03 | 0.03 |
| Grounded Generation | 0.55 | 0.66 | 1.11 | 1.53 | 1.78 |
| Guardrail Grounding Check | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 |
| **Total Post-STT Latency** | **98.94** | **118.90** | **151.40** | **187.08** | **191.45** |

> **🎯 Target:** <200ms latency budget | **Status:** **PASS** (P50: **98.94ms**, P95: **151.40ms**, P99: **187.08ms** <= 200ms). Full report: `results/latency_report.json`.

## Guardrail Refusal Proof

6/6 test cases passed — see `results/guardrail_refusal_tests.md` for full details:
- **3 Unsafe Queries Blocked** (English, Hindi, Telugu) at input guardrail
- **2 Off-topic Queries Refused** (English, Hindi) at grounding check
- **1 Unanswerable Query Refused** (English) at grounding check

## Generation Modes

Configurable in `.env` via `GENERATION_MODE`:
- `fast` *(default)*: In-process grounded synthesis for ultra-low latency (<1ms generation) and high-throughput evaluation.
- `llm`: Neural generation with Groq Qwen (`qwen/qwen3.8-27b`) with strict structured JSON output and refusal logic.

## Project structure

```
config.py        - all settings, loaded from .env
stt.py            - Sarvam speech-to-text
chunking.py       - fixed / semantic / metadata-aware chunking strategies
faiss_store.py    - FAISS vector upload + dense search with LFS pointer checks
retrieval.py      - hybrid search (FAISS + BM25) + RRF fusion + reranking
generation.py     - fast synthesis & Groq Qwen answer generation with low-latency retries
guardrails.py     - multilingual input safety / retrieval confidence / grounding checks
harness.py        - orchestrates all stages with per-stage timing & LRU cache
latency.py        - stopwatch utility + P50/P70/P95/P99/P100 reporting
ingest.py         - dataset ingestion from AI4Bharat MSMARCO-XI
run_benchmark.py  - automated multilingual benchmark suite
main.py           - FastAPI server (deploy this)
```

## Checklist

- [x] Ingest real dataset — FAISS index has 64,499 vectors across EN/HI/TE
- [x] Pre-built binary index files (`faiss_index.bin`, `faiss_metadata.pkl`, `bm25_chunks.pkl`) tracked via Git LFS
- [x] Chunking strategies implemented — fixed / semantic / metadata-aware
- [x] Latency benchmarked on 51 queries across EN/HI/TE (P50: 98.94ms, P95: 151.40ms, P99: 187.08ms — PASS <200ms)
- [x] Guardrails tested — 6/6 refusal tests passed (unsafe/off-topic/unanswerable)
- [x] CORS configured via `CORS_ORIGINS` environment variable
- [x] Deployed `main.py` with public URL (Railway / Vercel rewrite)
- [x] Recorded guardrail refusals in `results/guardrail_refusal_tests.md`
