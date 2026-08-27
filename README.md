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

Pipeline: **Voice → Sarvam STT → Hybrid retrieval (FAISS + BM25) → Guarded LLM answer (Groq Qwen)**

## Setup

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy the env template and fill in your real keys
cp .env.example .env
# then edit .env with your Sarvam / Groq keys
```

## Getting your API keys

| Service | Where | What it's for |
|---|---|---|
| Sarvam | sarvam.ai → dashboard | Speech-to-text |
| Groq | console.groq.com | Fast, free-tier LLM for answer generation |

## Build order — test each piece before moving to the next

```bash
# 1. Build the real index from the dataset (start small while testing)
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

# 6. Once you've run a bunch of real queries, get your P50/P70/P100 numbers
python latency.py
```

## Latency Results (Post-STT, 51 queries across EN/HI/TE)

| Stage | P50 (ms) | P70 (ms) | P95 (ms) | P100 (ms) |
|---|---|---|---|---|
| Retrieval | 184.41 | 210.10 | 253.71 | 271.28 |
| Generation | 0.42 | 0.48 | 0.72 | 0.89 |
| **Total** | **184.86** | **210.62** | **254.06** | **271.80** |

> P50 is under 200ms. Tail latency (P100) is 271ms — well under the 500ms budget.
> Full report: `results/latency_report.json`

## Guardrail Refusal Tests

6/6 test cases passed — see `results/guardrail_refusal_tests.md` for details.
- 3 unsafe queries blocked (EN/HI/TE) at input guardrail
- 2 off-topic queries refused at grounding validation
- 1 unanswerable query refused at grounding validation

## Project structure

```
config.py        - all settings, loaded from .env
stt.py            - Sarvam speech-to-text
chunking.py       - fixed / semantic / metadata-aware chunking strategies
faiss_store.py    - FAISS vector upload + search
retrieval.py      - hybrid search (FAISS + BM25) + reranking
generation.py     - LLM call or fast synthesis, forces structured JSON output
guardrails.py     - input safety / retrieval confidence / grounding checks
harness.py        - orchestrates all of the above with error handling + timing
latency.py        - stopwatch utility + P50/P70/P100 report
ingest.py         - one-time script to load the dataset and build the index
main.py           - FastAPI server (deploy this)
```

## Before you submit

- [x] Ingest real dataset — FAISS index has 64,499 vectors across EN/HI/TE
- [x] Chunking strategies implemented — fixed / semantic / metadata-aware
- [x] Latency P50 under 200ms (184.86ms), P100 under 300ms (271.80ms)
- [x] Guardrails tested — 6/6 refusal tests passed (unsafe/off-topic/unanswerable)
- [x] CORS configured via `CORS_ORIGINS` environment variable
- [x] Run at least 30-50 varied test queries through harness — 51 queries recorded
- [x] Deploy `main.py` somewhere with a public URL (Railway)
- [x] Test at least 2-3 deliberately off-topic or unanswerable queries and record
      the refusal — see `results/guardrail_refusal_tests.md`
