# Voice-Enabled RAG — HH Goa 2026

Pipeline: **Voice → Sarvam STT → Hybrid retrieval (vector + BM25 + rerank) → Guarded LLM answer**

## Setup

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy the env template and fill in your real keys
cp .env.example .env
# then edit .env with your Sarvam / Groq / Qdrant keys
```

## Getting your API keys

| Service | Where | What it's for |
|---|---|---|
| Sarvam | sarvam.ai → dashboard | Speech-to-text |
| Groq | console.groq.com | Fast, free-tier LLM for answer generation |
| Qdrant Cloud | cloud.qdrant.io | Free vector database cluster |

## Build order — test each piece before moving to the next

```bash
# 1. Confirm Qdrant connection works
python vectorstore.py

# 2. Confirm STT works (needs a real short .wav file)
python stt.py path/to/test_audio.wav

# 3. Build the real index from the dataset (start small while testing)
python ingest.py --strategy metadata --limit 500

# 4. Test retrieval + generation with TEXT (no mic needed)
python harness.py "your test question here"

# 5. Run the full API server
uvicorn main:app --reload

# 6. Test the full voice endpoint (from another terminal)
curl -X POST http://127.0.0.1:8000/ask-voice -F "file=@test_audio.wav"

# 7. Once you've run a bunch of real queries, get your P50/P70/P100 numbers
python latency.py
```

## Project structure

```
config.py        - all settings, loaded from .env
stt.py            - Sarvam speech-to-text
chunking.py       - fixed / semantic / metadata-aware chunking strategies
vectorstore.py    - Qdrant upload + vector search
retrieval.py      - hybrid search (vector + BM25) + reranking
generation.py     - LLM call, forces structured JSON output
guardrails.py     - input safety / retrieval confidence / grounding checks
harness.py        - orchestrates all of the above with error handling + timing
latency.py        - stopwatch utility + P50/P70/P100 report
ingest.py         - one-time script to load the dataset and build the index
main.py           - FastAPI server (deploy this)
```

## Before you submit

- [ ] Adjust `ingest.py` field names (`passage`, `text`, `language`) to match the
      actual MSMARCO-XI column names — check the Hugging Face dataset viewer first
- [ ] Tune `MIN_RETRIEVAL_SCORE` in `config.py` after seeing real retrieval scores
      on your dataset — the default is a starting guess, not a measured value
- [ ] Run at least 30-50 varied test queries through `/ask-voice` or `/ask-text`
      before generating your final latency numbers — don't submit a single run
- [ ] Deploy `main.py` somewhere with a public URL (Render, Railway, Fly.io all
      have free tiers that work for FastAPI)
- [ ] Test at least 2-3 deliberately off-topic or unanswerable queries and record
      the refusal — this is required proof per the task's guardrail requirement
- [ ] Tighten CORS `allow_origins` in `main.py` from `"*"` to your real frontend URL
