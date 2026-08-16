"""
The actual web server. This is what you deploy and what your frontend talks to.

Run locally with:
    uvicorn main:app --reload

Two endpoints:
    POST /ask-text   -> send {"query": "..."} for text-only testing (no mic needed)
    POST /ask-voice  -> send an audio file, runs STT -> harness end to end
"""
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import tempfile

from stt import transcribe_audio
from harness import answer_query
from latency import timed_stage


def _normalize_language_code(sarvam_code: str) -> str:
    """Sarvam returns codes like 'en-IN' / 'hi-IN', but our indexed chunks are
    tagged with plain 2-letter codes like 'en' / 'hi' (see ingest.py). Without
    this, the language filter would silently never match anything."""
    if not sarvam_code:
        return None
    return sarvam_code.split("-")[0].lower()

app = FastAPI(title="HH Goa 2026 - Voice RAG")

# allow your frontend (running on a different port/domain) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your actual frontend URL before final submission
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextQuery(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask-text")
def ask_text(payload: TextQuery):
    """Text-in, text-out. Use this to test/demo retrieval and generation
    without needing a working mic or audio file."""
    return answer_query(payload.query)


@app.post("/ask-voice")
def ask_voice(file: UploadFile = File(...)):
    """The real end-to-end path: audio in, transcribes, retrieves, answers."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    with timed_stage() as t:
        stt_result = transcribe_audio(tmp_path)
    stt_ms = round(t["ms"], 1)

    response = answer_query(stt_result["text"], language_filter=_normalize_language_code(stt_result.get("language")))
    response["transcript"] = stt_result["text"]
    response["timings"]["stt_ms"] = stt_ms
    return response