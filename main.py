"""
FastAPI Server for Voice-Enabled Multilingual RAG System.
Endpoints:
    GET  /           -> Serves interactive demo web UI
    GET  /health     -> Health check
    POST /ask-text   -> JSON query text-in, answer-out
    POST /ask-voice  -> Audio file voice-in, STT -> harness answer-out
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Must import torch first on Windows
import torch

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import shutil
import tempfile

from stt import transcribe_audio
from harness import answer_query
from latency import timed_stage


def _normalize_language_code(sarvam_code: str) -> Optional[str]:
    if not sarvam_code:
        return None
    return sarvam_code.split("-")[0].lower()


app = FastAPI(title="HH Goa 2026 - Voice RAG")

_cors_origins_env = os.getenv("CORS_ORIGINS", "").strip()
allowed_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class TextQuery(BaseModel):
    query: str
    language: Optional[str] = None


class TranslateRequest(BaseModel):
    text: str
    target_language: str



@app.get("/")
def serve_frontend():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Voice RAG API is running. Frontend located at /static/index.html"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    fav_path = os.path.join(STATIC_DIR, "favicon.ico")
    if os.path.exists(fav_path):
        return FileResponse(fav_path)
    return FileResponse(os.path.join(os.path.dirname(__file__), "favicon.ico"))


@app.on_event("startup")
def warm_up_models():
    """Pre-warms FAISS, BM25, and neural models into memory at server startup."""
    try:
        from harness import answer_query
        print("Pre-warming vector store and language models...")
        answer_query("warmup initialization query", language_filter=None)
        print("Warm-up complete.")
    except Exception as e:
        print(f"Warm-up skipped: {e}")


@app.post("/ask-text")
def ask_text(payload: TextQuery):
    """Text-in, text-out testing endpoint."""
    return answer_query(payload.query, language_filter=payload.language)


@app.post("/translate")
def translate_api(payload: TranslateRequest):
    """On-demand translation of answers into English, Hindi, Telugu, or Tamil."""
    from generation import translate_text
    translated = translate_text(payload.text, payload.target_language)
    return {"translated_text": translated, "target_language": payload.target_language}



@app.post("/ask-voice")
def ask_voice(file: UploadFile = File(...), language: Optional[str] = Form(None)):
    """End-to-end voice pipeline: Audio In -> Sarvam STT -> Harness -> Response."""
    real_content_type = file.content_type or "audio/wav"
    suffix = ".webm" if "webm" in real_content_type else ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        with timed_stage() as t:
            stt_result = transcribe_audio(tmp_path, content_type=real_content_type)
        stt_ms = round(t["ms"], 1)

        detected_lang = language or _normalize_language_code(stt_result.get("language"))
        response = answer_query(stt_result["text"], language_filter=detected_lang)
        response["transcript"] = stt_result["text"]
        response["detected_language"] = stt_result.get("language")
        response["timings"]["stt_ms"] = stt_ms
        if "total_ms" in response["timings"]:
            response["timings"]["total_ms"] = round(response["timings"]["total_ms"] + stt_ms, 1)
        return response
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"Starting Voice RAG server on http://127.0.0.1:{port} ...")
    uvicorn.run(app, host="0.0.0.0", port=port)