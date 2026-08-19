"""
Step 4: Multilingual Answer Generation & Translation module.
Generates grounded answers and on-demand high quality translations in 4 languages:
English, Hindi (हिंदी), Telugu (తెలుగు), and Tamil (தமிழ்).
"""
import json
import re
import requests
from config import GENERATION_MODE, LLM_PROVIDER, GROQ_API_KEY, GROQ_MODEL, GEMINI_API_KEY, GEMINI_MODEL

SYSTEM_PROMPT = """You are an accurate multilingual voice assistant that answers questions using ONLY the provided context passages.

Output a valid JSON object matching this schema:
{
  "answer": "1-3 sentence grounded answer in the language the question was asked in",
  "grounded": true,
  "used_chunk_ids": ["chunk_id_1"]
}
"""

LANG_NAME_MAP = {
    "en": "English",
    "hi": "Hindi (हिंदी)",
    "te": "Telugu (తెలుగు)",
    "ta": "Tamil (தமிழ்)"
}

_trans_cache = {}


def _clean_text(raw: str) -> str:
    """Strips <think>...</think> reasoning blocks and markdown fences."""
    if not raw:
        return ""
    if "</think>" in raw:
        text = raw.split("</think>")[-1]
    elif "<think>" in raw:
        text = re.sub(r'<think>.*', '', raw, flags=re.DOTALL)
    else:
        text = raw
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'^```[a-z]*\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)
    return text.strip()


def _fast_synthesize(query: str, chunks: list) -> dict:
    """In-process ultra-fast (<1ms) grounded answer extraction directly from top chunks."""
    if not chunks:
        return {
            "answer": "No relevant information found in the dataset.",
            "grounded": False,
            "used_chunk_ids": [],
        }

    top_chunk = chunks[0]
    text = top_chunk.get("text", "").strip()
    
    sentences = [s.strip() for s in re.split(r'[.!?।॥]+', text) if len(s.strip()) > 10]
    answer_text = ". ".join(sentences[:2]) + "." if sentences else text[:250]

    return {
        "answer": answer_text,
        "grounded": True,
        "used_chunk_ids": [c["chunk_id"] for c in chunks[:2]],
    }


def _build_user_prompt(query: str, chunks: list) -> str:
    context_blocks = []
    for c in chunks:
        cid = c.get("chunk_id", "unknown")
        txt = c.get("text", "").strip()
        context_blocks.append(f"[{cid}]\n{txt}")
    context_str = "\n\n".join(context_blocks)
    return f"Context:\n{context_str}\n\nQuestion: {query}\n\nRespond with valid JSON containing answer, grounded, and used_chunk_ids:"


def _extract_json(text: str) -> dict:
    text = _clean_text(text)
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def _call_groq(system_prompt: str, user_prompt: str) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set.")
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
        },
        timeout=14,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    return _clean_text(raw)


def _fallback_translate(text: str, target_language: str) -> str:
    """Fast, zero-rate-limit fallback translation for Indic and English languages."""
    try:
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl={target_language}&dt=t&q={requests.utils.quote(text)}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            parts = [item[0] for item in resp.json()[0] if item and len(item) > 0 and item[0]]
            return "".join(parts).strip()
    except Exception as e:
        print(f"Fallback translation error: {e}")
    return text


def translate_text(text: str, target_language: str) -> str:
    """Translates any answer into English, Hindi, Telugu, or Tamil with caching & fallback."""
    if not text or not text.strip():
        return ""
    
    target_code = target_language.lower().strip()
    cache_key = (text.strip(), target_code)
    if cache_key in _trans_cache:
        return _trans_cache[cache_key]

    target_lang_name = LANG_NAME_MAP.get(target_code, target_language)

    sys_msg = (
        f"You are a professional multilingual translator. "
        f"Translate the provided text directly into natural {target_lang_name}. "
        f"Always write in the native script of {target_lang_name} (e.g., Devanagari for Hindi, Telugu script for Telugu, Tamil script for Tamil, English for English). "
        "Output ONLY the final translated text. Do not output explanations, markdown formatting, or quotes."
    )

    try:
        if GROQ_API_KEY:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": text}
                    ],
                    "temperature": 0.1
                },
                timeout=10
            )
            if resp.status_code == 200:
                raw_content = resp.json()["choices"][0]["message"]["content"]
                translated = _clean_text(raw_content)
                if translated and len(translated) > 3:
                    _trans_cache[cache_key] = translated
                    return translated
    except Exception as e:
        print(f"Primary translation ({target_code}) error: {e}")

    # Seamless instant fallback
    fallback_res = _fallback_translate(text, target_code)
    if fallback_res:
        _trans_cache[cache_key] = fallback_res
        return fallback_res

    return text


def generate_answer(query: str, chunks: list, mode: str = None) -> dict:
    """Generates a grounded answer from retrieved chunks."""
    if not chunks:
        return {
            "answer": "No relevant context was found to answer this question.",
            "grounded": False,
            "used_chunk_ids": [],
        }

    use_mode = mode or GENERATION_MODE
    if use_mode == "fast":
        return _fast_synthesize(query, chunks)

    user_prompt = _build_user_prompt(query, chunks)

    try:
        raw = _call_groq(SYSTEM_PROMPT, user_prompt)
        parsed = _extract_json(raw)
        grounded_val = parsed.get("grounded", False)
        is_grounded = grounded_val in [True, "true", "True", 1, "TRUE"]
        
        main_answer = str(parsed.get("answer", "")).strip()

        return {
            "answer": main_answer,
            "grounded": is_grounded,
            "used_chunk_ids": parsed.get("used_chunk_ids", [c["chunk_id"] for c in chunks[:2]]),
        }
    except Exception:
        return _fast_synthesize(query, chunks)