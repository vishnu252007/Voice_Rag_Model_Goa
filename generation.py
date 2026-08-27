"""
Step 4: Multilingual Answer Generation & Translation module.
Generates grounded answers and on-demand high quality translations in 3 languages:
English, Hindi (हिंदी), and Telugu (తెలుగు).
"""
import json
import re
import requests
from config import GENERATION_MODE, GROQ_API_KEY, GROQ_MODEL

SYSTEM_PROMPT = """You are an extremely strict factual RAG assistant.
Your job is to answer ONLY if the provided context passages explicitly and directly contain the exact answer to the user's question.

CRITICAL RULES:
1. First, check: Does the provided context DIRECTLY and SPECIFICALLY contain the exact answer to the question?
2. If the context is only loosely related, shares vague keywords, or lacks the exact specific facts needed to answer:
   - You MUST set "grounded": false
   - You MUST set "answer": "I do not have sufficient information in the provided context to answer this question."
   - You MUST set "used_chunk_ids": []
3. If and only if the context directly and unambiguously answers the exact question:
   - Set "grounded": true
   - Write a concise 1-2 sentence factual answer in the language the question was asked.
   - Set "used_chunk_ids" to the list of matching chunk IDs.

Output ONLY a valid JSON object matching this schema:
{
  "answer": "concise answer or honest refusal",
  "grounded": true,
  "used_chunk_ids": ["chunk_id_1"]
}"""

LANG_NAME_MAP = {
    "en": "English",
    "hi": "Hindi (हिंदी)",
    "te": "Telugu (తెలుగు)"
}

_session = None
_trans_cache = {}


def _get_session() -> requests.Session:
    """Persistent HTTP session for connection pooling and low-latency API calls."""
    global _session
    if _session is None:
        _session = requests.Session()
        if GROQ_API_KEY:
            _session.headers.update({
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            })
    return _session


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
    """In-process ultra-fast (<1ms) grounded answer extraction directly from top chunks with strict relevance validation."""
    if not chunks:
        return {
            "answer": "I do not have sufficient information in the provided context to answer this question.",
            "grounded": False,
            "used_chunk_ids": [],
        }

    stopwords = {
        'what', 'is', 'the', 'a', 'an', 'in', 'on', 'of', 'for', 'to', 'how',
        'do', 'does', 'and', 'or', 'by', 'with', 'from', 'at', 'this', 'that',
        'these', 'those', 'it', 'its', 'as', 'are', 'was', 'were', 'be', 'been',
        'kya', 'hai', 'ka', 'ki', 'ke', 'mein', 'se', 'ko', 'find'
    }
    q_words = [w for w in re.findall(r'\w+', query.lower()) if w not in stopwords]
    if not q_words:
        q_words = re.findall(r'\w+', query.lower())

    best_chunk = None
    best_score = 0

    for c in chunks:
        c_text = c.get("text", "").lower()
        c_words = set(re.findall(r'\w+', c_text))
        matched = sum(1 for w in q_words if w in c_words)
        score = matched / max(1, len(q_words))
        if score > best_score:
            best_score = score
            best_chunk = c

    # Require comprehensive match of query terms (>= 0.75) for grounding
    if best_score < 0.75 or not best_chunk:
        return {
            "answer": "I do not have sufficient information in the provided context to answer this question.",
            "grounded": False,
            "used_chunk_ids": [],
        }

    text = best_chunk.get("text", "").strip()
    sentences = [s.strip() for s in re.split(r'[.!?।॥\n]+', text) if len(s.strip()) > 10]
    
    # Pick the most relevant sentence from the chunk
    best_sent = sentences[0] if sentences else text[:200]
    for s in sentences:
        s_words = set(re.findall(r'\w+', s.lower()))
        if sum(1 for w in q_words if w in s_words) >= max(1, len(q_words) * 0.5):
            best_sent = s
            break

    return {
        "answer": best_sent if best_sent.endswith(('.', '।')) else f"{best_sent}.",
        "grounded": True,
        "used_chunk_ids": [best_chunk.get("chunk_id", "chunk_0")],
    }


def _build_user_prompt(query: str, chunks: list) -> str:
    context_blocks = []
    for c in chunks:
        cid = c.get("chunk_id", "unknown")
        txt = c.get("text", "").strip()
        context_blocks.append(f"[{cid}]\n{txt}")
    context_str = "\n\n".join(context_blocks)
    return f"Context:\n{context_str}\n\nQuestion: {query}\n\nRespond with valid JSON:"


def _extract_json(text: str) -> dict:
    text = _clean_text(text)
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def _call_groq(system_prompt: str, user_prompt: str, max_tokens: int = 150) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set.")

    import time
    session = _get_session()
    max_retries = 2
    base_delay = 0.5

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"}
    }

    for attempt in range(max_retries):
        resp = session.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            timeout=5,
        )
        if resp.status_code == 429:
            if attempt < max_retries - 1:
                retry_after = resp.headers.get("Retry-After")
                try:
                    sleep_time = min(0.3, float(retry_after)) if retry_after else (base_delay * (2 ** attempt))
                except (ValueError, TypeError):
                    sleep_time = base_delay * (2 ** attempt)
                sleep_time = min(sleep_time, 0.5)
                time.sleep(sleep_time)
                continue
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        return _clean_text(raw)

    resp.raise_for_status()


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
    """Translates any answer into English, Hindi, or Telugu with caching & fallback."""
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
        f"Always write in the native script of {target_lang_name} (e.g., Devanagari for Hindi, Telugu script for Telugu, English for English). "
        "Output ONLY the final translated text. Do not output explanations, markdown formatting, or quotes."
    )

    try:
        if GROQ_API_KEY:
            session = _get_session()
            resp = session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": text}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 200
                },
                timeout=8
            )
            if resp.status_code == 200:
                raw_content = resp.json()["choices"][0]["message"]["content"]
                translated = _clean_text(raw_content)
                if translated and len(translated) > 3:
                    _trans_cache[cache_key] = translated
                    return translated
    except Exception as e:
        print(f"Primary translation ({target_code}) error: {e}")

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
        raw = _call_groq(SYSTEM_PROMPT, user_prompt, max_tokens=150)
        parsed = _extract_json(raw)
        grounded_val = parsed.get("grounded", False)
        is_grounded = grounded_val in [True, "true", "True", 1, "TRUE"]

        main_answer = str(parsed.get("answer", "")).strip()
        used_ids = parsed.get("used_chunk_ids", [c["chunk_id"] for c in chunks[:2]])

        return {
            "answer": main_answer,
            "grounded": is_grounded,
            "used_chunk_ids": used_ids,
        }
    except Exception as e:
        print(f"    [generate_answer] primary call failed ({e}) — failing safe with a refusal", flush=True)
        return {
            "answer": "I do not have sufficient information in the context to answer this question.",
            "grounded": False,
            "used_chunk_ids": [],
        }