"""
Step 4: the actual "write an answer using only these chunks" call.
Kept in its own file and behind one function (generate_answer) so swapping
providers later (Groq -> Gemini -> anything else) only touches this file.

Forces the LLM to reply in JSON so the harness can parse it reliably instead
of hoping the model formats things consistently.
"""
import json
import requests
from config import LLM_PROVIDER, GROQ_API_KEY, GROQ_MODEL, GEMINI_API_KEY, GEMINI_MODEL

SYSTEM_PROMPT = """You are a careful, helpful voice assistant that answers ONLY using the provided context.

Answer style rules:
- Always answer in a complete, natural sentence (or 2-3 sentences) — never a bare sentence fragment.
  Bad: "the first nuclear weapons"
  Good: "The Manhattan Project produced the first nuclear weapons during World War II."
- Restate enough of the question in your answer that it makes sense on its own, since this will
  be read aloud to someone who may not see the original question text.
- Add one relevant supporting detail from the context if it's available (a date, a cause, a
  consequence) so the answer feels complete rather than minimal — but stay strictly within
  what the context actually says, never add outside facts.
- Keep it to 1-3 sentences total — thorough but not a lecture.

Grounding rules (these matter more than style):
- If the context does not contain enough information to answer, say so honestly in the "answer"
  field instead of guessing, and set "grounded" to false.
- Never add facts that aren't in the context, even to make the answer sound more complete.

Respond with ONLY a JSON object, no other text, in this exact shape:
{"answer": "...", "grounded": true or false, "used_chunk_ids": ["..."]}
"""


def _build_user_prompt(query: str, chunks: list) -> str:
    context_block = "\n\n".join(
        f"[{c['chunk_id']}] {c['text']}" for c in chunks
    )
    return f"Context:\n{context_block}\n\nQuestion: {query}\n\nRespond with the JSON object only."


def _call_groq(system_prompt: str, user_prompt: str) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set. Add it to your .env file.")
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        },
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_gemini(system_prompt: str, user_prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set. Add it to your .env file.")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    resp = requests.post(
        url,
        json={
            "contents": [{"parts": [{"text": system_prompt + "\n\n" + user_prompt}]}],
            "generationConfig": {"temperature": 0.1, "response_mime_type": "application/json"},
        },
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def generate_answer(query: str, chunks: list) -> dict:
    """Returns {"answer": str, "grounded": bool, "used_chunk_ids": [...]}.
    Falls back to a safe refusal shape if the LLM call or JSON parsing fails,
    so the harness never crashes on a bad model response."""
    user_prompt = _build_user_prompt(query, chunks)

    try:
        if LLM_PROVIDER == "groq":
            raw = _call_groq(SYSTEM_PROMPT, user_prompt)
        elif LLM_PROVIDER == "gemini":
            raw = _call_gemini(SYSTEM_PROMPT, user_prompt)
        else:
            raise RuntimeError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")

        parsed = json.loads(raw)
        return {
            "answer": parsed.get("answer", ""),
            "grounded": bool(parsed.get("grounded", False)),
            "used_chunk_ids": parsed.get("used_chunk_ids", []),
        }
    except Exception as e:
        return {
            "answer": f"Sorry, I couldn't generate an answer right now ({e}).",
            "grounded": False,
            "used_chunk_ids": [],
        }