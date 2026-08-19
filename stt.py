"""
Step 1 of the pipeline: turn an audio file into text using Sarvam's Saaras v3 model.

HOW TO TEST THIS FILE ON ITS OWN:
    python stt.py path/to/your_recording.wav

Record a short wav on your phone/laptop, drop it in the project folder, and run this
before wiring it into the full app — confirm you can actually get text out first.
"""
import sys
import requests
from config import SARVAM_API_KEY

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"


def transcribe_audio(file_path: str, mode: str = "transcribe", content_type: str = "audio/wav") -> dict:
    """
    Sends an audio file to Sarvam and returns {"text": ..., "language": ...}.

    mode options:
      "transcribe" -> returns text in whatever language was spoken
      "translate"  -> returns text translated to English

    content_type: the REAL mime type of the audio (e.g. "audio/wav" for files you
    recorded yourself, "audio/webm" for browser microphone recordings — browsers
    record webm by default, not wav, so this must match or Sarvam may reject or
    mis-transcribe the file).
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set. Add it to your .env file.")

    with open(file_path, "rb") as f:
        files = {"file": (file_path, f, content_type)}
        data = {"model": "saaras:v3", "mode": mode}
        headers = {"api-subscription-key": SARVAM_API_KEY}

        response = requests.post(
            SARVAM_STT_URL, headers=headers, files=files, data=data, timeout=30
        )

    if response.status_code != 200:
        raise RuntimeError(f"Sarvam STT failed ({response.status_code}): {response.text}")

    result = response.json()
    return {
        "text": result.get("transcript", ""),
        "language": result.get("language_code", "unknown"),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stt.py path/to/audio.wav")
        sys.exit(1)

    out = transcribe_audio(sys.argv[1])
    print(f"Language detected: {out['language']}")
    print(f"Transcript: {out['text']}")