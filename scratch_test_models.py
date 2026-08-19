import requests, json, sys
sys.stdout.reconfigure(encoding='utf-8')
from config import GROQ_API_KEY, GROQ_MODEL

text = "కార్పొరేషన్లు వాటి స్టాక్ హోల్డర్ల (వాటాదారుల) యాజమాన్యంలో ఉంటాయి."
langs = {"en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu"}

for code, lang in langs.items():
    r = requests.post(
        'https://api.groq.com/openai/v1/chat/completions',
        headers={'Authorization': f'Bearer {GROQ_API_KEY}', 'Content-Type': 'application/json'},
        json={
            'model': GROQ_MODEL,
            'messages': [
                {'role': 'system', 'content': f'You are a professional multilingual translator. Translate the given text directly into natural {lang}. Output ONLY the translated text in {lang} native script. Do not output JSON, explanations, or quotes.'},
                {'role': 'user', 'content': text}
            ],
            'temperature': 0.1
        },
        timeout=12
    )
    print(code, lang, 'Status:', r.status_code)
    if r.status_code == 200:
        ans = r.json()['choices'][0]['message']['content'].strip()
        print(code, '->', ans)
    else:
        print('Error:', r.text)
