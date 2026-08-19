import sys, requests, json
sys.stdout.reconfigure(encoding='utf-8')
from config import GROQ_API_KEY, GROQ_MODEL

text = 'కార్పొరేషన్లు వాటి స్టాక్ హోల్డర్ల (వాటాదారుల) యాజమాన్యంలో ఉంటాయి.'
langs = {'en': 'English', 'hi': 'Hindi (हिन्दी)', 'ta': 'Tamil (தமிழ்)', 'te': 'Telugu (తెలుగు)'}

for code, name in langs.items():
    sys_msg = f"You are a professional multilingual translator. Translate the text into {name}. You MUST write in the native script of {name}. Respond with ONLY a JSON object: {{\"translated_text\": \"<translated text>\"}}"
    resp = requests.post(
        'https://api.groq.com/openai/v1/chat/completions',
        headers={'Authorization': f'Bearer {GROQ_API_KEY}'},
        json={
            'model': GROQ_MODEL,
            'messages': [
                {'role': 'system', 'content': sys_msg},
                {'role': 'user', 'content': f"Translate this text to {name}:\n{text}"}
            ],
            'temperature': 0.1,
            'response_format': {'type': 'json_object'}
        },
        timeout=10
    )
    d = json.loads(resp.json()['choices'][0]['message']['content'])
    print(code, name, '->', d.get('translated_text'))
