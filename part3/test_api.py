import os
from dotenv import load_dotenv
import requests

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

url = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": "llama-3.1-8b-instant",
    "messages": [
        {"role": "user", "content": "Say hello in one sentence."}
    ]
}

response = requests.post(url, headers=headers, json=payload)
print(response.status_code)
print(response.json())