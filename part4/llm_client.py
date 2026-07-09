import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get('GROQ_API_KEY')
URL = "https://api.groq.com/openai/v1/chat/completions"

def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(URL, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return None
    return response.json()['choices'][0]['message']['content']


if __name__ == "__main__":
    result = call_llm("You are a helpful assistant.", "Reply with only the word: hello")
    print(result)