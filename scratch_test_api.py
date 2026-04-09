import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL")

print(f"Testing with Model: {PRIMARY_MODEL}")

response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    },
    json={
        "model": PRIMARY_MODEL,
        "messages": [
            {"role": "user", "content": "Say hello world"}
        ]
    }
)

print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")
