import os
import requests
from typing import List
from dotenv import load_dotenv

load_dotenv()

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("LLM_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
EMBED_TIMEOUT_SECONDS = float(os.getenv("EMBED_TIMEOUT_SECONDS", "6"))

class EmbedService:
    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        self.url = "https://openrouter.ai/api/v1/embeddings"

    def embed_text(self, text: str) -> List[float]:
        """
        Calls OpenRouter to get text embeddings.
        """
        if not self.api_key:
            print("[WARNING] OPENROUTER_API_KEY not found for embeddings")
            return [0.0] * 2048 # Default for nvidia/llama-nemotron-embed-vl-1b-v2:free
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": EMBEDDING_MODEL,
            "input": text
        }
        
        try:
            response = requests.post(
                self.url,
                headers=headers,
                json=payload,
                timeout=EMBED_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()
            embedding = data["data"][0]["embedding"]
            print(f"--- [DEBUG] Query Embedding Dimension: {len(embedding)} ---")
            return embedding
        except Exception as e:
            print(f"[ERROR] Embedding failed: {e}")
            return [0.0] * 2048

embed_service = EmbedService()
