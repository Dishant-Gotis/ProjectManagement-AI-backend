import os
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") # Use Service Role for RAG

class SupabaseService:
    def __init__(self):
        if not SUPABASE_URL or not SUPABASE_KEY:
            print("[WARNING] Supabase credentials not found in .env")
            self.client = None
        else:
            self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    def search_knowledge_base(self, query_vector: List[float], threshold: float = 0.5, limit: int = 5) -> List[Dict]:
        """
        Performs vector search in the knowledge_base table.
        """
        if not self.client:
            return []
        
        try:
            rpc_params = {
                "query_embedding": query_vector,
                "match_threshold": threshold,
                "match_count": limit
            }
            response = self.client.rpc("match_knowledge_base", rpc_params).execute()
            return response.data
        except Exception as e:
            print(f"[ERROR] KB search failed: {e}")
            return []

    def save_query_history(self, user_id: str, query: str, card_json: Dict, route: str):
        """
        Persists a generated card to query_history.
        """
        if not self.client:
            return
        
        try:
            data = {
                "user_id": user_id,
                "query": query,
                "card_json": card_json,
                "route": route
            }
            self.client.table("query_history").insert(data).execute()
        except Exception as e:
            print(f"[ERROR] Saving history failed: {e}")

    def get_user_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """
        Retrieves recent query history for a user.
        """
        if not self.client:
            return []
        
        try:
            response = self.client.table("query_history") \
                .select("query, card_json, route, created_at") \
                .eq("user_id", user_id) \
                .order("created_at", desc=True) \
                .limit(limit) \
                .execute()
            return response.data
        except Exception as e:
            print(f"[ERROR] Fetching history failed: {e}")
            return []

    def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """
        Retrieves and formats history for LangChain context window.
        """
        if not self.client:
            return []
            
        try:
            # Fetch last N turns, ordered descending from database, then reverse for chronological order
            response = self.client.table("query_history") \
                .select("query, card_json") \
                .eq("user_id", user_id) \
                .order("created_at", desc=True) \
                .limit(limit) \
                .execute()
                
            history = []
            for row in reversed(response.data):
                # User turn
                if row.get("query"):
                    history.append({"role": "user", "content": row["query"]})
                
                # Assistant turn
                card = row.get("card_json") or {}
                if "response" in card:
                    history.append({"role": "assistant", "content": card["response"]})
                elif "concept" in card:
                    history.append({"role": "assistant", "content": f"Generated Case Study: {card['concept']}"})
                    
            return history
        except Exception as e:
            print(f"[ERROR] Fetching conversation failed: {e}")
            return []

supabase_service = SupabaseService()
