from typing import Dict, Any, List
from services.supabase_service import supabase_service
from services.embed_service import embed_service
import requests
import os

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def get_tool_schemas() -> List[Dict[str, Any]]:
    """
    Returns the JSON schemas for the tools available to the LLM.
    """
    return [
        {
            "name": "retrieve_kb",
            "description": "Search the internal PM knowledge base for theory and definitions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The concept to search for"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "web_search",
            "description": "Search the live web for real-world company examples or current project events using Tavily.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The specific company or event to search for"}
                },
                "required": ["query"]
            }
        }
    ]

def execute_tool(name: str, args: Dict[str, Any]) -> Any:
    """
    Dispatcher to execute the real tool logic.
    """
    print(f"--- [SERVICE] Executing Tool: {name} (Args: {args}) ---")
    
    if name == "retrieve_kb":
        query = args.get("query")
        # 1. Embed the query
        vector = embed_service.embed_text(query)
        # 2. Search Supabase
        results = supabase_service.search_knowledge_base(vector)
        return {
            "source": "knowledge_base",
            "results": results
        }
        
    elif name == "web_search":
        query = args.get("query")
        if not TAVILY_API_KEY:
            return {"error": "Tavily API key not configured"}
            
        try:
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "basic",
                "max_results": 3
            }
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return {
                "source": "web_search",
                "results": data.get("results", [])
            }
        except Exception as e:
            return {"error": f"Web search failed: {e}"}
    
    return {"error": "Tool not found"}
