import json
import os
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from services.tool_service import execute_tool, get_tool_schemas
from models.card import CaseStudyCard
from dotenv import load_dotenv

load_dotenv()

# Configuration
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "qwen/qwen-2.5-72b-instruct:free")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")

class LLMService:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=PRIMARY_MODEL,
            api_key=LLM_API_KEY,
            base_url=BASE_URL,
            temperature=0.7
        )
        self.tools = get_tool_schemas()
        self.bound_llm = self.llm.bind_tools(self.tools)

    def process_query(self, query: str, user_id: str = "terminal_user") -> Any:
        import concurrent.futures
        from pydantic import BaseModel, Field
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        from services.supabase_service import supabase_service

        # 1. Fetch from Supabase
        db_history = supabase_service.get_conversation_history(user_id, limit=6)
        
        # 2. Format Native LangChain History
        formatted_history = []
        for msg in db_history:
            if msg["role"] == "user":
                formatted_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                formatted_history.append(AIMessage(content=msg["content"]))
                    
        class UserIntent(BaseModel):
            is_conversational: bool = Field(..., description="True if the user is chatting normally without explicitly asking to learn a PM topic.")
            pm_topic: str = Field(..., description="If is_conversational is False, the core PM concept to fetch from databases (e.g. 'Agile', 'Scrum').")

        print("[DEBUG] Agent is determining query intent...")
        intent_llm = self.llm.with_structured_output(UserIntent)
        
        try:
            router_messages = [SystemMessage(content="Determine user intent. Pay attention to conversation history to understand context.")] + formatted_history + [HumanMessage(content=query)]
            intent = intent_llm.invoke(router_messages)
        except Exception as e:
            print(f"[ERROR] Intent routing failed: {e}")
            return "I'm having trouble understanding right now. Please try again."
        
        # 3. Conversational Route
        if getattr(intent, 'is_conversational', False):
            print("[DEBUG] Chatbot routed to Conversational Reply.")
            convo_messages = [SystemMessage(content="You are a friendly Project Management Learning Assistant. Chat naturally.")] + formatted_history + [HumanMessage(content=query)]
            response_text = self.llm.invoke(convo_messages).content
            
            # Save to Supabase
            supabase_service.save_query_history(user_id, query, {"response": response_text}, "chat")
            return response_text

        # 4. High-Performance Single-Shot Case Study Pipeline
        topic = getattr(intent, 'pm_topic', query)
        if not topic or topic.lower() == "unspecified":
             topic = query
             
        print(f"[DEBUG] Chatbot routed to Case Study Synthesis. Topic: '{topic}'. Fetching RAG and Web Context concurrently...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            kb_future = executor.submit(execute_tool, "retrieve_kb", {"query": topic})
            web_future = executor.submit(execute_tool, "web_search", {"query": f"{topic} methodology real world company examples case study"})
            kb_result = kb_future.result()
            web_result = web_future.result()
            
        system_instruction = (
            "You are an expert Project Management Learning Assistant.\n"
            "Your goal is to help students understand PM concepts through real-world stories.\n"
            "Below is the theory and real-world database context fetched for the topic.\n"
            "Synthesize this strictly into a CaseStudyCard format.\n"
            "GROUND ALL STORIES IN REAL CONTEXT. Do not hallucinate names or dates.\n\n"
            f"--- KNOWLEDGE BASE ---\n{json.dumps(kb_result)}\n\n"
            f"--- WEB SEARCH ---\n{json.dumps(web_result)}\n"
        )
        
        synthesis_messages = [SystemMessage(content=system_instruction)] + formatted_history + [HumanMessage(content=query)]

        print("[DEBUG] Generating structured final Case Study Card (Single-Shot)...")
        structured_llm = self.llm.with_structured_output(CaseStudyCard)
        
        try:
            final_card = structured_llm.invoke(synthesis_messages)
            # Save to Supabase
            supabase_service.save_query_history(user_id, query, final_card.model_dump(), "case_study")
            return final_card
        except Exception as e:
            print(f"[ERROR] Structured synthesis failed: {e}. Attempting to fallback parse...")
            fallback_response = self.llm.invoke(synthesis_messages)
            content = fallback_response.content if hasattr(fallback_response, 'content') else str(fallback_response)
            if "```json" in content:
                content = content.split("```json")[-1].split("```")[0].strip()
            data = json.loads(content)
            card = CaseStudyCard.model_validate(data)
            
            supabase_service.save_query_history(user_id, query, card.model_dump(), "case_study")
            return card

llm_service = LLMService()
