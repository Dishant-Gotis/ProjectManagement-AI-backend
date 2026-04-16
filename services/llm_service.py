import concurrent.futures
import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from models.card import CaseStudyCard
from services.tool_service import execute_tool, get_tool_schemas

load_dotenv()


def _env_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


# Configuration
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "qwen/qwen-2.5-72b-instruct:free")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "1"))
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "4"))
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "220"))
TOOL_TIMEOUT_SECONDS = float(os.getenv("TOOL_TIMEOUT_SECONDS", "8"))
ENABLE_WEB_SEARCH = _env_flag("ENABLE_WEB_SEARCH", True)
ENABLE_GREETING_FAST_PATH = _env_flag("ENABLE_GREETING_FAST_PATH", True)

PM_HINT_TERMS = {
    "agile",
    "scrum",
    "kanban",
    "waterfall",
    "stakeholder",
    "sprint",
    "backlog",
    "retrospective",
    "risk",
    "scope",
    "timeline",
    "gantt",
    "milestone",
    "pmp",
    "evm",
    "critical path",
    "project management",
}

LEARNING_TRIGGERS = (
    "what is",
    "explain",
    "tell me about",
    "how does",
    "difference between",
    "case study",
    "example",
    "best practice",
)

GREETING_TERMS = {
    "hi",
    "hii",
    "hello",
    "hey",
    "yo",
    "good morning",
    "good evening",
    "how are you",
    "whats up",
    "what's up",
}


class LLMService:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=PRIMARY_MODEL,
            api_key=LLM_API_KEY,
            base_url=BASE_URL,
            temperature=0.5,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_RETRIES,
        )
        self.chat_llm = self.llm.bind(max_tokens=CHAT_MAX_TOKENS)
        self.tools = get_tool_schemas()
        self.bound_llm = self.llm.bind_tools(self.tools)

    @staticmethod
    def _extract_text(response: Any) -> str:
        if response is None:
            return ""
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, str):
                return content
            return str(content)
        return str(response)

    @staticmethod
    def _is_short_greeting(query: str) -> bool:
        lowered = query.strip().lower()
        compact = " ".join(lowered.split())
        if compact in GREETING_TERMS:
            return True
        tokens = compact.split()
        return len(tokens) <= 3 and any(token in GREETING_TERMS for token in tokens)

    @staticmethod
    def _should_generate_case_study(query: str) -> bool:
        lowered = " ".join(query.strip().lower().split())
        if not lowered:
            return False

        if any(trigger in lowered for trigger in LEARNING_TRIGGERS):
            return True

        if any(term in lowered for term in PM_HINT_TERMS):
            return True

        return len(lowered.split()) >= 8 and "?" in lowered

    @staticmethod
    def _safe_future_result(
        future: concurrent.futures.Future,
        timeout_seconds: float,
        fallback: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            result = future.result(timeout=timeout_seconds)
            if isinstance(result, dict):
                return result
            return {"result": result}
        except Exception as exc:
            timeout_fallback = dict(fallback)
            timeout_fallback["warning"] = f"tool call failed or timed out: {exc}"
            return timeout_fallback

    def _load_history(self, user_id: str) -> List[Any]:
        from services.supabase_service import supabase_service

        db_history = supabase_service.get_conversation_history(user_id, limit=HISTORY_LIMIT)
        formatted_history: List[Any] = []
        for message in db_history:
            if message.get("role") == "user":
                formatted_history.append(HumanMessage(content=message.get("content", "")))
            elif message.get("role") == "assistant":
                formatted_history.append(AIMessage(content=message.get("content", "")))
        return formatted_history

    def process_query(self, query: str, user_id: str = "terminal_user") -> Any:
        from services.supabase_service import supabase_service

        normalized_query = (query or "").strip()
        if not normalized_query:
            return "Please share a project management question, and I will help."

        formatted_history = self._load_history(user_id)

        if ENABLE_GREETING_FAST_PATH and self._is_short_greeting(normalized_query):
            response_text = (
                "Hi there. Ask me any project management topic and I can explain it or generate a practical case study."
            )
            supabase_service.save_query_history(user_id, normalized_query, {"response": response_text}, "chat")
            return response_text

        if not self._should_generate_case_study(normalized_query):
            print("[DEBUG] Routed to conversational reply (fast path).")
            conversation_messages = [
                SystemMessage(
                    content=(
                        "You are a friendly Project Management Learning Assistant. "
                        "Keep responses concise unless the user asks for depth."
                    )
                )
            ] + formatted_history + [HumanMessage(content=normalized_query)]

            response = self.chat_llm.invoke(conversation_messages)
            response_text = self._extract_text(response).strip()
            if not response_text:
                response_text = "I am here and ready to help with any project management question."

            supabase_service.save_query_history(user_id, normalized_query, {"response": response_text}, "chat")
            return response_text

        topic = normalized_query
        print(f"[DEBUG] Routed to case study synthesis. Topic: '{topic}'")

        web_query = f"{topic} methodology real world company examples case study"
        max_workers = 2 if ENABLE_WEB_SEARCH else 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            kb_future = executor.submit(execute_tool, "retrieve_kb", {"query": topic})
            kb_result = self._safe_future_result(
                kb_future,
                TOOL_TIMEOUT_SECONDS,
                {"source": "knowledge_base", "results": []},
            )

            if ENABLE_WEB_SEARCH:
                web_future = executor.submit(execute_tool, "web_search", {"query": web_query})
                web_result = self._safe_future_result(
                    web_future,
                    TOOL_TIMEOUT_SECONDS,
                    {"source": "web_search", "results": []},
                )
            else:
                web_result = {"source": "web_search", "results": [], "note": "disabled by config"}

        system_instruction = (
            "You are an expert Project Management Learning Assistant.\n"
            "Generate a CaseStudyCard grounded in the provided context.\n"
            "If evidence is weak, stay transparent and avoid fabricated details.\n\n"
            f"--- KNOWLEDGE BASE ---\n{json.dumps(kb_result)}\n\n"
            f"--- WEB SEARCH ---\n{json.dumps(web_result)}\n"
        )

        synthesis_messages = [SystemMessage(content=system_instruction)] + formatted_history + [
            HumanMessage(content=normalized_query)
        ]

        structured_llm = self.llm.with_structured_output(CaseStudyCard)
        try:
            final_card = structured_llm.invoke(synthesis_messages)
            supabase_service.save_query_history(user_id, normalized_query, final_card.model_dump(), "case_study")
            return final_card
        except Exception as exc:
            print(f"[ERROR] Structured synthesis failed: {exc}. Falling back to text response.")
            fallback_response = self.chat_llm.invoke(synthesis_messages)
            content = self._extract_text(fallback_response)

            if "```json" in content:
                content = content.split("```json")[-1].split("```")[0].strip()

            try:
                data = json.loads(content)
                card = CaseStudyCard.model_validate(data)
                supabase_service.save_query_history(user_id, normalized_query, card.model_dump(), "case_study")
                return card
            except Exception:
                fallback_text = content.strip() or "I found context but could not format a case study this time."
                supabase_service.save_query_history(user_id, normalized_query, {"response": fallback_text}, "chat")
                return fallback_text

llm_service = LLMService()
