import os
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import Optional, Any
from dotenv import load_dotenv

from services.llm_service import llm_service
from services.supabase_service import supabase_service
from middleware.auth import AuthMiddleware
import uvicorn

# Load Environment Variables
load_dotenv()


def get_cors_origins() -> list[str]:
    raw_origins = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
    if raw_origins:
        origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
        if origins:
            return origins

    frontend_origin = os.getenv("FRONTEND_ORIGIN", "").strip()
    if frontend_origin:
        return [frontend_origin]

    return ["*"]


def validate_env():
    required_keys = [
        "TAVILY_API_KEY", 
        "SUPABASE_URL", 
        "SUPABASE_SERVICE_ROLE_KEY"
    ]
    missing = [key for key in required_keys if not os.getenv(key)]
    if missing:
        print(f"[CRITICAL] Missing environment variables: {', '.join(missing)}")
        # In production, we might exit here
        # sys.exit(1)

validate_env()

app = FastAPI(title="PM Learning Agent API")

# Add Middlewares
cors_origins = get_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=cors_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AuthMiddleware)

# --- Models ---

class GenerateRequest(BaseModel):
    query: str
    user_id: Optional[str] = "anonymous"

class GenerateResponse(BaseModel):
    type: str  # "chat" or "case_study"
    content: Any  # CaseStudyCard dict or str
    status: str = "success"


class HistoryResponse(BaseModel):
    history: list[dict[str, Any]]

# --- Routes ---

@app.get("/")
async def root():
    return {"message": "PM Learning Agent API is online"}


@app.get("/api/health")
async def health():
    """
    Lightweight health endpoint for frontend connectivity checks.
    """
    return {
        "status": "ok",
        "service": "pm-learning-agent-api",
        "supabase_connected": bool(supabase_service.client),
    }

@app.post("/api/generate", response_model=GenerateResponse)
async def generate_card(request: GenerateRequest):
    """
    Core endpoint to generate a case study card or conversational response.
    """
    try:
        # LLM Service now handles internal Supabase DB tracking natively
        result = await run_in_threadpool(
            llm_service.process_query,
            request.query,
            request.user_id or "anonymous",
        )
        
        if isinstance(result, str):
            return GenerateResponse(type="chat", content=result)
        else:
            return GenerateResponse(type="case_study", content=result)
            
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
async def get_history(request: Request):
    """
    Retrieves history for the authenticated user.
    """
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required for history")
    
    try:
        history = supabase_service.get_user_history(user.id)
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history/public", response_model=HistoryResponse)
async def get_public_history(
    user_id: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50),
):
    """
    Prototype-friendly history endpoint for anonymous frontend sessions.
    """
    try:
        history = supabase_service.get_user_history(user_id, limit=limit)
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
