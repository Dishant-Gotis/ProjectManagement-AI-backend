from fastapi import Request, HTTPException
from services.supabase_service import supabase_service
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

class AuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate Supabase JWT and inject user_id into Request state.
    """
    async def dispatch(self, request: Request, call_next):
        # Exclude root and public endpoints
        if request.url.path in [
            "/",
            "/docs",
            "/openapi.json",
            "/api/generate",
            "/api/health",
            "/api/history/public",
        ]:
            # Note: For prototype, /api/generate is public or uses anonymous user_id
            return await call_next(request)
            
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"detail": "Missing or invalid authorization header"})
            
        token = auth_header.split(" ")[1]
        
        try:
            # Verify token with Supabase
            if not supabase_service.client:
                raise Exception("Supabase client not initialized")
                
            user_response = supabase_service.client.auth.get_user(token)
            if not user_response or not user_response.user:
                raise Exception("Invalid user token")
                
            # Inject user info into request state
            request.state.user = user_response.user
            return await call_next(request)
            
        except Exception as e:
            return JSONResponse(status_code=401, content={"detail": f"Authentication failed: {str(e)}"})
