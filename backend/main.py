from fastapi import FastAPI
from .routes import router

app = FastAPI()
app.include_router(router)
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import router

app = FastAPI(title="AI Photo Studio Pro")

# CORS – allow all origins (no credentials required)
# This fixes the "CORS Missing Allow Origin" error.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Allows any frontend domain
    allow_credentials=False,      # Set to False because "*" cannot be used with True
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check – prevents Render from restarting on startup delays
@app.get("/")
async def health_check():
    return {"status": "ok", "message": "AI Photo Studio is running"}

app.include_router(router)
