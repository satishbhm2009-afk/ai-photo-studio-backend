from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import router

app = FastAPI(title="AI Photo Studio")

# CORS – allow all origins for now (update to your domain in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root health endpoint – prevents Render from restarting due to 404 on /
@app.get("/")
async def health_check():
    return {"status": "ok", "message": "AI Photo Studio is running"}

# Include all API routes
app.include_router(router)
