from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import router

app = FastAPI(title="AI Photo Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint – prevents Render from restarting
@app.get("/")
async def health_check():
    return {"status": "ok", "message": "AI Photo Studio is running"}

app.include_router(router)
