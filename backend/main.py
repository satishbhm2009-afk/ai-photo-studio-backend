import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.config import settings
from backend.routes import router as api_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Production Softonix AI Media Enhancement Engine",
    version="1.0.0"
)

# CORS Setup
origins = [origin.strip() for origin in settings.ALLOWED_ORIGINS.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if "*" not in origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Mount Static Results & Frontend Files
app.mount("/results", StaticFiles(directory=settings.RESULTS_DIR), name="results")

if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def serve_index():
    index_path = os.path.join("frontend", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "running", "app": settings.PROJECT_NAME}

@app.get("/health")
def root_health():
    return {"status": "healthy"}
