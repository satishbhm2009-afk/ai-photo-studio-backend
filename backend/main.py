import os
import shutil
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import router
from backend.config import settings
from backend.logger import logger
from backend.utils import ensure_directory

app = FastAPI(title=settings.APP_NAME, debug=settings.DEBUG)

# CORS for frontend (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


# --------------------- HEALTH CHECK ---------------------
@app.get("/health")
async def health_check():
    """Lightweight endpoint for Render's port scan."""
    return {"status": "healthy"}


# --------------------- STARTUP / SHUTDOWN ---------------------
@app.on_event("startup")
async def startup_event():
    ensure_directory(settings.TEMP_DIR)
    logger.info(f"Started {settings.APP_NAME}")

    # OPTIONAL: Kick off heavy model loading in background
    # asyncio.create_task(load_heavy_models())


@app.on_event("shutdown")
async def shutdown_event():
    # Cleanup temp dir on shutdown
    try:
        shutil.rmtree(settings.TEMP_DIR, ignore_errors=True)
    except Exception:
        pass
    logger.info("Shutdown complete")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
