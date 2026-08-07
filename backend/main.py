import os
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

@app.on_event("startup")
async def startup_event():
    ensure_directory(settings.TEMP_DIR)
    logger.info(f"Started {settings.APP_NAME}")

@app.on_event("shutdown")
async def shutdown_event():
    # Cleanup temp dir on shutdown (optional)
    import shutil
    try:
        shutil.rmtree(settings.TEMP_DIR, ignore_errors=True)
    except:
        pass
    logger.info("Shutdown complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)