import logging
import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------
# Logging
# ---------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("AI-Photo-Studio")

logger.info("=" * 60)
logger.info("Starting AI Photo Studio Backend")
logger.info("Python Version : %s", sys.version)
logger.info("Working Dir    : %s", os.getcwd())
logger.info("=" * 60)

# ---------------------------------------------------
# Dependency Diagnostics
# ---------------------------------------------------

try:
    import cv2

    logger.info("OpenCV Version : %s", cv2.__version__)

except Exception as e:
    logger.exception("OpenCV failed to load")
    raise

try:
    import mediapipe as mp

    logger.info("MediaPipe Version : %s", getattr(mp, "__version__", "Unknown"))
    logger.info("MediaPipe File    : %s", getattr(mp, "__file__", "Unknown"))
    logger.info("Has solutions     : %s", hasattr(mp, "solutions"))

except Exception:
    logger.exception("MediaPipe import failed")
    raise

try:
    import torch

    logger.info("Torch Version : %s", torch.__version__)
    logger.info("CUDA Available : %s", torch.cuda.is_available())

except Exception:
    logger.exception("Torch import failed")
    raise

# ---------------------------------------------------
# Import API Routes
# ---------------------------------------------------

from backend.routes import router

# ---------------------------------------------------
# FastAPI
# ---------------------------------------------------

app = FastAPI(
    title="AI Photo Studio Pro",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# Routes
# ---------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "running",
        "service": "AI Photo Studio Pro",
        "version": "2.0.0",
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "opencv": True,
        "mediapipe": True,
        "torch": True,
    }


app.include_router(router)

logger.info("Backend started successfully.")
