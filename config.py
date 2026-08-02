import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "SOFTONIX AI Photo Studio"
    VERSION: str = "1.0.0"
    
    # Absolute paths for standard environment setup
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_DIR: str = os.path.join(BASE_DIR, "uploads")
    RESULT_DIR: str = os.path.join(BASE_DIR, "results")
    
    # Image constraints
    MAX_FILE_SIZE_MB: int = 15
    ALLOWED_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".webp"}
    JPEG_QUALITY: int = 95

    class Config:
        env_file = ".env"

settings = Settings()

# Ensure required folders exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.RESULT_DIR, exist_ok=True)"""
==================================================
SOFTONIX AI PHOTO STUDIO
Configuration
==================================================
"""

from pathlib import Path

# -------------------------------------------------
# Base Directory
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# -------------------------------------------------
# Storage
# -------------------------------------------------

UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# File Settings
# -------------------------------------------------

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}

MAX_UPLOAD_SIZE = 10 * 1024 * 1024   # 10 MB

# -------------------------------------------------
# Image Export
# -------------------------------------------------

JPEG_QUALITY = 95
PNG_COMPRESSION = 3

# -------------------------------------------------
# AI Processing Options
# -------------------------------------------------

ENABLE_AUTO_CONTRAST = True
ENABLE_DENOISE = True
ENABLE_SHARPEN = True
ENABLE_COLOR_BOOST = True

# -------------------------------------------------
# API
# -------------------------------------------------

API_TITLE = "Softonix AI Photo Studio API"
API_VERSION = "1.0.0"

# Frontend URL (CORS)
ALLOWED_ORIGINS = [
    "*"
]

# -------------------------------------------------
# Cleanup
# -------------------------------------------------

AUTO_DELETE_HOURS = 24
