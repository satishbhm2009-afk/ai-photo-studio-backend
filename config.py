"""
==================================================
SOFTONIX AI PHOTO STUDIO
Configuration File
==================================================
"""

from pathlib import Path

# ==================================================
# BASE DIRECTORY
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

# ==================================================
# FOLDERS
# ==================================================

UPLOAD_FOLDER = BASE_DIR / "uploads"
RESULT_FOLDER = BASE_DIR / "results"

UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULT_FOLDER.mkdir(exist_ok=True)

# ==================================================
# IMAGE SETTINGS
# ==================================================

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}

MAX_FILE_SIZE = 10 * 1024 * 1024      # 10 MB

JPEG_QUALITY = 95
PNG_COMPRESSION = 3

# ==================================================
# AI PROCESSING
# ==================================================

ENABLE_AUTO_CONTRAST = True
ENABLE_DENOISE = True
ENABLE_SHARPEN = True
ENABLE_COLOR_BOOST = True

# ==================================================
# API
# ==================================================

API_TITLE = "Softonix AI Photo Studio API"

API_VERSION = "1.0.0"

# ==================================================
# CLEANUP
# ==================================================

AUTO_DELETE_AFTER_HOURS = 24
