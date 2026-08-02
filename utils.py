"""
==================================================
SOFTONIX AI PHOTO STUDIO
Utility Functions
==================================================
"""

import uuid
import shutil
from pathlib import Path
from fastapi import UploadFile

from config import (
    ALLOWED_EXTENSIONS,
    MAX_UPLOAD_SIZE,
    UPLOAD_DIR,
    RESULT_DIR
)


# ==================================================
# VALIDATE FILE EXTENSION
# ==================================================

def is_allowed_file(filename: str) -> bool:
    """
    Check if file extension is allowed.
    """

    extension = Path(filename).suffix.lower()

    return extension in ALLOWED_EXTENSIONS


# ==================================================
# VALIDATE FILE SIZE
# ==================================================

def is_valid_size(size: int) -> bool:
    """
    Check upload size.
    """

    return size <= MAX_UPLOAD_SIZE


# ==================================================
# GENERATE UNIQUE NAME
# ==================================================

def generate_filename(filename: str) -> str:
    """
    Create random filename.
    """

    extension = Path(filename).suffix.lower()

    return f"{uuid.uuid4().hex}{extension}"


# ==================================================
# SAVE FILE
# ==================================================

def save_upload_file(file: UploadFile, filename: str) -> Path:
    """
    Save uploaded image.
    """

    destination = UPLOAD_DIR / filename

    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return destination


# ==================================================
# RESULT PATH
# ==================================================

def get_result_path(filename: str) -> Path:
    """
    Result image location.
    """

    return RESULT_DIR / filename


# ==================================================
# FILE EXISTS
# ==================================================

def file_exists(path: Path) -> bool:

    return path.exists()


# ==================================================
# DELETE FILE
# ==================================================

def delete_file(path: Path):

    try:

        if path.exists():

            path.unlink()

    except Exception:

        pass


# ==================================================
# CLEANUP PLACEHOLDER
# ==================================================

def cleanup_old_files():

    """
    Future automatic cleanup.
    """

    return True
