"""
==================================================
SOFTONIX AI PHOTO STUDIO
Utility Functions
==================================================
"""

import os
import uuid
import shutil
from pathlib import Path

from config import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE,
    UPLOAD_FOLDER,
    RESULT_FOLDER
)


# ==================================================
# FILE VALIDATION
# ==================================================

def is_allowed_file(filename: str) -> bool:
    """
    Check if uploaded file extension is allowed.
    """

    extension = Path(filename).suffix.lower()

    return extension in ALLOWED_EXTENSIONS


# ==================================================
# FILE SIZE VALIDATION
# ==================================================

def validate_file_size(size: int) -> bool:
    """
    Validate uploaded file size.
    """

    return size <= MAX_FILE_SIZE


# ==================================================
# RANDOM FILE NAME
# ==================================================

def generate_filename(original_name: str) -> str:
    """
    Generate unique filename.
    """

    extension = Path(original_name).suffix.lower()

    return f"{uuid.uuid4().hex}{extension}"


# ==================================================
# SAVE UPLOADED FILE
# ==================================================

def save_upload(upload_file, filename: str) -> Path:
    """
    Save uploaded file.
    """

    destination = UPLOAD_FOLDER / filename

    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return destination


# ==================================================
# RESULT PATH
# ==================================================

def result_path(filename: str) -> Path:
    """
    Result image path.
    """

    return RESULT_FOLDER / filename


# ==================================================
# FILE EXISTS
# ==================================================

def file_exists(path: Path) -> bool:
    """
    Check file exists.
    """

    return path.exists()


# ==================================================
# DELETE FILE
# ==================================================

def delete_file(path: Path):
    """
    Delete file safely.
    """

    try:

        if path.exists():

            path.unlink()

    except Exception:

        pass


# ==================================================
# CLEAN TEMP FILES
# ==================================================

def cleanup():
    """
    Placeholder for automatic cleanup.
    """

    return True
