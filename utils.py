import os
import uuid
from fastapi import UploadFile, HTTPException
from config import settings

def validate_image_file(file: UploadFile):
    """Validates file extension and size limit."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported image format. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )
    return ext

def generate_unique_filename(extension: str) -> str:
    """Generates a secure unique filename using UUID4."""
    return f"{uuid.uuid4().hex}{extension}"

def cleanup_file(filepath: str):
    """Removes temporary files safely."""
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except Exception:
            pass
