import os
import uuid
import shutil
import aiofiles
from typing import List
from fastapi import UploadFile, HTTPException
from backend.config import settings

def generate_unique_id() -> str:
    """Generates a secure unique session identifier."""
    return str(uuid.uuid4())

def get_file_extension(filename: str) -> str:
    """Extracts lowercase file extension without dot."""
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[1].lower()

def is_allowed_file(filename: str) -> bool:
    """Validates if file extension is supported."""
    ext = get_file_extension(filename)
    return ext in settings.ALLOWED_EXTENSIONS

def is_video_file(filename: str) -> bool:
    """Checks if filename has a video extension."""
    ext = get_file_extension(filename)
    return ext in settings.ALLOWED_VIDEO_EXTENSIONS

async def save_upload_file(upload_file: UploadFile, destination_path: str) -> str:
    """Saves an incoming FastAPI UploadFile asynchronously to disk."""
    async with aiofiles.open(destination_path, 'wb') as out_file:
        while content := await upload_file.read(1024 * 1024):  # Read in 1MB chunks
            await out_file.write(content)
    return destination_path

def cleanup_directory(dir_path: str):
    """Safely removes a directory and all contained files."""
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path, ignore_errors=True)