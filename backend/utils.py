import os
import shutil
import tempfile
import hashlib
import base64
from typing import List
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import io

from backend.config import settings
from backend.logger import logger


def get_temp_file(suffix: str = ".mp4") -> str:
    """Create a temporary file with given suffix."""
    fd, path = tempfile.mkstemp(suffix=suffix, dir=settings.TEMP_DIR)
    os.close(fd)
    return path


def cleanup_temp_files(paths: List[str]) -> None:
    """Safely delete temporary files and directories."""
    for p in paths:
        if p and os.path.exists(p):
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    os.unlink(p)
            except Exception as e:
                logger.warning(f"Cleanup failed for {p}: {e}")


def image_to_base64(image: np.ndarray, format: str = "JPEG") -> str:
    """Convert OpenCV image (BGR) to base64 string."""
    if image.ndim == 3 and image.shape[2] == 3:
        # BGR -> RGB for PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    pil_img = Image.fromarray(image_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def compute_hash(image: np.ndarray, hash_size: int = 8) -> str:
    """Perceptual hash (dHash) for duplicate detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return hashlib.md5(diff.tobytes()).hexdigest()


def ensure_directory(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
