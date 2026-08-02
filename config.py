import os

class Settings:
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_DIR: str = os.path.join(BASE_DIR, "uploads")
    RESULT_DIR: str = os.path.join(BASE_DIR, "results")
    ALLOWED_VIDEO_EXTENSIONS: set = {".mp4", ".mov", ".avi", ".mkv"}

settings = Settings()

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.RESULT_DIR, exist_ok=True)
