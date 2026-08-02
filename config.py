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
os.makedirs(settings.RESULT_DIR, exist_ok=True)
