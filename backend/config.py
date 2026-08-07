import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # General
    APP_NAME: str = "AI Photo Studio"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Directories
    TEMP_DIR: str = "/tmp/ai_photo_studio"
    MAX_UPLOAD_SIZE: int = 1024 * 1024 * 1024  # 1 GB

    # MediaPipe
    MP_MIN_DETECTION_CONFIDENCE: float = 0.5
    MP_MIN_TRACKING_CONFIDENCE: float = 0.5

    # CLIP model
    CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"
    CLIP_DEVICE: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

    # Pipeline defaults
    DEFAULT_NUM_FRAMES: int = 10
    FRAME_SAMPLE_INTERVAL: float = 0.5  # seconds
    SCENE_CHANGE_THRESHOLD: float = 0.3
    BLUR_THRESHOLD: float = 100.0
    DUPLICATE_HASH_SIZE: int = 8

    # Downloader
    YT_DLP_PATH: str = "yt-dlp"
    DOWNLOAD_TIMEOUT: int = 300
    MAX_RETRIES: int = 3

    # Cache
    CACHE_TTL: int = 3600

    class Config:
        env_file = ".env"

settings = Settings()