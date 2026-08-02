import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Softonix AI Photo & Video Studio"
    API_V1_STR: str = "/api/v1"
    
    # Directory paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", os.path.join("backend", "uploads"))
    RESULTS_DIR: str = os.getenv("RESULTS_DIR", os.path.join("backend", "results"))
    TEMP_DIR: str = os.getenv("TEMP_DIR", os.path.join("backend", "temp"))
    
    # Processing limits & constraints
    MAX_CONTENT_LENGTH: int = 100 * 1024 * 1024  # 100MB max payload
    MAX_PHOTOS: int = 20
    MIN_PHOTOS: int = 1
    
    ALLOWED_IMAGE_EXTENSIONS: set = {"jpg", "jpeg", "png", "webp"}
    ALLOWED_VIDEO_EXTENSIONS: set = {"mp4", "mov", "avi"}
    ALLOWED_EXTENSIONS: set = ALLOWED_IMAGE_EXTENSIONS.union(ALLOWED_VIDEO_EXTENSIONS)
    
    # Provider options: mock, replicate, fal, huggingface
    AI_PROVIDER: str = os.getenv("AI_PROVIDER", "mock")
    REPLICATE_API_TOKEN: str = os.getenv("REPLICATE_API_TOKEN", "")
    FAL_KEY: str = os.getenv("FAL_KEY", "")
    HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY", "")
    
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "*")

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# Ensure mandatory directories exist
for path in [settings.UPLOAD_DIR, settings.RESULTS_DIR, settings.TEMP_DIR]:
    os.makedirs(path, exist_ok=True)
    gitkeep_file = os.path.join(path, ".gitkeep")
    if not os.path.exists(gitkeep_file):
        with open(gitkeep_file, "w") as f:
            f.write("")