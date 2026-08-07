from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any
from datetime import datetime

class VideoInfo(BaseModel):
    url: Optional[str] = None
    title: Optional[str] = None
    duration: Optional[float] = None
    resolution: Optional[str] = None
    format: Optional[str] = None
    size: Optional[int] = None

class FrameResult(BaseModel):
    index: int
    timestamp: float
    score: float = Field(..., description="Overall combined score")
    quality: float = Field(..., description="Sharpness/brightness/face quality")
    semantic: float = Field(..., description="CLIP similarity to prompt")
    image_base64: str

class ExtractionResponse(BaseModel):
    success: bool
    total_frames: int
    selected_frames: List[FrameResult]
    processing_time: float
    video_info: Optional[VideoInfo] = None
    error: Optional[str] = None

class DownloadRequest(BaseModel):
    url: HttpUrl
    cookies: Optional[Dict[str, str]] = None

class UploadResponse(BaseModel):
    success: bool
    video_path: Optional[str] = None
    message: str
