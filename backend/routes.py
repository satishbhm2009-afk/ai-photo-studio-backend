from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from backend.models import DownloadRequest, ExtractionResponse, UploadResponse
from backend.downloader import SocialDownloader, validate_url
from backend.pipeline import process_video
from backend.utils import cleanup_temp_files, get_temp_file
from backend.logger import logger
import shutil
import os

router = APIRouter()

@router.post("/upload-video", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file directly.
    """
    if not file.filename.endswith(('.mp4', '.mov', '.avi', '.mkv')):
        raise HTTPException(400, detail="Unsupported file format. Please upload MP4, MOV, AVI, or MKV.")

    try:
        # Save uploaded file to temp
        temp_video = get_temp_file(suffix=".mp4")
        with open(temp_video, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return UploadResponse(success=True, video_path=temp_video, message="File uploaded successfully")
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(500, detail=str(e))

@router.post("/download-video", response_model=UploadResponse)
async def download_video(request: DownloadRequest):
    """
    Download video from a social media URL.
    """
    url = str(request.url)
    if not validate_url(url):
        raise HTTPException(400, detail="Unsupported URL or invalid format.")

    result = SocialDownloader.download(url, cookies=request.cookies)
    if not result["success"]:
        raise HTTPException(400, detail=result.get("error", "Download failed"))

    return UploadResponse(
        success=True,
        video_path=result["video_path"],
        message="Download successful"
    )

@router.post("/extract-top-frames", response_model=ExtractionResponse)
async def extract_top_frames(
    video_path: str,
    prompt: str = "a person looking good",
    num_frames: int = 10
):
    """
    Process a video (local path) and return top frames.
    """
    if not os.path.exists(video_path):
        raise HTTPException(404, detail="Video file not found")

    try:
        result = process_video(video_path, prompt, num_frames)
        # Cleanup the video file after processing
        cleanup_temp_files([video_path])
        if not result["success"]:
            raise HTTPException(400, detail=result.get("error", "Processing failed"))
        return ExtractionResponse(**result)
    except Exception as e:
        logger.error(f"Extraction error: {e}")
        cleanup_temp_files([video_path])
        raise HTTPException(500, detail=str(e))

@router.get("/health")
async def health():
    return {"status": "healthy"}

@router.get("/version")
async def version():
    return {"version": "1.0.0", "app": "AI Photo Studio"}